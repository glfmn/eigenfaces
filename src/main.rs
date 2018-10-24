#[macro_use]
extern crate clap;
extern crate gnuplot;
extern crate image;
extern crate nalgebra as na;
extern crate rand;
extern crate typenum as tn;

use gnuplot::{Figure, Caption, Color, AxesCommon};
use image::{DynamicImage, GenericImageView};
use rand::distributions::Uniform;
use rand::prelude::*;
use std::fs::{self, File};
use std::ops::RangeInclusive;

/// Dynamically sized matrix to store images
///
/// Fallback to heap allocation for image that are a different size than the 112x92 pixels from the
/// AT&T databse of faces training data set.
type ImageVector = na::Matrix<f64, na::Dynamic, na::U1, na::MatrixVec<f64, na::Dynamic, na::U1>>;

type DataSet = na::DMatrix<f64>;

fn image_vector(image: DynamicImage) -> ImageVector {
    let image = image.to_luma();
    let (w, h) = image.dimensions();
    let len = (w * h) as usize;
    let pixels = image.pixels().map(|p| p[0] as f64);
    ImageVector::from_iterator(len, pixels)
}

/// Read a dataset from the given path
///
/// Expects the path to contain a folder for each set of images, where the folders are called s
/// with the set number appended to it.  In other words, `s1` is set 1.  The inclusive range sets
/// determines
fn read_dataset<S>(
    path: S,
    sets: RangeInclusive<usize>,
    range: RangeInclusive<usize>,
) -> Result<(DataSet, (u32, u32)), String>
where
    S: AsRef<str> + std::fmt::Display,
{
    let mut images: Vec<ImageVector> = Vec::with_capacity(sets.end() - sets.start() + 1);

    let mut bounds = (0, 0);

    // Stride the range of included sets in this dataset
    for s in sets {
        // Load the images from the set which correspond to the training set
        for i in range.clone() {
            if let Ok(img) = image::open(format!("{}/s{}/{}.pgm", path, s, i)) {
                bounds = img.dimensions();
                images.push(image_vector(img));
            } else {
                return Err(format!("Failed to open image s{}/{}.pgm", s, i));
            }
        }
    }

    Ok((DataSet::from_columns(images.as_slice()), bounds))
}

/// Calculate the mean image from the training set
///
/// Mean image is the mean for each pixel across the entire data set.
fn mean_image(data_set: &DataSet) -> ImageVector {
    let mut mean = ImageVector::zeros(data_set.nrows());

    let size = data_set.ncols() as f64;
    for r in 0..data_set.nrows() {
        let sum: f64 = data_set.row(r).iter().sum();
        mean[r] = sum / size;
    }

    mean
}

/// Center a dataset around the mean image
fn mean_center<'a>(mean_image: &ImageVector, data_set: &'a mut DataSet) -> &'a DataSet {
    for c in 0..data_set.ncols() {
        let mut image = data_set.column_mut(c);
        image -= mean_image;
    }

    data_set
}

/// Sort the eigenvectors and eigenvalues by the magnitude of the eigenvalues
fn sort_by_eigenvalue(values: &mut na::DVector<f64>, vectors: &mut na::DMatrix<f64>) {
    fn greater(a: f64, b: f64) -> bool {
        match (a, b) {
            (x, y) if x.is_nan() || y.is_nan() => panic!("NaN found in sort"),
            (_, _) => a < b,
        }
    }

    for e in 1..values.nrows() {
        let mut j = e;
        while j > 0 && greater(values[j - 1], values[j]) {
            values.swap_rows(j - 1, j);
            vectors.swap_rows(j - 1, j);
            j -= 1;
        }
    }
}

/// Get the principal components of a data_set
fn image_eigen(data_set: &DataSet) -> (DataSet, na::DVector<f64>) {
    let covariance = data_set.transpose() * data_set.clone();
    let na::SymmetricEigen {
        mut eigenvectors,
        mut eigenvalues,
    } = covariance.symmetric_eigen();
    sort_by_eigenvalue(&mut eigenvalues, &mut eigenvectors);

    let eigen = data_set.clone() * eigenvectors;

    (eigen, eigenvalues)
}

fn image_pca(basis_size: usize, basis: DataSet) -> DataSet {
    let size = basis.ncols();
    if basis_size >= size {
        panic!("Basis size larger than number of images in data set");
    }

    basis.remove_columns(basis_size - 1, size as usize - basis_size)
}

fn calculate_image<R, C, S>(
    image: &na::Matrix<f64, R, C, S>,
    min: f64,
    max: f64,
) -> Vec<u8>
where
    R: na::Dim,
    C: na::Dim,
    S: na::storage::Storage<f64, R, C>,
{
    image
        .iter()
        .map(|p| ((*p - min) / (max - min) * 255f64) as u8)
        .collect()
}

fn write_image<P>(path: P, buffer: &Vec<u8>, w: u32, h: u32) -> std::io::Result<()>
where
    P: AsRef<std::path::Path> + std::fmt::Display,
{
    use image::{jpeg, ColorType};
    println!(" create {}", path);
    let mut file = File::create(path).unwrap();
    jpeg::JPEGEncoder::new(&mut file).encode(buffer.as_slice(), w, h, ColorType::Gray(8))?;

    Ok(())
}

fn pca_load<S>(image: &na::Matrix<f64, na::Dynamic, na::U1, S>, pca: &DataSet) -> na::DVector<f64>
where
    S: na::storage::Storage<f64, na::Dynamic, na::U1>,
{
    let mut pca_load = na::DVector::<f64>::from_element(pca.ncols(), 0.);

    for c in 0..pca.ncols() {
        let pc = pca.column(c);
        pca_load[c] = (pc.transpose() * image)[0];
    }

    pca_load
}

fn pca_project<S>(
    pca_loading: &na::Matrix<f64, na::Dynamic, na::U1, S>,
    pca: &DataSet
) -> ImageVector
where
    S: na::storage::Storage<f64, na::Dynamic, na::U1>,
{

    let mut projected = ImageVector::from_element(pca.nrows(), 0.);
    for c in 0..pca.ncols() {
        let pc = pca.column(c);
        projected += pca_loading[c] * pc;
    }

    projected
}

struct Mahalanobis<'a> {
    eigenvalues: &'a na::DVector<f64>,
}

impl<'a> Mahalanobis<'a> {
    pub fn new(eigenvalues: &'a na::DVector<f64>) -> Self {
        // Mahalanobis distance
        Mahalanobis { eigenvalues }
    }

    /// Claculate the malahanobis distance for two vectors
    pub fn distance<S>(
        &self,
        x: &na::Matrix<f64, na::Dynamic, na::U1, S>,
        y: &na::Matrix<f64, na::Dynamic, na::U1, S>,
    ) -> f64
    where
        S: na::storage::Storage<f64, na::Dynamic, na::U1>,
    {
        (x - y)
            .iter()
            .enumerate()
            .fold(0., |acc, (i, d)| acc + (1. / self.eigenvalues[i]) * d * d)
            .abs()
            .sqrt()
    }

    /// Calculate Mahalanobis distance for each column vector in each matrix
    pub fn distance_matrix(&self, x: &na::DMatrix<f64>, y: &na::DMatrix<f64>) -> na::DMatrix<f64> {
        let mut distance = na::DMatrix::<f64>::from_element(x.ncols(), y.ncols(), 0.);

        for i in 0..x.ncols() {
            for j in 0..y.ncols() {
                distance[(i, j)] = self.distance(&x.column(i), &y.column(j));
            }
        }

        distance
    }
}

fn main() {
    let matches = clap_app!(eigenfaces =>
        (version: crate_version!())
        (author: crate_authors!())
        (about: crate_description!())
        (@arg DATA: "Path to the data set")
        (@arg OUTPUT: -o --output +takes_value "Path to place output files")
    ).get_matches();

    let input = matches.value_of("DATA").unwrap_or("orl_faces");
    let output = matches.value_of("OUTPUT").unwrap_or("target/eigenfaces");

    let mut errors = vec![pca_main(true, input, &format!("{}/a", output), 50, (3..=6, 1..=2, 7..=10))];
    for s in vec![40, 30, 20, 10, 5] {
        errors.push(pca_main(false, input, &format!("{}/b/{}", output, s), s, (3..=6, 1..=2, 7..=10)));
    }
    pca_main(true, input, &format!("{}/c", output), 50, (1..=4, 9..=10, 5..=8));

    let mut fg = Figure::new();
    fg.set_terminal("epscairo", format!("{}/error.eps", output).as_str())
        .axes2d()
        .set_title("Error vs PCA Size", &[])
        .lines(
            &[50f64, 40., 30., 20., 10., 5.],
            errors.as_slice(),
            &[Caption("Error"), Color("red")]
        );
    fg.show();

    fn pca_main<P>(
        write_faces: bool,
        input: P,
        output: P,
        pca_size: usize,
        experiment: (RangeInclusive<usize>, RangeInclusive<usize>, RangeInclusive<usize>)
    ) -> f64
    where P: AsRef<std::path::Path> + std::fmt::Display + AsRef<str> + Clone
    {
        fs::create_dir_all(format!("{}", output)).unwrap();
        fs::create_dir_all(format!("{}/eigenfaces", output)).unwrap();
        fs::create_dir_all(format!("{}/matched", output)).unwrap();

        // Load data-sets
        let (mut training_set, (w, h)) = read_dataset(input.clone(), 1..=40, experiment.0).unwrap();
        let (testing_set, _) = read_dataset(input.clone(), 1..=40, experiment.1).unwrap();
        let (gallery_set, _) = read_dataset(input.clone(), 1..=40, experiment.2).unwrap();

        let mean = mean_image(&training_set);

        if write_faces {
            let buffer: Vec<u8> = mean.iter().map(|p| *p as u8).collect();
            write_image(format!("{}/mean_image.jpg", output), &buffer, w, h)
                .expect("failed to encode image");
        }

        mean_center(&mean, &mut training_set);

        let (eigen, eigenvalues) = image_eigen(&training_set);

        let pca = image_pca(pca_size, eigen.clone());

        let (min, max) = pca.iter().fold((std::f64::MAX, std::f64::MIN), |b, p| {
            (b.0.min(*p), b.1.max(*p))
        });

        // Create and write images for eigenfaces

        if write_faces {
            // 10 most significant eigenfaces
            for i in 0..10.min(pca.ncols()) {
                let buffer: Vec<u8> = calculate_image(&pca.column(i), min, max);
                write_image(format!("{}/eigenfaces/{}.jpg", output, i + 1), &buffer, w, h)
                .expect("failed to encode image");
            }

            // 10 least significant eigenfaces
            for i in (pca.ncols() - 10).max(0)..pca.ncols() {
                let buffer: Vec<u8> = calculate_image(&pca.column(i), min, max);
                write_image(format!("{}/eigenfaces/{}.jpg", output, i + 1), &buffer, w, h)
                .expect("failed to encode image");
            }
        }

        let mahalanobis = Mahalanobis::new(&eigenvalues);

        let mut testing_pca_load: Vec<na::DVector<f64>> = vec![];
        for i in 0..testing_set.ncols() {
            testing_pca_load.push(pca_load(&testing_set.column(i), &pca));
        }
        let testing_pca_load: DataSet = DataSet::from_columns(testing_pca_load.as_slice());

        let mut gallery_pca_load: Vec<na::DVector<f64>> = vec![];
        for i in 0..gallery_set.ncols() {
            gallery_pca_load.push(pca_load(&gallery_set.column(i), &pca));
        }
        let gallery_pca_load: DataSet = DataSet::from_columns(gallery_pca_load.as_slice());

        let dist = mahalanobis.distance_matrix(&testing_pca_load, &gallery_pca_load);

        let mut matches = vec![(0, 0.); dist.nrows()];
        for r in 0..dist.nrows() {
            matches[r] = dist
            .row(r)
            .iter()
            .enumerate()
            .fold(
                (0, std::f64::MAX),
                |acc, (i, d)| {
                    if *d < acc.1 {
                        (i, *d)
                    } else {
                        acc
                    }
                },
            );
        }

        if write_faces {
            let mut rng = thread_rng();
            let uniform = Uniform::new(0, matches.len());
            let samples = rng.sample_iter(&uniform);
            let selection: Vec<usize> = samples.take(2).collect();

            let mut name = 0;
            for s in 0..gallery_set.ncols() {
                name += 1;
                let m = matches[s].0;
                let selection: Vec<u8> = testing_set.column(s).iter().map(|p| *p as u8).collect();
                let matched: Vec<u8> = gallery_set.column(m).iter().map(|p| *p as u8).collect();
                let select_proj: Vec<u8> = calculate_image(
                    &pca_project(&testing_pca_load.column(s), &pca), min, max
                ).iter().map(|p| *p as u8).collect();
                write_image(format!("{}/matched/proj_{}.jpg", output, name), &select_proj, w, h,)
                    .expect("failed to write image");
                write_image(format!("{}/matched/{}_selection.jpg", output, name), &selection, w, h)
                    .expect("failed to write image");
                write_image(format!("{}/matched/{}_match.jpg", output, name), &matched, w, h)
                    .expect("failed to write image");
            }
        }

        let mut match_count = 0;
        for (s, (m, _)) in matches.iter().enumerate() {
            if s/2 == *m/4 {
                match_count += 1;
            }
        }
        match_count as f64 / matches.len() as f64
    }
}
