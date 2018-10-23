#[macro_use]
extern crate clap;
extern crate image;
extern crate nalgebra as na;
extern crate rand;
extern crate typenum as tn;

use image::{DynamicImage, GenericImageView};
use rand::distributions::{Distribution, Uniform};
use rand::prelude::*;
use std::fs::File;
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
            (_, _) => a > b,
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

fn calculate_image<R, C, SR, SC>(
    image: &na::MatrixSlice<f64, R, C, SR, SC>,
    min: f64,
    max: f64,
) -> Vec<u8>
where
    SR: na::Dim,
    SC: na::Dim,
    C: na::Dim,
    R: na::Dim,
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

fn pca_project(pca_loading: &na::DVector<f64>, pca: &DataSet) -> ImageVector {
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
    // Try to calculate the inverse of the covariance matrix and store it as the basis for
    /// Create a mahalanobis distance calculator if the covariance matrix has an inverse
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
        let difference = x - y;
        let d = difference
            .iter()
            .enumerate()
            .fold(0., |acc, (i, d)| acc + (1. / self.eigenvalues[i]) * d * d)
            .abs()
            .sqrt();
        d
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
    let _matches = clap_app!(eigenfaces =>
        (version: crate_version!())
        (author: crate_authors!())
        (about: crate_description!())
        (@arg DATA: +required "Path to the data set")
        (@arg OUTPUT: -o --output +takes_value "Path to place output files")
    ).get_matches();

    let output = "output";

    // Load data-sets
    let (mut training_set, (w, h)) = match read_dataset("orl_faces", 1..=40, 3..=6) {
        Ok(data) => data,
        Err(e) => {
            println!("{}", e);
            std::process::exit(1)
        }
    };
    let (testing_set, _) = read_dataset("orl_faces", 1..=40, 1..=2).unwrap();
    let (gallery_set, _) = read_dataset("orl_faces", 1..=40, 7..=10).unwrap();

    let mean = mean_image(&training_set);

    {
        let buffer: Vec<u8> = mean.iter().map(|p| *p as u8).collect();
        write_image(format!("{}/mean_image.png", output), &buffer, w, h)
            .expect("failed to encode image");
    }

    mean_center(&mean, &mut training_set);

    let (eigen, eigenvalues) = image_eigen(&training_set);
    let pca = image_pca(50, eigen.clone());

    let (min, max) = pca.iter().fold((std::f64::MAX, std::f64::MIN), |b, p| {
        (b.0.min(*p), b.1.max(*p))
    });

    // Create and write images for eigenfaces

    // 10 most significant eigenfaces
    for i in 0..10 {
        let buffer: Vec<u8> = calculate_image(&pca.column(i), min, max);
        write_image(format!("{}/eigenfaces/{}.png", output, i + 1), &buffer, w, h)
            .expect("failed to encode image");
    }

    // 10 least significant eigenfaces
    for i in 39..50 {
        let buffer: Vec<u8> = calculate_image(&pca.column(i), min, max);
        write_image(format!("{}/eigenfaces/{}.png", output, i + 1), &buffer, w, h)
            .expect("failed to encode image");
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

    let mut rng = thread_rng();
    let uniform = Uniform::new(0, matches.len());
    let samples = rng.sample_iter(&uniform);
    let selection: Vec<usize> = samples.take(2).collect();

    let mut name = 0;
    for s in &selection {
        name += 1;
        let c = matches[*s].0;
        let selection: Vec<u8> = calculate_image(&gallery_set.column(*s), min, max);
        let matched: Vec<u8> = calculate_image(&gallery_set.column(c), min, max);
        write_image(format!("{}/matched/{}_selection.png", output, name), &selection, w, h)
            .expect("failed to write image");
        write_image(format!("{}/matched/{}_match.png", output, name), &matched, w, h)
            .expect("failed to write image");
    }
}
