#[macro_use]
extern crate clap;
extern crate image;
extern crate nalgebra as na;
extern crate typenum as tn;

use image::{DynamicImage, GenericImageView};
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

fn image_pca(basis_size: usize, data_set: &DataSet) -> DataSet {
    let size = data_set.ncols();
    if basis_size >= size {
        panic!("Basis size larger than number of images in data set");
    }

    let covariance = data_set.transpose() * data_set.clone();
    let na::SymmetricEigen {
        mut eigenvectors,
        mut eigenvalues,
    } = covariance.symmetric_eigen();
    sort_by_eigenvalue(&mut eigenvalues, &mut eigenvectors);

    let eigen = data_set.clone() * eigenvectors;

    eigen.remove_columns(basis_size - 1, size as usize - basis_size)
}

fn main() {
    let _matches = clap_app!(eigenfaces =>
        (version: crate_version!())
        (author: crate_authors!())
        (about: crate_description!())
        (@arg DATA: +required "Path to the data set")
        (@arg OUTPUT: -o --output +takes_value "Path to place output files")
    ).get_matches();

    // Load data-sets
    let (mut training_set, (w, h)) = match read_dataset("orl_faces", 1..=40, 3..=6) {
        Ok(data) => data,
        Err(e) => {
            println!("{}", e);
            std::process::exit(1)
        }
    };
    let (mut testing_set, _) = read_dataset("orl_faces", 1..=40, 1..=2).unwrap();
    let (mut gallery_set, _) = read_dataset("orl_faces", 1..=40, 7..=10).unwrap();

    let mean = mean_image(&training_set);

    mean_center(&mean, &mut training_set);
    mean_center(&mean, &mut testing_set);
    mean_center(&mean, &mut gallery_set);

    let eigen = image_pca(50, &training_set);
    for i in 0..10 {
        use image::{jpeg, ColorType};
        let mut file = File::create(format!("output/eigenfaces/{}.png", i)).unwrap();
        let (min, max) = eigen.iter().fold((std::f64::MAX, std::f64::MIN), |b, p| {
            (b.0.min(*p), b.1.max(*p))
        });
        let buffer: Vec<u8> = eigen
            .column(i)
            .iter()
            .map(|p| ((*p - min) / (max - min) * 255f64) as u8)
            .collect();
        jpeg::JPEGEncoder::new(&mut file)
            .encode(buffer.as_slice(), w, h, ColorType::Gray(8))
            .expect("failed to encode image");
        println!("Creating image output/eigenfaces/{}.png", i + 1);
    }
}
