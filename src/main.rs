#[macro_use]
extern crate clap;
extern crate image;
extern crate nalgebra as na;
extern crate typenum as tn;

use image::{DynamicImage, GenericImage, GenericImageView};

/// Dynamically sized matrix to store images
///
/// Fallback to heap allocation for image that are a different size than the 112x92 pixels from the
/// AT&T databse of faces training data set.
type ImageVector = na::Matrix<f64, na::U1, na::Dynamic, na::MatrixVec<f64, na::U1, na::Dynamic>>;

type DataSet = na::DMatrix<f64>;

fn image_vector(image: DynamicImage) -> ImageVector {
    let image = image.to_luma();
    let (w, h) = image.dimensions();
    let pixels = image.pixels().map(|p| p[0] as f64);
    ImageVector::from_iterator((w*h) as usize, pixels)
}

fn read_dataset(sets: usize, range: usize) -> Result<DataSet, String> {
    let mut data_set = DataSet::zeros(10304, 10);

    for s in 1..(sets + 1) {
        for i in 1..(range + 1) {
            if let Ok(img) = image::open(format!("orl_faces/s{}/{}.pgm", s, i)) {
                image_vector(img)
            } else {
                return Err(format!("Failed to open image s{}/{}.pgm", s, i));
            };
        }
    }

    Ok(data_set)
}

fn main() {
    let matches = clap_app!(eigenfaces =>
        (version: crate_version!())
        (author: crate_authors!())
        (about: crate_description!())
        (@arg DATA: +required "Path to the data set")
        (@arg OUTPUT: -o --output +takes_value "Path to place output files")
    ).get_matches();

    let training_set = match read_dataset(40, 2) {
        Ok(data) => data,
        Err(e) => {
            println!("{}", e);
            std::process::exit(1)
        },
    };
}
