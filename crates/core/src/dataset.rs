//! Where the data comes from (chosen on the Mac, downloaded only on the GPU machine). Saved as `dataset.json` in the project.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Dataset {
    /// "builtin" | "huggingface" | "kaggle"
    pub source: String,
    /// "classify" (one label per image) | "segment" (a class for every pixel: images + masks)
    #[serde(default = "classify")]
    pub task: String,
    /// built-in sets: resize to image_size / convert to channels (e.g. CIFAR-10 at 224×224 for AlexNet); other sources always resize
    #[serde(default)]
    pub resize: bool,
    /// builtin: "mnist" | "fashion_mnist" | "cifar10"
    #[serde(default)]
    pub builtin: String,
    /// huggingface: repository id ("ylecun/mnist"), optional config, splits and columns
    #[serde(default)]
    pub hf_repo: String,
    #[serde(default)]
    pub hf_config: String,
    #[serde(default = "train")]
    pub hf_train_split: String,
    /// empty = take 10% of the training split
    #[serde(default)]
    pub hf_test_split: String,
    #[serde(default = "image")]
    pub hf_image_col: String,
    #[serde(default = "label")]
    pub hf_label_col: String,
    /// segmentation: the column with the mask image (pixel value = class index; for two classes any non-zero pixel = class 1)
    #[serde(default = "mask")]
    pub hf_mask_col: String,
    /// kaggle: "owner/dataset" and the folder (inside it) with one subfolder per class; optional separate test folder
    #[serde(default)]
    pub kaggle_slug: String,
    #[serde(default)]
    pub kaggle_train_dir: String,
    #[serde(default)]
    pub kaggle_test_dir: String,
    /// segmentation: the folder of masks (same file names as the images in kaggle_train_dir)
    #[serde(default)]
    pub kaggle_mask_dir: String,
    /// images are resized to this (height, width) and converted to this many channels (1 = grey, 3 = colour); ignored for built-in sets
    #[serde(default = "size64")]
    pub image_size: [usize; 2],
    #[serde(default = "three")]
    pub channels: usize,
    /// class names in label order (Hugging Face: from the ClassLabel feature; Kaggle: the class folder names)
    #[serde(default)]
    pub classes: Vec<String>,
}

fn train() -> String {
    "train".into()
}
fn image() -> String {
    "image".into()
}
fn label() -> String {
    "label".into()
}
fn mask() -> String {
    "mask".into()
}
fn classify() -> String {
    "classify".into()
}
fn size64() -> [usize; 2] {
    [64, 64]
}
fn three() -> usize {
    3
}

pub struct Builtin {
    pub key: &'static str,
    pub label: &'static str,
    pub shape: [usize; 3],
    pub classes: &'static [&'static str],
}

pub const BUILTINS: &[Builtin] = &[
    Builtin { key: "mnist", label: "MNIST handwritten digits", shape: [1, 28, 28], classes: &["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"] },
    Builtin {
        key: "fashion_mnist",
        label: "Fashion-MNIST clothing",
        shape: [1, 28, 28],
        classes: &["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
    },
    Builtin {
        key: "cifar10",
        label: "CIFAR-10 small colour images",
        shape: [3, 32, 32],
        classes: &["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
    },
];

impl Default for Dataset {
    fn default() -> Self {
        Dataset {
            source: "builtin".into(),
            task: classify(),
            resize: false,
            builtin: "mnist".into(),
            hf_repo: String::new(),
            hf_config: String::new(),
            hf_train_split: train(),
            hf_test_split: String::new(),
            hf_image_col: image(),
            hf_label_col: label(),
            hf_mask_col: mask(),
            kaggle_slug: String::new(),
            kaggle_train_dir: String::new(),
            kaggle_test_dir: String::new(),
            kaggle_mask_dir: String::new(),
            image_size: size64(),
            channels: 3,
            classes: vec![],
        }
    }
}

impl Dataset {
    pub fn builtin_info(&self) -> Option<&'static Builtin> {
        BUILTINS.iter().find(|b| b.key == self.builtin)
    }

    /// (channels, height, width) the model receives
    pub fn input_shape(&self) -> Vec<usize> {
        match (self.source.as_str(), self.builtin_info()) {
            ("builtin", Some(b)) if !self.resize => b.shape.to_vec(),
            _ => vec![self.channels, self.image_size[0], self.image_size[1]],
        }
    }

    pub fn class_names(&self) -> Vec<String> {
        match (self.source.as_str(), self.builtin_info()) {
            ("builtin", Some(b)) => b.classes.iter().map(|s| s.to_string()).collect(),
            _ => self.classes.clone(),
        }
    }

    /// plain-words problems that stop a run (the design itself is checked by shapes::analyze)
    pub fn problems(&self) -> Vec<String> {
        let mut p = vec![];
        let ok_id = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || "-_./".contains(c)) && !s.contains("..");
        match self.source.as_str() {
            "builtin" => {
                if self.builtin_info().is_none() {
                    p.push("Choose a built-in dataset.".into())
                }
            }
            "huggingface" => {
                if !ok_id(&self.hf_repo) || !self.hf_repo.contains('/') {
                    p.push("Hugging Face: give the dataset id, like ylecun/mnist.".into())
                }
            }
            "kaggle" => {
                if !ok_id(&self.kaggle_slug) || self.kaggle_slug.matches('/').count() != 1 {
                    p.push("Kaggle: give the dataset as owner/name.".into())
                }
                for d in [&self.kaggle_train_dir, &self.kaggle_test_dir, &self.kaggle_mask_dir] {
                    if !d.is_empty() && !ok_id(d) {
                        p.push("Kaggle: folders may contain letters, digits and - _ . / only.".into())
                    }
                }
            }
            _ => p.push("Choose where the data comes from.".into()),
        }
        if self.task == "segment" {
            match self.source.as_str() {
                "builtin" => p.push("The built-in datasets are for classification: choose a segmentation dataset from Hugging Face or Kaggle.".into()),
                "huggingface" if self.hf_mask_col.is_empty() => p.push("Hugging Face: choose the mask column.".into()),
                "kaggle" if self.kaggle_train_dir.is_empty() || self.kaggle_mask_dir.is_empty() => {
                    p.push("Kaggle: give the images folder and the masks folder (same file names).".into())
                }
                _ => {}
            }
        }
        if self.source != "builtin" || self.resize {
            if self.class_names().len() < 2 {
                p.push("Give the class names (at least two): read them from the dataset with 'Read info', or type them.".into())
            }
            if !(1..=3).contains(&self.channels) || self.channels == 2 {
                p.push("Channels must be 1 (grey) or 3 (colour).".into())
            }
            if self.image_size.iter().any(|&s| !(8..=1024).contains(&s)) {
                p.push("Image size must be between 8 and 1024 pixels.".into())
            }
        }
        p
    }
}
