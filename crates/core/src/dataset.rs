//! Where the data comes from (chosen on the Mac, downloaded only on the GPU machine). Saved as `dataset.json` in the project.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Dataset {
    /// "builtin" | "huggingface" | "kaggle"
    pub source: String,
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
    /// kaggle: "owner/dataset" and the folder (inside it) with one subfolder per class; optional separate test folder
    #[serde(default)]
    pub kaggle_slug: String,
    #[serde(default)]
    pub kaggle_train_dir: String,
    #[serde(default)]
    pub kaggle_test_dir: String,
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
            builtin: "mnist".into(),
            hf_repo: String::new(),
            hf_config: String::new(),
            hf_train_split: train(),
            hf_test_split: String::new(),
            hf_image_col: image(),
            hf_label_col: label(),
            kaggle_slug: String::new(),
            kaggle_train_dir: String::new(),
            kaggle_test_dir: String::new(),
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
            ("builtin", Some(b)) => b.shape.to_vec(),
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
                for d in [&self.kaggle_train_dir, &self.kaggle_test_dir] {
                    if !d.is_empty() && !ok_id(d) {
                        p.push("Kaggle: folders may contain letters, digits and - _ . / only.".into())
                    }
                }
            }
            _ => p.push("Choose where the data comes from.".into()),
        }
        if self.source != "builtin" {
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
