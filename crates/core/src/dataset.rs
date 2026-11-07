//! Where the data comes from (chosen on the Mac, downloaded only on the GPU machine). Saved as `dataset.json` in the project.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Dataset {
    /// "builtin" | "huggingface" | "kaggle" | "local" (the owner's own files, e.g. GeoTIFFs exported from Google Earth Engine)
    pub source: String,
    /// "classify" (one label per image) | "multilabel" (any number of classes per image: sigmoid + BCE) | "segment" (a class for every
    /// pixel: images + masks) | "regress" (numbers per image from columns of the labels table; `classes` holds the target columns) |
    /// "detect" (boxes: an images folder and a labels folder of YOLO .txt files, "class cx cy w h" normalised, one per image)
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
    /// kaggle: "owner/dataset". kaggle and local: the folder (inside the dataset) with one subfolder per class; optional separate test folder
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
    /// multispectral / radar images (Sentinel-2, Sentinel-1, …) read from GeoTIFF or .npy files: the bands the model receives, in this
    /// order (the number of bands is the number of channels). Empty = ordinary images (grey or colour, read with PIL).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub bands: Vec<String>,
    /// the order of the bands in the files (each file has these channels; `bands` picks from them by name)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub band_order: Vec<String>,
    /// "s2" (Sentinel-2 optical) | "s1" (Sentinel-1 radar) | "other"; empty for ordinary images
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub sensor: String,
    /// Sentinel-1: the files hold linear backscatter, converted to decibels (10·log10) before normalising (files that already hold
    /// decibels, as Earth Engine's COPERNICUS/S1_GRD does, are left as they are)
    #[serde(default)]
    pub s1_db: bool,
    /// local: the folder with the files, on this Mac (copied to the training machine, or uploaded to Kaggle as a private dataset, when a
    /// run starts) or, with `local_on_machine`, a folder that is already on the training machine
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub local_dir: String,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub local_on_machine: bool,
    /// local: the files are in the cloud and the training machine fetches them itself (nothing passes through the Mac):
    /// gs://bucket/folder (Earth Engine's Cloud Storage export; a private bucket needs the key from the Connection tab), a shared
    /// Google Drive folder or file link (https://drive.google.com/…), or rclone:remote:path (rclone set up on the machine)
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub cloud_url: String,
    /// "" = files (one subfolder per class, or images + masks) | "scene" = large images (an Earth Engine export, possibly split into
    /// several files) with a label image on the same grid, cut into tiles of `image_size` (segmentation)
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub layout: String,
    /// scene: the image file or folder of files, and the label file or folder (same file names), relative to the dataset folder
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub scene_image: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub scene_label: String,
    /// scene: how the test tiles are chosen: "" = 10 % at random (optimistic: they touch training tiles) | "strip" = the eastern tenth of
    /// every image, one column of tiles left out between | "files" = a separately exported test region (scene_test_image / _label)
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub scene_split: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub scene_test_image: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub scene_test_label: String,
    /// multilabel (kaggle / local): the CSV with the labels, relative to the dataset: a file column (path relative to the CSV's folder or
    /// the training folder, with or without extension) and either a column of class names separated by ; , | or spaces, or one 0/1
    /// column per class. Hugging Face: the label column is a list of class labels.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub labels_csv: String,
    /// several outputs (multi-task): extra targets next to the main task, each read from columns of the labels table (Hugging Face:
    /// of the dataset's rows) and given to the Output block with the same `head` number (1, 2, ...); the main task is head 0
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub heads: Vec<Head>,
    /// with extra targets and the main task "classify": the labels table's column with each image's class (empty = the name of the
    /// folder the image is in)
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub label_column: String,
    /// fusion (e.g. Sentinel-1 + Sentinel-2): a second band group, read from `fuse_dir` (a folder with the same relative file names,
    /// any extension; scenes: the second image folder) and stacked after the first group's bands. Input blocks with group 1 / 2 take
    /// one group each (mid fusion); an Input with group 0 takes both (early fusion).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub bands2: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub band_order2: Vec<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub sensor2: String,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub s1_db2: bool,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub fuse_dir: String,
    /// the second band group's own image size [height, width] (empty = resized to image_size): the two groups then reach the network
    /// as separate inputs (one Input block each), e.g. Sentinel-1 at 32×32 next to Sentinel-2 at 64×64. In the batch tensor group 2
    /// fills the top-left corner of its channels (NaN / 0 elsewhere), which the generated forward cuts out, so it may not be larger
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub image_size2: Vec<usize>,
    /// time series: the number of dates per sample (0 = single images). Each file holds all dates: a .npy (dates, bands, H, W) or
    /// (dates, H, W, bands), or a GeoTIFF whose bands go date by date (all bands of date 1, then date 2, ...); the network's input is
    /// (bands, dates, height, width)
    #[serde(default, skip_serializing_if = "is_zero")]
    pub time_steps: usize,
}

/// Sentinel-2 L1C bands in the order EuroSAT-MS stores them (B8A last); `SENTINEL2_10_20` are the 10 m and 20 m bands
pub const SENTINEL2_BANDS: &[&str] = &["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"];
pub const SENTINEL2_10_20: &[&str] = &["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"];
pub const SENTINEL1_BANDS: &[&str] = &["VV", "VH"];
/// Sentinel-2 as Google Earth Engine names and exports it (COPERNICUS/S2_SR_HARMONIZED: no B10; the TOA collection adds B10 after B9)
pub const SENTINEL2_GEE: &[&str] = &["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"];
/// the most bands an image may have
pub const MAX_BANDS: usize = 32;

/// what is wrong with a cloud location (None: empty or fine)
pub fn cloud_url_problem(u: &str) -> Option<String> {
    if u.is_empty() {
        return None;
    }
    if u.chars().any(|c| c.is_whitespace() || c.is_control() || "\"'`$\\;|&<>".contains(c)) {
        return Some("The cloud location may not contain spaces, quotes or shell characters.".into());
    }
    let ok = match u.split_once("://") {
        Some(("gs", rest)) => rest.split('/').next().is_some_and(|b| b.len() >= 3),
        Some(("https", rest)) => rest.starts_with("drive.google.com/"),
        Some(("file", rest)) => rest.starts_with('/'),
        _ => u.strip_prefix("rclone:").is_some_and(|r| r.split_once(':').is_some_and(|(remote, _)| !remote.is_empty())),
    };
    (!ok).then(|| "The cloud location: gs://bucket/folder, a Google Drive link (https://drive.google.com/…), or rclone:remote:path.".into())
}

/// a band name for comparing: upper case, and Earth Engine's B1 … B9 written as B01 … B09 (so B4 and B04 are the same band)
pub fn band_key(b: &str) -> String {
    let u = b.trim().to_ascii_uppercase();
    match u.strip_prefix('B') {
        Some(rest) if rest.len() == 1 && rest.chars().all(|c| c.is_ascii_digit()) => format!("B0{rest}"),
        _ => u,
    }
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

/// one extra output of a multi-task network (Dataset::heads)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct Head {
    /// shown in the results and on the Output block, e.g. "biomass"
    pub name: String,
    /// "classify" (one class from a column) | "multilabel" (a column of class lists, or one 0/1 column per class) | "regress" (numbers)
    pub task: String,
    /// the table's columns: classify / multilabel: the labels column (multilabel with none: the classes are 0/1 columns); regress:
    /// the numeric columns, one number each
    #[serde(default)]
    pub columns: Vec<String>,
    /// classify / multilabel: the class names (in order); regress: unused (the columns name the numbers)
    #[serde(default)]
    pub classes: Vec<String>,
    /// "" = the task's usual loss (cross-entropy, binary cross-entropy, MSE) | "focal" (classify) | "l1" | "huber" (regress)
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub loss: String,
    /// this head's share of the total loss (the main task counts 1)
    #[serde(default = "one")]
    pub weight: f64,
}

fn is_zero(v: &usize) -> bool {
    *v == 0
}

fn one() -> f64 {
    1.0
}

impl Head {
    /// the number of values the head's Output gives: one score per class, or one number per target column
    pub fn size(&self) -> usize {
        if self.task == "regress" { self.columns.len() } else { self.classes.len() }
    }
}

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
            bands: vec![],
            band_order: vec![],
            sensor: String::new(),
            s1_db: false,
            local_dir: String::new(),
            local_on_machine: false,
            cloud_url: String::new(),
            layout: String::new(),
            scene_image: String::new(),
            scene_label: String::new(),
            scene_split: String::new(),
            scene_test_image: String::new(),
            scene_test_label: String::new(),
            labels_csv: String::new(),
            heads: vec![],
            label_column: String::new(),
            bands2: vec![],
            band_order2: vec![],
            sensor2: String::new(),
            s1_db2: false,
            fuse_dir: String::new(),
            image_size2: vec![],
            time_steps: 0,
        }
    }
}

impl Dataset {
    /// the number of values Output block `head` gives (0 = the main task's classes or targets; None = no such head)
    pub fn head_size(&self, head: usize) -> Option<usize> {
        if head == 0 { Some(self.class_names().len()) } else { self.heads.get(head - 1).map(|h| h.size()) }
    }

    fn head_problems(&self) -> Vec<String> {
        let mut p = vec![];
        let ok_col = |s: &str| !s.is_empty() && s.len() <= 128 && !s.contains(['"', '\\', '\n', '\r']);
        if !matches!(self.source.as_str(), "kaggle" | "local" | "huggingface") {
            p.push("Extra outputs read their targets from a labels table (own files or Kaggle) or from Hugging Face columns.".into());
        }
        if matches!(self.source.as_str(), "kaggle" | "local") && self.labels_csv.is_empty() {
            p.push("Extra outputs: give the labels table (CSV) that has their columns.".into());
        }
        if self.task == "segment" {
            p.push("Extra outputs are for one value per image: a segmentation network has its single per-pixel output.".into());
        }
        if self.heads.len() > 8 {
            p.push("At most 8 extra outputs.".into());
        }
        if !self.label_column.is_empty() && !ok_col(&self.label_column) {
            p.push("The class column's name may not contain quotes or backslashes.".into());
        }
        for (i, h) in self.heads.iter().enumerate() {
            let name = if h.name.trim().is_empty() { format!("output {}", i + 1) } else { h.name.clone() };
            if h.name.trim().is_empty() || h.name.len() > 40 || !ok_col(&h.name) {
                p.push(format!("Extra output {}: give it a short name (no quotes).", i + 1));
            }
            if self.heads[..i].iter().any(|o| o.name == h.name) {
                p.push(format!("Two extra outputs are called {name}."));
            }
            if !h.columns.iter().all(|c| ok_col(c)) {
                p.push(format!("{name}: a column name may not contain quotes or backslashes."));
            }
            match h.task.as_str() {
                "classify" => {
                    if h.columns.len() != 1 {
                        p.push(format!("{name}: give the one column with the class of each image."));
                    }
                    if h.classes.len() < 2 {
                        p.push(format!("{name}: give its class names (at least two)."));
                    }
                }
                "multilabel" => {
                    if h.columns.len() > 1 {
                        p.push(format!("{name}: give one labels column, or none when the classes are 0/1 columns."));
                    }
                    if h.classes.is_empty() {
                        p.push(format!("{name}: give its class names."));
                    }
                }
                "regress" => {
                    if h.columns.is_empty() {
                        p.push(format!("{name}: give the column(s) with the numbers to predict."));
                    }
                }
                _ => p.push(format!("{name}: choose its task (classes, several classes, or numbers).")),
            }
            let losses: &[&str] = match h.task.as_str() {
                "classify" => &["", "focal"],
                "regress" => &["", "mse", "l1", "huber"],
                _ => &[""],
            };
            if !losses.contains(&h.loss.as_str()) {
                p.push(format!("{name}: the loss '{}' is not for this task.", h.loss));
            }
            if !(h.weight.is_finite() && (0.0..=100.0).contains(&h.weight)) {
                p.push(format!("{name}: its weight must be from 0 to 100."));
            }
        }
        p
    }

    pub fn builtin_info(&self) -> Option<&'static Builtin> {
        BUILTINS.iter().find(|b| b.key == self.builtin)
    }

    /// (channels, height, width) the model receives
    pub fn input_shape(&self) -> Vec<usize> {
        match (self.source.as_str(), self.builtin_info()) {
            ("builtin", Some(b)) if !self.resize => b.shape.to_vec(),
            _ if !self.bands.is_empty() && self.time_steps > 0 => vec![self.bands.len(), self.time_steps, self.image_size[0], self.image_size[1]],
            _ if !self.bands.is_empty() => vec![self.bands.len() + self.bands2.len(), self.image_size[0], self.image_size[1]],
            _ => vec![self.channels, self.image_size[0], self.image_size[1]],
        }
    }

    /// multispectral / radar images (bands chosen), not ordinary pictures
    pub fn multiband(&self) -> bool {
        !self.bands.is_empty()
    }

    /// two band groups (e.g. Sentinel-2 + Sentinel-1)
    pub fn fused(&self) -> bool {
        self.multiband() && !self.bands2.is_empty()
    }

    /// the second band group has its own size (separate inputs, not stacked on one grid)
    pub fn separate_inputs(&self) -> bool {
        self.fused() && self.image_size2.len() == 2
    }

    /// the shape an Input block of this group receives (channels, height, width): group 2 at its own size when it has one
    pub fn group_shape(&self, group: u64) -> Option<Vec<usize>> {
        let c = self.group_channels(group)?;
        let mut s = self.input_shape();
        s[0] = c;
        if group == 2 && self.separate_inputs() {
            s[1] = self.image_size2[0];
            s[2] = self.image_size2[1];
        }
        Some(s)
    }

    /// the channels an Input block of this group receives: 0 = all, 1 = the first band group, 2 = the second (None: no such group)
    pub fn group_channels(&self, group: u64) -> Option<usize> {
        match group {
            0 => Some(self.input_shape()[0]),
            1 if self.fused() => Some(self.bands.len()),
            2 if self.fused() => Some(self.bands2.len()),
            _ => None,
        }
    }

    /// local: the name the files have on the training machine (<data>/local/<key>) and on Kaggle (<owner>/<key>, a private dataset):
    /// "nnb-" + the folder's name + a short hash of its full path, in the letters Kaggle allows (3 to 50 characters)
    pub fn local_key(&self) -> String {
        let path = if self.cloud_url.is_empty() { self.local_dir.as_str() } else { self.cloud_url.as_str() };
        let path = path.trim_end_matches(['/', '\\']);
        let base = path.rsplit(['/', '\\']).next().unwrap_or("");
        let mut name: String = base.to_ascii_lowercase().chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '-' }).collect();
        while name.contains("--") {
            name = name.replace("--", "-");
        }
        let name: String = name.trim_matches('-').chars().take(32).collect();
        // FNV-1a: the same folder always gets the same key, and two folders with the same name do not collide
        let h = path.bytes().fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3));
        format!("nnb-{}{:08x}", if name.is_empty() { String::new() } else { format!("{}-", name.trim_end_matches('-')) }, h as u32)
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
            "local" => {
                let d = self.local_dir.trim();
                let absolute = d.starts_with('/') || d.starts_with("~/") || (d.len() > 2 && d.as_bytes()[1] == b':');
                if let Some(problem) = cloud_url_problem(&self.cloud_url) {
                    p.push(problem)
                } else if !self.cloud_url.is_empty() {
                    // fetched by the training machine: no folder on this Mac
                } else if d.is_empty() || !absolute {
                    p.push("Your own files: give the full path of the folder (Choose… picks it).".into())
                } else if d.chars().any(|c| c.is_control() || "\"'`$".contains(c)) {
                    p.push("Your own files: the folder's path may not contain quotes, ` or $.".into())
                }
                for d in [&self.kaggle_train_dir, &self.kaggle_test_dir, &self.kaggle_mask_dir, &self.scene_image, &self.scene_label] {
                    if !d.is_empty() && !ok_id(d) {
                        p.push("Folders and files inside the dataset may contain letters, digits and - _ . / only.".into())
                    }
                }
                if self.layout == "scene" {
                    if self.task != "segment" || !self.multiband() {
                        p.push("Large images with a label image are for segmentation of GeoTIFF bands: choose Segmentation and the bands.".into())
                    }
                    if self.scene_image.is_empty() || self.scene_label.is_empty() {
                        p.push("Give the image (file or folder) and the label image (file or folder with the same file names).".into())
                    }
                    if !matches!(self.scene_split.as_str(), "" | "strip" | "files") {
                        p.push("Choose how the test tiles are chosen.".into())
                    }
                    if self.scene_split == "files" && (self.scene_test_image.is_empty() || self.scene_test_label.is_empty()) {
                        p.push("Give the test region's image and label image (exported separately).".into())
                    }
                    for d in [&self.scene_test_image, &self.scene_test_label] {
                        if !d.is_empty() && !ok_id(d) {
                            p.push("Folders and files inside the dataset may contain letters, digits and - _ . / only.".into())
                        }
                    }
                }
            }
            _ => p.push("Choose where the data comes from.".into()),
        }
        if !matches!(self.task.as_str(), "classify" | "multilabel" | "segment" | "regress" | "detect") {
            p.push("Choose the task.".into())
        }
        if self.task == "multilabel" {
            match self.source.as_str() {
                "builtin" => p.push("The built-in datasets have one label per image: choose a multi-label dataset.".into()),
                "kaggle" | "local" if self.labels_csv.is_empty() => p.push("Give the CSV file with the labels of every image.".into()),
                "kaggle" | "local" if !ok_id(&self.labels_csv) => p.push("The labels file may contain letters, digits and - _ . / only.".into()),
                _ => {}
            }
        }
        if self.task == "detect" {
            match self.source.as_str() {
                "kaggle" | "local" if self.kaggle_train_dir.is_empty() || self.kaggle_mask_dir.is_empty() => {
                    p.push("Give the images folder and the labels folder (YOLO .txt files with the same names: class cx cy w h per box).".into())
                }
                "kaggle" | "local" => {}
                _ => p.push("Object detection reads boxes from YOLO label files: use your own files or Kaggle.".into()),
            }
            if !self.heads.is_empty() || self.time_steps > 0 || self.fused() {
                p.push("Object detection works on single images (no extra outputs, time series or second band group yet).".into());
            }
        }
        if self.task == "regress" {
            match self.source.as_str() {
                "builtin" => p.push("The built-in datasets have classes, not numbers: choose a regression dataset.".into()),
                "kaggle" | "local" if self.labels_csv.is_empty() => p.push("Give the CSV file with the target numbers of every image.".into()),
                "kaggle" | "local" if !ok_id(&self.labels_csv) => p.push("The labels file may contain letters, digits and - _ . / only.".into()),
                _ => {}
            }
        }
        if !self.heads.is_empty() {
            p.extend(self.head_problems());
        }
        if self.time_steps > 0 {
            if !self.multiband() {
                p.push("A time series is a stack of satellite images: choose its bands (Dataset tab: bands).".into());
            }
            if self.time_steps < 2 || self.time_steps > 400 {
                p.push("A time series has 2 to 400 dates per sample.".into());
            }
            if self.fused() {
                p.push("Time series with a second band group are not supported yet: put all bands in one group.".into());
            }
            if self.source == "huggingface" {
                p.push("Time series are read from files (own files or Kaggle), not from Hugging Face yet.".into());
            }
            if self.layout == "scene" {
                p.push("Time series come as one file per sample (all dates in it), not as a large scene.".into());
            }
        }
        if self.task == "segment" {
            match self.source.as_str() {
                "builtin" => p.push("The built-in datasets are for classification: choose a segmentation dataset from Hugging Face or Kaggle.".into()),
                "huggingface" if self.hf_mask_col.is_empty() => p.push("Hugging Face: choose the mask column.".into()),
                "kaggle" | "local" if self.layout != "scene" && (self.kaggle_train_dir.is_empty() || self.kaggle_mask_dir.is_empty()) => {
                    p.push("Give the images folder and the masks folder (same file names).".into())
                }
                _ => {}
            }
        }
        if self.source != "builtin" || self.resize {
            if self.task == "detect" {
                if self.classes.is_empty() {
                    p.push("Give the object classes, one per line, in the order of the class numbers of the label files.".into())
                }
            } else if self.task == "regress" {
                if self.classes.is_empty() {
                    p.push("Choose the target columns (the numbers to predict): read them from the labels file, or type them.".into())
                }
            } else if self.class_names().len() < 2 {
                p.push("Give the class names (at least two): read them from the dataset with 'Read info', or type them.".into())
            }
            if self.multiband() {
                if self.bands.len() > MAX_BANDS {
                    p.push(format!("At most {MAX_BANDS} bands."))
                }
                let ok_name = |b: &String| !b.is_empty() && b.len() <= 16 && b.chars().all(|c| c.is_ascii_alphanumeric() || "_-".contains(c));
                if !self.bands.iter().chain(&self.band_order).all(ok_name) {
                    p.push("Band names may contain letters, digits, - and _ only.".into())
                }
                let order: Vec<String> = self.band_order.iter().map(|b| band_key(b)).collect();
                if let Some(b) = self.bands.iter().find(|b| !order.contains(&band_key(b))) {
                    p.push(format!("The band {b} is not in the bands of the files ({}).", self.band_order.join(" ")))
                }
                if order.iter().enumerate().any(|(i, b)| order[..i].contains(b)) {
                    p.push("A band appears twice in the bands of the files.".into())
                }
                if self.bands.iter().enumerate().any(|(i, b)| self.bands[..i].iter().any(|x| band_key(x) == band_key(b))) {
                    p.push("A band is chosen twice.".into())
                }
                if !self.bands2.is_empty() {
                    let order2: Vec<String> = self.band_order2.iter().map(|b| band_key(b)).collect();
                    if let Some(b) = self.bands2.iter().find(|b| !order2.contains(&band_key(b))) {
                        p.push(format!("The band {b} is not in the bands of the second files ({}).", self.band_order2.join(" ")))
                    }
                    if !self.band_order2.iter().all(ok_name) {
                        p.push("Band names may contain letters, digits, - and _ only.".into())
                    }
                    if self.bands.len() + self.bands2.len() > MAX_BANDS {
                        p.push(format!("At most {MAX_BANDS} bands in both groups together."))
                    }
                    let keys: Vec<String> = self.bands.iter().chain(&self.bands2).map(|b| band_key(b)).collect();
                    // the same bands twice is a pair in time (before / after the same place: change detection); otherwise a name may
                    // be in one group only
                    let pair = self.bands2.iter().map(|b| band_key(b)).collect::<Vec<_>>() == self.bands.iter().map(|b| band_key(b)).collect::<Vec<_>>();
                    if !pair && keys.iter().enumerate().any(|(i, b)| keys[..i].contains(b)) {
                        p.push("The two band groups share a band name: rename one (e.g. VV in the second group only).".into())
                    }
                    if !self.kaggle_test_dir.is_empty() && self.task != "multilabel" && self.task != "regress" {
                        p.push("With two band groups, leave the test folder empty (10% of the images are the test data).".into())
                    }
                    if !self.image_size2.is_empty() {
                        if self.image_size2.len() != 2 || self.image_size2.iter().any(|&s| !(4..=1024).contains(&s)) {
                            p.push("The second group's image size must be two numbers from 4 to 1024.".into())
                        } else if self.image_size2[0] > self.image_size[0] || self.image_size2[1] > self.image_size[1] {
                            p.push(format!("The second group's size ({}×{}) may not be larger than the first's ({}×{}): make the sharper sensor group 1.",
                                self.image_size2[0], self.image_size2[1], self.image_size[0], self.image_size[1]))
                        }
                        if self.layout == "scene" {
                            p.push("A scene and its second group are cut into tiles on one grid: leave the second group's own size empty.".into())
                        }
                    }
                    if self.fuse_dir.is_empty() || !ok_id(&self.fuse_dir) {
                        p.push("Give the folder with the second group's files (same file names as the first).".into())
                    } else if self.fuse_dir.starts_with('/') || self.fuse_dir.starts_with('~') {
                        // (only the dataset folder is copied or uploaded: a full path is not there on the training machine)
                        p.push("The second group's folder is a folder inside the dataset folder (e.g. s1), not a full path.".into())
                    }
                }
                if !matches!(self.source.as_str(), "kaggle" | "local" | "huggingface") {
                    p.push("Multispectral and radar images come from your own files, a Kaggle dataset or a Hugging Face dataset with band arrays.".into())
                }
                if self.fused() && self.source == "huggingface" {
                    p.push("Two band groups (fusion) are read from files: use your own files or Kaggle.".into())
                }
            } else if !(1..=3).contains(&self.channels) || self.channels == 2 {
                p.push("Channels must be 1 (grey) or 3 (colour).".into())
            }
            if self.image_size.iter().any(|&s| !(8..=1024).contains(&s)) {
                p.push("Image size must be between 8 and 1024 pixels.".into())
            }
        }
        p
    }
}
