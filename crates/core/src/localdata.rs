//! The owner's own dataset folder (for example Sentinel-1/2 GeoTIFFs exported from Google Earth Engine): what is in it, read on the Mac
//! without opening an image library (file list, class folders, and the header of a GeoTIFF: size, number of bands, band names), and a
//! signature that tells whether the folder changed since it was last copied or uploaded.

use crate::cloud;
use anyhow::{bail, Result};
use serde_json::{json, Value};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// at most this many files are listed (a bigger folder is still used whole; only the description stops here)
const MAX_LISTED: usize = 200_000;

/// the files under `dir` as (path relative to dir with '/', size); hidden files and folders are skipped
pub fn files(dir: &Path) -> Result<Vec<(String, u64)>> {
    if !dir.is_dir() {
        bail!("The folder {} does not exist.", dir.display());
    }
    let mut out = vec![];
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let mut entries: Vec<_> = std::fs::read_dir(&d)?.flatten().collect();
        entries.sort_by_key(|e| e.file_name());
        for e in entries {
            let name = e.file_name().to_string_lossy().to_string();
            if name.starts_with('.') {
                continue;
            }
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if let Ok(m) = p.metadata() {
                let rel = p.strip_prefix(dir).unwrap_or(&p).to_string_lossy().replace('\\', "/");
                out.push((rel, m.len()));
                if out.len() >= MAX_LISTED {
                    return Ok(out);
                }
            }
        }
    }
    out.sort();
    Ok(out)
}

/// number of files, total bytes and the newest change: equal signatures = nothing to copy again
pub fn signature(dir: &Path) -> Result<String> {
    let fs = files(dir)?;
    let newest = fs.iter().filter_map(|(f, _)| dir.join(f).metadata().ok()?.modified().ok())
        .filter_map(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_secs()).max().unwrap_or(0);
    Ok(format!("{} files, {} bytes, newest {}", fs.len(), fs.iter().map(|f| f.1).sum::<u64>(), newest))
}

/// what the Dataset tab shows for a folder: files, size, folders with one subfolder per class, other folders, and the first GeoTIFF's header
pub fn describe(dir: &Path) -> Result<Value> {
    let fs = files(dir)?;
    let listed: Vec<(String, String)> = fs.iter().map(|(f, s)| (f.clone(), s.to_string())).collect();
    let folders = cloud::class_folders(&listed);
    let mut dirs: Vec<String> = fs.iter().filter_map(|(f, _)| f.rsplit_once('/').map(|(d, _)| d.to_string())).collect();
    dirs.sort();
    dirs.dedup();
    let tifs: Vec<&String> = fs.iter().map(|f| &f.0).filter(|f| { let l = f.to_ascii_lowercase(); l.ends_with(".tif") || l.ends_with(".tiff") }).collect();
    let tif = tifs.first().and_then(|f| tiff_info(&dir.join(f)).ok().map(|i| json!({"file": f, "width": i.width, "height": i.height,
        "bands": i.bands, "bits": i.bits, "float": i.float, "names": i.names})));
    Ok(json!({
        "files": fs.len(), "bytes": fs.iter().map(|f| f.1).sum::<u64>(), "complete": fs.len() < MAX_LISTED,
        "tifs": tifs.len(), "npys": fs.iter().filter(|f| f.0.to_ascii_lowercase().ends_with(".npy")).count(),
        "folders": folders.iter().map(|(p, c)| json!({"folder": p, "classes": c})).collect::<Vec<_>>(),
        "dirs": dirs.into_iter().take(40).collect::<Vec<_>>(),
        "top": fs.iter().filter(|f| !f.0.contains('/')).map(|f| &f.0).take(40).collect::<Vec<_>>(),
        "tif": tif,
    }))
}

#[derive(Debug, Default, PartialEq)]
pub struct TiffInfo {
    pub width: u64,
    pub height: u64,
    pub bands: u64,
    pub bits: u64,
    pub float: bool,
    /// band names written by GDAL / Earth Engine (the DESCRIPTION items of GDAL_METADATA); empty when the file has none
    pub names: Vec<String>,
}

/// the first image of a TIFF / BigTIFF file: size, samples per pixel, bits and the band names GDAL keeps in its metadata tag (42112)
pub fn tiff_info(path: &Path) -> Result<TiffInfo> {
    let mut f = std::fs::File::open(path)?;
    let mut head = [0u8; 16];
    f.read_exact(&mut head[..8])?;
    let le = match &head[..2] {
        b"II" => true,
        b"MM" => false,
        _ => bail!("not a TIFF file"),
    };
    let u16_at = |b: &[u8]| if le { u16::from_le_bytes([b[0], b[1]]) } else { u16::from_be_bytes([b[0], b[1]]) } as u64;
    let u32_at = |b: &[u8]| if le { u32::from_le_bytes(b[..4].try_into().unwrap()) } else { u32::from_be_bytes(b[..4].try_into().unwrap()) } as u64;
    let u64_at = |b: &[u8]| if le { u64::from_le_bytes(b[..8].try_into().unwrap()) } else { u64::from_be_bytes(b[..8].try_into().unwrap()) };
    let big = match u16_at(&head[2..4]) {
        42 => false,
        43 => true,
        _ => bail!("not a TIFF file"),
    };
    let ifd = if big {
        f.read_exact(&mut head[8..16])?;
        u64_at(&head[8..16])
    } else {
        u32_at(&head[4..8])
    };
    f.seek(SeekFrom::Start(ifd))?;
    let (count_len, entry_len, value_len) = if big { (8, 20, 8) } else { (2, 12, 4) };
    let mut c = [0u8; 8];
    f.read_exact(&mut c[..count_len])?;
    let n = if big { u64_at(&c) } else { u16_at(&c) }.min(4096) as usize;
    let mut entries = vec![0u8; n * entry_len];
    f.read_exact(&mut entries)?;
    let mut info = TiffInfo { bands: 1, bits: 8, ..Default::default() };
    let mut gdal = String::new();
    for e in entries.chunks(entry_len) {
        let (tag, typ) = (u16_at(&e[0..2]), u16_at(&e[2..4]));
        let count = if big { u64_at(&e[4..12]) } else { u32_at(&e[4..8]) };
        let v = &e[entry_len - value_len..];
        // the first value of a SHORT / LONG / LONG8 field stored in the entry
        let first = match typ {
            3 => u16_at(v),
            16 => u64_at(v),
            _ => u32_at(v),
        };
        match tag {
            256 => info.width = first,
            257 => info.height = first,
            258 => info.bits = first,
            277 => info.bands = first,
            339 => info.float = first == 3,
            42112 if typ == 2 && count < 1 << 20 => {
                let mut text = vec![0u8; count as usize];
                if count as usize <= value_len {
                    text.copy_from_slice(&v[..count as usize]);
                } else {
                    let at = f.stream_position()?;
                    f.seek(SeekFrom::Start(if big { u64_at(v) } else { u32_at(v) }))?;
                    f.read_exact(&mut text)?;
                    f.seek(SeekFrom::Start(at))?;
                }
                gdal = String::from_utf8_lossy(&text).trim_end_matches('\0').to_string();
            }
            _ => {}
        }
    }
    info.names = band_names(&gdal, info.bands as usize);
    Ok(info)
}

/// <Item name="DESCRIPTION" sample="0" role="description">B4</Item> … in GDAL's metadata XML: the band names in band order, only when
/// every band has one
fn band_names(xml: &str, bands: usize) -> Vec<String> {
    let mut names = vec![String::new(); bands];
    for item in xml.split("<Item").skip(1) {
        let (attrs, rest) = match item.split_once('>') {
            Some(x) => x,
            None => continue,
        };
        if !attrs.contains("name=\"DESCRIPTION\"") {
            continue;
        }
        let sample = attrs.split("sample=\"").nth(1).and_then(|s| s.split('"').next()).and_then(|s| s.parse::<usize>().ok());
        let text = rest.split("</Item>").next().unwrap_or("").trim();
        if let Some(i) = sample.filter(|&i| i < bands) {
            names[i] = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">");
        }
    }
    if names.iter().all(|n| !n.is_empty()) {
        names
    } else {
        vec![]
    }
}
