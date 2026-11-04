//! Automatic updates from the public GitHub releases (adapted from the LaTeX Editor's updater).
//!
//! `check` reads the latest published release and picks the file for this system; `download` fetches it (size and the release's
//! SHA-256 `digest` checked); `install` puts the new version in place; the window then restarts the app.
//!
//! | System  | Asset                         | Install                                                           |
//! |---------|-------------------------------|-------------------------------------------------------------------|
//! | macOS   | `*_aarch64.app.zip`           | unzip next to the running `.app`, then swap the two by renaming   |
//! | Windows | `*_x64-setup.exe`             | run the installer after the app quits                             |
//! | Linux   | `*_amd64.AppImage` / `.deb`   | replace the AppImage file / open the package in the installer     |
//!
//! Only the program is replaced: projects, settings and keychain entries live elsewhere.

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

pub const REPO: &str = "varunjha089/NN-Builder";
/// overrides the "latest release" address (tests use a local server)
const URL_ENV: &str = "NN_BUILDER_UPDATE_URL";

pub type Version = (u64, u64, u64);

#[derive(Clone, Debug, Serialize)]
pub struct Release {
    pub version: String,
    pub newer: bool,
    pub notes: String,
    pub page: String,
    pub asset: Option<Asset>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Asset {
    pub name: String,
    pub url: String,
    pub size: u64,
    pub sha256: Option<String>,
}

/// what happens after the files are in place
#[derive(Clone, Debug, PartialEq)]
pub enum After {
    /// restart the app (macOS bundle swapped, AppImage replaced)
    Restart,
    /// run this installer, then quit (Windows)
    Installer(PathBuf),
    /// the system's package installer took over (.deb)
    Nothing,
}

pub fn current_version() -> Version {
    parse_version(env!("CARGO_PKG_VERSION")).unwrap_or_default()
}

/// "v1.2.3", "1.2" or "1.2.3-beta" -> (1, 2, 3)
pub fn parse_version(s: &str) -> Option<Version> {
    let core = s.trim().trim_start_matches(['v', 'V']).split(['-', '+']).next()?;
    let mut parts = core.split('.').map(|p| p.parse::<u64>().ok());
    let major = parts.next()??;
    let minor = parts.next().unwrap_or(Some(0))?;
    let patch = parts.next().unwrap_or(Some(0))?;
    Some((major, minor, patch))
}

fn get(url: &str, accept: &str) -> Result<ureq::http::Response<ureq::Body>, String> {
    let agent = ureq::Agent::config_builder()
        .user_agent(concat!("nn-builder/", env!("CARGO_PKG_VERSION")))
        .timeout_connect(Some(std::time::Duration::from_secs(15)))
        .http_status_as_error(false)
        .build()
        .new_agent();
    agent.get(url).header("Accept", accept).call().map_err(|e| format!("GitHub could not be reached ({e}). Check the internet connection."))
}

/// the latest published release; Ok(None) when none is published yet
pub fn check() -> Result<Option<Release>, String> {
    let url = std::env::var(URL_ENV).unwrap_or_else(|_| format!("https://api.github.com/repos/{REPO}/releases/latest"));
    let mut r = get(&url, "application/vnd.github+json")?;
    match r.status().as_u16() {
        200 => {}
        404 => return Ok(None),
        403 | 429 => return Err("GitHub is limiting requests right now; try again in an hour.".into()),
        c => return Err(format!("GitHub answered with HTTP {c}; try again later.")),
    }
    let body = r.body_mut().read_to_string().map_err(|e| e.to_string())?;
    parse_release(&body, std::env::consts::OS, std::env::consts::ARCH, std::env::var_os("APPIMAGE").is_some()).map(Some)
}

pub fn parse_release(json: &str, os: &str, arch: &str, appimage: bool) -> Result<Release, String> {
    let v: serde_json::Value = serde_json::from_str(json).map_err(|e| format!("Unexpected answer from GitHub: {e}"))?;
    let tag = v["tag_name"].as_str().ok_or("The release has no version tag.")?;
    let version = parse_version(tag).ok_or_else(|| format!("{tag:?} is not a version number."))?;
    let assets: Vec<Asset> = v["assets"].as_array().map(|l| l.iter().filter_map(|a| Some(Asset {
        name: a["name"].as_str()?.to_owned(),
        url: a["browser_download_url"].as_str()?.to_owned(),
        size: a["size"].as_u64().unwrap_or(0),
        sha256: a["digest"].as_str().and_then(|d| d.strip_prefix("sha256:")).map(str::to_ascii_lowercase),
    })).collect()).unwrap_or_default();
    Ok(Release {
        version: format!("{}.{}.{}", version.0, version.1, version.2),
        newer: version > current_version(),
        notes: v["body"].as_str().unwrap_or_default().trim().to_owned(),
        page: v["html_url"].as_str().map_or_else(|| format!("https://github.com/{REPO}/releases/latest"), str::to_owned),
        asset: pick_asset(&assets, os, arch, appimage).cloned(),
    })
}

/// the file for this system, by the names the release workflow gives them
pub fn pick_asset<'a>(assets: &'a [Asset], os: &str, arch: &str, appimage: bool) -> Option<&'a Asset> {
    let find = |suffix: &str, archs: &[&str]| assets.iter().find(|a| {
        let n = a.name.to_ascii_lowercase();
        n.ends_with(suffix) && archs.iter().any(|x| n.contains(x))
    });
    match (os, arch) {
        ("macos", "aarch64") => find(".app.zip", &["aarch64", "arm64", "universal"]),
        ("macos", "x86_64") => find(".app.zip", &["x64", "x86_64", "universal"]),
        ("windows", "x86_64") => find("-setup.exe", &["x64", "x86_64"]),
        ("linux", "x86_64") if appimage => find(".appimage", &["amd64", "x86_64"]),
        ("linux", "x86_64") => find(".deb", &["amd64", "x86_64"]),
        _ => None,
    }
}

pub fn work_dir() -> PathBuf {
    std::env::temp_dir().join("nn-builder-update")
}

/// download into `dir`; checks the size and (when the release lists it) the SHA-256
pub fn download(asset: &Asset, dir: &Path, log: &dyn Fn(String)) -> Result<PathBuf, String> {
    fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let name = Path::new(&asset.name).file_name().ok_or("The download has no file name.")?;
    let (dest, part) = (dir.join(name), dir.join(format!("{}.part", name.to_string_lossy())));
    let r = get(&asset.url, "application/octet-stream")?;
    if r.status() != 200 {
        return Err(format!("Download failed: HTTP {}", r.status().as_u16()));
    }
    let mut reader = r.into_body().into_with_config().limit(2 * 1024 * 1024 * 1024).reader();
    let mut file = fs::File::create(&part).map_err(|e| e.to_string())?;
    let (mut hasher, mut buf, mut total, mut shown) = (Sha256::new(), vec![0u8; 64 * 1024], 0u64, 0u64);
    loop {
        let n = reader.read(&mut buf).map_err(|e| format!("Download failed: {e}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).map_err(|e| e.to_string())?;
        hasher.update(&buf[..n]);
        total += n as u64;
        if total - shown > 2 * 1024 * 1024 {
            shown = total;
            log(format!("downloaded {:.1} of {:.1} MB", total as f64 / 1e6, asset.size as f64 / 1e6));
        }
    }
    drop(file);
    let fail = |m: String| {
        let _ = fs::remove_file(&part);
        Err(m)
    };
    if asset.size > 0 && total != asset.size {
        return fail(format!("The download is incomplete ({total} of {} bytes).", asset.size));
    }
    let digest: String = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect();
    if let Some(expected) = &asset.sha256 {
        if *expected != digest {
            return fail("The download is damaged (its SHA-256 does not match the release).".into());
        }
    }
    fs::rename(&part, &dest).map_err(|e| e.to_string())?;
    Ok(dest)
}

pub fn install(file: &Path) -> Result<After, String> {
    let name = file.file_name().map(|n| n.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
    if name.ends_with(".app.zip") {
        let bundle = running_bundle().ok_or("This copy is not running from an .app bundle, so it cannot replace itself.")?;
        install_app_zip(file, &bundle).map(|_| After::Restart)
    } else if name.ends_with(".exe") {
        Ok(After::Installer(file.to_path_buf()))
    } else if name.ends_with(".appimage") {
        let target = std::env::var_os("APPIMAGE").map(PathBuf::from).ok_or("This copy is not running as an AppImage.")?;
        replace_file(file, &target).map(|_| After::Restart)
    } else if name.ends_with(".deb") {
        Command::new("xdg-open").arg(file).spawn().map(|_| After::Nothing).map_err(|e| e.to_string())
    } else {
        Err(format!("Don't know how to install {}.", file.display()))
    }
}

/// the .app this process runs from (X.app/Contents/MacOS/exe)
fn running_bundle() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?.canonicalize().ok()?;
    let b = exe.parent()?.parent()?.parent()?;
    (b.extension()? == "app" && b.join("Contents/Info.plist").is_file()).then(|| b.to_path_buf())
}

/// unzip next to the bundle (same volume), check it, swap by renaming; the old bundle goes only once the new one is in place
pub fn install_app_zip(zip: &Path, bundle: &Path) -> Result<(), String> {
    let stage = sibling(bundle, "unzip");
    let _ = fs::remove_dir_all(&stage);
    run(Command::new("ditto").args(["-x", "-k"]).arg(zip).arg(&stage))?;
    let app = fs::read_dir(&stage).map_err(|e| e.to_string())?.flatten().map(|e| e.path()).find(|p| p.extension().is_some_and(|x| x == "app"))
        .ok_or("The download contains no .app.")?;
    if !app.join("Contents/Info.plist").is_file() {
        let _ = fs::remove_dir_all(&stage);
        return Err("The new app in the download is incomplete.".into());
    }
    let old = sibling(bundle, "old");
    let _ = fs::remove_dir_all(&old);
    fs::rename(bundle, &old).map_err(|e| format!("Could not move the current app aside: {e}"))?;
    if let Err(e) = fs::rename(&app, bundle) {
        let _ = fs::rename(&old, bundle);
        let _ = fs::remove_dir_all(&stage);
        return Err(format!("Could not put the new app in place: {e}"));
    }
    let _ = fs::remove_dir_all(&old);
    let _ = fs::remove_dir_all(&stage);
    let _ = Command::new("xattr").args(["-dr", "com.apple.quarantine"]).arg(bundle).status();
    Ok(())
}

fn replace_file(new: &Path, target: &Path) -> Result<(), String> {
    let staged = sibling(target, "new");
    fs::copy(new, &staged).map_err(|e| e.to_string())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&staged, fs::Permissions::from_mode(0o755)).map_err(|e| e.to_string())?;
    }
    fs::rename(&staged, target).map_err(|e| {
        let _ = fs::remove_file(&staged);
        e.to_string()
    })
}

/// /dir/Name.app -> /dir/.Name.app.<tag> (hidden, same volume)
fn sibling(p: &Path, tag: &str) -> PathBuf {
    p.with_file_name(format!(".{}.{tag}", p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()))
}

fn run(cmd: &mut Command) -> Result<(), String> {
    let out = cmd.output().map_err(|e| format!("Could not run {:?}: {e}", cmd.get_program()))?;
    if out.status.success() { Ok(()) } else { Err(format!("{:?} failed: {}", cmd.get_program(), String::from_utf8_lossy(&out.stderr).trim())) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asset(name: &str) -> Asset {
        Asset { name: name.into(), url: format!("https://example.invalid/{name}"), size: 1, sha256: None }
    }

    #[test]
    fn versions_and_assets() {
        assert_eq!(parse_version("v0.10.2"), Some((0, 10, 2)));
        assert!(parse_version("v0.10.0") > parse_version("v0.9.9"));
        assert_eq!(parse_version("latest"), None);
        let assets: Vec<Asset> = ["NN-Builder_0.2.0_aarch64.app.zip", "NN.Builder_0.2.0_x64-setup.exe", "nn-builder_0.2.0_amd64.deb", "nn-builder_0.2.0_amd64.AppImage"]
            .into_iter().map(asset).collect();
        let pick = |os, arch, ai| pick_asset(&assets, os, arch, ai).map(|a| a.name.as_str());
        assert_eq!(pick("macos", "aarch64", false), Some("NN-Builder_0.2.0_aarch64.app.zip"));
        assert_eq!(pick("macos", "x86_64", false), None);
        assert_eq!(pick("windows", "x86_64", false), Some("NN.Builder_0.2.0_x64-setup.exe"));
        assert_eq!(pick("linux", "x86_64", false), Some("nn-builder_0.2.0_amd64.deb"));
        assert_eq!(pick("linux", "x86_64", true), Some("nn-builder_0.2.0_amd64.AppImage"));
    }

    #[test]
    fn reads_a_release() {
        let json = r#"{"tag_name":"v99.1.0","html_url":"https://github.com/o/r/releases/tag/v99.1.0","body":" Notes ",
            "assets":[{"name":"NN-Builder_99.1.0_aarch64.app.zip","browser_download_url":"https://x/a.zip","size":42,"digest":"sha256:ABC"}]}"#;
        let r = parse_release(json, "macos", "aarch64", false).unwrap();
        assert!(r.newer);
        assert_eq!((r.version.as_str(), r.notes.as_str()), ("99.1.0", "Notes"));
        assert_eq!(r.asset.unwrap().sha256.as_deref(), Some("abc"));
        assert!(parse_release("{}", "macos", "aarch64", false).is_err());
    }

    /// one HTTP response per connection from a local port
    fn serve(bodies: Vec<Vec<u8>>) -> String {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        std::thread::spawn(move || {
            for body in bodies {
                let (mut s, _) = l.accept().unwrap();
                let mut req = [0u8; 4096];
                let _ = s.read(&mut req);
                let _ = s.write_all(format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len()).as_bytes());
                let _ = s.write_all(&body);
            }
        });
        format!("http://{addr}")
    }

    #[test]
    fn downloads_and_verifies() {
        let data = b"new version".to_vec();
        let sha: String = Sha256::digest(&data).iter().map(|b| format!("{b:02x}")).collect();
        let base = serve(vec![data.clone(), data.clone()]);
        let dir = std::env::temp_dir().join(format!("nnb-dl-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let good = Asset { name: "../x.app.zip".into(), url: format!("{base}/a"), size: data.len() as u64, sha256: Some(sha) };
        let f = download(&good, &dir, &|_| {}).unwrap();
        assert_eq!(f, dir.join("x.app.zip"), "the name stays inside the folder");
        assert_eq!(fs::read(&f).unwrap(), data);
        let bad = Asset { name: "y.app.zip".into(), sha256: Some("00".repeat(32)), ..good };
        assert!(download(&bad, &dir, &|_| {}).unwrap_err().contains("damaged"));
        assert!(!dir.join("y.app.zip").exists() && !dir.join("y.app.zip.part").exists());
        fs::remove_dir_all(&dir).ok();
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn swaps_an_app_bundle() {
        let d = std::env::temp_dir().join(format!("nnb-swap-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        let make = |root: &Path, v: &str| {
            fs::create_dir_all(root.join("NN Builder.app/Contents")).unwrap();
            fs::write(root.join("NN Builder.app/Contents/Info.plist"), v).unwrap();
        };
        make(&d.join("new"), "v2");
        make(&d.join("installed"), "v1");
        let zip = d.join("u.app.zip");
        run(Command::new("ditto").args(["-c", "-k", "--keepParent"]).arg(d.join("new/NN Builder.app")).arg(&zip)).unwrap();
        install_app_zip(&zip, &d.join("installed/NN Builder.app")).unwrap();
        assert_eq!(fs::read_to_string(d.join("installed/NN Builder.app/Contents/Info.plist")).unwrap(), "v2");
        assert_eq!(fs::read_dir(d.join("installed")).unwrap().count(), 1, "no leftovers");
        fs::remove_dir_all(&d).ok();
    }
}
