//! Prints a famous architecture as model.json (development helper):  cargo run -p nnb-core --example template_json -- unet
fn main() {
    let key = std::env::args().nth(1).unwrap_or_else(|| "lenet5".into());
    let g = nnb_core::templates::build(&key).unwrap_or_else(|| panic!("unknown template {key}"));
    println!("{}", serde_json::to_string_pretty(&g).unwrap());
}
