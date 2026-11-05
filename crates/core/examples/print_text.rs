//! Prints a demo's design text (for looking at the printer):  cargo run -p nnb-core --example print_text -- resnet18
fn main() {
    let key = std::env::args().nth(1).unwrap_or_else(|| "resnet18".into());
    let g = nnb_core::templates::build(&key).expect("demo key");
    print!("{}", nnb_core::dsl::sync(&g).text);
}
