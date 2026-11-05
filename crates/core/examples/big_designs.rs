//! Builds large ViT-style designs from text and times what the app does with them (no PyTorch, no training):
//!     cargo run --release -p nnb-core --example big_designs [-- <folder>]   (with a folder: also writes each as a project's model.json)
use nnb_core::dataset::Dataset;
use std::time::Instant;

fn vit(d: usize, depth: usize, heads: usize) -> String {
    format!("net \"ViT d{d} x{depth}\"\n1 input\n2 conv {d} 14x14 stride 14 valid none\n3 to_tokens\n4 cls_pos\n5 repeat {depth} {{\n  a ln\n  b attention heads {heads}\n  c add <- in, b\n  d ln\n  e dense {m} gelu\n  f dense {d} none\n  g add <- c, f\n}}\n6 ln\n7 token_pool\n8 output\n", m = 4 * d)
}

fn main() {
    let out = std::env::args().nth(1);
    let ds = Dataset { source: "huggingface".into(), hf_repo: "x/y".into(), image_size: [224, 224], channels: 3,
        classes: (0..10).map(|i| i.to_string()).collect(), ..Default::default() };
    println!("{:<22} {:>16} {:>7} {:>9} {:>9} {:>9} {:>9} {:>10} {:>10}", "design", "params", "blocks", "expand", "analyze", "sync", "print", "codegen", "json KB");
    for (name, d, depth, heads) in [("1B  d1408 x40", 1408, 40, 16), ("5B  d4096 x24", 4096, 24, 32), ("25B d8192 x32", 8192, 32, 64),
        ("25B d4608 x100", 4608, 100, 36), ("50B d8192 x64", 8192, 64, 64), ("100B d12288 x55", 12288, 55, 96), ("25B d2048 x500", 2048, 500, 16)] {
        let text = vit(d, depth, heads);
        let t = Instant::now();
        let g = nnb_core::dsl::apply(&text, &Default::default()).unwrap();
        let t_expand = t.elapsed();
        let t = Instant::now();
        let a = nnb_core::shapes::analyze(&g, &ds);
        let t_analyze = t.elapsed();
        if !a.ok {
            let bad: Vec<_> = a.nodes.iter().filter(|n| n.status != "ok").take(4).map(|n| format!("{} {}: {:?}", n.title, n.status, n.messages)).collect();
            println!("{name}: NOT OK {:?} {bad:?}", a.problems);
            continue;
        }
        let mut moved = g.clone();
        moved.nodes[3].x += 5.0;
        let t = Instant::now();
        nnb_core::dsl::sync(&moved);
        let t_sync = t.elapsed();
        let mut cleared = g.clone();
        cleared.text.clear();
        let t = Instant::now();
        nnb_core::dsl::sync(&cleared);
        let t_print = t.elapsed();
        let t = Instant::now();
        let code = nnb_core::codegen::train_script(&g, &ds, &a, &Default::default(), name).unwrap();
        let t_code = t.elapsed();
        let json = serde_json::to_string(&g).unwrap();
        let ajson = serde_json::to_string(&a).unwrap();
        println!("{:<22} {:>16} {:>7} {:>8.1?} {:>8.1?} {:>8.1?} {:>8.1?} {:>9.1?} {:>5}+{:<5}", name, a.total_params, g.nodes.len(), t_expand, t_analyze, t_sync, t_print, t_code,
            json.len() / 1024, ajson.len() / 1024);
        let _ = code;
        if let Some(dir) = &out {
            let p = format!("{dir}/{}", name.replace(' ', "_"));
            std::fs::create_dir_all(&p).unwrap();
            // a project folder the app can open (for measuring the window's memory with this design loaded)
            std::fs::create_dir_all(format!("{p}/runs")).unwrap();
            std::fs::write(format!("{p}/project.json"), serde_json::json!({"name": name, "app": "NN Builder"}).to_string()).unwrap();
            std::fs::write(format!("{p}/model.json"), serde_json::to_string(&nnb_core::dsl::sync(&g)).unwrap()).unwrap();
            std::fs::write(format!("{p}/dataset.json"), serde_json::to_string(&ds).unwrap()).unwrap();
        }
    }
}
