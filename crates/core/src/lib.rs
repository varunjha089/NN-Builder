//! NN Builder core: the design (graph), the dataset choice, the shape check, Python code generation, and remote training control
//! (Kaggle, SSH machines, AWS EC2). No UI and no PyTorch: the Mac only designs; training runs elsewhere.

pub mod cloud;
pub mod codegen;
pub mod dataset;
pub mod dsl;
pub mod jobs;
pub mod model;
pub mod remote;
pub mod runs;
pub mod shapes;
pub mod store;
pub mod templates;
