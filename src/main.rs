#![forbid(unsafe_code)]
use types::Library;

mod marshal;
mod parse;
mod types;

struct Environment {}

impl Environment {
  fn add_lib(&mut self, lib: &Library, _check: bool) {
    println!("{} proofs", lib.opaques.len())
    // todo
  }
}

fn main() {
  let mut env = Environment {};

  let lib = Library::from_file("../metacoq/pcuic/theories/PCUICAlpha.vo").unwrap();
  env.add_lib(&lib, true)
}
