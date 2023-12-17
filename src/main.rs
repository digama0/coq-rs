#![forbid(unsafe_code)]
use std::path::{Path, PathBuf};
use std::process::Command;
use types::DirPath;
mod coqproject;
mod environment;
mod marshal;
mod parse;
mod types;

fn main() {
  let root = {
    let script = "eval $(opam env --cli=2.1 --shell=sh); echo $OPAM_SWITCH_PREFIX";
    let out = Command::new("sh").arg("-c").arg(script).output().unwrap().stdout;
    PathBuf::from(String::from_utf8(out).unwrap().trim())
  };
  let coqlib = std::env::var("COQLIB").map_or_else(|_| root.join("lib/coq"), PathBuf::from);
  let coqcorelib = if let Ok(s) = std::env::var("COQCORELIB") {
    PathBuf::from(s)
  } else {
    let plugins = coqlib.join("plugins");
    if plugins.exists() {
      plugins
    } else {
      coqlib.join("../coq-core")
    }
  };
  let theories_dir = coqlib.join("theories");
  assert!(theories_dir.join("Init/Prelude.vo").exists());
  assert!(coqcorelib.join("plugins").exists());
  let base: &Path = "../metacoq/pcuic".as_ref();
  let mut env = environment::Environment::default();
  env.paths.includes.push((coqlib.join("theories"), "Coq".into()));
  env.paths.includes.push((coqlib.join("user-contrib"), DirPath(vec![])));
  env.paths.parse(&base.join("_CoqProject"), base).unwrap();
  let target = DirPath::from("MetaCoq.PCUIC.PCUICAlpha");
  env.get_or_load_lib(&target, true);
  std::mem::forget(env); // rust has trouble dropping deeply nested exprs
}
