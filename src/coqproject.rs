use std::{
  io,
  path::{Path, PathBuf},
};

use crate::types::DirPath;

#[derive(Default, Debug)]
pub struct SearchPaths {
  pub includes: Vec<(PathBuf, DirPath)>,
}

impl SearchPaths {
  pub fn process_args(&mut self, base: &Path, mut it: &[&str]) -> io::Result<()> {
    loop {
      match it {
        [] => break,
        ["-impredicative-set", _] =>
          panic!("Use \"-arg -impredicative-set\" instead of \"-impredicative-set\""),
        &["-Q", phys_path, log_path, ref rest @ ..] => {
          let log_path = &*String::leak(log_path.into());
          self.includes.push((base.join(phys_path), DirPath::from(log_path)));
          it = rest;
        }
        &["-I", _phys_path, ref rest @ ..] => {
          // self.ml_includes.push(base.join(phys_path));
          it = rest;
        }
        &["-R", phys_path, log_path, ref rest @ ..] => {
          let log_path = &*String::leak(log_path.into());
          self.includes.push((base.join(phys_path), DirPath::from(log_path)));
          it = rest;
        }
        &["-native-compiler", ref rest @ ..] => {
          println!("ignoring -native-compiler");
          it = rest;
        }
        &["-f", file, ref rest @ ..] => {
          let file = base.join(file);
          self.parse(&file, file.parent().unwrap())?;
          it = rest;
        }
        &["-o", file, ref rest @ ..] => {
          println!("ignoring -o {file}");
          it = rest;
        }
        &["-docroot", _, ref rest @ ..] => {
          it = rest;
        }
        &["-generate-meta-for-package", m, ref rest @ ..] => {
          println!("ignoring generate-meta-for-package {m}");
          it = rest;
        }
        &[v, "=", def, ref rest @ ..] => {
          println!("ignoring {v}={def}");
          it = rest;
        }
        &["-arg", a, ref rest @ ..] => {
          println!("ignoring arg {a}");
          it = rest;
        }
        [v_file, rest @ ..] => {
          let file = base.join(v_file);
          match file.extension().and_then(|s| s.to_str()) {
            Some("v") => {}
            _ => println!("ignoring file {}", file.display()),
          }
          it = rest;
        }
      }
    }
    Ok(())
  }

  pub fn parse(&mut self, path: &Path, base: &Path) -> io::Result<()> {
    let file = std::fs::read_to_string(path)?;
    let mut args = vec![];
    let mut it = file.as_bytes();
    loop {
      if let Some(i) = it.iter().position(|x| !x.is_ascii_whitespace()) {
        it = &it[i..]
      }
      match it.first() {
        None => break,
        Some(b'#') => {
          let Some(i) = it.iter().position(|&x| x == b'\n') else { break };
          it = &it[i + 1..]
        }
        Some(b'"') => {
          let i = it.iter().position(|&x| x == b'"').expect("parse error, unterminated string");
          args.push(std::str::from_utf8(&it[..i]).unwrap());
          it = &it[i + 1..]
        }
        Some(_) => {
          let i = it.iter().position(|x| x.is_ascii_whitespace()).unwrap_or(it.len());
          args.push(std::str::from_utf8(&it[..i]).unwrap());
          it = &it[i..]
        }
      }
    }
    self.process_args(base, &args)
  }

  pub fn find_path(&self, path: &DirPath) -> Option<PathBuf> {
    for (phys, log) in &self.includes {
      if let Some(subdir) = path.0.strip_prefix(&*log.0) {
        let mut phys = phys.to_owned();
        subdir.iter().for_each(|path| phys.push(path));
        phys.set_extension("vo");
        if phys.exists() {
          return Some(phys)
        }
      }
    }
    None
  }
}
