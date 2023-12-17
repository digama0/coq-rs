use crate::{coqproject::SearchPaths, types::*};
use std::{collections::HashMap, sync::Arc};

#[derive(Default)]
pub struct Environment {
  pub paths: SearchPaths,
  libs: HashMap<DirPath, Arc<Library>>,
  consts: HashMap<UserKerPair, Arc<ConstBody>>,
  inds: HashMap<UserKerPair, Arc<MutIndBody>>,
  mods: HashMap<ModPath, Arc<ModBody>>,
  modtypes: HashMap<ModPath, Arc<ModTypeBody>>,
}

impl Environment {
  pub fn get_or_load_lib(&mut self, dir: &DirPath, check: bool) -> Arc<Library> {
    if let Some(lib) = self.libs.get(dir) {
      return lib.clone()
    }
    // println!("loading {} ...", dir);
    let lib = Arc::new(self.paths.load_lib(dir).unwrap());
    self.add_lib(dir, &lib, check);
    self.libs.entry(dir.clone()).or_insert(lib).clone()
  }

  fn add_lib(&mut self, name: &DirPath, lib: &Library, check: bool) {
    for (dep, digest) in &lib.deps {
      let deplib = self.get_or_load_lib(dep, false);
      assert!(deplib.digest == *digest, "digest failure, {name} imports {dep}");
    }
    self.add_module(&lib.compiled.mod_)
  }

  fn add_retroknowledge(&mut self, act: &RetroAction) {
    //
  }

  fn add_module(&mut self, mod_: &Arc<ModBody>) {
    self.mods.insert(mod_.path.clone(), mod_.clone());
    if let ModSig::NoFunctor(struc) = &mod_.ty {
      self.add_structure(&mod_.path, &struc.0, &mod_.delta);
      for act in &mod_.retro.0 .0 {
        self.add_retroknowledge(act)
      }
    }
  }

  fn add_structure(
    &mut self, path: &ModPath, struc: &[(Label, StructFieldBody)], resolver: &DeltaResolver,
  ) {
    for (l, sfb) in struc {
      match sfb {
        StructFieldBody::Const(cb) => {
          // todo!("{cb:#?}, {resolver:#?}")
        }
        StructFieldBody::MutInd(ind) => {
          // todo!("{ind:#?}, {resolver:#?}")
        }
        StructFieldBody::Module(mb) => self.add_module(mb),
        StructFieldBody::ModType(_) => {
          // todo!()
        }
      }
    }
  }
}
