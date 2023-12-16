use crate::coqproject::SearchPaths;
use crate::marshal::{parse, parse_objects, parse_str, Data, Object};
use crate::types::{
  Cache, CompiledLibrary, DirPath, Library, List, OpaqueProof, Summary, VoDigest,
};
use byteorder::{ReadBytesExt, BE};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::{io, sync::Arc};
use zerocopy::big_endian::{U32, U64};
use zerocopy::{FromBytes, FromZeroes, Ref, Unaligned};

impl Data {
  pub fn dest_block<'a>(self, mem: &'a [Object<'_>]) -> (u8, &'a [Data]) {
    match self {
      Data::Atom(tag) => (tag, &[]),
      Data::Pointer(p) => match &mem[p] {
        Object::Struct(tag, data) => (*tag, data),
        k => panic!("bad data: {k:?}"),
      },
      Data::Int(tag) => (tag.try_into().unwrap(), &[]),
      _ => panic!("bad data"),
    }
  }

  pub fn dest_str<'a>(self, mem: &[Object<'a>]) -> &'a [u8] {
    match self {
      Data::Pointer(p) => match &mem[p] {
        Object::Str(data) => data,
        _ => panic!("bad data"),
      },
      _ => panic!("bad data"),
    }
  }
  pub fn dest_int(self, mem: &[Object<'_>]) -> i64 {
    match self {
      Data::Pointer(p) => match mem[p] {
        Object::Int64(data) => data,
        _ => panic!("bad data"),
      },
      Data::Int(n) => n,
      _ => panic!("bad data"),
    }
  }
  pub fn dest_float(self, mem: &[Object<'_>]) -> f64 {
    match self {
      Data::Pointer(p) => match mem[p] {
        Object::Float(data) => data,
        _ => panic!("bad data"),
      },
      _ => panic!("bad data"),
    }
  }
}

pub trait Cacheable {
  fn get_mut(cache: &mut Cache) -> &mut HashMap<usize, Arc<Self>>;
}

impl Cache {
  pub fn try_from_data<T: Cacheable + ?Sized>(&mut self, d: Data) -> Option<Arc<T>> {
    if let Data::Pointer(p) = d {
      return Some(T::get_mut(self).get(&p)?.clone())
    }
    None
  }
}

pub trait FromData {
  fn from_data(d: Data, mem: &[Object<'_>], cache: &mut Cache) -> Self;
}

impl FromData for i64 {
  fn from_data(d: Data, mem: &[Object<'_>], _: &mut Cache) -> Self { d.dest_int(mem) }
}
impl FromData for u32 {
  fn from_data(d: Data, mem: &[Object<'_>], _: &mut Cache) -> Self {
    d.dest_int(mem).try_into().unwrap()
  }
}
impl FromData for f64 {
  fn from_data(d: Data, mem: &[Object<'_>], _: &mut Cache) -> Self { d.dest_float(mem) }
}

macro_rules! impl_from_data_tuple {
  ($(($($ty:ident),*);)*) => {
    $(impl<$($ty: FromData),*> FromData for ($($ty,)*) {
      #[allow(non_snake_case)]
      fn from_data(d: Data, mem: &[Object<'_>], cache: &mut Cache) -> Self {
        match d.dest_block(mem) {
          (0, [$($ty),*]) => ($(FromData::from_data(*$ty, mem, cache),)*),
          k => panic!("bad tag in {}: {k:?}", stringify!(($($ty),*)))
        }
      }
    })*
  }
}
impl_from_data_tuple! { (A); (A, B); (A, B, C); (A, B, C, D); }

impl<T: FromData> FromData for Vec<T> {
  fn from_data(d: Data, mem: &[Object<'_>], cache: &mut Cache) -> Self {
    let (_, args) = d.dest_block(mem);
    args.iter().map(|d| T::from_data(*d, mem, cache)).collect()
  }
}

impl<T: FromData> FromData for Box<T> {
  fn from_data(d: Data, mem: &[Object<'_>], cache: &mut Cache) -> Self {
    Box::new(T::from_data(d, mem, cache))
  }
}

macro_rules! impl_from_data_enum {
  ($(enum$(<$($g:ident),*>)? for $name:ty {
    $($n:literal$(($($a:ident),*))? => $e:expr,)*
  })*) => {
    $(impl$(<$($g: FromData),*>)? FromData for $name {
      fn from_data(d: Data, mem: &[Object<'_>], _cache: &mut Cache) -> Self {
        match d.dest_block(mem) {
          $(($n, [$($($a),*)?]) => {
            let ($($($a,)*)?) = ($($(FromData::from_data(*$a, mem, _cache),)*)?);
            $e
          },)*
          k => panic!("bad tag: {k:?}"),
        }
      }
    })*
  }
}

impl<T: FromData> FromData for List<T> {
  fn from_data(mut d: Data, mem: &[Object<'_>], cache: &mut Cache) -> Self {
    let mut out = vec![];
    loop {
      match d.dest_block(mem) {
        (0, []) => {
          out.reverse();
          return List(out)
        }
        (0, [a, b]) => {
          out.push(FromData::from_data(*a, mem, cache));
          d = *b;
        }
        k => panic!("bad tag: {k:?}"),
      }
    }
  }
}

impl_from_data_enum! {
  enum for () { 0() => (), }
  enum for bool { 0() => false, 1() => true, }
  enum<T> for Option<T> { 0() => None, 0(a) => Some(a), }
}

impl FromData for String {
  fn from_data(d: Data, mem: &[Object<'_>], _: &mut Cache) -> Self {
    Self::from_utf8(d.dest_str(mem).to_vec()).unwrap()
  }
}

impl FromData for Box<[u8]> {
  fn from_data(d: Data, mem: &[Object<'_>], _: &mut Cache) -> Self {
    d.dest_str(mem).to_vec().into()
  }
}

impl<T: FromData + Ord> FromData for BTreeSet<T> {
  fn from_data(mut d: Data, mem: &[Object<'_>], cache: &mut Cache) -> Self {
    let mut stack = vec![];
    let mut out = BTreeSet::new();
    loop {
      match d.dest_block(mem) {
        (0, []) => match stack.pop() {
          Some(v) => d = v,
          None => return out,
        },
        (0, [l, v, r, _]) => {
          out.insert(T::from_data(*v, mem, cache));
          d = *l;
          stack.push(*r);
        }
        k => panic!("bad tag: {k:?}"),
      }
    }
  }
}

impl<K: FromData + Ord, V: FromData> FromData for BTreeMap<K, V> {
  fn from_data(mut d: Data, mem: &[Object<'_>], cache: &mut Cache) -> Self {
    let mut stack = vec![];
    let mut out = BTreeMap::new();
    loop {
      match d.dest_block(mem) {
        (0, []) => match stack.pop() {
          Some(v) => d = v,
          None => return out,
        },
        (0, [l, k, v, r, _]) => {
          out.insert(K::from_data(*k, mem, cache), V::from_data(*v, mem, cache));
          d = *l;
          stack.push(*r);
        }
        k => panic!("bad tag: {k:?}"),
      }
    }
  }
}

impl FromData for Data {
  fn from_data(d: Data, _: &[Object<'_>], _: &mut Cache) -> Self { d }
}

#[derive(Debug, FromZeroes, FromBytes)]
#[repr(C)]
struct Header {
  magic: [u8; 4],
  version: U32,
  summary_pos: U64,
}

#[derive(Debug, FromZeroes, FromBytes, Unaligned)]
#[repr(C)]
struct Segment {
  pos: U64,
  len: U64,
  hash: [u8; 16],
}

impl Library {
  pub fn from_file(path: impl AsRef<std::path::Path>, name2: &DirPath) -> io::Result<Library> {
    let buf = std::fs::read(path)?;
    let (header, _) = Ref::<_, Header>::new_from_prefix(&*buf).unwrap();
    assert!(header.magic == *b"Coq!");
    match header.version.get() {
      81800 => {}
      n => panic!("unsupported version {n}"),
    }
    fn parse_seg<'a>(summary: &mut &'a [u8], name: &'static [u8]) -> &'a Segment {
      assert!(parse_str(parse::<U32>(summary).get() as usize, summary) == name);
      parse(summary)
    }
    fn parse_as<T: FromData>(buf: &[u8], seg: &Segment) -> T {
      let (mut pos, next) = &buf[seg.pos.get() as usize..].split_at(seg.len.get() as usize);
      let mut memory = vec![];
      let root = parse_objects(&mut pos, &mut memory);
      assert!(pos.is_empty() && next[..16] == seg.hash);
      T::from_data(root, &memory, &mut Cache::default())
    }
    let mut summary = &buf[header.summary_pos.get() as usize..];
    assert!(summary.read_u32::<BE>()? == 5);
    let seg_md = parse_seg(&mut summary, b"library");
    let compiled = parse_as::<(CompiledLibrary, Data, Data)>(&buf, seg_md).0;
    let opaques = parse_as::<Vec<Option<OpaqueProof>>>(&buf, parse_seg(&mut summary, b"opaques"));
    let Summary { name, deps, .. } = parse_as::<Summary>(&buf, parse_seg(&mut summary, b"summary"));
    parse_as::<()>(&buf, parse_seg(&mut summary, b"tasks"));
    parse_as::<()>(&buf, parse_seg(&mut summary, b"universes"));
    assert!(name == *name2);
    // println!("{:#?}", seg_sd);
    // println!("{:#?}", deps);
    Ok(Library { compiled, opaques, deps, digest: VoDigest::VoOrVi(Box::new(seg_md.hash)) })
  }
}

impl SearchPaths {
  pub fn load_lib(&self, name: &DirPath) -> io::Result<Library> {
    let Some(path) = self.find_path(name) else { panic!("{name} not found in loadpath") };
    Library::from_file(path, name)
  }
}
