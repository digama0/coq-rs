use crate::coqproject::SearchPaths;
use crate::marshal::{parse, parse_objects, parse_str, Data, Object};
use crate::types::{
  Cache, CompiledLibrary, DirPath, Library, List, OpaqueProof, RList, Summary, VoDigest,
};
use byteorder::{ReadBytesExt, BE};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::{io, sync::Arc};
use zerocopy::big_endian::{U32, U64};
use zerocopy::{FromBytes, FromZeroes, Ref, Unaligned};

impl Data {
  pub fn dest_block(self, mem: &'static [Object]) -> (u8, &'static [Data]) {
    match self {
      Data::Atom(tag) => (tag, &[]),
      Data::Pointer(p) => match &mem[p as usize] {
        Object::Struct(tag, data) => (*tag, data),
        k => panic!("bad data: {k:?}"),
      },
      Data::Int(tag) => (tag.try_into().unwrap(), &[]),
      _ => panic!("bad data"),
    }
  }

  pub fn dest_str(self, mem: &'static [Object]) -> &'static [u8] {
    match self {
      Data::Pointer(p) => match &mem[p as usize] {
        Object::Str(data) => data,
        _ => panic!("bad data"),
      },
      _ => panic!("bad data"),
    }
  }
  pub fn dest_int(self, mem: &'static [Object]) -> i64 {
    match self {
      Data::Pointer(p) => match mem[p as usize] {
        Object::Int64(data) => data,
        _ => panic!("bad data"),
      },
      Data::Int(n) => n,
      _ => panic!("bad data"),
    }
  }
  pub fn dest_float(self, mem: &'static [Object]) -> f64 {
    match self {
      Data::Pointer(p) => match mem[p as usize] {
        Object::Float(data) => data,
        _ => panic!("bad data"),
      },
      _ => panic!("bad data"),
    }
  }
}

pub trait Cacheable {
  fn get_mut(cache: &mut Cache) -> &mut HashMap<u32, Arc<Self>>;
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
  fn from_data(d: Data, mem: &'static [Object], cache: &mut Cache) -> Self;
}

impl FromData for i64 {
  fn from_data(d: Data, mem: &'static [Object], _: &mut Cache) -> Self { d.dest_int(mem) }
}
impl FromData for u32 {
  fn from_data(d: Data, mem: &'static [Object], _: &mut Cache) -> Self {
    d.dest_int(mem).try_into().unwrap()
  }
}
impl FromData for f64 {
  fn from_data(d: Data, mem: &'static [Object], _: &mut Cache) -> Self { d.dest_float(mem) }
}

macro_rules! impl_from_data_tuple {
  ($(($($ty:ident),*);)*) => {
    $(impl<$($ty: FromData),*> FromData for ($($ty,)*) {
      #[allow(non_snake_case)]
      fn from_data(d: Data, mem: &'static [Object], cache: &mut Cache) -> Self {
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
  fn from_data(d: Data, mem: &'static [Object], cache: &mut Cache) -> Self {
    let (_, args) = d.dest_block(mem);
    args.iter().map(|d| T::from_data(*d, mem, cache)).collect()
  }
}

impl<T: FromData> FromData for Box<T> {
  fn from_data(d: Data, mem: &'static [Object], cache: &mut Cache) -> Self {
    Box::new(T::from_data(d, mem, cache))
  }
}

macro_rules! impl_from_data_enum {
  ($(enum$(<$($g:ident),*>)? for $name:ty {
    $($n:literal$(($($a:ident),*))? => $e:expr,)*
  })*) => {
    $(impl$(<$($g: FromData),*>)? FromData for $name {
      fn from_data(d: Data, mem: &'static [Object], _cache: &mut Cache) -> Self {
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
  fn from_data(mut d: Data, mem: &'static [Object], cache: &mut Cache) -> Self {
    let mut out = vec![];
    loop {
      match d.dest_block(mem) {
        (0, []) => return List(out),
        (0, [a, b]) => {
          out.push(FromData::from_data(*a, mem, cache));
          d = *b;
        }
        k => panic!("bad tag: {k:?}"),
      }
    }
  }
}

impl<T: FromData> FromData for RList<T> {
  fn from_data(d: Data, mem: &'static [Object], cache: &mut Cache) -> Self {
    let List(mut out) = List::from_data(d, mem, cache);
    out.reverse();
    RList(out)
  }
}

impl_from_data_enum! {
  enum for () { 0() => (), }
  enum for bool { 0() => false, 1() => true, }
  enum<T> for Option<T> { 0() => None, 0(a) => Some(a), }
}

impl FromData for &'static str {
  fn from_data(d: Data, mem: &'static [Object], _: &mut Cache) -> Self {
    std::str::from_utf8(d.dest_str(mem)).unwrap()
  }
}

impl FromData for String {
  fn from_data(d: Data, mem: &'static [Object], _: &mut Cache) -> Self {
    Self::from_utf8(d.dest_str(mem).to_vec()).unwrap()
  }
}

impl FromData for Box<[u8]> {
  fn from_data(d: Data, mem: &'static [Object], _: &mut Cache) -> Self {
    d.dest_str(mem).to_vec().into()
  }
}

fn parse_set(
  mut d: Data, mem: &'static [Object], cache: &mut Cache, stack: &mut Vec<Data>,
  f: &mut impl FnMut(Data, &mut Cache),
) {
  loop {
    match d.dest_block(mem) {
      (0, []) => match stack.pop() {
        Some(v) => d = v,
        None => return,
      },
      (0, [l, k, r, _]) => {
        f(*k, cache);
        d = *l;
        stack.push(*r);
      }
      k => panic!("bad tag: {k:?}"),
    }
  }
}

fn parse_map(
  mut d: Data, mem: &'static [Object], cache: &mut Cache, stack: &mut Vec<Data>,
  f: &mut impl FnMut(Data, Data, &mut Cache),
) {
  loop {
    match d.dest_block(mem) {
      (0, []) => match stack.pop() {
        Some(v) => d = v,
        None => return,
      },
      (0, [l, k, v, r, _]) => {
        f(*k, *v, cache);
        d = *l;
        stack.push(*r);
      }
      k => panic!("bad tag: {k:?}"),
    }
  }
}

impl<T: FromData + Ord> FromData for BTreeSet<T> {
  fn from_data(d: Data, mem: &'static [Object], cache: &mut Cache) -> Self {
    let mut out = BTreeSet::new();
    parse_set(d, mem, cache, &mut vec![], &mut |k, cache| {
      out.insert(T::from_data(k, mem, cache));
    });
    out
  }
}

impl<K: FromData + Ord, V: FromData> FromData for BTreeMap<K, V> {
  fn from_data(d: Data, mem: &'static [Object], cache: &mut Cache) -> Self {
    let mut out = BTreeMap::new();
    parse_map(d, mem, cache, &mut vec![], &mut |k, v, cache| {
      out.insert(K::from_data(k, mem, cache), V::from_data(v, mem, cache));
    });
    out
  }
}

impl<T: FromData + std::hash::Hash + Eq> FromData for HashSet<T> {
  fn from_data(d: Data, mem: &'static [Object], cache: &mut Cache) -> Self {
    let mut stack = vec![];
    let mut out = HashSet::new();
    parse_map(d, mem, cache, &mut vec![], &mut |_, v, cache| {
      parse_set(v, mem, cache, &mut stack, &mut |k, cache| {
        out.insert(T::from_data(k, mem, cache));
      })
    });
    out
  }
}

impl<K: FromData + std::hash::Hash + Eq, V: FromData> FromData for HashMap<K, V> {
  fn from_data(d: Data, mem: &'static [Object], cache: &mut Cache) -> Self {
    let mut stack = vec![];
    let mut out = HashMap::new();
    parse_map(d, mem, cache, &mut vec![], &mut |_, v, cache| {
      parse_map(v, mem, cache, &mut stack, &mut |k, v, cache| {
        out.insert(K::from_data(k, mem, cache), V::from_data(v, mem, cache));
      })
    });
    out
  }
}

#[derive(Debug)]
pub struct Any;

pub struct Segment {
  pub hash: [u8; 16],
  pub root: Data,
  pub mem: &'static [Object],
  pub cache: Cache,
}
impl Segment {
  pub fn get<T: FromData>(&mut self, d: Data) -> T { T::from_data(d, self.mem, &mut self.cache) }
  pub fn root<T: FromData>(&mut self) -> T { self.get(self.root) }
}

impl FromData for Any {
  fn from_data(_: Data, _: &[Object], _: &mut Cache) -> Self { Any }
}

#[derive(Clone, Debug)]
pub enum Lazy<T> {
  Unloaded(Data),
  Loaded(T),
}
impl<T> FromData for Lazy<T> {
  fn from_data(d: Data, _: &'static [Object], _: &mut Cache) -> Self { Self::Unloaded(d) }
}
impl<T: FromData> Lazy<T> {
  pub fn get(&mut self, seg: &mut Segment) -> &mut T {
    if let Self::Unloaded(d) = *self {
      *self = Self::Loaded(seg.get(d));
    }
    let Self::Loaded(t) = self else { unreachable!() };
    t
  }
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
struct SegmentHeader {
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
    fn parse_seg(
      summary: &mut &[u8], buf: &[u8], arena: &'static typed_arena::Arena<Data>,
      name: &'static [u8],
    ) -> Box<Segment> {
      assert!(parse_str(parse::<U32>(summary).get() as usize, summary) == name);
      let seg = parse::<SegmentHeader>(summary);
      let (mut pos, next) = &buf[seg.pos.get() as usize..].split_at(seg.len.get() as usize);
      let mut memory = vec![];
      let root = parse_objects(&mut pos, &mut memory, arena);
      assert!(pos.is_empty() && next[..16] == seg.hash);
      Box::new(Segment { root, hash: seg.hash, mem: Vec::leak(memory), cache: Cache::default() })
    }
    let arena = Box::leak(Box::new(typed_arena::Arena::new()));
    let mut summary = &buf[header.summary_pos.get() as usize..];
    assert!(summary.read_u32::<BE>()? == 5);
    let mut seg_md = parse_seg(&mut summary, &buf, arena, b"library");
    let compiled = seg_md.root::<(CompiledLibrary, Any, Any)>().0;
    let opaques =
      parse_seg(&mut summary, &buf, arena, b"opaques").root::<Vec<Option<OpaqueProof>>>();
    let Summary { name, deps, .. } = parse_seg(&mut summary, &buf, arena, b"summary").root();
    parse_seg(&mut summary, &buf, arena, b"tasks").root::<()>();
    parse_seg(&mut summary, &buf, arena, b"universes").root::<()>();
    assert!(name == *name2);
    // println!("{:#?}", seg_sd);
    // println!("{:#?}", deps);
    Ok(Library { compiled, opaques, deps, digest: VoDigest::VoOrVi(Box::new(seg_md.hash)) })
  }
}

impl SearchPaths {
  pub fn load_lib(&self, name: &DirPath) -> io::Result<Library> {
    let Some(path) = self.find_path(name) else { panic!("{name} not found in loadpath") };
    let mut v = path.clone();
    v.set_extension("v");
    println!("loading {name} -> {}", v.display());
    Library::from_file(path, name)
  }
}
