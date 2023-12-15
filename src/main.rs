#![allow(unused)]
use std::{
  any::TypeId,
  collections::{BTreeMap, BTreeSet, HashMap, HashSet},
  fs::File,
  io::{self, BufRead, BufReader, Read, Seek},
  marker::PhantomData,
  rc::Rc,
};

use byteorder::{ReadBytesExt, BE, LE};
use memmap2::Mmap;
use zerocopy::{
  big_endian::{I16, I32, I64, U16, U32, U64},
  FromBytes, FromZeroes, Ref, Unaligned, F64,
};

fn read_string(r: &mut &[u8], len: usize) -> std::io::Result<String> {
  let mut out = vec![0; len];
  r.read_exact(&mut out)?;
  Ok(String::from_utf8(out).unwrap())
}

#[derive(Debug, FromZeroes, FromBytes, Unaligned)]
#[repr(C)]
struct Segment {
  pos: U64,
  len: U64,
  hash: [u8; 16],
}

#[derive(Debug, FromZeroes, FromBytes)]
#[repr(C)]
struct Header {
  magic: [u8; 4],
  version: U32,
  summary_pos: U64,
}

fn parse<'a, T: FromBytes>(pos: &mut &'a [u8]) -> &'a T {
  let (t, pos2) = Ref::<_, T>::new_from_prefix(*pos).unwrap();
  *pos = pos2;
  t.into_ref()
}

fn parse_u8(pos: &mut &[u8]) -> u8 {
  let (t, pos2) = pos.split_first().unwrap();
  *pos = pos2;
  *t
}
fn parse_str<'a>(n: usize, pos: &mut &'a [u8]) -> &'a [u8] {
  let (t, pos2) = pos.split_at(n);
  *pos = pos2;
  t
}

mod tag {
  pub(crate) const PREFIX_SMALL_BLOCK: u8 = 0x80;
  pub(crate) const PREFIX_SMALL_INT: u8 = 0x40;
  pub(crate) const PREFIX_SMALL_STRING: u8 = 0x20;
  pub(crate) const CODE_INT8: u8 = 0x00;
  pub(crate) const CODE_INT16: u8 = 0x01;
  pub(crate) const CODE_INT32: u8 = 0x02;
  pub(crate) const CODE_INT64: u8 = 0x03;
  pub(crate) const CODE_SHARED8: u8 = 0x04;
  pub(crate) const CODE_SHARED16: u8 = 0x05;
  pub(crate) const CODE_SHARED32: u8 = 0x06;
  pub(crate) const CODE_DOUBLE_ARRAY32_LITTLE: u8 = 0x07;
  pub(crate) const CODE_BLOCK32: u8 = 0x08;
  pub(crate) const CODE_STRING8: u8 = 0x09;
  pub(crate) const CODE_STRING32: u8 = 0x0A;
  pub(crate) const CODE_DOUBLE_BIG: u8 = 0x0B;
  pub(crate) const CODE_DOUBLE_LITTLE: u8 = 0x0C;
  pub(crate) const CODE_DOUBLE_ARRAY8_BIG: u8 = 0x0D;
  pub(crate) const CODE_DOUBLE_ARRAY8_LITTLE: u8 = 0x0E;
  pub(crate) const CODE_DOUBLE_ARRAY32_BIG: u8 = 0x0F;
  pub(crate) const CODE_CODEPOINTER: u8 = 0x10;
  pub(crate) const CODE_INFIXPOINTER: u8 = 0x11;
  pub(crate) const CODE_CUSTOM: u8 = 0x12;
  pub(crate) const CODE_BLOCK64: u8 = 0x13;
  pub(crate) const CODE_SHARED64: u8 = 0x14;
  pub(crate) const CODE_STRING64: u8 = 0x15;
  pub(crate) const CODE_DOUBLE_ARRAY64_BIG: u8 = 0x16;
  pub(crate) const CODE_DOUBLE_ARRAY64_LITTLE: u8 = 0x17;
  pub(crate) const CODE_CUSTOM_LEN: u8 = 0x18;
  pub(crate) const CODE_CUSTOM_FIXED: u8 = 0x19;
}

#[derive(Debug)]
enum Tag<'a> {
  Block(u8, usize),
  Int(i64),
  Str(&'a [u8]),
  Pointer(usize),
  Code(usize),
  Int64(i64),
  Float(f64),
}

fn parse_object<'a>(pos: &mut &'a [u8]) -> Tag<'a> {
  let tag = parse_u8(pos);
  match tag {
    tag::PREFIX_SMALL_BLOCK..=u8::MAX => Tag::Block(tag & 0xf, ((tag >> 4) & 0x7) as usize),
    tag::PREFIX_SMALL_INT..=u8::MAX => Tag::Int((tag & 0x3f) as i64),
    tag::PREFIX_SMALL_STRING..=u8::MAX => Tag::Str(parse_str((tag & 0x1f) as usize, pos)),
    tag::CODE_INT8 => Tag::Int((parse_u8(pos) as i8).into()),
    tag::CODE_INT16 => Tag::Int(parse::<I16>(pos).get().into()),
    tag::CODE_INT32 => Tag::Int(parse::<I32>(pos).get().into()),
    tag::CODE_INT64 => Tag::Int(parse::<I64>(pos).get()),
    tag::CODE_SHARED8 => Tag::Pointer(parse_u8(pos).into()),
    tag::CODE_SHARED16 => Tag::Pointer(parse::<U16>(pos).get().into()),
    tag::CODE_SHARED32 => Tag::Pointer(parse::<U32>(pos).get() as usize),
    tag::CODE_BLOCK32 => {
      let val = parse::<U32>(pos).get();
      Tag::Block(val as u8, val as usize >> 10)
    }
    tag::CODE_BLOCK64 => {
      let val = parse::<U64>(pos).get();
      Tag::Block(val as u8, val as usize >> 10)
    }
    tag::CODE_STRING8 => Tag::Str(parse_str(parse_u8(pos) as usize, pos)),
    tag::CODE_STRING32 => Tag::Str(parse_str(parse::<U32>(pos).get() as usize, pos)),
    tag::CODE_CODEPOINTER =>
      Tag::Code((parse::<U32>(pos).get(), parse::<[u8; 16]>(pos)).0 as usize),
    tag::CODE_CUSTOM | tag::CODE_CUSTOM_FIXED => {
      let s = std::ffi::CStr::from_bytes_until_nul(pos).unwrap().to_bytes();
      *pos = &pos[s.len() + 1..];
      match s {
        b"_j" => Tag::Int64(parse::<I64>(pos).get()),
        s => panic!("unhandled custom code: {}", String::from_utf8_lossy(s)),
      }
    }
    tag::CODE_DOUBLE_BIG => Tag::Float(parse::<F64<BE>>(pos).get()),
    tag::CODE_DOUBLE_LITTLE => Tag::Float(parse::<F64<LE>>(pos).get()),
    tag::CODE_DOUBLE_ARRAY32_LITTLE
    | tag::CODE_DOUBLE_ARRAY8_BIG
    | tag::CODE_DOUBLE_ARRAY8_LITTLE
    | tag::CODE_DOUBLE_ARRAY32_BIG
    | tag::CODE_INFIXPOINTER
    | tag::CODE_SHARED64
    | tag::CODE_STRING64
    | tag::CODE_DOUBLE_ARRAY64_BIG
    | tag::CODE_DOUBLE_ARRAY64_LITTLE
    | tag::CODE_CUSTOM_LEN => panic!("unhandled tag {tag}"),
    _ => panic!("unknown tag {tag}"),
  }
}

#[derive(Debug, FromZeroes, FromBytes)]
struct ObjectHeader {
  magic: [u8; 4],
  length: U32,
  objects: U32,
  size32: U32,
  size64: U32,
}

#[derive(Copy, Clone, Debug)]
enum Data {
  Atom(u8),
  Int(i64),
  Pointer(usize),
  Code(usize),
}

enum Object<'a> {
  Struct(u8, Box<[Data]>),
  Int64(i64),
  Float(f64),
  Str(&'a [u8]),
}

impl<'a> std::fmt::Debug for Object<'a> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Struct(arg0, arg1) => f.debug_tuple("Struct").field(arg0).field(arg1).finish(),
      Self::Int64(arg0) => f.debug_tuple("Int64").field(arg0).finish(),
      Self::Float(arg0) => f.debug_tuple("Float").field(arg0).finish(),
      Self::Str(arg0) => f.debug_tuple("Str").field(&String::from_utf8_lossy(arg0)).finish(),
    }
  }
}

fn parse_objects<'a>(pos: &mut &'a [u8], objects: &mut Vec<Object<'a>>) -> Data {
  fn push_vec<T>(vec: &mut Vec<T>, val: T) -> usize { (vec.len(), vec.push(val)).0 }
  let header = parse::<ObjectHeader>(pos);
  assert_eq!(header.magic, [132, 149, 166, 190]);
  let mut stack = vec![];
  let mut result = Data::Atom(0);
  loop {
    let (data, to_push) = match parse_object(pos) {
      Tag::Pointer(n) => (Data::Pointer(objects.len() - n), None),
      Tag::Int(n) => (Data::Int(n), None),
      Tag::Str(s) => (Data::Pointer(push_vec(objects, Object::Str(s))), None),
      Tag::Block(tag, 0) => (Data::Atom(tag), None),
      Tag::Block(tag, len) => {
        let p = push_vec(objects, Object::Struct(tag, vec![Data::Atom(0); len].into_boxed_slice()));
        (Data::Pointer(p), Some(p))
      }
      Tag::Code(addr) => (Data::Code(addr), None),
      Tag::Int64(n) => (Data::Pointer(push_vec(objects, Object::Int64(n))), None),
      Tag::Float(n) => (Data::Pointer(push_vec(objects, Object::Float(n))), None),
    };
    if let Some((p, off)) = stack.last_mut() {
      let Object::Struct(_, args) = &mut objects[*p] else { unreachable!() };
      args[*off] = data;
      *off += 1;
      if *off == args.len() {
        stack.pop();
      }
    } else {
      result = data
    }
    if let Some(p) = to_push {
      stack.push((p, 0))
    } else if stack.is_empty() {
      return result
    }
  }
}

impl Data {
  fn dest_block<'a>(self, mem: &'a [Object<'_>]) -> (u8, &'a [Data]) {
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

  fn dest_str<'a>(self, mem: &[Object<'a>]) -> &'a [u8] {
    match self {
      Data::Pointer(p) => match &mem[p] {
        Object::Str(data) => data,
        _ => panic!("bad data"),
      },
      _ => panic!("bad data"),
    }
  }
  fn dest_int(self, mem: &[Object<'_>]) -> i64 {
    match self {
      Data::Pointer(p) => match mem[p] {
        Object::Int64(data) => data,
        _ => panic!("bad data"),
      },
      Data::Int(n) => n,
      _ => panic!("bad data"),
    }
  }
  fn dest_float(self, mem: &[Object<'_>]) -> f64 {
    match self {
      Data::Pointer(p) => match mem[p] {
        Object::Float(data) => data,
        _ => panic!("bad data"),
      },
      _ => panic!("bad data"),
    }
  }
}

trait Cacheable {
  fn get_mut(cache: &mut Cache) -> &mut HashMap<usize, Rc<Self>>;
}
macro_rules! mk_cache {
  (struct $cache:ident { $($name:ident: $ty:ty $(: $from:ty)?,)* }) => {
    #[derive(Default)]
    struct $cache {
      // used: HashMap<usize, std::backtrace::Backtrace>,
      $($name: HashMap<usize, Rc<$ty>>,)*
    }
    $(
      impl Cacheable for $ty {
        fn get_mut(cache: &mut $cache) -> &mut HashMap<usize, Rc<Self>> { &mut cache.$name }
      }
      mk_cache!(@impl $ty $(: $from)?);
    )*
  };
  (@impl $ty:ty: $from:ty) => {
    impl FromData for Rc<$ty> {
      fn from_data(d: Data, mem: &[Object<'_>], cache: &mut Cache) -> Self {
        if let Some(val) = cache.try_from_data(d, mem) {
          return val
        }
        let result: Self = <$from>::from_data(d, mem, cache).into();
        if let Data::Pointer(p) = d {
          <$ty as Cacheable>::get_mut(cache).insert(p, result.clone());
        }
        result
      }
    }
  };
  (@impl $ty:ty) => { mk_cache!(@impl $ty: $ty); };
}
mk_cache! {
  struct Cache {
    str: str: String,
    expr: ExprKind,
    u32s: [u32]: Vec<u32>,
    exprs: [Expr]: Vec<Expr>,
    mod_path: ModPathKind,
    ker_name: KerNameKind,
    ind_name: IndNameKind,
    ctor_name: CtorNameKind,
    ker_pair: KerPairKind,
    level: LevelKind,
    levels: [Level]: Vec<Level>,
    binder: BinderAnnot<Name>,
    typing: TypingFlags,
    abstr_inst_info: AbstrInstInfo,
    case_info: CaseInfo,
    case_branch: [CaseBranch]: Vec<CaseBranch>,
    case_return: [Rc<BinderAnnot<Name>>]: Vec<Rc<BinderAnnot<Name>>>,
    named_decl: NamedDecl,
    delayed_univs: DelayedUniverses,
    proj_repr: ProjRepr,
    proj: (Rc<ProjRepr>, bool),
  }
}
impl Cache {
  fn try_from_data<T: Cacheable + ?Sized>(&mut self, d: Data, mem: &[Object<'_>]) -> Option<Rc<T>> {
    if let Data::Pointer(p) = d {
      return Some(T::get_mut(self).get(&p)?.clone())
    }
    None
  }
}

trait FromData {
  fn from_data(d: Data, mem: &[Object<'_>], cache: &mut Cache) -> Self;
}

impl FromData for i64 {
  fn from_data(d: Data, mem: &[Object<'_>], cache: &mut Cache) -> Self { d.dest_int(mem) }
}
impl FromData for u32 {
  fn from_data(d: Data, mem: &[Object<'_>], cache: &mut Cache) -> Self {
    d.dest_int(mem).try_into().unwrap()
  }
}
impl FromData for f64 {
  fn from_data(d: Data, mem: &[Object<'_>], cache: &mut Cache) -> Self { d.dest_float(mem) }
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
      fn from_data(d: Data, mem: &[Object<'_>], cache: &mut Cache) -> Self {
        match d.dest_block(mem) {
          $(($n, [$($($a),*)?]) => {
            let ($($($a,)*)?) = ($($(FromData::from_data(*$a, mem, cache),)*)?);
            $e
          },)*
          _ => panic!("bad tag"),
        }
      }
    })*
  }
}

macro_rules! from_data_enum {
  ($($(#[$doc:meta])* enum $name:ident { $($a:ident$(($($b:ident: $ty:ty),*))? = $val:literal,)* })*) => {
    $(
      $(#[$doc])* enum $name { $($a$(($($ty),*))?),* }
      impl FromData for $name {
        fn from_data(d: Data, mem: &[Object<'_>], cache: &mut Cache) -> Self {
          match d.dest_block(mem) {
            $(($val, [$($($b),*)?]) => $name::$a$(($(FromData::from_data(*$b, mem, cache)),*))?,)*
            k => panic!("bad tag in {}: {k:?}", stringify!($name))
          }
        }
      }
    )*
  }
}

macro_rules! impl_from_data_map {
  ($($(<$($a:ident),*>)? for $name:ty = $p:pat in $ty:ty => $e:expr;)*) => {
    $(impl$(<$($a: FromData),*>)? FromData for $name {
      fn from_data(d: Data, mem: &[Object<'_>]) -> Self {
        let $p = <$ty>::from_data(d, mem);
        $e
      }
    })*
  }
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct List<T>(Vec<T>);

impl<T: std::fmt::Debug> std::fmt::Debug for List<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { self.0.fmt(f) }
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

macro_rules! from_data_struct {
  ($($(#[$doc:meta])* struct $name:ident { $($a:ident: $ty:ty,)* })*) => {
    $(
      $(#[$doc])* struct $name { $($a: $ty),* }
      impl FromData for $name {
        fn from_data(d: Data, mem: &[Object<'_>], cache: &mut Cache) -> Self {
          let (_, [$($a),*]) = d.dest_block(mem) else {
            panic!("bad tag in {}: {:?}", stringify!($name), d.dest_block(mem))
          };
          $name {$($a: FromData::from_data(*$a, mem, cache),)* }
        }
      }
    )*
  }
}

impl FromData for String {
  fn from_data(d: Data, mem: &[Object<'_>], cache: &mut Cache) -> Self {
    Self::from_utf8(d.dest_str(mem).to_vec()).unwrap()
  }
}

impl FromData for Box<[u8]> {
  fn from_data(d: Data, mem: &[Object<'_>], cache: &mut Cache) -> Self {
    d.dest_str(mem).to_vec().into()
  }
}

type DirPath = List<Rc<str>>;
type CompilationUnitName = DirPath;

impl std::fmt::Debug for VoDigest {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::VoOrVi(lib) => f.debug_struct("VoOrVi").finish(),
      Self::ViVo(lib, univ) => f.debug_struct("ViVo").finish(),
    }
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

type HSet<V> = BTreeMap<i64, BTreeSet<V>>;
type HMap<K, V> = BTreeMap<i64, BTreeMap<K, V>>;

impl FromData for Data {
  fn from_data(mut d: Data, _: &[Object<'_>], _: &mut Cache) -> Self { d }
}

#[derive(Debug)]
struct Dyn {
  tag: i64,
  data: Data,
}
impl FromData for Dyn {
  fn from_data(mut d: Data, mem: &[Object<'_>], cache: &mut Cache) -> Self {
    let (0, &[Data::Int(tag), data]) = d.dest_block(mem) else { panic!() };
    Dyn { tag, data }
  }
}

#[derive(Debug)]
enum Panic {}
impl FromData for Panic {
  fn from_data(mut d: Data, mem: &[Object<'_>], cache: &mut Cache) -> Self { panic!("{d:?}") }
}

type Id = Rc<str>;
type LibObjects = List<LibObject>;
type LibraryObjects = (LibObjects, LibObjects);

type UId = (i64, Id, DirPath);
type Label = Id;

type MutIndName = KerPair;
type Univ = List<LevelExpr>;
type Constant = KerPair;

type UnivConstraint = (Level, ConstraintType, Level);
type Constraints = BTreeSet<UnivConstraint>;
type Constrained<T> = (T, Constraints);
type UniverseSet = HSet<Level>;
type ContextSet = Constrained<UniverseSet>;
type Instance = Rc<[Level]>;
// type Instance = (Vec<Quality>, Vec<Level>); // added in 8.20
type PUniv<T> = (T, Instance);

type BinderAnnot<T> = (T, Relevance);
type Fix = ((Vec<u32>, u32), RecDecl);
type CoFix = (u32, RecDecl);

type RelContext = List<RelDecl>;
type NamedContext = List<Rc<NamedDecl>>;
type CompactedContext = List<CompactedDecl>;

type DeltaResolver = (BTreeMap<ModPath, ModPath>, HMap<KerName, DeltaHint>);

type FieldInfo = (Id, Vec<Label>, Vec<Relevance>, Vec<Type>);
type StructBody = List<(Label, StructFieldBody)>;

type Expr = Rc<ExprKind>;
type ModPath = Rc<ModPathKind>;
type KerPair = Rc<KerPairKind>;

from_data_enum! {
  #[derive(Debug)]
  enum Name {
    Anonymous = 0,
    Name(a: Id) = 0,
  }

  #[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
  enum ModPathKind {
    File(a: DirPath) = 0,
    Bound(a: UId) = 1,
    Dot(a: ModPath, b: Label) = 2,
  }

  #[derive(PartialEq, Eq, PartialOrd, Ord)]
  enum KerPairKind {
    Same(a: KerName) = 0,
    Dual(user: KerName, canon: KerName) = 1,
  }

  #[derive(PartialEq, Eq, PartialOrd, Ord)]
  enum RawLevel {
    Set = 0,
    Level(a: GlobalLevel) = 0,
    Var(a: u32) = 1,
  }

  #[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
  enum QVar {
    Var(a: u32) = 0,
    Unif(a: String, b: u32) = 1,
  }

  #[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
  enum ConstQuality {
    Prop = 0,
    SProp = 1,
    Type = 2,
  }

  #[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
  enum Quality {
    Var(a: QVar) = 0,
    Const(a: ConstQuality) = 1,
  }

  #[derive(Debug)]
  enum Variance {
    Irrelevant = 0,
    Covariant = 1,
    Invariant = 2,
  }

  #[allow(clippy::enum_variant_names)]
  #[derive(Debug)]
  enum Sort {
    SProp = 0,
    Prop = 1,
    Set = 2,
    Type(a: Univ) = 0,
    QSort(a: QVar, b: Univ) = 0,
  }

  #[derive(Debug)]
  enum Relevance {
    Relevant = 0,
    Irrelevant = 1,
    Var(a: QVar) = 0,
  }

  #[derive(Debug)]
  enum CaseStyle {
    Let = 0,
    If = 1,
    LetPat = 2,
    Match = 3,
    Regular = 4,
  }

  #[derive(Debug)]
  enum CastKind {
    Vm = 0,
    Native = 1,
    Default = 2,
  }

  #[derive(Debug)]
  enum ExprKind {
    Rel(a: u32) = 0,
    Var(a: Id) = 1,
    Sort(a: Sort) = 4,
    Cast(a: Expr, b: CastKind, c: Expr) = 5,
    Prod(a: Rc<BinderAnnot<Name>>, b: Type, c: Type) = 6,
    Lambda(a: Rc<BinderAnnot<Name>>, b: Type, c: Expr) = 7,
    LetIn(a: Rc<BinderAnnot<Name>>, b: Expr, c: Type, d: Expr) = 8,
    App(a: Expr, args: Rc<[Expr]>) = 9,
    Const(a: PUniv<Constant>) = 10,
    Ind(a: PUniv<IndName>) = 11,
    Ctor(a: PUniv<CtorName>) = 12,
    Case(
      a: Rc<CaseInfo>, b: Instance, c: Rc<[Expr]>, d: Box<CaseReturn>,
      e: CaseInvert, f: Expr, g: Rc<[CaseBranch]>
    ) = 13,
    Fix(a: Fix) = 14,
    CoFix(a: CoFix) = 15,
    Proj(a: Proj, c: Expr) = 16,
    // Proj(a: Proj, b: Relevance, c: Expr) = 16, // new in 8.20
    Int(a: i64) = 17,
    Float(a: f64) = 18,
    Array(a: Instance, b: Vec<Expr>, c: Expr, d: Type) = 19,
  }

  #[derive(Debug)]
  enum RelDecl {
    LocalAssum(a: Rc<BinderAnnot<Name>>, b: Type) = 0,
    LocalDef(a: Rc<BinderAnnot<Name>>, b: Expr, c: Type) = 1,
  }

  #[derive(Debug)]
  enum NamedDecl {
    LocalAssum(a: BinderAnnot<Id>, b: Type) = 0,
    LocalDef(a: BinderAnnot<Id>, b: Expr, c: Type) = 1,
  }

  #[derive(Debug)]
  enum CompactedDecl {
    LocalAssum(a: List<BinderAnnot<Id>>, b: Type) = 0,
    LocalDef(a: List<BinderAnnot<Id>>, b: Expr, c: Type) = 1,
  }

  #[derive(Debug)]
  enum DeltaHint {
    Inline(a: u32, b: Option<UnivAbstracted<Expr>>) = 0,
    Equiv(a: KerName) = 1,
  }

  #[derive(Debug)]
  enum Opaque {
    Indirect(subst: List<ModSubst>, discharge: List<CookingInfo>, lib: DirPath, index: u32) = 0,
  }

  #[derive(Debug)]
  enum Transparency {
    Expand = 0,
    Level(lvl: u32) = 0,
    Opaque = 1,
  }

  #[derive(Debug)]
  enum Primitive {
    Int63Head0 = 0,
    Int63Tail0 = 1,
    Int63Add = 2,
    Int63Sub = 3,
    Int63Mul = 4,
    Int63Div = 5,
    Int63Mod = 6,
    Int63Divs = 7,
    Int63Mods = 8,
    Int63Lsr = 9,
    Int63Lsl = 10,
    Int63Asr = 11,
    Int63LAnd = 12,
    Int63LOr = 13,
    Int63LXor = 14,
    Int63AddC = 15,
    Int63SubC = 16,
    Int63AddCarryC = 17,
    Int63SubCarryC = 18,
    Int63MulC = 19,
    Int63DivEucl = 20,
    Int63Div21 = 21,
    Int63AddMulDiv = 22,
    Int63Eq = 23,
    Int63Lt = 24,
    Int63Le = 25,
    Int63LtS = 26,
    Int63LeS = 27,
    Int63Compare = 28,
    Int63CompareS = 29,
    Float64Opp = 30,
    Float64Abs = 31,
    Float64Eq = 32,
    Float64Lt = 33,
    Float64Le = 34,
    Float64Compare = 35,
    Float64Equal = 36,
    Float64Classify = 37,
    Float64Add = 38,
    Float64Sub = 39,
    Float64Mul = 40,
    Float64Div = 41,
    Float64Sqrt = 42,
    Float64OfUint63 = 43,
    Float64NormFrMantissa = 44,
    Float64FrShiftExp = 45,
    Float64LdShiftExp = 46,
    Float64NextUp = 47,
    Float64NextDown = 48,
    ArrayMake = 49,
    ArrayGet = 50,
    ArrayDefault = 51,
    ArraySet = 52,
    ArrayCopy = 53,
    ArrayLength = 54,
  }

  #[derive(Debug)]
  enum ConstDef {
    Undef(a: Option<u32>) = 0,
    Def(a: Expr) = 1,
    OpaqueDef(a: Opaque) = 2,
    Primitive(a: Primitive) = 3,
  }

  #[derive(Debug)]
  enum Universes {
    Monomorphic = 0,
    Polymorphic(a: AbstractContext) = 0,
  }

  #[derive(Debug)]
  enum NestedType {
    NestedInd(a: IndName) = 0,
    NestedPrim(a: Constant) = 1,
  }

  #[derive(Debug)]
  enum RecArg {
    NoRec = 0,
    MutRec(a: IndName) = 1,
    Nested(a: NestedType) = 2,
  }

  #[derive(Debug)]
  enum WfPaths {
    Var(a: u32, b: u32) = 0,
    Node(a: RecArg, b: Vec<Vec<WfPaths>>) = 1,
    Rec(a: u32, b: Vec<WfPaths>) = 2,
  }

  #[derive(Debug)]
  enum IndArity {
    Regular(a: MonoIndArity) = 0,
    Template(a: TemplateArity) = 1,
  }

  #[derive(Debug)]
  enum SquashInfo {
    AlwaysSquashed = 0,
    SometimesSquashed(a: BTreeSet<Quality>) = 0,
  }

  #[derive(Debug)]
  enum RecursivityKind {
    Inductive = 0,
    Coinductive = 1,
    NonRecursive = 2,
  }

  #[derive(Debug)]
  enum RecordInfo {
    Not = 0,
    Fake = 1,
    Prim(a: Vec<FieldInfo>) = 0,
  }

  #[derive(Debug)]
  enum PrimInd {
    Bool = 0,
    Carry = 1,
    Pair = 2,
    Cmp = 3,
    FCmp = 4,
    FClass = 5,
  }

  #[derive(Debug)]
  enum PrimType {
    Int63 = 0,
    Float64 = 1,
    Array = 2,
  }

  #[derive(Debug)]
  enum RetroAction {
    RegisterInd(prim: PrimInd, ind: IndName) = 0,
    RegisterType(prim: PrimType, cst: Constant) = 1,
  }

  #[derive(Debug)]
  enum WithDecl {
    Mod(a: List<Id>, b: ModPath) = 0,
    Def(a: List<Id>, b: (Expr, Option<AbstractContext>)) = 1,
  }

  #[derive(Debug)]
  enum ModAlgExpr {
    Ident(a: ModPath) = 0,
    Apply(a: Box<ModAlgExpr>, b: ModPath) = 1,
    With(a: Box<ModAlgExpr>, b: WithDecl) = 2,
  }

  #[derive(Debug)]
  enum StructFieldBody {
    Const(a: Box<ConstBody>) = 0,
    MutInd(a: Box<MutIndBody>) = 1,
    Module(a: Box<ModBody>) = 2,
    ModType(a: Box<ModTypeBody>) = 3,
  }

  #[derive(Debug)]
  enum ModSig {
    NoFunctor(a: StructBody) = 0,
    MoreFunctor(a: Id, b: Box<ModTypeBody>, c: Box<ModSig>) = 1,
  }

  #[derive(Debug)]
  enum ModExpr {
    NoFunctor(a: Box<ModSig>) = 0,
    MoreFunctor(a: Box<ModExpr>) = 1,
  }

  #[derive(Debug)]
  enum ModImpl {
    Abstract = 0,
    Algebraic(a: ModExpr) = 0,
    Struct(a: StructBody) = 1,
    FullStruct = 1,
  }

  enum VoDigest {
    VoOrVi(lib: Box<[u8]>) = 0,
    ViVo(lib: Box<[u8]>, univ: Box<[u8]>) = 1,
  }

  #[derive(Debug)]
  enum OpenFilter {
    Unfiltered = 0,
    Filtered(a: Predicate<String>) = 1,
  }

  #[derive(Debug)]
  enum AlgebraicObjs {
    Objs(a: List<LibObject>) = 0,
    Ref(a: ModPath, b: ModSubst) = 1,
  }

  #[derive(Debug)]
  enum LibObject {
    Module(a: Id, b: SubstObjs) = 0,
    ModuleType(a: Id, b: SubstObjs) = 1,
    Include(a: AlgebraicObjs) = 2,
    Keep(a: Id, b: List<LibObject>) = 3,
    Export(a: List<(OpenFilter, ModPath)>) = 4,
    Atomic(a: Dyn) = 5,
  }

  #[derive(Debug)]
  enum OrderRequest {
    Equal = 0,
    Leq = 1,
  }

  #[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
  enum ConstraintType {
    Lt = 0,
    Le = 1,
    Eq = 2,
  }

  #[derive(Debug)]
  enum DelayedUniverses {
    Monomorphic(a: ()) = 0,
    Polymorphic(a: ContextSet) = 1,
  }

  #[derive(Debug)]
  enum LibraryInfo {
    Deprecation(since: Option<String>, note: Option<String>) = 0,
  }
}

type SubstObjs = (List<UId>, AlgebraicObjs);
type ModSubst = BTreeMap<ModPath, (ModPath, DeltaResolver)>;

type Type = Expr;

type UnivConstraints = BTreeSet<(Quality, OrderRequest, Quality)>;

type Predicate<T> = (bool, BTreeSet<T>);

type BoundNames = Vec<Name>;
// type BoundNames = (Vec<Name>, Vec<Name>); // added in 8.20
type AbstractContext = Constrained<BoundNames>;
type UnivAbstracted<T> = (T, AbstractContext);

type Proj = Rc<(Rc<ProjRepr>, bool)>;

type CaseInvert = Option<Vec<Expr>>;
type CaseBranch = (Rc<[Rc<BinderAnnot<Name>>]>, Expr);
type CaseReturn = (Rc<[Rc<BinderAnnot<Name>>]>, Type);
// type CaseReturn = ((Vec<Rc<BinderAnnot<Name>>>, Type), Relevance); // new in 8.20

type EntryMap<T> = (HMap<Constant, T>, HMap<MutIndName, T>);
type ExpandInfo = EntryMap<Rc<AbstrInstInfo>>;

impl std::fmt::Debug for KerNameKind {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.label) }
}

impl std::fmt::Debug for KerPairKind {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      KerPairKind::Same(a) | KerPairKind::Dual(a, _) => write!(f, "{:?}", a),
    }
  }
}

impl std::fmt::Debug for GlobalLevel {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "g{:?}", self.uid)
  }
}
impl std::fmt::Debug for RawLevel {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      RawLevel::Set => write!(f, "Set"),
      RawLevel::Level(l) => write!(f, "{l:?}"),
      RawLevel::Var(v) => write!(f, "l{v:?}"),
    }
  }
}
impl std::fmt::Debug for LevelKind {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { self.data.fmt(f) }
}

type KerName = Rc<KerNameKind>;
type IndName = Rc<IndNameKind>;
type CtorName = Rc<CtorNameKind>;
type Level = Rc<LevelKind>;

from_data_struct! {
  #[derive(PartialEq, Eq, PartialOrd, Ord)]
  struct KerNameKind {
    path: ModPath,
    label: Label,
    hash: i64,
  }

  #[derive(Debug)]
  struct IndNameKind {
    name: MutIndName,
    index: u32,
  }

  #[derive(Debug)]
  struct CtorNameKind {
    name: IndName,
    index: u32,
  }

  #[derive(PartialEq, Eq, PartialOrd, Ord)]
  struct GlobalLevel {
    lib: DirPath,
    process: Rc<str>,
    uid: u32,
  }

  #[derive(PartialEq, Eq, PartialOrd, Ord)]
  struct LevelKind {
    hash: i64,
    data: RawLevel,
  }

  #[derive(Debug)]
  struct LevelExpr {
    base: Level,
    off: u32,
  }

  #[derive(Debug)]
  struct CasePrinting {
    ind_tags: List<bool>,
    cstr_tags: Vec<List<bool>>,
    style: CaseStyle,
  }

  #[derive(Debug)]
  struct CaseInfo {
    ind: IndName,
    npar: u32,
    cstr_ndecls: Rc<[u32]>,
    cstr_nargs: Rc<[u32]>,
    relevance: Relevance, // removed in 8.20
    pp_info: CasePrinting,
  }

  #[derive(Debug)]
  struct ProjRepr {
    ind: IndName,
    relevant: bool, // removed in 8.20
    npars: u32,
    arg: u32,
    name: Label,
    // name: Constant, // new in 8.20
  }

  #[derive(Debug)]
  struct RecDecl {
    binders: Vec<Rc<BinderAnnot<Name>>>,
    types: Vec<Type>,
    exprs: Vec<Expr>,
  }

  #[derive(Debug)]
  struct AbstrInfo {
    ctx: NamedContext,
    au_ctx: AbstractContext,
    au_subst: Instance,
  }

  #[derive(Debug)]
  struct AbstrInstInfo {
    rev_inst: List<Id>,
    uinst: Instance,
  }

  #[derive(Debug)]
  struct CookingInfo {
    expand: ExpandInfo,
    abstr: AbstrInfo,
  }

  #[derive(Debug)]
  struct ConvOracle {
    var_opacity: BTreeMap<Id, Transparency>,
    cst_opacity: HMap<Constant, Transparency>,
    var_trstate: Predicate<Id>,
    cst_trstate: Predicate<Constant>,
  }

  #[derive(Debug)]
  struct TemplateArity {
    level: Sort,
  }

  #[derive(Debug)]
  struct TemplateUnivs {
    param_levels: List<Option<Level>>,
    context: ContextSet,
  }

  #[derive(Debug)]
  struct TypingFlags {
    check_guarded: bool,
    check_positive: bool,
    check_universes: bool,
    conv_oracle: ConvOracle,
    share_reduction: bool,
    enable_vm: bool,
    enable_native_compiler: bool,
    indices_matter: bool,
    impredicative_set: bool,
    sprop_allowed: bool,
    allow_uip: bool,
  }

  #[derive(Debug)]
  struct ConstBody {
    hyps: (),
    univ_hyps: Instance,
    body: ConstDef,
    ty: Type,
    relevance: Relevance,
    code: Option<Data>, // Option<BodyCode>
    univs: Universes,
    inline_code: bool,
    typing_flags: Rc<TypingFlags>,
  }

  #[derive(Debug)]
  struct MonoIndArity {
    user_arity: Type,
    sort: Sort,
  }

  #[derive(Debug)]
  struct OneIndBody {
    name: Id,
    arity_ctx: RelContext,
    arity: IndArity,
    ctor_names: Vec<Id>,
    user_lc: Vec<Type>,
    n_real_args: u32,
    n_real_decls: u32,
    squashed: Option<SquashInfo>,
    nf_lc: Vec<(RelContext, Type)>,
    cons_n_real_args: Vec<u32>,
    cons_n_real_decls: Vec<u32>,
    rec_args: WfPaths,
    relevance: Relevance,
    nb_constant: u32,
    nb_args: u32,
    reloc_tbl: Data,
  }

  #[derive(Debug)]
  struct MutIndBody {
    packets: Vec<OneIndBody>,
    record: RecordInfo,
    recursivity: RecursivityKind,
    ntypes: u32,
    hyps: (),
    univ_hyps: Instance,
    n_params: u32,
    n_params_rec: u32,
    params_ctx: RelContext,
    univs: Universes,
    template: Option<TemplateUnivs>,
    variance: Option<Vec<Variance>>,
    sec_variance: Option<Vec<Variance>>,
    private: Option<bool>,
    flags: Rc<TypingFlags>,
  }

  #[derive(Debug)]
  struct ModBody {
    path: ModPath,
    expr: ModImpl,
    ty: ModSig,
    ty_alg: Option<ModExpr>,
    delta: DeltaResolver,
    retro: (List<RetroAction>,),
  }

  #[derive(Debug)]
  struct ModTypeBody {
    path: ModPath,
    expr: (),
    ty: ModSig,
    ty_alg: Option<ModExpr>,
    delta: DeltaResolver,
    retro: (),
  }

  #[derive(Debug)]
  struct CompiledLibrary {
    name: DirPath,
    mod_: ModBody,
    univs: ContextSet,
    deps: Vec<(CompilationUnitName, VoDigest)>,
  }

  #[derive(Debug)]
  struct Summary {
    name: CompilationUnitName,
    deps: Vec<(CompilationUnitName, VoDigest)>,
    ocaml: String,
    // info: Vec<LibraryInfo>, // added in 8.20
  }
}

type OpaqueProof = (Expr, Rc<DelayedUniverses>);

struct Library {
  name: CompilationUnitName,
  compiled: CompiledLibrary,
  opaques: Vec<Option<OpaqueProof>>,
  deps: Vec<(CompilationUnitName, VoDigest)>,
  digest: VoDigest,
}

impl Library {
  fn from_file(path: impl AsRef<std::path::Path>) -> io::Result<Library> {
    let buf = unsafe { Mmap::map(&File::open(path)?)? };
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
    // println!("{:#?}", seg_sd);
    // println!("{:#?}", deps);
    Ok(Library { name, compiled, opaques, deps, digest: VoDigest::VoOrVi(Box::new(seg_md.hash)) })
  }
}

struct Environment {}

impl Environment {
  fn add_lib(&mut self, lib: &Library, check: bool) {
    println!("{} proofs", lib.opaques.len())
    // todo
  }
}

fn main() {
  let mut env = Environment {};
  let lib = Library::from_file("../metacoq/pcuic/theories/PCUICAlpha.vo").unwrap();
  env.add_lib(&lib, true)
}
