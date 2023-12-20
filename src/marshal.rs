use byteorder::{BE, LE};
use typed_arena::Arena;
use zerocopy::big_endian::{I16, I32, I64, U16, U32, U64};
use zerocopy::{FromBytes, FromZeroes, Ref, F64};

pub fn parse<'a, T: FromBytes>(pos: &mut &'a [u8]) -> &'a T {
  let (t, pos2) = Ref::<_, T>::new_from_prefix(*pos).unwrap();
  *pos = pos2;
  t.into_ref()
}

pub fn parse_u8(pos: &mut &[u8]) -> u8 {
  let (t, pos2) = pos.split_first().unwrap();
  *pos = pos2;
  *t
}
pub fn parse_str<'a>(n: usize, pos: &mut &'a [u8]) -> &'a [u8] {
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
  magic: U32,
  _length: U32,
  _objects: U32,
  _size32: U32,
  _size64: U32,
}

#[derive(Copy, Clone, Debug)]
pub enum Data {
  Atom(u8),
  Int(i64),
  Pointer(u32),
  Code(usize),
}

pub enum Object {
  Struct(u8, &'static mut [Data]),
  Int64(i64),
  Float(f64),
  Str(&'static [u8]),
}

impl std::fmt::Debug for Object {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Struct(arg0, arg1) => f.debug_tuple("Struct").field(arg0).field(arg1).finish(),
      Self::Int64(arg0) => f.debug_tuple("Int64").field(arg0).finish(),
      Self::Float(arg0) => f.debug_tuple("Float").field(arg0).finish(),
      Self::Str(arg0) => f.debug_tuple("Str").field(&String::from_utf8_lossy(arg0)).finish(),
    }
  }
}

pub fn parse_objects(
  pos: &mut &[u8], objects: &mut Vec<Object>, arena: &'static Arena<Data>,
) -> Data {
  fn push_vec<T>(vec: &mut Vec<T>, val: T) -> usize { (vec.len(), vec.push(val)).0 }
  let header = parse::<ObjectHeader>(pos);
  assert_eq!(header.magic.get(), 0x8495a6be);
  let mut stack = vec![];
  let mut result = Data::Atom(0);
  loop {
    let (data, to_push) = match parse_object(pos) {
      Tag::Pointer(n) => (Data::Pointer((objects.len() - n).try_into().unwrap()), None),
      Tag::Int(n) => (Data::Int(n), None),
      Tag::Str(s) => {
        let s = Vec::leak(Vec::from(s));
        (Data::Pointer(push_vec(objects, Object::Str(s)).try_into().unwrap()), None)
      }
      Tag::Block(tag, 0) => (Data::Atom(tag), None),
      Tag::Block(tag, len) => {
        let args = arena.alloc_extend(std::iter::repeat(Data::Atom(0)).take(len));
        let p = push_vec(objects, Object::Struct(tag, args)).try_into().unwrap();
        (Data::Pointer(p), Some(p))
      }
      Tag::Code(addr) => (Data::Code(addr), None),
      Tag::Int64(n) =>
        (Data::Pointer(push_vec(objects, Object::Int64(n)).try_into().unwrap()), None),
      Tag::Float(n) =>
        (Data::Pointer(push_vec(objects, Object::Float(n)).try_into().unwrap()), None),
    };
    if let Some((p, off)) = stack.last_mut() {
      let Object::Struct(_, args) = &mut objects[*p as usize] else { unreachable!() };
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
