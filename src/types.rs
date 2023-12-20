use crate::marshal::{Data, Object};
use crate::parse::{Any, Cacheable, FromData, Lazy};
use itertools::Itertools;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;

macro_rules! from_data_enum {
  ($($(#[$doc:meta])* pub enum $name:ident { $($a:ident$(($($b:ident: $ty:ty),*))? = $val:literal,)* })*) => {
    $(
      $(#[$doc])* pub enum $name { $($a$(($($ty),*))?),* }
      impl FromData for $name {
        fn from_data(d: Data, mem: &'static [Object], _cache: &mut Cache) -> Self {
          match d.dest_block(mem) {
            $(($val, [$($($b),*)?]) => $name::$a$(($(FromData::from_data(*$b, mem, _cache)),*))?,)*
            k => panic!("bad tag in {}: {k:?}", stringify!($name))
          }
        }
      }
    )*
  }
}

macro_rules! from_data_enum_rec {
  ($($(#[$doc:meta])* pub enum $name:ident { $($a:ident$(($($b:ident: $ty:ty),*))? = $val:literal,)* })*) => {
    $(
      $(#[$doc])* pub enum $name { $($a$(($($ty),*))?),* }
      impl FromData for $name {
        fn from_data(d: Data, mem: &'static [Object], _cache: &mut Cache) -> Self {
          stacker::maybe_grow(1024 * 1024, 16 * 1024, || {
            match d.dest_block(mem) {
              $(($val, [$($($b),*)?]) => $name::$a$(($(FromData::from_data(*$b, mem, _cache)),*))?,)*
              k => panic!("bad tag in {}: {k:?}", stringify!($name))
            }
          })
        }
      }
    )*
  }
}

macro_rules! from_data_struct {
  ($($(#[$doc:meta])* pub struct $name:ident { $(pub $a:ident: $ty:ty,)* })*) => {
    $(
      $(#[$doc])* pub struct $name { $(pub $a: $ty),* }
      impl FromData for $name {
        fn from_data(d: Data, mem: &'static [Object], cache: &mut Cache) -> Self {
          let (_, [$($a),*]) = d.dest_block(mem) else {
            panic!("bad tag in {}: {:?}", stringify!($name), d.dest_block(mem))
          };
          $name {$($a: FromData::from_data(*$a, mem, cache),)* }
        }
      }
    )*
  }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct List<T>(pub Vec<T>);

impl<T: std::fmt::Debug> std::fmt::Debug for List<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { self.0.fmt(f) }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct RList<T>(pub Vec<T>);

impl<T: std::fmt::Debug> std::fmt::Debug for RList<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { self.0.fmt(f) }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DirPath(pub Vec<&'static str>);
pub type CompilationUnitName = DirPath;

impl FromData for DirPath {
  fn from_data(d: Data, mem: &'static [Object], cache: &mut Cache) -> Self {
    Self(<RList<&'static str>>::from_data(d, mem, cache).0)
  }
}

impl std::fmt::Display for DirPath {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{}", self.0.iter().format("."))
  }
}
impl std::fmt::Debug for DirPath {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{}", self.0.iter().format("."))
  }
}

impl From<&'static str> for DirPath {
  fn from(value: &'static str) -> Self { DirPath(value.split('.').map(From::from).collect()) }
}

impl std::fmt::Debug for VoDigest {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::VoOrVi(_) => f.debug_struct("VoOrVi").finish(),
      Self::ViVo(..) => f.debug_struct("ViVo").finish(),
    }
  }
}

pub type Id = &'static str;

pub type UId = (i64, Id, DirPath);
pub type Label = Id;

pub type MutIndName = KerPair;
pub type Univ = List<LevelExpr>;
pub type Constant = KerPair;

pub type UnivConstraint = (Level, ConstraintType, Level);
pub type Constraints = BTreeSet<UnivConstraint>;
pub type Constrained<T> = (T, Constraints);
pub type UniverseSet = HashSet<Level>;
pub type ContextSet = Constrained<UniverseSet>;
pub type Instance = Arc<[Level]>;
// pub type Instance = (Vec<Quality>, Vec<Level>); // added in 8.20
pub type PUniv<T> = (T, Instance);

pub type BinderAnnot<T> = (T, Relevance);
pub type Fix = ((Vec<u32>, u32), RecDecl);
pub type CoFix = (u32, RecDecl);

pub type RelContext = RList<RelDecl>;
pub type NamedContext = RList<Arc<NamedDecl>>;

pub type FieldInfo = (Id, Vec<Label>, Vec<Relevance>, Vec<Type>);
pub type StructBody = List<(Label, StructFieldBody)>;

pub type Expr = Arc<ExprKind>;
pub type ModPath = Arc<ModPathKind>;

from_data_enum! {
  #[derive(Debug)]
  pub enum Name {
    Anonymous = 0,
    Name(a: Id) = 0,
  }
}

from_data_enum_rec! {
  #[derive(Hash, Debug, PartialEq, Eq, PartialOrd, Ord)]
  pub enum ModPathKind {
    File(a: DirPath) = 0,
    Bound(a: UId) = 1,
    Dot(a: ModPath, b: Label) = 2,
  }
}

from_data_enum! {
  #[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
  pub enum KerPairKind {
    Same(a: KerName) = 0,
    Dual(user: KerName, canon: KerName) = 1,
  }
}
impl KerPairKind {
  pub fn user(&self) -> &KerName {
    let (KerPairKind::Same(a) | KerPairKind::Dual(a, _)) = self;
    a
  }
  pub fn new(user: KerName, canon: KerName) -> KerPair {
    Arc::new(if user == canon { Self::Same(user) } else { Self::Dual(user, canon) })
  }
}

impl std::fmt::Debug for KerPairKind {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { self.user().fmt(f) }
}

pub type KerPair = Arc<KerPairKind>;

#[derive(Eq)]
pub struct UserKerPair(pub KerPair);

impl std::hash::Hash for UserKerPair {
  fn hash<H: std::hash::Hasher>(&self, state: &mut H) { self.0.user().hash(state) }
}
impl PartialEq for UserKerPair {
  fn eq(&self, other: &Self) -> bool { self.0.user() == other.0.user() }
}
impl PartialOrd for UserKerPair {
  fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
}
impl Ord for UserKerPair {
  fn cmp(&self, other: &Self) -> std::cmp::Ordering { self.0.user().cmp(other.0.user()) }
}

from_data_enum! {
  #[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
  pub enum RawLevel {
    Set = 0,
    Level(a: GlobalLevel) = 0,
    Var(a: u32) = 1,
  }

  #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
  pub enum QVar {
    Var(a: u32) = 0,
    Unif(a: &'static str, b: u32) = 1,
  }

  #[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
  pub enum ConstQuality {
    Prop = 0,
    SProp = 1,
    Type = 2,
  }

  #[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
  pub enum Quality {
    Var(a: QVar) = 0,
    Const(a: ConstQuality) = 1,
  }

  #[derive(Debug)]
  pub enum Variance {
    Irrelevant = 0,
    Covariant = 1,
    Invariant = 2,
  }

  #[allow(clippy::enum_variant_names)]
  #[derive(Debug)]
  pub enum Sort {
    SProp = 0,
    Prop = 1,
    Set = 2,
    Type(a: Univ) = 0,
    QSort(a: QVar, b: Univ) = 0,
  }

  #[derive(Debug)]
  pub enum SortFamily {
    SProp = 0,
    Prop = 1,
    Set = 2,
    Type = 3,
    QSort = 4,
  }

  #[derive(Debug, Clone)]
  pub enum Relevance {
    Relevant = 0,
    Irrelevant = 1,
    Var(a: QVar) = 0,
  }

  #[derive(Debug)]
  pub enum CaseStyle {
    Let = 0,
    If = 1,
    LetPat = 2,
    Match = 3,
    Regular = 4,
  }

  #[derive(Debug)]
  pub enum CastKind {
    Vm = 0,
    Native = 1,
    Default = 2,
  }
}

from_data_enum_rec! {
  #[derive(Debug)]
  pub enum ExprKind {
    Rel(a: u32) = 0,
    Var(a: Id) = 1,
    Sort(a: Sort) = 4,
    Cast(a: Expr, b: CastKind, c: Expr) = 5,
    Prod(a: Arc<BinderAnnot<Name>>, b: Type, c: Type) = 6,
    Lambda(a: Arc<BinderAnnot<Name>>, b: Type, c: Expr) = 7,
    LetIn(a: Arc<BinderAnnot<Name>>, b: Expr, c: Type, d: Expr) = 8,
    App(a: Expr, args: Arc<[Expr]>) = 9,
    Const(a: PUniv<Constant>) = 10,
    Ind(a: PUniv<IndName>) = 11,
    Ctor(a: PUniv<CtorName>) = 12,
    Case(
      a: Arc<CaseInfo>, b: Instance, c: Arc<[Expr]>, d: Box<CaseReturn>,
      e: CaseInvert, f: Expr, g: Arc<[CaseBranch]>
    ) = 13,
    Fix(a: Fix) = 14,
    CoFix(a: CoFix) = 15,
    Proj(a: Proj, c: Expr) = 16,
    // Proj(a: Proj, b: Relevance, c: Expr) = 16, // new in 8.20
    Int(a: i64) = 17,
    Float(a: f64) = 18,
    Array(a: Instance, b: Vec<Expr>, c: Expr, d: Type) = 19,
  }
}

from_data_enum! {
  #[derive(Debug)]
  pub enum RelDecl {
    LocalAssum(a: Arc<BinderAnnot<Name>>, b: Type) = 0,
    LocalDef(a: Arc<BinderAnnot<Name>>, b: Expr, c: Type) = 1,
  }

  #[derive(Debug)]
  pub enum NamedDecl {
    LocalAssum(a: BinderAnnot<Id>, b: Type) = 0,
    LocalDef(a: BinderAnnot<Id>, b: Expr, c: Type) = 1,
  }

  #[derive(Debug)]
  pub enum CompactedDecl {
    LocalAssum(a: RList<BinderAnnot<Id>>, b: Type) = 0,
    LocalDef(a: RList<BinderAnnot<Id>>, b: Expr, c: Type) = 1,
  }

  #[derive(Debug)]
  pub enum DeltaHint {
    Inline(a: u32, b: Option<UnivAbstracted<Expr>>) = 0,
    Equiv(a: KerName) = 1,
  }

  #[derive(Debug)]
  pub enum Opaque {
    Indirect(subst: RList<ModSubst>, discharge: RList<CookingInfo>, lib: DirPath, index: u32) = 0,
  }

  #[derive(Debug)]
  pub enum Transparency {
    Expand = 0,
    Level(lvl: u32) = 0,
    Opaque = 1,
  }

  #[derive(Debug)]
  pub enum Primitive {
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
  pub enum ConstDef {
    Undef(a: Option<u32>) = 0,
    Def(a: Expr) = 1,
    OpaqueDef(a: Opaque) = 2,
    Primitive(a: Primitive) = 3,
  }

  #[derive(Debug)]
  pub enum Universes {
    Monomorphic = 0,
    Polymorphic(a: AbstractContext) = 0,
  }

  #[derive(Debug)]
  pub enum NestedType {
    NestedInd(a: IndName) = 0,
    NestedPrim(a: Constant) = 1,
  }

  #[derive(Debug)]
  pub enum RecArg {
    NoRec = 0,
    MutRec(a: IndName) = 0,
    Nested(a: NestedType) = 1,
  }
}

from_data_enum_rec! {
  #[derive(Debug)]
  pub enum WfPaths {
    Var(a: u32, b: u32) = 0,
    Node(a: RecArg, b: Vec<WfPaths>) = 1,
    // Node(a: RecArg, b: Vec<Vec<WfPaths>>) = 1, // new in 8.20
    Rec(a: u32, b: Vec<WfPaths>) = 2,
  }
}

from_data_enum! {
  #[derive(Debug)]
  pub enum IndArity {
    Regular(a: MonoIndArity) = 0,
    Template(a: TemplateArity) = 1,
  }

  #[derive(Debug)]
  pub enum SquashInfo {
    AlwaysSquashed = 0,
    SometimesSquashed(a: BTreeSet<Quality>) = 0,
  }

  #[derive(Debug)]
  pub enum RecursivityKind {
    Inductive = 0,
    Coinductive = 1,
    NonRecursive = 2,
  }

  #[derive(Debug)]
  pub enum RecordInfo {
    Not = 0,
    Fake = 1,
    Prim(a: Vec<FieldInfo>) = 0,
  }

  #[derive(Debug)]
  pub enum PrimInd {
    Bool = 0,
    Carry = 1,
    Pair = 2,
    Cmp = 3,
    FCmp = 4,
    FClass = 5,
  }

  #[derive(Debug)]
  pub enum PrimType {
    Int63 = 0,
    Float64 = 1,
    Array = 2,
  }

  #[derive(Debug)]
  pub enum RetroAction {
    RegisterInd(prim: PrimInd, ind: IndName) = 0,
    RegisterType(prim: PrimType, cst: Constant) = 1,
  }

  #[derive(Debug)]
  pub enum WithDecl {
    Mod(a: List<Id>, b: ModPath) = 0,
    Def(a: List<Id>, b: (Lazy<Expr>, Option<AbstractContext>)) = 1,
  }
}

from_data_enum_rec! {
  #[derive(Debug)]
  pub enum ModAlgExpr {
    Ident(a: ModPath) = 0,
    Apply(a: Box<ModAlgExpr>, b: ModPath) = 1,
    With(a: Box<ModAlgExpr>, b: WithDecl) = 2,
  }

  #[derive(Debug)]
  pub enum StructFieldBody {
    Const(a: Arc<ConstBody>) = 0,
    MutInd(a: Lazy<Arc<MutIndBody>>) = 1,
    Module(a: Arc<ModBody>) = 2,
    ModType(a: Lazy<Arc<ModTypeBody>>) = 3,
  }

  #[derive(Debug)]
  pub enum ModSig {
    NoFunctor(a: StructBody) = 0,
    MoreFunctor(a: UId, b: Box<ModTypeBody>, c: Box<ModSig>) = 1,
  }

  #[derive(Debug)]
  pub enum ModExpr {
    NoFunctor(a: Box<ModAlgExpr>) = 0,
    MoreFunctor(a: Lazy<Box<ModExpr>>) = 1,
  }
}

from_data_enum! {
  #[derive(Debug)]
  pub enum ModImpl {
    Abstract = 0,
    Algebraic(a: ModExpr) = 0,
    Struct(a: StructBody) = 1,
    FullStruct = 1,
  }

  #[derive(PartialEq, Eq)]
  pub enum VoDigest {
    VoOrVi(lib: Box<[u8]>) = 0,
    ViVo(lib: Box<[u8]>, univ: Box<[u8]>) = 1,
  }

  #[derive(Debug)]
  pub enum OpenFilter {
    Unfiltered = 0,
    Filtered(a: Predicate<String>) = 1,
  }

  // #[derive(Debug)]
  // pub enum AlgebraicObjs {
  //   Objs(a: RList<LibObject>) = 0,
  //   Ref(a: ModPath, b: ModSubst) = 1,
  // }
}

// from_data_enum_rec! {
//   #[derive(Debug)]
//   pub enum LibObject {
//     Module(a: Id, b: SubstObjs) = 0,
//     ModuleType(a: Id, b: SubstObjs) = 1,
//     Include(a: AlgebraicObjs) = 2,
//     Keep(a: Id, b: List<LibObject>) = 3,
//     Export(a: List<(OpenFilter, ModPath)>) = 4,
//     Atomic(a: Any) = 5,
//   }
// }

from_data_enum! {
  #[derive(Debug)]
  pub enum OrderRequest {
    Equal = 0,
    Leq = 1,
  }

  #[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
  pub enum ConstraintType {
    Lt = 0,
    Le = 1,
    Eq = 2,
  }

  #[derive(Debug)]
  pub enum DelayedUniverses {
    Monomorphic(a: ()) = 0,
    Polymorphic(a: ContextSet) = 1,
  }

  #[derive(Debug)]
  pub enum LibraryInfo {
    Deprecation(since: Option<String>, note: Option<String>) = 0,
  }
}

// pub type SubstObjs = (List<UId>, AlgebraicObjs);
pub type ModSubst = BTreeMap<ModPath, (ModPath, DeltaResolver)>;

pub type Type = Expr;

pub type Predicate<T> = (bool, BTreeSet<T>);

pub type BoundNames = Vec<Name>;
// pub type BoundNames = (Vec<Name>, Vec<Name>); // added in 8.20
pub type AbstractContext = Constrained<BoundNames>;
pub type UnivAbstracted<T> = (T, AbstractContext);

pub type Proj = Arc<(Arc<ProjRepr>, bool)>;

pub type CaseInvert = Option<Vec<Expr>>;
pub type CaseBranch = (Arc<[Arc<BinderAnnot<Name>>]>, Expr);
pub type CaseReturn = (Arc<[Arc<BinderAnnot<Name>>]>, Type);
// pub type CaseReturn = ((Vec<Arc<BinderAnnot<Name>>>, Type), Relevance); // new in 8.20

pub type EntryMap<T> = (Lazy<HashMap<Constant, T>>, Lazy<HashMap<MutIndName, T>>);
pub type ExpandInfo = EntryMap<Arc<AbstrInstInfo>>;

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

impl std::fmt::Debug for KerNameKind {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.label) }
}

pub type KerName = Arc<KerNameKind>;
pub type IndName = Arc<IndNameKind>;
pub type CtorName = Arc<CtorNameKind>;
pub type Level = Arc<LevelKind>;

from_data_struct! {
  #[derive(Hash, PartialEq, Eq, PartialOrd, Ord)]
  pub struct KerNameKind {
    pub path: ModPath,
    pub label: Label,
    pub hash: i64,
  }
}

impl KerNameKind {
  pub fn new(path: ModPath, label: Label) -> KerName {
    Arc::new(KerNameKind { path, label, hash: 0 }) // todo
  }
}

from_data_struct! {
  #[derive(Debug)]
  pub struct IndNameKind {
    pub name: MutIndName,
    pub index: u32,
  }

  #[derive(Debug)]
  pub struct CtorNameKind {
    pub name: IndName,
    pub index: u32,
  }

  #[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
  pub struct GlobalLevel {
    pub lib: DirPath,
    pub process: &'static str,
    pub uid: u32,
  }

  #[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
  pub struct LevelKind {
    pub hash: i64,
    pub data: RawLevel,
  }

  #[derive(Debug)]
  pub struct LevelExpr {
    pub base: Level,
    pub off: u32,
  }

  #[derive(Debug)]
  pub struct CasePrinting {
    pub ind_tags: List<bool>,
    pub cstr_tags: Vec<List<bool>>,
    pub style: CaseStyle,
  }

  #[derive(Debug)]
  pub struct CaseInfo {
    pub ind: IndName,
    pub npar: u32,
    pub cstr_ndecls: Arc<[u32]>,
    pub cstr_nargs: Arc<[u32]>,
    pub relevance: Relevance, // removed in 8.20
    pub pp_info: CasePrinting,
  }

  #[derive(Debug)]
  pub struct ProjRepr {
    pub ind: IndName,
    pub relevant: bool, // removed in 8.20
    pub npars: u32,
    pub arg: u32,
    pub name: Label,
    // pub name: Constant, // new in 8.20
  }

  #[derive(Debug)]
  pub struct RecDecl {
    pub binders: Vec<Arc<BinderAnnot<Name>>>,
    pub types: Vec<Type>,
    pub exprs: Vec<Expr>,
  }

  #[derive(Debug)]
  pub struct AbstrInfo {
    pub ctx: NamedContext,
    pub au_ctx: AbstractContext,
    pub au_subst: Instance,
  }

  #[derive(Debug)]
  pub struct AbstrInstInfo {
    pub inst: RList<Id>,
    pub uinst: Instance,
  }

  #[derive(Debug)]
  pub struct CookingInfo {
    pub expand: ExpandInfo,
    pub abstr: AbstrInfo,
  }

  #[derive(Debug)]
  pub struct ConvOracle {
    pub var_opacity: BTreeMap<Id, Transparency>,
    pub cst_opacity: Lazy<HashMap<Constant, Transparency>>,
    pub var_trstate: Predicate<Id>,
    pub cst_trstate: Predicate<Constant>,
  }

  #[derive(Debug)]
  pub struct TemplateArity {
    pub level: Sort,
  }

  #[derive(Debug)]
  pub struct TemplateUnivs {
    pub param_levels: RList<Option<Level>>,
    pub context: ContextSet,
  }

  #[derive(Debug)]
  pub struct TypingFlags {
    pub check_guarded: bool,
    pub check_positive: bool,
    pub check_universes: bool,
    pub conv_oracle: ConvOracle,
    pub share_reduction: bool,
    pub enable_vm: bool,
    pub enable_native_compiler: bool,
    pub indices_matter: bool,
    pub impredicative_set: bool,
    pub sprop_allowed: bool,
    pub allow_uip: bool,
  }

  #[derive(Debug)]
  pub struct ConstBody {
    pub hyps: (),
    pub univ_hyps: Instance,
    pub body: Lazy<ConstDef>,
    pub ty: Lazy<Type>,
    pub relevance: Relevance,
    pub code: Option<Any>, // Option<BodyCode>
    pub univs: Universes,
    pub inline_code: bool,
    pub typing_flags: Arc<TypingFlags>,
  }

  #[derive(Debug)]
  pub struct MonoIndArity {
    pub user_arity: Type,
    pub sort: Sort,
  }

  #[derive(Debug)]
  pub struct OneIndBody {
    pub name: Id,
    pub arity_ctx: RelContext,
    pub arity: IndArity,
    pub ctor_names: Vec<Id>,
    pub user_lc: Vec<Type>,
    pub n_real_args: u32,
    pub n_real_decls: u32,
    pub kelim: SortFamily, // removed in 8.20
    // pub squashed: Option<SquashInfo>, // added in 8.20
    pub nf_lc: Vec<(RelContext, Type)>,
    pub cons_n_real_args: Vec<u32>,
    pub cons_n_real_decls: Vec<u32>,
    pub rec_args: WfPaths,
    pub relevance: Relevance,
    pub nb_constant: u32,
    pub nb_args: u32,
    pub reloc_tbl: Any,
  }

  #[derive(Debug)]
  pub struct MutIndBody {
    pub packets: Vec<OneIndBody>,
    pub record: RecordInfo,
    pub recursivity: RecursivityKind,
    pub ntypes: u32,
    pub hyps: (),
    pub univ_hyps: Instance,
    pub n_params: u32,
    pub n_params_rec: u32,
    pub params_ctx: RelContext,
    pub univs: Universes,
    pub template: Option<TemplateUnivs>,
    pub variance: Option<Vec<Variance>>,
    pub sec_variance: Option<Vec<Variance>>,
    pub private: Option<bool>,
    pub flags: Arc<TypingFlags>,
  }

  #[derive(Debug)]
  pub struct DeltaResolver {
    pub mod_: BTreeMap<ModPath, ModPath>,
    pub name: HashMap<KerName, DeltaHint>,
  }
}

impl DeltaResolver {
  fn find_prefix(&self, mp: &ModPath) -> Option<ModPath> {
    if let Some(res) = self.mod_.get(mp) {
      return Some(res.clone())
    }
    if let ModPathKind::Dot(mp, n) = &**mp {
      Some(Arc::new(ModPathKind::Dot(self.find_prefix(mp)?, n.clone())))
    } else {
      None
    }
  }

  fn resolve_kername_core(&self, kn: &KerName) -> Result<KerName, (u32, &UnivAbstracted<Expr>)> {
    match self.name.get(kn) {
      Some(DeltaHint::Inline(lev, Some(c))) => Err((*lev, c)),
      Some(DeltaHint::Equiv(kn1)) => Ok(kn1.clone()),
      Some(DeltaHint::Inline(_, None)) | None => {
        if let Some(mp) = self.find_prefix(&kn.path).filter(|p| *p != kn.path) {
          Ok(KerNameKind::new(mp, kn.label.clone()))
        } else {
          Ok(kn.clone())
        }
      }
    }
  }

  pub fn resolve_kername(&self, kn: &KerName) -> KerName {
    self.resolve_kername_core(kn).unwrap_or(kn.clone())
  }

  pub fn resolve_kerpair(&self, kn: &KerName) -> KerPair {
    KerPairKind::new(kn.clone(), self.resolve_kername(kn).clone())
  }
}

from_data_struct! {
  #[derive(Debug)]
  pub struct ModBody {
    pub path: ModPath,
    pub expr: ModImpl,
    pub ty: ModSig,
    pub ty_alg: Option<ModExpr>,
    pub delta: DeltaResolver,
    pub retro: (List<RetroAction>,),
  }

  #[derive(Debug)]
  pub struct ModTypeBody {
    pub path: ModPath,
    pub expr: (),
    pub ty: ModSig,
    pub ty_alg: Option<ModExpr>,
    pub delta: DeltaResolver,
    pub retro: (),
  }

  #[derive(Debug)]
  pub struct CompiledLibrary {
    pub name: DirPath,
    pub mod_: Arc<ModBody>,
    pub univs: ContextSet,
    pub deps: Vec<(CompilationUnitName, VoDigest)>,
  }

  #[derive(Debug)]
  pub struct Summary {
    pub name: CompilationUnitName,
    pub deps: Vec<(CompilationUnitName, VoDigest)>,
    pub ocaml: String,
    // pub info: Vec<LibraryInfo>, // added in 8.20
  }
}

pub type OpaqueProof = (Expr, Arc<DelayedUniverses>);

pub struct Library {
  pub compiled: CompiledLibrary,
  pub opaques: Vec<Option<OpaqueProof>>,
  pub deps: Vec<(CompilationUnitName, VoDigest)>,
  pub digest: VoDigest,
}

macro_rules! mk_cache {
  (pub struct $cache:ident { $($name:ident: $ty:ty $(: $from:ty)?,)* }) => {
    #[derive(Default)]
    pub struct $cache {
      // used: HashMap<usize, std::backtrace::Backtrace>,
      $($name: HashMap<u32, Arc<$ty>>,)*
    }
    $(
      impl Cacheable for $ty {
        fn get_mut(cache: &mut $cache) -> &mut HashMap<u32, Arc<Self>> {
          &mut cache.$name
        }
      }
      mk_cache!(@impl $ty $(: $from)?);
    )*
  };
  (@impl $ty:ty: $from:ty) => {
    impl FromData for Arc<$ty> {
      fn from_data(d: Data, mem: &'static [Object], cache: &mut Cache) -> Self {
        if let Some(val) = cache.try_from_data(d) {
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
  pub struct Cache {
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
    case_return: [Arc<BinderAnnot<Name>>]: Vec<Arc<BinderAnnot<Name>>>,
    named_decl: NamedDecl,
    delayed_univs: DelayedUniverses,
    proj_repr: ProjRepr,
    proj: (Arc<ProjRepr>, bool),
    mod_body: ModBody,
    mod_type_body: ModTypeBody,
    const_body: ConstBody,
    mut_ind_body: MutIndBody,
  }
}
