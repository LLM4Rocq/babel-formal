/-
BENCHMARK_ID: TINY_MATHLIB_BATCH05_COMBINATORICS_FLAG_ALGEBRA_DENSITY_LIKE
PAIR_STEM: combinatorics_flag_algebra_density_like
MATH_DOMAIN: Combinatorics
SOURCE_MATHLIB: Mathlib/Combinatorics/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class CombStruct_flag_algebra_density (X : Type u) where
  le : X -> X -> Prop
  le_refl : forall x : X, le x x
  le_trans : forall {x y z : X}, le x y -> le y z -> le x z
  add : X -> X -> X
  mul : X -> X -> X
  entropy : X -> X
  density : X -> X
  regularize : X -> X
  container_axiom : forall x : X, le x (regularize x)
  entropy_axiom : forall x y : X,
      le (entropy (add x y)) (add (entropy x) (entropy y))
  density_axiom : forall x : X, le (density (regularize x)) (density x)
  sparse_regularity_axiom : forall x : X, le (regularize (regularize x)) (regularize x)
  tverberg_axiom : forall x y : X, le (mul x y) (add x y)
  counting_axiom : forall x y : X,
      le (density (mul x y)) (mul (density x) (density y))
  extremal_axiom : forall x y : X,
      le (mul (density x) (density y)) (density (add x y))

infix:50 " ≼ " => CombStruct_flag_algebra_density.le
infixl:65 " ⊞ " => CombStruct_flag_algebra_density.add
infixl:70 " ⊠ " => CombStruct_flag_algebra_density.mul

def HypergraphObj_flag_algebra_density
    {X : Type u} [CombStruct_flag_algebra_density X] (x : X) : X :=
  x

def EntropyObj_flag_algebra_density
    {X : Type u} [CombStruct_flag_algebra_density X] (x : X) : X :=
  CombStruct_flag_algebra_density.entropy x

def DensityObj_flag_algebra_density
    {X : Type u} [CombStruct_flag_algebra_density X] (x : X) : X :=
  CombStruct_flag_algebra_density.density x

def RegularityObj_flag_algebra_density
    {X : Type u} [CombStruct_flag_algebra_density X] (x : X) : X :=
  CombStruct_flag_algebra_density.regularize x

theorem container_step_flag_algebra_density
    {X : Type u} [CombStruct_flag_algebra_density X]
    (x : X) :
    HypergraphObj_flag_algebra_density x ≼
      RegularityObj_flag_algebra_density x := by
  have hraw : CombStruct_flag_algebra_density.le x
      (CombStruct_flag_algebra_density.regularize x) :=
    CombStruct_flag_algebra_density.container_axiom x
  calc
    HypergraphObj_flag_algebra_density x = x := by
      rfl
    _ ≼ CombStruct_flag_algebra_density.regularize x := hraw
    _ = RegularityObj_flag_algebra_density x := by
          rfl

theorem entropy_lemma_step_flag_algebra_density
    {X : Type u} [CombStruct_flag_algebra_density X]
    (x y : X) :
    EntropyObj_flag_algebra_density (x ⊞ y) ≼
      EntropyObj_flag_algebra_density x ⊞
        EntropyObj_flag_algebra_density y := by
  have hent : CombStruct_flag_algebra_density.le
      (CombStruct_flag_algebra_density.entropy (CombStruct_flag_algebra_density.add x y))
      (CombStruct_flag_algebra_density.add
        (CombStruct_flag_algebra_density.entropy x)
        (CombStruct_flag_algebra_density.entropy y)) :=
    CombStruct_flag_algebra_density.entropy_axiom x y
  calc
    EntropyObj_flag_algebra_density (x ⊞ y)
        = CombStruct_flag_algebra_density.entropy
            (CombStruct_flag_algebra_density.add x y) := by
              rfl
    _ ≼ CombStruct_flag_algebra_density.add
          (CombStruct_flag_algebra_density.entropy x)
          (CombStruct_flag_algebra_density.entropy y) := hent
    _ = EntropyObj_flag_algebra_density x ⊞
          EntropyObj_flag_algebra_density y := by
          rfl

theorem flag_density_step_flag_algebra_density
    {X : Type u} [CombStruct_flag_algebra_density X]
    (x : X)
    (hreg : RegularityObj_flag_algebra_density x ≼
      RegularityObj_flag_algebra_density x) :
    DensityObj_flag_algebra_density
      (RegularityObj_flag_algebra_density x) ≼
      DensityObj_flag_algebra_density x := by
  have hden : CombStruct_flag_algebra_density.le
      (CombStruct_flag_algebra_density.density
        (CombStruct_flag_algebra_density.regularize x))
      (CombStruct_flag_algebra_density.density x) :=
    CombStruct_flag_algebra_density.density_axiom x
  have _ : RegularityObj_flag_algebra_density x ≼
      RegularityObj_flag_algebra_density x := hreg
  calc
    DensityObj_flag_algebra_density
      (RegularityObj_flag_algebra_density x)
        = CombStruct_flag_algebra_density.density
            (CombStruct_flag_algebra_density.regularize x) := by
              rfl
    _ ≼ CombStruct_flag_algebra_density.density x := hden
    _ = DensityObj_flag_algebra_density x := by
          rfl

theorem sparse_regularity_step_flag_algebra_density
    {X : Type u} [CombStruct_flag_algebra_density X]
    (x : X) :
    RegularityObj_flag_algebra_density
      (RegularityObj_flag_algebra_density x) ≼
      RegularityObj_flag_algebra_density x /\
    HypergraphObj_flag_algebra_density x ≼
      RegularityObj_flag_algebra_density x := by
  have hsparse : CombStruct_flag_algebra_density.le
      (CombStruct_flag_algebra_density.regularize
        (CombStruct_flag_algebra_density.regularize x))
      (CombStruct_flag_algebra_density.regularize x) :=
    CombStruct_flag_algebra_density.sparse_regularity_axiom x
  have hcont : HypergraphObj_flag_algebra_density x ≼
      RegularityObj_flag_algebra_density x :=
    container_step_flag_algebra_density x
  constructor
  · calc
      RegularityObj_flag_algebra_density
        (RegularityObj_flag_algebra_density x)
          = CombStruct_flag_algebra_density.regularize
              (CombStruct_flag_algebra_density.regularize x) := by
                rfl
      _ ≼ CombStruct_flag_algebra_density.regularize x := hsparse
      _ = RegularityObj_flag_algebra_density x := by
            rfl
  · exact hcont

theorem tverberg_partition_step_flag_algebra_density
    {X : Type u} [CombStruct_flag_algebra_density X]
    (x y : X) :
    HypergraphObj_flag_algebra_density x ⊠
      HypergraphObj_flag_algebra_density y ≼
      HypergraphObj_flag_algebra_density x ⊞
        HypergraphObj_flag_algebra_density y := by
  have htver : CombStruct_flag_algebra_density.le
      (CombStruct_flag_algebra_density.mul x y)
      (CombStruct_flag_algebra_density.add x y) :=
    CombStruct_flag_algebra_density.tverberg_axiom x y
  calc
    HypergraphObj_flag_algebra_density x ⊠
      HypergraphObj_flag_algebra_density y
        = CombStruct_flag_algebra_density.mul x y := by
            rfl
    _ ≼ CombStruct_flag_algebra_density.add x y := htver
    _ = HypergraphObj_flag_algebra_density x ⊞
          HypergraphObj_flag_algebra_density y := by
          rfl

theorem counting_upgrade_flag_algebra_density
    {X : Type u} [CombStruct_flag_algebra_density X]
    (x y : X) :
    DensityObj_flag_algebra_density (x ⊠ y) ≼
      DensityObj_flag_algebra_density x ⊠
        DensityObj_flag_algebra_density y := by
  have hcount : CombStruct_flag_algebra_density.le
      (CombStruct_flag_algebra_density.density
        (CombStruct_flag_algebra_density.mul x y))
      (CombStruct_flag_algebra_density.mul
        (CombStruct_flag_algebra_density.density x)
        (CombStruct_flag_algebra_density.density y)) :=
    CombStruct_flag_algebra_density.counting_axiom x y
  calc
    DensityObj_flag_algebra_density (x ⊠ y)
        = CombStruct_flag_algebra_density.density
            (CombStruct_flag_algebra_density.mul x y) := by
              rfl
    _ ≼ CombStruct_flag_algebra_density.mul
          (CombStruct_flag_algebra_density.density x)
          (CombStruct_flag_algebra_density.density y) := hcount
    _ = DensityObj_flag_algebra_density x ⊠
          DensityObj_flag_algebra_density y := by
          rfl

theorem extremal_conclusion_flag_algebra_density
    {X : Type u} [CombStruct_flag_algebra_density X]
    (x y : X) :
    DensityObj_flag_algebra_density (x ⊠ y) ≼
      DensityObj_flag_algebra_density (x ⊞ y) := by
  have h1 : DensityObj_flag_algebra_density (x ⊠ y) ≼
      DensityObj_flag_algebra_density x ⊠ DensityObj_flag_algebra_density y :=
    counting_upgrade_flag_algebra_density x y
  have h2 : CombStruct_flag_algebra_density.le
      (CombStruct_flag_algebra_density.mul
        (CombStruct_flag_algebra_density.density x)
        (CombStruct_flag_algebra_density.density y))
      (CombStruct_flag_algebra_density.density
        (CombStruct_flag_algebra_density.add x y)) :=
    CombStruct_flag_algebra_density.extremal_axiom x y
  have h2' : DensityObj_flag_algebra_density x ⊠ DensityObj_flag_algebra_density y ≼
      DensityObj_flag_algebra_density (x ⊞ y) := by
    simpa using h2
  exact CombStruct_flag_algebra_density.le_trans h1 h2'
