/-
BENCHMARK_ID: TINY_MATHLIB_BATCH05_COMBINATORICS_SPARSE_REGULARIZATION_AXIOMATIC_LIKE
PAIR_STEM: combinatorics_sparse_regularization_axiomatic_like
MATH_DOMAIN: Combinatorics
SOURCE_MATHLIB: Mathlib/Combinatorics/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class CombStruct_sparse_regularization (V : Type u) where
  weight : V -> Nat
  density : (V -> Nat) -> Nat
  entropy : (V -> Nat) -> Nat
  regularize : (V -> Nat) -> (V -> Nat)
  container : (V -> Nat) -> (V -> Nat)
  density_mono : forall {f g : V -> Nat},
      (forall v, f v <= g v) -> density f <= density g
  entropy_nonneg : forall f : V -> Nat, 0 <= entropy f
  regularize_majorizes : forall (f : V -> Nat) (v : V), f v <= regularize f v
  regularize_idem : forall (f : V -> Nat) (v : V),
      regularize (regularize f) v = regularize f v
  container_controls : forall (f : V -> Nat) (v : V),
      regularize f v <= container f v
  counting_axiom : forall f : V -> Nat,
      density (regularize f) <= density (container f) + entropy f
  nat_le_trans_axiom : forall a b c : Nat, a <= b -> b <= c -> a <= c

def HypergraphObj_sparse_regularization {V : Type u}
    [CombStruct_sparse_regularization V] : Type u :=
  V -> Nat

def EntropyObj_sparse_regularization {V : Type u}
    [CombStruct_sparse_regularization V]
    (f : HypergraphObj_sparse_regularization (V := V)) : Nat :=
  CombStruct_sparse_regularization.entropy f

def DensityObj_sparse_regularization {V : Type u}
    [CombStruct_sparse_regularization V]
    (f : HypergraphObj_sparse_regularization (V := V)) : Nat :=
  CombStruct_sparse_regularization.density f

def RegularityObj_sparse_regularization {V : Type u}
    [CombStruct_sparse_regularization V]
    (f : HypergraphObj_sparse_regularization (V := V)) :
    HypergraphObj_sparse_regularization (V := V) :=
  CombStruct_sparse_regularization.regularize f

theorem container_step_sparse_regularization {V : Type u}
    [CombStruct_sparse_regularization V]
    (f : HypergraphObj_sparse_regularization (V := V)) :
    forall v : V,
      RegularityObj_sparse_regularization f v <=
        CombStruct_sparse_regularization.container f v := by
  intro v
  have hraw :
      CombStruct_sparse_regularization.regularize f v <=
        CombStruct_sparse_regularization.container f v :=
    CombStruct_sparse_regularization.container_controls f v
  simpa [RegularityObj_sparse_regularization] using hraw

theorem entropy_lemma_step_sparse_regularization {V : Type u}
    [CombStruct_sparse_regularization V]
    (f : HypergraphObj_sparse_regularization (V := V)) :
    0 <= EntropyObj_sparse_regularization f ∧
      DensityObj_sparse_regularization f <=
        DensityObj_sparse_regularization (RegularityObj_sparse_regularization f) := by
  have hEntropy : 0 <= CombStruct_sparse_regularization.entropy f :=
    CombStruct_sparse_regularization.entropy_nonneg f
  have hPoint : forall v : V,
      f v <= CombStruct_sparse_regularization.regularize f v :=
    CombStruct_sparse_regularization.regularize_majorizes f
  have hDense :
      CombStruct_sparse_regularization.density f <=
        CombStruct_sparse_regularization.density (CombStruct_sparse_regularization.regularize f) :=
    CombStruct_sparse_regularization.density_mono hPoint
  constructor
  · simpa [EntropyObj_sparse_regularization] using hEntropy
  · simpa [DensityObj_sparse_regularization, RegularityObj_sparse_regularization] using hDense

theorem flag_density_step_sparse_regularization {V : Type u}
    [CombStruct_sparse_regularization V]
    (f : HypergraphObj_sparse_regularization (V := V)) :
    forall v : V,
      f v <= RegularityObj_sparse_regularization f v := by
  intro v
  have hMajor : f v <= CombStruct_sparse_regularization.regularize f v :=
    CombStruct_sparse_regularization.regularize_majorizes f v
  simpa [RegularityObj_sparse_regularization] using hMajor

theorem sparse_regularity_step_sparse_regularization {V : Type u}
    [CombStruct_sparse_regularization V]
    (f : HypergraphObj_sparse_regularization (V := V)) :
    forall v : V,
      f v <= RegularityObj_sparse_regularization f v ∧
      RegularityObj_sparse_regularization f v <=
        CombStruct_sparse_regularization.container f v := by
  intro v
  have hMajor : f v <= CombStruct_sparse_regularization.regularize f v :=
    CombStruct_sparse_regularization.regularize_majorizes f v
  have hCont : CombStruct_sparse_regularization.regularize f v <=
      CombStruct_sparse_regularization.container f v :=
    CombStruct_sparse_regularization.container_controls f v
  constructor
  · simpa [RegularityObj_sparse_regularization] using hMajor
  · simpa [RegularityObj_sparse_regularization] using hCont

theorem tverberg_partition_step_sparse_regularization {V : Type u}
    [CombStruct_sparse_regularization V]
    (f : HypergraphObj_sparse_regularization (V := V)) :
    forall v : V,
      RegularityObj_sparse_regularization
        (RegularityObj_sparse_regularization f) v =
      RegularityObj_sparse_regularization f v := by
  intro v
  have hidem :
      CombStruct_sparse_regularization.regularize
        (CombStruct_sparse_regularization.regularize f) v =
      CombStruct_sparse_regularization.regularize f v :=
    CombStruct_sparse_regularization.regularize_idem f v
  simpa [RegularityObj_sparse_regularization] using hidem

theorem counting_upgrade_sparse_regularization {V : Type u}
    [CombStruct_sparse_regularization V]
    (f : HypergraphObj_sparse_regularization (V := V)) :
    DensityObj_sparse_regularization (RegularityObj_sparse_regularization f) <=
      DensityObj_sparse_regularization (CombStruct_sparse_regularization.container f) +
        EntropyObj_sparse_regularization f := by
  have hcount :
      CombStruct_sparse_regularization.density
        (CombStruct_sparse_regularization.regularize f) <=
      CombStruct_sparse_regularization.density
        (CombStruct_sparse_regularization.container f) +
          CombStruct_sparse_regularization.entropy f :=
    CombStruct_sparse_regularization.counting_axiom f
  simpa [DensityObj_sparse_regularization, RegularityObj_sparse_regularization,
    EntropyObj_sparse_regularization] using hcount

theorem extremal_conclusion_sparse_regularization {V : Type u}
    [h : CombStruct_sparse_regularization V]
    (f : HypergraphObj_sparse_regularization (V := V)) :
    DensityObj_sparse_regularization f <=
      DensityObj_sparse_regularization (CombStruct_sparse_regularization.container f) +
        EntropyObj_sparse_regularization f := by
  have hSeed :
      DensityObj_sparse_regularization f <=
        DensityObj_sparse_regularization (RegularityObj_sparse_regularization f) :=
    (entropy_lemma_step_sparse_regularization f).2
  have hCount :
      DensityObj_sparse_regularization (RegularityObj_sparse_regularization f) <=
      DensityObj_sparse_regularization (CombStruct_sparse_regularization.container f) +
        EntropyObj_sparse_regularization f :=
    counting_upgrade_sparse_regularization f
  exact h.nat_le_trans_axiom
    (a := DensityObj_sparse_regularization f)
    (b := DensityObj_sparse_regularization (RegularityObj_sparse_regularization f))
    (c := DensityObj_sparse_regularization (CombStruct_sparse_regularization.container f) +
      EntropyObj_sparse_regularization f)
    hSeed hCount
