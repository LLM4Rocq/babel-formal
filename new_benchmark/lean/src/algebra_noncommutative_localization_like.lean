/-
BENCHMARK_ID: TINY_MATHLIB_BATCH05_ALGEBRA_NONCOMMUTATIVE_LOCALIZATION_LIKE
PAIR_STEM: algebra_noncommutative_localization_like
MATH_DOMAIN: Algebra
SOURCE_MATHLIB: Mathlib/Algebra/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class AlgebraStruct_noncommutative_localization (A : Type u) where
  rel : A -> A -> Prop
  rel_refl : forall x : A, rel x x
  rel_trans : forall {x y z : A}, rel x y -> rel y z -> rel x z
  derived : A -> A
  tensor : A -> A -> A
  localize : A -> A
  cohom : A -> A
  exact_derived_axiom : forall {x y : A}, rel x y -> rel (derived x) (derived y)
  bar_unit_axiom : forall x : A, rel x (tensor x (derived x))
  bar_counit_axiom : forall x : A, rel (tensor (derived x) x) x
  localization_axiom : forall x : A, rel (localize x) (localize (derived x))
  crystalline_axiom : forall x : A, rel (cohom (localize x)) (localize (cohom x))
  tilt_axiom : forall x : A, rel (derived (localize x)) (localize (derived x))
  reconstruction_axiom : forall x : A, rel (cohom (derived x)) (derived (cohom x))

infix:50 " ≺ " => AlgebraStruct_noncommutative_localization.rel

def DerivedObj_noncommutative_localization
    {A : Type u} [AlgebraStruct_noncommutative_localization A] (x : A) : A :=
  AlgebraStruct_noncommutative_localization.derived x

def TensorObj_noncommutative_localization
    {A : Type u} [AlgebraStruct_noncommutative_localization A] (x y : A) : A :=
  AlgebraStruct_noncommutative_localization.tensor x y

def LocalizationObj_noncommutative_localization
    {A : Type u} [AlgebraStruct_noncommutative_localization A] (x : A) : A :=
  AlgebraStruct_noncommutative_localization.localize x

def CohomologyObj_noncommutative_localization
    {A : Type u} [AlgebraStruct_noncommutative_localization A] (x : A) : A :=
  AlgebraStruct_noncommutative_localization.cohom x

theorem derived_functor_exact_noncommutative_localization
    {A : Type u} [AlgebraStruct_noncommutative_localization A]
    {x y : A} (hxy : x ≺ y) :
    DerivedObj_noncommutative_localization x ≺
      DerivedObj_noncommutative_localization y := by
  have hraw : AlgebraStruct_noncommutative_localization.rel
      (AlgebraStruct_noncommutative_localization.derived x)
      (AlgebraStruct_noncommutative_localization.derived y) :=
    AlgebraStruct_noncommutative_localization.exact_derived_axiom hxy
  calc
    DerivedObj_noncommutative_localization x =
        AlgebraStruct_noncommutative_localization.derived x := by
          rfl
    _ ≺ AlgebraStruct_noncommutative_localization.derived y := hraw
    _ = DerivedObj_noncommutative_localization y := by
          rfl

theorem bar_cobar_unit_noncommutative_localization
    {A : Type u} [AlgebraStruct_noncommutative_localization A]
    (x : A) :
    x ≺ TensorObj_noncommutative_localization x
      (DerivedObj_noncommutative_localization x) := by
  have hunit : AlgebraStruct_noncommutative_localization.rel x
      (AlgebraStruct_noncommutative_localization.tensor x
        (AlgebraStruct_noncommutative_localization.derived x)) :=
    AlgebraStruct_noncommutative_localization.bar_unit_axiom x
  calc
    x ≺ AlgebraStruct_noncommutative_localization.tensor x
      (AlgebraStruct_noncommutative_localization.derived x) := hunit
    _ = TensorObj_noncommutative_localization x
        (DerivedObj_noncommutative_localization x) := by
          rfl

theorem bar_cobar_counit_noncommutative_localization
    {A : Type u} [AlgebraStruct_noncommutative_localization A]
    (x : A) :
    TensorObj_noncommutative_localization
      (DerivedObj_noncommutative_localization x) x ≺ x := by
  have hcounit : AlgebraStruct_noncommutative_localization.rel
      (AlgebraStruct_noncommutative_localization.tensor
        (AlgebraStruct_noncommutative_localization.derived x) x) x :=
    AlgebraStruct_noncommutative_localization.bar_counit_axiom x
  calc
    TensorObj_noncommutative_localization
      (DerivedObj_noncommutative_localization x) x
        = AlgebraStruct_noncommutative_localization.tensor
            (AlgebraStruct_noncommutative_localization.derived x) x := by
              rfl
    _ ≺ x := hcounit

theorem localization_universal_noncommutative_localization
    {A : Type u} [AlgebraStruct_noncommutative_localization A]
    (x : A) :
    LocalizationObj_noncommutative_localization x ≺
      LocalizationObj_noncommutative_localization
        (DerivedObj_noncommutative_localization x) := by
  have hloc : AlgebraStruct_noncommutative_localization.rel
      (AlgebraStruct_noncommutative_localization.localize x)
      (AlgebraStruct_noncommutative_localization.localize
        (AlgebraStruct_noncommutative_localization.derived x)) :=
    AlgebraStruct_noncommutative_localization.localization_axiom x
  calc
    LocalizationObj_noncommutative_localization x =
        AlgebraStruct_noncommutative_localization.localize x := by
          rfl
    _ ≺ AlgebraStruct_noncommutative_localization.localize
          (AlgebraStruct_noncommutative_localization.derived x) := hloc
    _ = LocalizationObj_noncommutative_localization
          (DerivedObj_noncommutative_localization x) := by
          rfl

theorem crystalline_descent_noncommutative_localization
    {A : Type u} [AlgebraStruct_noncommutative_localization A]
    (x : A) :
    CohomologyObj_noncommutative_localization
      (LocalizationObj_noncommutative_localization x) ≺
      LocalizationObj_noncommutative_localization
        (CohomologyObj_noncommutative_localization x) := by
  have hcrys : AlgebraStruct_noncommutative_localization.rel
      (AlgebraStruct_noncommutative_localization.cohom
        (AlgebraStruct_noncommutative_localization.localize x))
      (AlgebraStruct_noncommutative_localization.localize
        (AlgebraStruct_noncommutative_localization.cohom x)) :=
    AlgebraStruct_noncommutative_localization.crystalline_axiom x
  calc
    CohomologyObj_noncommutative_localization
      (LocalizationObj_noncommutative_localization x)
        = AlgebraStruct_noncommutative_localization.cohom
            (AlgebraStruct_noncommutative_localization.localize x) := by
              rfl
    _ ≺ AlgebraStruct_noncommutative_localization.localize
          (AlgebraStruct_noncommutative_localization.cohom x) := hcrys
    _ = LocalizationObj_noncommutative_localization
          (CohomologyObj_noncommutative_localization x) := by
          rfl

theorem tilt_equivalence_core_noncommutative_localization
    {A : Type u} [AlgebraStruct_noncommutative_localization A]
    (x : A) :
    DerivedObj_noncommutative_localization
      (LocalizationObj_noncommutative_localization x) ≺
      LocalizationObj_noncommutative_localization
        (DerivedObj_noncommutative_localization x) := by
  have htilt : AlgebraStruct_noncommutative_localization.rel
      (AlgebraStruct_noncommutative_localization.derived
        (AlgebraStruct_noncommutative_localization.localize x))
      (AlgebraStruct_noncommutative_localization.localize
        (AlgebraStruct_noncommutative_localization.derived x)) :=
    AlgebraStruct_noncommutative_localization.tilt_axiom x
  calc
    DerivedObj_noncommutative_localization
      (LocalizationObj_noncommutative_localization x)
        = AlgebraStruct_noncommutative_localization.derived
            (AlgebraStruct_noncommutative_localization.localize x) := by
              rfl
    _ ≺ AlgebraStruct_noncommutative_localization.localize
          (AlgebraStruct_noncommutative_localization.derived x) := htilt
    _ = LocalizationObj_noncommutative_localization
          (DerivedObj_noncommutative_localization x) := by
          rfl

theorem algebraic_reconstruction_noncommutative_localization
    {A : Type u} [AlgebraStruct_noncommutative_localization A]
    (x : A) :
    CohomologyObj_noncommutative_localization
      (DerivedObj_noncommutative_localization x) ≺
      DerivedObj_noncommutative_localization
        (CohomologyObj_noncommutative_localization x) := by
  have hrec : AlgebraStruct_noncommutative_localization.rel
      (AlgebraStruct_noncommutative_localization.cohom
        (AlgebraStruct_noncommutative_localization.derived x))
      (AlgebraStruct_noncommutative_localization.derived
        (AlgebraStruct_noncommutative_localization.cohom x)) :=
    AlgebraStruct_noncommutative_localization.reconstruction_axiom x
  calc
    CohomologyObj_noncommutative_localization
      (DerivedObj_noncommutative_localization x)
        = AlgebraStruct_noncommutative_localization.cohom
            (AlgebraStruct_noncommutative_localization.derived x) := by
              rfl
    _ ≺ AlgebraStruct_noncommutative_localization.derived
          (AlgebraStruct_noncommutative_localization.cohom x) := hrec
    _ = DerivedObj_noncommutative_localization
          (CohomologyObj_noncommutative_localization x) := by
          rfl
