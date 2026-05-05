/-
BENCHMARK_ID: TINY_MATHLIB_BATCH05_ALGEBRA_INFINITY_ALGEBRA_BAR_COBAR_LIKE
PAIR_STEM: algebra_infinity_algebra_bar_cobar_like
MATH_DOMAIN: Algebra
SOURCE_MATHLIB: Mathlib/Algebra/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class AlgebraStruct_infinity_algebra_bar (A : Type u) where
  mul : A -> A -> A
  one : A
  d : A -> A
  bar : A -> A
  cobar : A -> A
  loc : A -> A
  coh : A -> A
  d_mul_axiom : forall x y : A, d (mul x y) = mul (d x) y
  bar_cobar_unit_axiom : forall x : A, cobar (bar x) = x
  bar_cobar_counit_axiom : forall x : A, bar (cobar x) = x
  loc_mul_axiom : forall x y : A, loc (mul x y) = mul (loc x) (loc y)
  coh_loc_comm_axiom : forall x : A, coh (loc x) = loc (coh x)
  coh_bar_comm_axiom : forall x : A, coh (bar x) = bar (coh x)

def DerivedObj_infinity_algebra_bar {A : Type u}
    [AlgebraStruct_infinity_algebra_bar A] (x : A) : A :=
  AlgebraStruct_infinity_algebra_bar.d x

def TensorObj_infinity_algebra_bar {A : Type u}
    [AlgebraStruct_infinity_algebra_bar A] (x y : A) : A :=
  AlgebraStruct_infinity_algebra_bar.mul x y

def LocalizationObj_infinity_algebra_bar {A : Type u}
    [AlgebraStruct_infinity_algebra_bar A] (x : A) : A :=
  AlgebraStruct_infinity_algebra_bar.loc x

def CohomologyObj_infinity_algebra_bar {A : Type u}
    [AlgebraStruct_infinity_algebra_bar A] (x : A) : A :=
  AlgebraStruct_infinity_algebra_bar.coh x

theorem derived_functor_exact_infinity_algebra_bar {A : Type u}
    [AlgebraStruct_infinity_algebra_bar A]
    (x y : A) :
    DerivedObj_infinity_algebra_bar (TensorObj_infinity_algebra_bar x y) =
      TensorObj_infinity_algebra_bar (DerivedObj_infinity_algebra_bar x) y := by
  have hRaw :
      AlgebraStruct_infinity_algebra_bar.d
        (AlgebraStruct_infinity_algebra_bar.mul x y) =
      AlgebraStruct_infinity_algebra_bar.mul
        (AlgebraStruct_infinity_algebra_bar.d x) y :=
    AlgebraStruct_infinity_algebra_bar.d_mul_axiom x y
  calc
    DerivedObj_infinity_algebra_bar (TensorObj_infinity_algebra_bar x y)
        = AlgebraStruct_infinity_algebra_bar.d
            (AlgebraStruct_infinity_algebra_bar.mul x y) := by
          rfl
    _ = AlgebraStruct_infinity_algebra_bar.mul
          (AlgebraStruct_infinity_algebra_bar.d x) y := hRaw
    _ = TensorObj_infinity_algebra_bar (DerivedObj_infinity_algebra_bar x) y := by
          rfl

theorem bar_cobar_unit_infinity_algebra_bar {A : Type u}
    [AlgebraStruct_infinity_algebra_bar A]
    (x : A) :
    AlgebraStruct_infinity_algebra_bar.cobar
      (AlgebraStruct_infinity_algebra_bar.bar x) = x := by
  have hUnit :
      AlgebraStruct_infinity_algebra_bar.cobar
        (AlgebraStruct_infinity_algebra_bar.bar x) = x :=
    AlgebraStruct_infinity_algebra_bar.bar_cobar_unit_axiom x
  calc
    AlgebraStruct_infinity_algebra_bar.cobar
        (AlgebraStruct_infinity_algebra_bar.bar x)
        = x := hUnit

theorem bar_cobar_counit_infinity_algebra_bar {A : Type u}
    [AlgebraStruct_infinity_algebra_bar A]
    (x : A) :
    AlgebraStruct_infinity_algebra_bar.bar
      (AlgebraStruct_infinity_algebra_bar.cobar x) = x := by
  have hCounit :
      AlgebraStruct_infinity_algebra_bar.bar
        (AlgebraStruct_infinity_algebra_bar.cobar x) = x :=
    AlgebraStruct_infinity_algebra_bar.bar_cobar_counit_axiom x
  calc
    AlgebraStruct_infinity_algebra_bar.bar
        (AlgebraStruct_infinity_algebra_bar.cobar x)
        = x := hCounit

theorem localization_universal_infinity_algebra_bar {A : Type u}
    [AlgebraStruct_infinity_algebra_bar A]
    (x y : A) :
    LocalizationObj_infinity_algebra_bar (TensorObj_infinity_algebra_bar x y) =
      TensorObj_infinity_algebra_bar
        (LocalizationObj_infinity_algebra_bar x)
        (LocalizationObj_infinity_algebra_bar y) := by
  have hRaw :
      AlgebraStruct_infinity_algebra_bar.loc
        (AlgebraStruct_infinity_algebra_bar.mul x y) =
      AlgebraStruct_infinity_algebra_bar.mul
        (AlgebraStruct_infinity_algebra_bar.loc x)
        (AlgebraStruct_infinity_algebra_bar.loc y) :=
    AlgebraStruct_infinity_algebra_bar.loc_mul_axiom x y
  calc
    LocalizationObj_infinity_algebra_bar (TensorObj_infinity_algebra_bar x y)
        = AlgebraStruct_infinity_algebra_bar.loc
            (AlgebraStruct_infinity_algebra_bar.mul x y) := by
          rfl
    _ = AlgebraStruct_infinity_algebra_bar.mul
          (AlgebraStruct_infinity_algebra_bar.loc x)
          (AlgebraStruct_infinity_algebra_bar.loc y) := hRaw
    _ = TensorObj_infinity_algebra_bar
          (LocalizationObj_infinity_algebra_bar x)
          (LocalizationObj_infinity_algebra_bar y) := by
          rfl

theorem crystalline_descent_infinity_algebra_bar {A : Type u}
    [AlgebraStruct_infinity_algebra_bar A]
    (x : A) :
    CohomologyObj_infinity_algebra_bar (LocalizationObj_infinity_algebra_bar x) =
      LocalizationObj_infinity_algebra_bar (CohomologyObj_infinity_algebra_bar x) := by
  have hRaw :
      AlgebraStruct_infinity_algebra_bar.coh
        (AlgebraStruct_infinity_algebra_bar.loc x) =
      AlgebraStruct_infinity_algebra_bar.loc
        (AlgebraStruct_infinity_algebra_bar.coh x) :=
    AlgebraStruct_infinity_algebra_bar.coh_loc_comm_axiom x
  calc
    CohomologyObj_infinity_algebra_bar (LocalizationObj_infinity_algebra_bar x)
        = AlgebraStruct_infinity_algebra_bar.coh
            (AlgebraStruct_infinity_algebra_bar.loc x) := by
          rfl
    _ = AlgebraStruct_infinity_algebra_bar.loc
          (AlgebraStruct_infinity_algebra_bar.coh x) := hRaw
    _ = LocalizationObj_infinity_algebra_bar (CohomologyObj_infinity_algebra_bar x) := by
          rfl

theorem tilt_equivalence_core_infinity_algebra_bar {A : Type u}
    [AlgebraStruct_infinity_algebra_bar A]
    (x : A) :
    CohomologyObj_infinity_algebra_bar
      (AlgebraStruct_infinity_algebra_bar.bar x) =
      AlgebraStruct_infinity_algebra_bar.bar (CohomologyObj_infinity_algebra_bar x) := by
  have hRaw :
      AlgebraStruct_infinity_algebra_bar.coh
        (AlgebraStruct_infinity_algebra_bar.bar x) =
      AlgebraStruct_infinity_algebra_bar.bar
        (AlgebraStruct_infinity_algebra_bar.coh x) :=
    AlgebraStruct_infinity_algebra_bar.coh_bar_comm_axiom x
  calc
    CohomologyObj_infinity_algebra_bar
        (AlgebraStruct_infinity_algebra_bar.bar x)
        = AlgebraStruct_infinity_algebra_bar.coh
            (AlgebraStruct_infinity_algebra_bar.bar x) := by
          rfl
    _ = AlgebraStruct_infinity_algebra_bar.bar
          (AlgebraStruct_infinity_algebra_bar.coh x) := hRaw
    _ = AlgebraStruct_infinity_algebra_bar.bar (CohomologyObj_infinity_algebra_bar x) := by
          rfl

theorem algebraic_reconstruction_infinity_algebra_bar {A : Type u}
    [AlgebraStruct_infinity_algebra_bar A]
    (x : A) :
    AlgebraStruct_infinity_algebra_bar.cobar
      (AlgebraStruct_infinity_algebra_bar.bar
        (LocalizationObj_infinity_algebra_bar (CohomologyObj_infinity_algebra_bar x))) =
      LocalizationObj_infinity_algebra_bar (CohomologyObj_infinity_algebra_bar x) := by
  have hUnit :
      AlgebraStruct_infinity_algebra_bar.cobar
          (AlgebraStruct_infinity_algebra_bar.bar
            (LocalizationObj_infinity_algebra_bar (CohomologyObj_infinity_algebra_bar x))) =
        LocalizationObj_infinity_algebra_bar (CohomologyObj_infinity_algebra_bar x) :=
    AlgebraStruct_infinity_algebra_bar.bar_cobar_unit_axiom
      (LocalizationObj_infinity_algebra_bar (CohomologyObj_infinity_algebra_bar x))
  have hShape :
      LocalizationObj_infinity_algebra_bar (CohomologyObj_infinity_algebra_bar x) =
        AlgebraStruct_infinity_algebra_bar.loc
          (AlgebraStruct_infinity_algebra_bar.coh x) := by
    rfl
  calc
    AlgebraStruct_infinity_algebra_bar.cobar
        (AlgebraStruct_infinity_algebra_bar.bar
          (LocalizationObj_infinity_algebra_bar (CohomologyObj_infinity_algebra_bar x)))
        = LocalizationObj_infinity_algebra_bar (CohomologyObj_infinity_algebra_bar x) := hUnit
    _ = AlgebraStruct_infinity_algebra_bar.loc
          (AlgebraStruct_infinity_algebra_bar.coh x) := hShape
    _ = LocalizationObj_infinity_algebra_bar (CohomologyObj_infinity_algebra_bar x) := by
          rfl
