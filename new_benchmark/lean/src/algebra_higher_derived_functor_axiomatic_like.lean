/-
BENCHMARK_ID: TINY_MATHLIB_BATCH05_ALGEBRA_HIGHER_DERIVED_FUNCTOR_AXIOMATIC_LIKE
PAIR_STEM: algebra_higher_derived_functor_axiomatic_like
MATH_DOMAIN: Algebra
SOURCE_MATHLIB: Mathlib/Algebra/Homology
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class AlgebraStruct_higher_derived_functor (A : Type u) where
  derived : A → A
  tensor : A → A → A
  localize : A → A
  cohomology : A → A
  localize_idem_axiom : ∀ X : A, localize (localize X) = localize X
  derived_functor_exact_axiom : ∀ X Y : A,
    derived (tensor X Y) = derived (tensor Y X)
  bar_cobar_unit_axiom : ∀ X : A,
    tensor (derived X) (localize X) = localize (tensor X X)
  bar_cobar_counit_axiom : ∀ X : A,
    tensor (localize X) (derived X) = localize (tensor X X)
  localization_universal_axiom : ∀ X : A,
    localize (derived X) = derived (localize X)
  crystalline_descent_axiom : ∀ X : A,
    cohomology (localize X) = cohomology X
  tilt_equivalence_axiom : ∀ X : A,
    derived (cohomology X) = cohomology (derived X)
  algebraic_reconstruction_axiom : ∀ X : A,
    tensor (cohomology X) (localize X) = tensor (localize X) (cohomology X)

def DerivedObj_higher_derived_functor {A : Type u}
    (S : AlgebraStruct_higher_derived_functor A) (X : A) : A :=
  S.derived X

def TensorObj_higher_derived_functor {A : Type u}
    (S : AlgebraStruct_higher_derived_functor A) (X Y : A) : A :=
  S.tensor X Y

def LocalizationObj_higher_derived_functor {A : Type u}
    (S : AlgebraStruct_higher_derived_functor A) (X : A) : A :=
  S.localize X

def CohomologyObj_higher_derived_functor {A : Type u}
    (S : AlgebraStruct_higher_derived_functor A) (X : A) : A :=
  S.cohomology X

theorem derived_functor_exact_higher_derived_functor {A : Type u}
    (S : AlgebraStruct_higher_derived_functor A)
    (X Y : A) :
    DerivedObj_higher_derived_functor S (TensorObj_higher_derived_functor S X Y)
      = DerivedObj_higher_derived_functor S (TensorObj_higher_derived_functor S Y X) := by
  have hExact : S.derived (S.tensor X Y) = S.derived (S.tensor Y X) :=
    S.derived_functor_exact_axiom X Y
  have hLeft :
      DerivedObj_higher_derived_functor S (TensorObj_higher_derived_functor S X Y)
        = S.derived (S.tensor X Y) := by
    rfl
  have hRight :
      DerivedObj_higher_derived_functor S (TensorObj_higher_derived_functor S Y X)
        = S.derived (S.tensor Y X) := by
    rfl
  calc
    DerivedObj_higher_derived_functor S (TensorObj_higher_derived_functor S X Y)
        = S.derived (S.tensor X Y) := hLeft
    _ = S.derived (S.tensor Y X) := hExact
    _ = DerivedObj_higher_derived_functor S (TensorObj_higher_derived_functor S Y X) := by
      symm
      exact hRight

theorem bar_cobar_unit_higher_derived_functor {A : Type u}
    (S : AlgebraStruct_higher_derived_functor A)
    (X : A) :
    TensorObj_higher_derived_functor S (DerivedObj_higher_derived_functor S X)
      (LocalizationObj_higher_derived_functor S X)
      = LocalizationObj_higher_derived_functor S (TensorObj_higher_derived_functor S X X) := by
  have hUnit : S.tensor (S.derived X) (S.localize X) = S.localize (S.tensor X X) :=
    S.bar_cobar_unit_axiom X
  change S.tensor (S.derived X) (S.localize X) = S.localize (S.tensor X X)
  exact hUnit

theorem bar_cobar_counit_higher_derived_functor {A : Type u}
    (S : AlgebraStruct_higher_derived_functor A)
    (X : A) :
    TensorObj_higher_derived_functor S (LocalizationObj_higher_derived_functor S X)
      (DerivedObj_higher_derived_functor S X)
      = LocalizationObj_higher_derived_functor S (TensorObj_higher_derived_functor S X X) := by
  have hCounit : S.tensor (S.localize X) (S.derived X) = S.localize (S.tensor X X) :=
    S.bar_cobar_counit_axiom X
  change S.tensor (S.localize X) (S.derived X) = S.localize (S.tensor X X)
  exact hCounit

theorem localization_universal_higher_derived_functor {A : Type u}
    (S : AlgebraStruct_higher_derived_functor A)
    (X : A) :
    LocalizationObj_higher_derived_functor S (DerivedObj_higher_derived_functor S X)
      = DerivedObj_higher_derived_functor S (LocalizationObj_higher_derived_functor S X) := by
  have hLoc : S.localize (S.derived X) = S.derived (S.localize X) :=
    S.localization_universal_axiom X
  change S.localize (S.derived X) = S.derived (S.localize X)
  exact hLoc

theorem crystalline_descent_higher_derived_functor {A : Type u}
    (S : AlgebraStruct_higher_derived_functor A)
    (X : A) :
    CohomologyObj_higher_derived_functor S
      (LocalizationObj_higher_derived_functor S (DerivedObj_higher_derived_functor S X))
      = CohomologyObj_higher_derived_functor S (DerivedObj_higher_derived_functor S X) := by
  have hStep :
      S.cohomology (S.localize (S.derived X)) = S.cohomology (S.derived X) :=
    S.crystalline_descent_axiom (S.derived X)
  have hExpandLeft :
      CohomologyObj_higher_derived_functor S
        (LocalizationObj_higher_derived_functor S (DerivedObj_higher_derived_functor S X))
      = S.cohomology (S.localize (S.derived X)) := by
    rfl
  have hExpandRight :
      CohomologyObj_higher_derived_functor S (DerivedObj_higher_derived_functor S X)
      = S.cohomology (S.derived X) := by
    rfl
  calc
    CohomologyObj_higher_derived_functor S
        (LocalizationObj_higher_derived_functor S (DerivedObj_higher_derived_functor S X))
        = S.cohomology (S.localize (S.derived X)) := hExpandLeft
    _ = S.cohomology (S.derived X) := hStep
    _ = CohomologyObj_higher_derived_functor S (DerivedObj_higher_derived_functor S X) := by
      symm
      exact hExpandRight

theorem tilt_equivalence_core_higher_derived_functor {A : Type u}
    (S : AlgebraStruct_higher_derived_functor A)
    (X : A) :
    DerivedObj_higher_derived_functor S (CohomologyObj_higher_derived_functor S X)
      = CohomologyObj_higher_derived_functor S (DerivedObj_higher_derived_functor S X) := by
  have hTilt : S.derived (S.cohomology X) = S.cohomology (S.derived X) :=
    S.tilt_equivalence_axiom X
  change S.derived (S.cohomology X) = S.cohomology (S.derived X)
  exact hTilt

theorem algebraic_reconstruction_higher_derived_functor {A : Type u}
    (S : AlgebraStruct_higher_derived_functor A)
    (X : A) :
    TensorObj_higher_derived_functor S
      (CohomologyObj_higher_derived_functor S (LocalizationObj_higher_derived_functor S X))
      (LocalizationObj_higher_derived_functor S X)
      =
    TensorObj_higher_derived_functor S
      (LocalizationObj_higher_derived_functor S X)
      (CohomologyObj_higher_derived_functor S (LocalizationObj_higher_derived_functor S X)) := by
  have hRecon :
      S.tensor (S.cohomology (S.localize X)) (S.localize (S.localize X))
        = S.tensor (S.localize (S.localize X)) (S.cohomology (S.localize X)) :=
    S.algebraic_reconstruction_axiom (S.localize X)
  have hIdemLeft : S.localize (S.localize X) = S.localize X :=
    S.localize_idem_axiom X
  have hIdemRight : S.localize (S.localize X) = S.localize X :=
    S.localize_idem_axiom X
  calc
    TensorObj_higher_derived_functor S
        (CohomologyObj_higher_derived_functor S (LocalizationObj_higher_derived_functor S X))
        (LocalizationObj_higher_derived_functor S X)
        = S.tensor (S.cohomology (S.localize X)) (S.localize X) := by
          rfl
    _ = S.tensor (S.cohomology (S.localize X)) (S.localize (S.localize X)) := by
          rw [hIdemLeft]
    _ = S.tensor (S.localize (S.localize X)) (S.cohomology (S.localize X)) := hRecon
    _ = S.tensor (S.localize X) (S.cohomology (S.localize X)) := by
          rw [hIdemRight]
    _ = TensorObj_higher_derived_functor S
          (LocalizationObj_higher_derived_functor S X)
          (CohomologyObj_higher_derived_functor S (LocalizationObj_higher_derived_functor S X)) := by
          rfl
