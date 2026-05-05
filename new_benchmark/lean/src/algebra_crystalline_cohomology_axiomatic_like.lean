/-
BENCHMARK_ID: TINY_MATHLIB_BATCH05_ALGEBRA_CRYSTALLINE_COHOMOLOGY_AXIOMATIC_LIKE
PAIR_STEM: algebra_crystalline_cohomology_axiomatic_like
MATH_DOMAIN: Algebra
SOURCE_MATHLIB: Mathlib/Algebra/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 17
-/

universe u

class AlgebraStruct_crystalline_cohomology (A : Type u) where
  rel : A -> A -> Prop
  rel_refl : forall x : A, rel x x
  rel_trans : forall {x y z : A}, rel x y -> rel y z -> rel x z
  Derived : A -> A
  Tensor : A -> A
  Localization : A -> A
  Cohomology : A -> A
  derived_exact_axiom : forall x : A, rel (Derived (Tensor x)) (Tensor (Derived x))
  unit_axiom : forall x : A, rel x (Tensor x)
  counit_axiom : forall x : A, rel (Tensor x) x
  localization_axiom : forall x : A,
      rel (Localization (Tensor x)) (Tensor (Localization x))
  descent_axiom : forall x : A,
      rel (Cohomology (Localization x)) (Localization (Cohomology x))
  tilt_axiom : forall x : A,
      rel (Derived (Cohomology x)) (Cohomology (Derived x))

def DerivedObj_crystalline_cohomology {A : Type u}
    [AlgebraStruct_crystalline_cohomology A] (x : A) : A :=
  AlgebraStruct_crystalline_cohomology.Derived x

def TensorObj_crystalline_cohomology {A : Type u}
    [AlgebraStruct_crystalline_cohomology A] (x : A) : A :=
  AlgebraStruct_crystalline_cohomology.Tensor x

def LocalizationObj_crystalline_cohomology {A : Type u}
    [AlgebraStruct_crystalline_cohomology A] (x : A) : A :=
  AlgebraStruct_crystalline_cohomology.Localization x

def CohomologyObj_crystalline_cohomology {A : Type u}
    [AlgebraStruct_crystalline_cohomology A] (x : A) : A :=
  AlgebraStruct_crystalline_cohomology.Cohomology x

theorem derived_functor_exact_crystalline_cohomology {A : Type u}
    [AlgebraStruct_crystalline_cohomology A]
    (x : A) :
    AlgebraStruct_crystalline_cohomology.rel
      (DerivedObj_crystalline_cohomology (TensorObj_crystalline_cohomology x))
      (TensorObj_crystalline_cohomology (DerivedObj_crystalline_cohomology x)) := by
  have hRaw :
      AlgebraStruct_crystalline_cohomology.rel
        (AlgebraStruct_crystalline_cohomology.Derived
          (AlgebraStruct_crystalline_cohomology.Tensor x))
        (AlgebraStruct_crystalline_cohomology.Tensor
          (AlgebraStruct_crystalline_cohomology.Derived x)) :=
    AlgebraStruct_crystalline_cohomology.derived_exact_axiom x
  simpa [DerivedObj_crystalline_cohomology, TensorObj_crystalline_cohomology] using hRaw

theorem bar_cobar_unit_crystalline_cohomology {A : Type u}
    [AlgebraStruct_crystalline_cohomology A]
    (x : A) :
    ∃ y : A,
      y = TensorObj_crystalline_cohomology x ∧
      AlgebraStruct_crystalline_cohomology.rel x y := by
  have hUnit :
      AlgebraStruct_crystalline_cohomology.rel
        x (AlgebraStruct_crystalline_cohomology.Tensor x) :=
    AlgebraStruct_crystalline_cohomology.unit_axiom x
  refine ⟨TensorObj_crystalline_cohomology x, rfl, ?_⟩
  simpa [TensorObj_crystalline_cohomology] using hUnit

theorem bar_cobar_counit_crystalline_cohomology {A : Type u}
    [AlgebraStruct_crystalline_cohomology A]
    (x : A) :
    AlgebraStruct_crystalline_cohomology.rel
      (TensorObj_crystalline_cohomology x) x ∧
    AlgebraStruct_crystalline_cohomology.rel x x := by
  have hCounit :
      AlgebraStruct_crystalline_cohomology.rel
        (AlgebraStruct_crystalline_cohomology.Tensor x) x :=
    AlgebraStruct_crystalline_cohomology.counit_axiom x
  have hRefl : AlgebraStruct_crystalline_cohomology.rel x x :=
    AlgebraStruct_crystalline_cohomology.rel_refl _
  constructor
  · simpa [TensorObj_crystalline_cohomology] using hCounit
  · exact hRefl

theorem localization_universal_crystalline_cohomology {A : Type u}
    [AlgebraStruct_crystalline_cohomology A]
    (x : A) :
    AlgebraStruct_crystalline_cohomology.rel
      (LocalizationObj_crystalline_cohomology (TensorObj_crystalline_cohomology x))
      (TensorObj_crystalline_cohomology (LocalizationObj_crystalline_cohomology x)) := by
  have hLoc :
      AlgebraStruct_crystalline_cohomology.rel
        (AlgebraStruct_crystalline_cohomology.Localization
          (AlgebraStruct_crystalline_cohomology.Tensor x))
        (AlgebraStruct_crystalline_cohomology.Tensor
          (AlgebraStruct_crystalline_cohomology.Localization x)) :=
    AlgebraStruct_crystalline_cohomology.localization_axiom x
  simpa [LocalizationObj_crystalline_cohomology, TensorObj_crystalline_cohomology] using hLoc

theorem crystalline_descent_crystalline_cohomology {A : Type u}
    [AlgebraStruct_crystalline_cohomology A]
    (x : A) :
    AlgebraStruct_crystalline_cohomology.rel
      (CohomologyObj_crystalline_cohomology
        (LocalizationObj_crystalline_cohomology x))
      (LocalizationObj_crystalline_cohomology
        (CohomologyObj_crystalline_cohomology x)) := by
  have hDesc :
      AlgebraStruct_crystalline_cohomology.rel
        (AlgebraStruct_crystalline_cohomology.Cohomology
          (AlgebraStruct_crystalline_cohomology.Localization x))
        (AlgebraStruct_crystalline_cohomology.Localization
          (AlgebraStruct_crystalline_cohomology.Cohomology x)) :=
    AlgebraStruct_crystalline_cohomology.descent_axiom x
  simpa [LocalizationObj_crystalline_cohomology, CohomologyObj_crystalline_cohomology] using hDesc

theorem tilt_equivalence_core_crystalline_cohomology {A : Type u}
    [AlgebraStruct_crystalline_cohomology A]
    (x : A) :
    AlgebraStruct_crystalline_cohomology.rel
      (DerivedObj_crystalline_cohomology
        (CohomologyObj_crystalline_cohomology x))
      (CohomologyObj_crystalline_cohomology
        (DerivedObj_crystalline_cohomology x)) := by
  have hTilt :
      AlgebraStruct_crystalline_cohomology.rel
        (AlgebraStruct_crystalline_cohomology.Derived
          (AlgebraStruct_crystalline_cohomology.Cohomology x))
        (AlgebraStruct_crystalline_cohomology.Cohomology
          (AlgebraStruct_crystalline_cohomology.Derived x)) :=
    AlgebraStruct_crystalline_cohomology.tilt_axiom x
  simpa [DerivedObj_crystalline_cohomology, CohomologyObj_crystalline_cohomology] using hTilt

theorem algebraic_reconstruction_crystalline_cohomology {A : Type u}
    [AlgebraStruct_crystalline_cohomology A]
    (x : A) :
    AlgebraStruct_crystalline_cohomology.rel
      (LocalizationObj_crystalline_cohomology
        (CohomologyObj_crystalline_cohomology
          (DerivedObj_crystalline_cohomology x)))
      (TensorObj_crystalline_cohomology
        (LocalizationObj_crystalline_cohomology
          (CohomologyObj_crystalline_cohomology
            (DerivedObj_crystalline_cohomology x)))) := by
  have h1 :
      AlgebraStruct_crystalline_cohomology.rel
        (DerivedObj_crystalline_cohomology
          (CohomologyObj_crystalline_cohomology x))
        (CohomologyObj_crystalline_cohomology
          (DerivedObj_crystalline_cohomology x)) :=
    tilt_equivalence_core_crystalline_cohomology x
  have h2 :
      AlgebraStruct_crystalline_cohomology.rel
        (LocalizationObj_crystalline_cohomology
          (CohomologyObj_crystalline_cohomology
            (DerivedObj_crystalline_cohomology x)))
        (TensorObj_crystalline_cohomology
          (LocalizationObj_crystalline_cohomology
            (CohomologyObj_crystalline_cohomology
              (DerivedObj_crystalline_cohomology x)))) := by
    have hUnitWitness :
        ∃ y : A,
          y =
            TensorObj_crystalline_cohomology
              (LocalizationObj_crystalline_cohomology
                (CohomologyObj_crystalline_cohomology
                  (DerivedObj_crystalline_cohomology x))) ∧
          AlgebraStruct_crystalline_cohomology.rel
            (LocalizationObj_crystalline_cohomology
              (CohomologyObj_crystalline_cohomology
                (DerivedObj_crystalline_cohomology x)))
            y :=
      bar_cobar_unit_crystalline_cohomology
        (LocalizationObj_crystalline_cohomology
          (CohomologyObj_crystalline_cohomology
            (DerivedObj_crystalline_cohomology x)))
    rcases hUnitWitness with ⟨y, hyEq, hyRel⟩
    rw [hyEq] at hyRel
    exact hyRel
  have h3 :
      AlgebraStruct_crystalline_cohomology.rel
        (LocalizationObj_crystalline_cohomology
          (CohomologyObj_crystalline_cohomology
            (DerivedObj_crystalline_cohomology x)))
        (LocalizationObj_crystalline_cohomology
          (CohomologyObj_crystalline_cohomology
            (DerivedObj_crystalline_cohomology x))) :=
    AlgebraStruct_crystalline_cohomology.rel_refl _
  have _ := crystalline_descent_crystalline_cohomology
      (DerivedObj_crystalline_cohomology x)
  have _ := h1
  exact AlgebraStruct_crystalline_cohomology.rel_trans h3 h2

theorem descent_then_tensorize_crystalline_cohomology {A : Type u}
    [AlgebraStruct_crystalline_cohomology A]
    (x : A) :
    AlgebraStruct_crystalline_cohomology.rel
      (CohomologyObj_crystalline_cohomology
        (LocalizationObj_crystalline_cohomology x))
      (TensorObj_crystalline_cohomology
        (LocalizationObj_crystalline_cohomology
          (CohomologyObj_crystalline_cohomology x))) := by
  have hDesc :
      AlgebraStruct_crystalline_cohomology.rel
        (CohomologyObj_crystalline_cohomology
          (LocalizationObj_crystalline_cohomology x))
        (LocalizationObj_crystalline_cohomology
          (CohomologyObj_crystalline_cohomology x)) :=
    crystalline_descent_crystalline_cohomology x
  have hUnit :
      AlgebraStruct_crystalline_cohomology.rel
        (LocalizationObj_crystalline_cohomology
          (CohomologyObj_crystalline_cohomology x))
        (TensorObj_crystalline_cohomology
          (LocalizationObj_crystalline_cohomology
            (CohomologyObj_crystalline_cohomology x))) := by
    have hRaw := AlgebraStruct_crystalline_cohomology.unit_axiom
      (LocalizationObj_crystalline_cohomology
        (CohomologyObj_crystalline_cohomology x))
    simpa [TensorObj_crystalline_cohomology] using hRaw
  exact AlgebraStruct_crystalline_cohomology.rel_trans hDesc hUnit

theorem localization_tensor_retract_crystalline_cohomology {A : Type u}
    [AlgebraStruct_crystalline_cohomology A]
    (x : A) :
    AlgebraStruct_crystalline_cohomology.rel
      (LocalizationObj_crystalline_cohomology
        (TensorObj_crystalline_cohomology x))
      (LocalizationObj_crystalline_cohomology x) := by
  have hLoc :
      AlgebraStruct_crystalline_cohomology.rel
        (LocalizationObj_crystalline_cohomology
          (TensorObj_crystalline_cohomology x))
        (TensorObj_crystalline_cohomology
          (LocalizationObj_crystalline_cohomology x)) :=
    localization_universal_crystalline_cohomology x
  have hCounit :
      AlgebraStruct_crystalline_cohomology.rel
        (TensorObj_crystalline_cohomology
          (LocalizationObj_crystalline_cohomology x))
        (LocalizationObj_crystalline_cohomology x) :=
    (bar_cobar_counit_crystalline_cohomology
      (LocalizationObj_crystalline_cohomology x)).1
  exact AlgebraStruct_crystalline_cohomology.rel_trans hLoc hCounit

theorem derived_tensor_retract_crystalline_cohomology {A : Type u}
    [AlgebraStruct_crystalline_cohomology A]
    (x : A) :
    AlgebraStruct_crystalline_cohomology.rel
      (DerivedObj_crystalline_cohomology
        (TensorObj_crystalline_cohomology x))
      (DerivedObj_crystalline_cohomology x) := by
  have hExact :
      AlgebraStruct_crystalline_cohomology.rel
        (DerivedObj_crystalline_cohomology
          (TensorObj_crystalline_cohomology x))
        (TensorObj_crystalline_cohomology
          (DerivedObj_crystalline_cohomology x)) :=
    derived_functor_exact_crystalline_cohomology x
  have hCounit :
      AlgebraStruct_crystalline_cohomology.rel
        (TensorObj_crystalline_cohomology
          (DerivedObj_crystalline_cohomology x))
        (DerivedObj_crystalline_cohomology x) :=
    (bar_cobar_counit_crystalline_cohomology
      (DerivedObj_crystalline_cohomology x)).1
  exact AlgebraStruct_crystalline_cohomology.rel_trans hExact hCounit

theorem tilt_then_tensorize_crystalline_cohomology {A : Type u}
    [AlgebraStruct_crystalline_cohomology A]
    (x : A) :
    AlgebraStruct_crystalline_cohomology.rel
      (DerivedObj_crystalline_cohomology
        (CohomologyObj_crystalline_cohomology x))
      (TensorObj_crystalline_cohomology
        (CohomologyObj_crystalline_cohomology
          (DerivedObj_crystalline_cohomology x))) := by
  have hTilt :
      AlgebraStruct_crystalline_cohomology.rel
        (DerivedObj_crystalline_cohomology
          (CohomologyObj_crystalline_cohomology x))
        (CohomologyObj_crystalline_cohomology
          (DerivedObj_crystalline_cohomology x)) :=
    tilt_equivalence_core_crystalline_cohomology x
  have hUnit :
      AlgebraStruct_crystalline_cohomology.rel
        (CohomologyObj_crystalline_cohomology
          (DerivedObj_crystalline_cohomology x))
        (TensorObj_crystalline_cohomology
          (CohomologyObj_crystalline_cohomology
            (DerivedObj_crystalline_cohomology x))) := by
    have hRaw := AlgebraStruct_crystalline_cohomology.unit_axiom
      (CohomologyObj_crystalline_cohomology
        (DerivedObj_crystalline_cohomology x))
    simpa [TensorObj_crystalline_cohomology] using hRaw
  exact AlgebraStruct_crystalline_cohomology.rel_trans hTilt hUnit

theorem tensor_retractions_pair_crystalline_cohomology {A : Type u}
    [AlgebraStruct_crystalline_cohomology A]
    (x : A) :
    AlgebraStruct_crystalline_cohomology.rel
      (LocalizationObj_crystalline_cohomology
        (TensorObj_crystalline_cohomology x))
      (LocalizationObj_crystalline_cohomology x) ∧
    AlgebraStruct_crystalline_cohomology.rel
      (DerivedObj_crystalline_cohomology
        (TensorObj_crystalline_cohomology x))
      (DerivedObj_crystalline_cohomology x) := by
  have hLoc :
      AlgebraStruct_crystalline_cohomology.rel
        (LocalizationObj_crystalline_cohomology
          (TensorObj_crystalline_cohomology x))
        (LocalizationObj_crystalline_cohomology x) :=
    localization_tensor_retract_crystalline_cohomology x
  have hDer :
      AlgebraStruct_crystalline_cohomology.rel
        (DerivedObj_crystalline_cohomology
          (TensorObj_crystalline_cohomology x))
        (DerivedObj_crystalline_cohomology x) :=
    derived_tensor_retract_crystalline_cohomology x
  constructor
  · exact hLoc
  · exact hDer
