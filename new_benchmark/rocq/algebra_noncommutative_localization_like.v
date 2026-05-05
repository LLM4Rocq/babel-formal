(*
BENCHMARK_ID: TINY_MATHLIB_BATCH05_ALGEBRA_NONCOMMUTATIVE_LOCALIZATION_LIKE
PAIR_STEM: algebra_noncommutative_localization_like
MATH_DOMAIN: Algebra
SOURCE_MATHLIB: Mathlib/Algebra/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class AlgebraStruct_noncommutative_localization (A : Type) := {
  rel : A -> A -> Prop;
  rel_refl : forall x : A, rel x x;
  rel_trans : forall {x y z : A}, rel x y -> rel y z -> rel x z;
  derived : A -> A;
  tensor : A -> A -> A;
  localize : A -> A;
  cohom : A -> A;
  exact_derived_axiom : forall {x y : A}, rel x y -> rel (derived x) (derived y);
  bar_unit_axiom : forall x : A, rel x (tensor x (derived x));
  bar_counit_axiom : forall x : A, rel (tensor (derived x) x) x;
  localization_axiom : forall x : A, rel (localize x) (localize (derived x));
  crystalline_axiom : forall x : A, rel (cohom (localize x)) (localize (cohom x));
  tilt_axiom : forall x : A, rel (derived (localize x)) (localize (derived x));
  reconstruction_axiom : forall x : A, rel (cohom (derived x)) (derived (cohom x))
}.

Arguments rel {A} {_} _ _.
Arguments derived {A} {_} _.
Arguments tensor {A} {_} _ _.
Arguments localize {A} {_} _.
Arguments cohom {A} {_} _.

Infix "≺" := rel (at level 70).

Definition DerivedObj_noncommutative_localization
    {A : Type} `{AlgebraStruct_noncommutative_localization A} (x : A) : A :=
  derived x.

Definition TensorObj_noncommutative_localization
    {A : Type} `{AlgebraStruct_noncommutative_localization A} (x y : A) : A :=
  tensor x y.

Definition LocalizationObj_noncommutative_localization
    {A : Type} `{AlgebraStruct_noncommutative_localization A} (x : A) : A :=
  localize x.

Definition CohomologyObj_noncommutative_localization
    {A : Type} `{AlgebraStruct_noncommutative_localization A} (x : A) : A :=
  cohom x.

Lemma derived_functor_exact_noncommutative_localization
    {A : Type} `{AlgebraStruct_noncommutative_localization A}
    {x y : A} (hxy : x ≺ y) :
    DerivedObj_noncommutative_localization x ≺
      DerivedObj_noncommutative_localization y.
Proof.
  assert (hraw : derived x ≺ derived y).
  { exact (exact_derived_axiom hxy). }
  unfold DerivedObj_noncommutative_localization.
  exact hraw.
Qed.

Lemma bar_cobar_unit_noncommutative_localization
    {A : Type} `{AlgebraStruct_noncommutative_localization A}
    (x : A) :
    x ≺ TensorObj_noncommutative_localization x
      (DerivedObj_noncommutative_localization x).
Proof.
  assert (hunit : x ≺ tensor x (derived x)).
  { exact (bar_unit_axiom x). }
  unfold TensorObj_noncommutative_localization, DerivedObj_noncommutative_localization.
  exact hunit.
Qed.

Lemma bar_cobar_counit_noncommutative_localization
    {A : Type} `{AlgebraStruct_noncommutative_localization A}
    (x : A) :
    TensorObj_noncommutative_localization
      (DerivedObj_noncommutative_localization x) x ≺ x.
Proof.
  assert (hcounit : tensor (derived x) x ≺ x).
  { exact (bar_counit_axiom x). }
  unfold TensorObj_noncommutative_localization, DerivedObj_noncommutative_localization.
  exact hcounit.
Qed.

Lemma localization_universal_noncommutative_localization
    {A : Type} `{AlgebraStruct_noncommutative_localization A}
    (x : A) :
    LocalizationObj_noncommutative_localization x ≺
      LocalizationObj_noncommutative_localization
        (DerivedObj_noncommutative_localization x).
Proof.
  assert (hloc : localize x ≺ localize (derived x)).
  { exact (localization_axiom x). }
  unfold LocalizationObj_noncommutative_localization, DerivedObj_noncommutative_localization.
  exact hloc.
Qed.

Lemma crystalline_descent_noncommutative_localization
    {A : Type} `{AlgebraStruct_noncommutative_localization A}
    (x : A) :
    CohomologyObj_noncommutative_localization
      (LocalizationObj_noncommutative_localization x) ≺
      LocalizationObj_noncommutative_localization
        (CohomologyObj_noncommutative_localization x).
Proof.
  assert (hcrys : cohom (localize x) ≺ localize (cohom x)).
  { exact (crystalline_axiom x). }
  unfold CohomologyObj_noncommutative_localization, LocalizationObj_noncommutative_localization.
  exact hcrys.
Qed.

Lemma tilt_equivalence_core_noncommutative_localization
    {A : Type} `{AlgebraStruct_noncommutative_localization A}
    (x : A) :
    DerivedObj_noncommutative_localization
      (LocalizationObj_noncommutative_localization x) ≺
      LocalizationObj_noncommutative_localization
        (DerivedObj_noncommutative_localization x).
Proof.
  assert (htilt : derived (localize x) ≺ localize (derived x)).
  { exact (tilt_axiom x). }
  unfold DerivedObj_noncommutative_localization, LocalizationObj_noncommutative_localization.
  exact htilt.
Qed.

Lemma algebraic_reconstruction_noncommutative_localization
    {A : Type} `{AlgebraStruct_noncommutative_localization A}
    (x : A) :
    CohomologyObj_noncommutative_localization
      (DerivedObj_noncommutative_localization x) ≺
      DerivedObj_noncommutative_localization
        (CohomologyObj_noncommutative_localization x).
Proof.
  assert (hrec : cohom (derived x) ≺ derived (cohom x)).
  { exact (reconstruction_axiom x). }
  unfold CohomologyObj_noncommutative_localization, DerivedObj_noncommutative_localization.
  exact hrec.
Qed.
