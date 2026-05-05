(*
BENCHMARK_ID: TINY_MATHLIB_BATCH05_ALGEBRA_CRYSTALLINE_COHOMOLOGY_AXIOMATIC_LIKE
PAIR_STEM: algebra_crystalline_cohomology_axiomatic_like
MATH_DOMAIN: Algebra
SOURCE_MATHLIB: Mathlib/Algebra/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 17
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class AlgebraStruct_crystalline_cohomology (A : Type) := {
  rel : A -> A -> Prop;
  rel_refl : forall x : A, rel x x;
  rel_trans : forall {x y z : A}, rel x y -> rel y z -> rel x z;
  Derived : A -> A;
  Tensor : A -> A;
  Localization : A -> A;
  Cohomology : A -> A;
  derived_exact_axiom : forall x : A, rel (Derived (Tensor x)) (Tensor (Derived x));
  unit_axiom : forall x : A, rel x (Tensor x);
  counit_axiom : forall x : A, rel (Tensor x) x;
  localization_axiom : forall x : A,
      rel (Localization (Tensor x)) (Tensor (Localization x));
  descent_axiom : forall x : A,
      rel (Cohomology (Localization x)) (Localization (Cohomology x));
  tilt_axiom : forall x : A,
      rel (Derived (Cohomology x)) (Cohomology (Derived x))
}.

Definition DerivedObj_crystalline_cohomology {A : Type}
    `{AlgebraStruct_crystalline_cohomology A} (x : A) : A :=
  Derived x.

Definition TensorObj_crystalline_cohomology {A : Type}
    `{AlgebraStruct_crystalline_cohomology A} (x : A) : A :=
  Tensor x.

Definition LocalizationObj_crystalline_cohomology {A : Type}
    `{AlgebraStruct_crystalline_cohomology A} (x : A) : A :=
  Localization x.

Definition CohomologyObj_crystalline_cohomology {A : Type}
    `{AlgebraStruct_crystalline_cohomology A} (x : A) : A :=
  Cohomology x.

Lemma derived_functor_exact_crystalline_cohomology {A : Type}
    `{AlgebraStruct_crystalline_cohomology A}
    (x : A) :
    rel
      (DerivedObj_crystalline_cohomology (TensorObj_crystalline_cohomology x))
      (TensorObj_crystalline_cohomology (DerivedObj_crystalline_cohomology x)).
Proof.
  change (rel (Derived (Tensor x)) (Tensor (Derived x))).
  apply derived_exact_axiom.
Qed.

Lemma bar_cobar_unit_crystalline_cohomology {A : Type}
    `{AlgebraStruct_crystalline_cohomology A}
    (x : A) :
    exists y : A,
      y = TensorObj_crystalline_cohomology x /\
      rel x y.
Proof.
  assert (hUnit : rel x (Tensor x)).
  { apply unit_axiom. }
  exists (TensorObj_crystalline_cohomology x).
  split.
  - reflexivity.
  - exact hUnit.
Qed.

Lemma bar_cobar_counit_crystalline_cohomology {A : Type}
    `{AlgebraStruct_crystalline_cohomology A}
    (x : A) :
    rel (TensorObj_crystalline_cohomology x) x /\
    rel x x.
Proof.
  assert (hCounit : rel (Tensor x) x).
  { apply counit_axiom. }
  assert (hRefl : rel x x).
  { apply rel_refl. }
  split.
  - exact hCounit.
  - exact hRefl.
Qed.

Lemma localization_universal_crystalline_cohomology {A : Type}
    `{AlgebraStruct_crystalline_cohomology A}
    (x : A) :
    rel
      (LocalizationObj_crystalline_cohomology (TensorObj_crystalline_cohomology x))
      (TensorObj_crystalline_cohomology (LocalizationObj_crystalline_cohomology x)).
Proof.
  remember (Localization (Tensor x)) as lhs.
  remember (Tensor (Localization x)) as rhs.
  pose proof (localization_axiom x) as hLoc.
  subst lhs rhs.
  exact hLoc.
Qed.

Lemma crystalline_descent_crystalline_cohomology {A : Type}
    `{AlgebraStruct_crystalline_cohomology A}
    (x : A) :
    rel
      (CohomologyObj_crystalline_cohomology
        (LocalizationObj_crystalline_cohomology x))
      (LocalizationObj_crystalline_cohomology
        (CohomologyObj_crystalline_cohomology x)).
Proof.
  change (rel (Cohomology (Localization x)) (Localization (Cohomology x))).
  specialize (descent_axiom x) as hDesc.
  exact hDesc.
Qed.

Lemma tilt_equivalence_core_crystalline_cohomology {A : Type}
    `{AlgebraStruct_crystalline_cohomology A}
    (x : A) :
    rel
      (DerivedObj_crystalline_cohomology
        (CohomologyObj_crystalline_cohomology x))
      (CohomologyObj_crystalline_cohomology
        (DerivedObj_crystalline_cohomology x)).
Proof.
  eapply rel_trans.
  - apply rel_refl.
  - apply tilt_axiom.
Qed.

Lemma algebraic_reconstruction_crystalline_cohomology {A : Type}
    `{AlgebraStruct_crystalline_cohomology A}
    (x : A) :
    rel
      (LocalizationObj_crystalline_cohomology
        (CohomologyObj_crystalline_cohomology
          (DerivedObj_crystalline_cohomology x)))
      (TensorObj_crystalline_cohomology
        (LocalizationObj_crystalline_cohomology
          (CohomologyObj_crystalline_cohomology
            (DerivedObj_crystalline_cohomology x)))).
Proof.
  assert (h1 : rel
      (DerivedObj_crystalline_cohomology
        (CohomologyObj_crystalline_cohomology x))
      (CohomologyObj_crystalline_cohomology
        (DerivedObj_crystalline_cohomology x))).
  { apply tilt_equivalence_core_crystalline_cohomology. }
  assert (h2 : rel
      (LocalizationObj_crystalline_cohomology
        (CohomologyObj_crystalline_cohomology
          (DerivedObj_crystalline_cohomology x)))
      (TensorObj_crystalline_cohomology
        (LocalizationObj_crystalline_cohomology
          (CohomologyObj_crystalline_cohomology
            (DerivedObj_crystalline_cohomology x))))).
  {
    destruct (bar_cobar_unit_crystalline_cohomology
      (LocalizationObj_crystalline_cohomology
        (CohomologyObj_crystalline_cohomology
          (DerivedObj_crystalline_cohomology x))))
      as [y [hyEq hyRel]].
    rewrite hyEq in hyRel.
    exact hyRel.
  }
  assert (h3 : rel
      (LocalizationObj_crystalline_cohomology
        (CohomologyObj_crystalline_cohomology
          (DerivedObj_crystalline_cohomology x)))
      (LocalizationObj_crystalline_cohomology
        (CohomologyObj_crystalline_cohomology
          (DerivedObj_crystalline_cohomology x)))).
  { apply rel_refl. }
  assert (hKeep : rel
      (CohomologyObj_crystalline_cohomology
        (LocalizationObj_crystalline_cohomology
          (DerivedObj_crystalline_cohomology x)))
      (LocalizationObj_crystalline_cohomology
        (CohomologyObj_crystalline_cohomology
          (DerivedObj_crystalline_cohomology x)))).
  { apply crystalline_descent_crystalline_cohomology. }
  assert (hUse : rel
      (LocalizationObj_crystalline_cohomology
        (CohomologyObj_crystalline_cohomology
          (DerivedObj_crystalline_cohomology x)))
      (TensorObj_crystalline_cohomology
        (LocalizationObj_crystalline_cohomology
          (CohomologyObj_crystalline_cohomology
            (DerivedObj_crystalline_cohomology x))))).
  { eapply rel_trans. exact h3. exact h2. }
  exact hUse.
Qed.

Lemma descent_then_tensorize_crystalline_cohomology {A : Type}
    `{AlgebraStruct_crystalline_cohomology A}
    (x : A) :
    rel
      (CohomologyObj_crystalline_cohomology
        (LocalizationObj_crystalline_cohomology x))
      (TensorObj_crystalline_cohomology
        (LocalizationObj_crystalline_cohomology
          (CohomologyObj_crystalline_cohomology x))).
Proof.
  assert (hDesc : rel
      (CohomologyObj_crystalline_cohomology
        (LocalizationObj_crystalline_cohomology x))
      (LocalizationObj_crystalline_cohomology
        (CohomologyObj_crystalline_cohomology x))).
  { apply crystalline_descent_crystalline_cohomology. }
  assert (hUnit : rel
      (LocalizationObj_crystalline_cohomology
        (CohomologyObj_crystalline_cohomology x))
      (TensorObj_crystalline_cohomology
        (LocalizationObj_crystalline_cohomology
          (CohomologyObj_crystalline_cohomology x)))).
  {
    unfold TensorObj_crystalline_cohomology.
    apply unit_axiom.
  }
  eapply rel_trans.
  - exact hDesc.
  - exact hUnit.
Qed.

Lemma localization_tensor_retract_crystalline_cohomology {A : Type}
    `{AlgebraStruct_crystalline_cohomology A}
    (x : A) :
    rel
      (LocalizationObj_crystalline_cohomology
        (TensorObj_crystalline_cohomology x))
      (LocalizationObj_crystalline_cohomology x).
Proof.
  assert (hLoc : rel
      (LocalizationObj_crystalline_cohomology
        (TensorObj_crystalline_cohomology x))
      (TensorObj_crystalline_cohomology
        (LocalizationObj_crystalline_cohomology x))).
  { apply localization_universal_crystalline_cohomology. }
  pose proof (bar_cobar_counit_crystalline_cohomology
      (LocalizationObj_crystalline_cohomology x)) as hPair.
  destruct hPair as [hCounit hRefl].
  eapply rel_trans.
  - exact hLoc.
  - exact hCounit.
Qed.

Lemma derived_tensor_retract_crystalline_cohomology {A : Type}
    `{AlgebraStruct_crystalline_cohomology A}
    (x : A) :
    rel
      (DerivedObj_crystalline_cohomology
        (TensorObj_crystalline_cohomology x))
      (DerivedObj_crystalline_cohomology x).
Proof.
  assert (hExact : rel
      (DerivedObj_crystalline_cohomology
        (TensorObj_crystalline_cohomology x))
      (TensorObj_crystalline_cohomology
        (DerivedObj_crystalline_cohomology x))).
  { apply derived_functor_exact_crystalline_cohomology. }
  pose proof (bar_cobar_counit_crystalline_cohomology
      (DerivedObj_crystalline_cohomology x)) as hPair.
  destruct hPair as [hCounit hRefl].
  eapply rel_trans.
  - exact hExact.
  - exact hCounit.
Qed.

Lemma tilt_then_tensorize_crystalline_cohomology {A : Type}
    `{AlgebraStruct_crystalline_cohomology A}
    (x : A) :
    rel
      (DerivedObj_crystalline_cohomology
        (CohomologyObj_crystalline_cohomology x))
      (TensorObj_crystalline_cohomology
        (CohomologyObj_crystalline_cohomology
          (DerivedObj_crystalline_cohomology x))).
Proof.
  assert (hTilt : rel
      (DerivedObj_crystalline_cohomology
        (CohomologyObj_crystalline_cohomology x))
      (CohomologyObj_crystalline_cohomology
        (DerivedObj_crystalline_cohomology x))).
  { apply tilt_equivalence_core_crystalline_cohomology. }
  assert (hUnit : rel
      (CohomologyObj_crystalline_cohomology
        (DerivedObj_crystalline_cohomology x))
      (TensorObj_crystalline_cohomology
        (CohomologyObj_crystalline_cohomology
          (DerivedObj_crystalline_cohomology x)))).
  {
    unfold TensorObj_crystalline_cohomology.
    apply unit_axiom.
  }
  eapply rel_trans.
  - exact hTilt.
  - exact hUnit.
Qed.

Lemma tensor_retractions_pair_crystalline_cohomology {A : Type}
    `{AlgebraStruct_crystalline_cohomology A}
    (x : A) :
    rel
      (LocalizationObj_crystalline_cohomology
        (TensorObj_crystalline_cohomology x))
      (LocalizationObj_crystalline_cohomology x) /\
    rel
      (DerivedObj_crystalline_cohomology
        (TensorObj_crystalline_cohomology x))
      (DerivedObj_crystalline_cohomology x).
Proof.
  assert (hLoc : rel
      (LocalizationObj_crystalline_cohomology
        (TensorObj_crystalline_cohomology x))
      (LocalizationObj_crystalline_cohomology x)).
  { apply localization_tensor_retract_crystalline_cohomology. }
  assert (hDer : rel
      (DerivedObj_crystalline_cohomology
        (TensorObj_crystalline_cohomology x))
      (DerivedObj_crystalline_cohomology x)).
  { apply derived_tensor_retract_crystalline_cohomology. }
  split.
  - exact hLoc.
  - exact hDer.
Qed.
