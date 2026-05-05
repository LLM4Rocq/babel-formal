(*
BENCHMARK_ID: TINY_MATHLIB_BATCH05_ALGEBRA_INFINITY_ALGEBRA_BAR_COBAR_LIKE
PAIR_STEM: algebra_infinity_algebra_bar_cobar_like
MATH_DOMAIN: Algebra
SOURCE_MATHLIB: Mathlib/Algebra/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class AlgebraStruct_infinity_algebra_bar (A : Type) := {
  mul : A -> A -> A;
  one : A;
  d : A -> A;
  bar : A -> A;
  cobar : A -> A;
  loc : A -> A;
  coh : A -> A;
  d_mul_axiom : forall x y : A, d (mul x y) = mul (d x) y;
  bar_cobar_unit_axiom : forall x : A, cobar (bar x) = x;
  bar_cobar_counit_axiom : forall x : A, bar (cobar x) = x;
  loc_mul_axiom : forall x y : A, loc (mul x y) = mul (loc x) (loc y);
  coh_loc_comm_axiom : forall x : A, coh (loc x) = loc (coh x);
  coh_bar_comm_axiom : forall x : A, coh (bar x) = bar (coh x)
}.

Definition DerivedObj_infinity_algebra_bar {A : Type}
    {AA : AlgebraStruct_infinity_algebra_bar A} (x : A) : A :=
  d x.

Definition TensorObj_infinity_algebra_bar {A : Type}
    {AA : AlgebraStruct_infinity_algebra_bar A} (x y : A) : A :=
  mul x y.

Definition LocalizationObj_infinity_algebra_bar {A : Type}
    {AA : AlgebraStruct_infinity_algebra_bar A} (x : A) : A :=
  loc x.

Definition CohomologyObj_infinity_algebra_bar {A : Type}
    {AA : AlgebraStruct_infinity_algebra_bar A} (x : A) : A :=
  coh x.

Lemma derived_functor_exact_infinity_algebra_bar {A : Type}
    {AA : AlgebraStruct_infinity_algebra_bar A}
    (x y : A) :
    DerivedObj_infinity_algebra_bar (TensorObj_infinity_algebra_bar x y) =
      TensorObj_infinity_algebra_bar (DerivedObj_infinity_algebra_bar x) y.
Proof.
  assert (hRaw : d (mul x y) = mul (d x) y).
  { apply d_mul_axiom. }
  unfold DerivedObj_infinity_algebra_bar, TensorObj_infinity_algebra_bar.
  exact hRaw.
Qed.

Lemma bar_cobar_unit_infinity_algebra_bar {A : Type}
    {AA : AlgebraStruct_infinity_algebra_bar A}
    (x : A) :
    cobar (bar x) = x.
Proof.
  apply bar_cobar_unit_axiom.
Qed.

Lemma bar_cobar_counit_infinity_algebra_bar {A : Type}
    {AA : AlgebraStruct_infinity_algebra_bar A}
    (x : A) :
    bar (cobar x) = x.
Proof.
  apply bar_cobar_counit_axiom.
Qed.

Lemma localization_universal_infinity_algebra_bar {A : Type}
    {AA : AlgebraStruct_infinity_algebra_bar A}
    (x y : A) :
    LocalizationObj_infinity_algebra_bar (TensorObj_infinity_algebra_bar x y) =
      TensorObj_infinity_algebra_bar
        (LocalizationObj_infinity_algebra_bar x)
        (LocalizationObj_infinity_algebra_bar y).
Proof.
  assert (hRaw : loc (mul x y) = mul (loc x) (loc y)).
  { apply loc_mul_axiom. }
  unfold LocalizationObj_infinity_algebra_bar, TensorObj_infinity_algebra_bar.
  exact hRaw.
Qed.

Lemma crystalline_descent_infinity_algebra_bar {A : Type}
    {AA : AlgebraStruct_infinity_algebra_bar A}
    (x : A) :
    CohomologyObj_infinity_algebra_bar (LocalizationObj_infinity_algebra_bar x) =
      LocalizationObj_infinity_algebra_bar (CohomologyObj_infinity_algebra_bar x).
Proof.
  assert (hRaw : coh (loc x) = loc (coh x)).
  { apply coh_loc_comm_axiom. }
  unfold CohomologyObj_infinity_algebra_bar, LocalizationObj_infinity_algebra_bar.
  exact hRaw.
Qed.

Lemma tilt_equivalence_core_infinity_algebra_bar {A : Type}
    {AA : AlgebraStruct_infinity_algebra_bar A}
    (x : A) :
    CohomologyObj_infinity_algebra_bar
      (bar x) =
      bar (CohomologyObj_infinity_algebra_bar x).
Proof.
  assert (hRaw : coh (bar x) = bar (coh x)).
  { apply coh_bar_comm_axiom. }
  unfold CohomologyObj_infinity_algebra_bar.
  exact hRaw.
Qed.

Lemma algebraic_reconstruction_infinity_algebra_bar {A : Type}
    {AA : AlgebraStruct_infinity_algebra_bar A}
    (x : A) :
    cobar
      (bar
        (LocalizationObj_infinity_algebra_bar (CohomologyObj_infinity_algebra_bar x))) =
      LocalizationObj_infinity_algebra_bar (CohomologyObj_infinity_algebra_bar x).
Proof.
  assert (hUnit :
      cobar
        (bar
          (LocalizationObj_infinity_algebra_bar (CohomologyObj_infinity_algebra_bar x))) =
      LocalizationObj_infinity_algebra_bar (CohomologyObj_infinity_algebra_bar x)).
  {
    apply bar_cobar_unit_axiom.
  }
  assert (hShape :
      LocalizationObj_infinity_algebra_bar (CohomologyObj_infinity_algebra_bar x) =
      loc (coh x)).
  { reflexivity. }
  clear hShape.
  exact hUnit.
Qed.
