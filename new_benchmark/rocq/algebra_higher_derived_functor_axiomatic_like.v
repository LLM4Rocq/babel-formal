(*
BENCHMARK_ID: TINY_MATHLIB_BATCH05_ALGEBRA_HIGHER_DERIVED_FUNCTOR_AXIOMATIC_LIKE
PAIR_STEM: algebra_higher_derived_functor_axiomatic_like
MATH_DOMAIN: Algebra
SOURCE_MATHLIB: Mathlib/Algebra/Homology
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class AlgebraStruct_higher_derived_functor (A : Type) := {
  derived : A -> A;
  tensor : A -> A -> A;
  localize : A -> A;
  cohomology : A -> A;
  localize_idem_axiom : forall X : A, localize (localize X) = localize X;
  derived_functor_exact_axiom : forall X Y : A,
    derived (tensor X Y) = derived (tensor Y X);
  bar_cobar_unit_axiom : forall X : A,
    tensor (derived X) (localize X) = localize (tensor X X);
  bar_cobar_counit_axiom : forall X : A,
    tensor (localize X) (derived X) = localize (tensor X X);
  localization_universal_axiom : forall X : A,
    localize (derived X) = derived (localize X);
  crystalline_descent_axiom : forall X : A,
    cohomology (localize X) = cohomology X;
  tilt_equivalence_axiom : forall X : A,
    derived (cohomology X) = cohomology (derived X);
  algebraic_reconstruction_axiom : forall X : A,
    tensor (cohomology X) (localize X) = tensor (localize X) (cohomology X)
}.

Definition DerivedObj_higher_derived_functor {A : Type}
    (S : AlgebraStruct_higher_derived_functor A) (X : A) : A :=
  derived X.

Definition TensorObj_higher_derived_functor {A : Type}
    (S : AlgebraStruct_higher_derived_functor A) (X Y : A) : A :=
  tensor X Y.

Definition LocalizationObj_higher_derived_functor {A : Type}
    (S : AlgebraStruct_higher_derived_functor A) (X : A) : A :=
  localize X.

Definition CohomologyObj_higher_derived_functor {A : Type}
    (S : AlgebraStruct_higher_derived_functor A) (X : A) : A :=
  cohomology X.

Lemma derived_functor_exact_higher_derived_functor {A : Type}
    (S : AlgebraStruct_higher_derived_functor A)
    (X Y : A) :
    DerivedObj_higher_derived_functor S (TensorObj_higher_derived_functor S X Y)
      = DerivedObj_higher_derived_functor S (TensorObj_higher_derived_functor S Y X).
Proof.
  assert (hExact : derived (tensor X Y) = derived (tensor Y X)).
  { apply derived_functor_exact_axiom. }
  assert (hLeft : DerivedObj_higher_derived_functor S (TensorObj_higher_derived_functor S X Y)
      = derived (tensor X Y)).
  { reflexivity. }
  assert (hRight : DerivedObj_higher_derived_functor S (TensorObj_higher_derived_functor S Y X)
      = derived (tensor Y X)).
  { reflexivity. }
  rewrite hLeft.
  rewrite hExact.
  rewrite hRight.
  reflexivity.
Qed.

Lemma bar_cobar_unit_higher_derived_functor {A : Type}
    (S : AlgebraStruct_higher_derived_functor A)
    (X : A) :
    TensorObj_higher_derived_functor S (DerivedObj_higher_derived_functor S X)
      (LocalizationObj_higher_derived_functor S X)
      = LocalizationObj_higher_derived_functor S (TensorObj_higher_derived_functor S X X).
Proof.
  assert (hUnit : tensor (derived X) (localize X) = localize (tensor X X)).
  { apply bar_cobar_unit_axiom. }
  change (tensor (derived X) (localize X) = localize (tensor X X)).
  exact hUnit.
Qed.

Lemma bar_cobar_counit_higher_derived_functor {A : Type}
    (S : AlgebraStruct_higher_derived_functor A)
    (X : A) :
    TensorObj_higher_derived_functor S (LocalizationObj_higher_derived_functor S X)
      (DerivedObj_higher_derived_functor S X)
      = LocalizationObj_higher_derived_functor S (TensorObj_higher_derived_functor S X X).
Proof.
  assert (hCounit : tensor (localize X) (derived X) = localize (tensor X X)).
  { apply bar_cobar_counit_axiom. }
  change (tensor (localize X) (derived X) = localize (tensor X X)).
  exact hCounit.
Qed.

Lemma localization_universal_higher_derived_functor {A : Type}
    (S : AlgebraStruct_higher_derived_functor A)
    (X : A) :
    LocalizationObj_higher_derived_functor S (DerivedObj_higher_derived_functor S X)
      = DerivedObj_higher_derived_functor S (LocalizationObj_higher_derived_functor S X).
Proof.
  assert (hLoc : localize (derived X) = derived (localize X)).
  { apply localization_universal_axiom. }
  change (localize (derived X) = derived (localize X)).
  exact hLoc.
Qed.

Lemma crystalline_descent_higher_derived_functor {A : Type}
    (S : AlgebraStruct_higher_derived_functor A)
    (X : A) :
    CohomologyObj_higher_derived_functor S
      (LocalizationObj_higher_derived_functor S (DerivedObj_higher_derived_functor S X))
      = CohomologyObj_higher_derived_functor S (DerivedObj_higher_derived_functor S X).
Proof.
  assert (hStep : cohomology (localize (derived X)) = cohomology (derived X)).
  { apply crystalline_descent_axiom. }
  assert (hExpandLeft :
      CohomologyObj_higher_derived_functor S
        (LocalizationObj_higher_derived_functor S (DerivedObj_higher_derived_functor S X))
      = cohomology (localize (derived X))).
  { reflexivity. }
  assert (hExpandRight :
      CohomologyObj_higher_derived_functor S (DerivedObj_higher_derived_functor S X)
      = cohomology (derived X)).
  { reflexivity. }
  rewrite hExpandLeft.
  rewrite hStep.
  rewrite hExpandRight.
  reflexivity.
Qed.

Lemma tilt_equivalence_core_higher_derived_functor {A : Type}
    (S : AlgebraStruct_higher_derived_functor A)
    (X : A) :
    DerivedObj_higher_derived_functor S (CohomologyObj_higher_derived_functor S X)
      = CohomologyObj_higher_derived_functor S (DerivedObj_higher_derived_functor S X).
Proof.
  assert (hTilt : derived (cohomology X) = cohomology (derived X)).
  { apply tilt_equivalence_axiom. }
  change (derived (cohomology X) = cohomology (derived X)).
  exact hTilt.
Qed.

Lemma algebraic_reconstruction_higher_derived_functor {A : Type}
    (S : AlgebraStruct_higher_derived_functor A)
    (X : A) :
    TensorObj_higher_derived_functor S
      (CohomologyObj_higher_derived_functor S (LocalizationObj_higher_derived_functor S X))
      (LocalizationObj_higher_derived_functor S X)
      =
    TensorObj_higher_derived_functor S
      (LocalizationObj_higher_derived_functor S X)
      (CohomologyObj_higher_derived_functor S (LocalizationObj_higher_derived_functor S X)).
Proof.
  assert (hRecon :
      tensor (cohomology (localize X)) (localize (localize X))
        = tensor (localize (localize X)) (cohomology (localize X))).
  { apply algebraic_reconstruction_axiom. }
  assert (hIdemLeft : localize (localize X) = localize X).
  { apply localize_idem_axiom. }
  assert (hIdemRight : localize (localize X) = localize X).
  { apply localize_idem_axiom. }
  repeat rewrite hIdemLeft in hRecon.
  repeat rewrite hIdemRight in hRecon.
  change (tensor (cohomology (localize X)) (localize X)
      = tensor (localize X) (cohomology (localize X))).
  exact hRecon.
Qed.
