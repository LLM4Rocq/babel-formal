(*
BENCHMARK_ID: TINY_MATHLIB_BATCH07_CATEGORY_BICATEGORY_PSEUDOFUNCTOR_COHERENCE_LIKE
PAIR_STEM: category_bicategory_pseudofunctor_coherence_like
MATH_DOMAIN: Category Theory
SOURCE_MATHLIB: Mathlib/CategoryTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Module CategoryBicategoryPseudofunctorCoherenceLike.

Class FrameworkStruct_category_bicategory_pseudofunctor_coherence (A : Type) := {
  comp : A -> A -> A;
  leftWhisker : A -> A;
  rightWhisker : A -> A;
  comp_assoc : forall a b c : A, comp (comp a b) c = comp a (comp b c);
  leftWhisker_comp : forall a b : A,
    leftWhisker (comp a b) = comp (leftWhisker a) (leftWhisker b);
  rightWhisker_comp : forall a b : A,
    rightWhisker (comp a b) = comp (rightWhisker a) (rightWhisker b);
  whisker_interchange : forall a : A,
    leftWhisker (rightWhisker a) = rightWhisker (leftWhisker a);
  leftWhisker_idem : forall a : A, leftWhisker (leftWhisker a) = leftWhisker a;
  rightWhisker_idem : forall a : A, rightWhisker (rightWhisker a) = rightWhisker a
}.

Record ContextData_category_bicategory_pseudofunctor_coherence
    (A : Type) := {
  eta_obj : A;
  theta_obj : A;
  iota_obj : A;
  kappa_obj : A
}.

Definition primary_map_category_bicategory_pseudofunctor_coherence
    {A : Type} `{FrameworkStruct_category_bicategory_pseudofunctor_coherence A}
    (d : ContextData_category_bicategory_pseudofunctor_coherence A) : A :=
  comp (leftWhisker (eta_obj d)) (rightWhisker (theta_obj d)).

Definition secondary_map_category_bicategory_pseudofunctor_coherence
    {A : Type} `{FrameworkStruct_category_bicategory_pseudofunctor_coherence A}
    (d : ContextData_category_bicategory_pseudofunctor_coherence A) : A :=
  comp (rightWhisker (theta_obj d)) (leftWhisker (iota_obj d)).

Definition tertiary_map_category_bicategory_pseudofunctor_coherence
    {A : Type} `{FrameworkStruct_category_bicategory_pseudofunctor_coherence A}
    (d : ContextData_category_bicategory_pseudofunctor_coherence A) : A :=
  comp
    (comp
      (primary_map_category_bicategory_pseudofunctor_coherence d)
      (secondary_map_category_bicategory_pseudofunctor_coherence d))
    (leftWhisker (kappa_obj d)).

Lemma stability_step_category_bicategory_pseudofunctor_coherence
    {A : Type} `{FrameworkStruct_category_bicategory_pseudofunctor_coherence A}
    (d : ContextData_category_bicategory_pseudofunctor_coherence A) :
    leftWhisker (primary_map_category_bicategory_pseudofunctor_coherence d) =
      comp (leftWhisker (leftWhisker (eta_obj d)))
        (leftWhisker (rightWhisker (theta_obj d))).
Proof.
  unfold primary_map_category_bicategory_pseudofunctor_coherence.
  transitivity (leftWhisker (comp (leftWhisker (eta_obj d)) (rightWhisker (theta_obj d)))).
  - reflexivity.
  - apply leftWhisker_comp.
Qed.

Lemma factorization_step_category_bicategory_pseudofunctor_coherence
    {A : Type} `{FrameworkStruct_category_bicategory_pseudofunctor_coherence A}
    (d : ContextData_category_bicategory_pseudofunctor_coherence A) :
    rightWhisker (secondary_map_category_bicategory_pseudofunctor_coherence d) =
      comp (rightWhisker (rightWhisker (theta_obj d)))
        (rightWhisker (leftWhisker (iota_obj d))).
Proof.
  unfold secondary_map_category_bicategory_pseudofunctor_coherence.
  transitivity (rightWhisker (comp (rightWhisker (theta_obj d)) (leftWhisker (iota_obj d)))).
  - reflexivity.
  - apply rightWhisker_comp.
Qed.

Lemma comparison_step_category_bicategory_pseudofunctor_coherence
    {A : Type} `{FrameworkStruct_category_bicategory_pseudofunctor_coherence A}
    (d : ContextData_category_bicategory_pseudofunctor_coherence A) :
    comp
        (leftWhisker (primary_map_category_bicategory_pseudofunctor_coherence d))
        (rightWhisker (secondary_map_category_bicategory_pseudofunctor_coherence d))
      =
    comp
      (comp (leftWhisker (leftWhisker (eta_obj d)))
        (leftWhisker (rightWhisker (theta_obj d))))
      (comp (rightWhisker (rightWhisker (theta_obj d)))
        (rightWhisker (leftWhisker (iota_obj d)))).
Proof.
  assert (hleft := stability_step_category_bicategory_pseudofunctor_coherence d).
  assert (hright := factorization_step_category_bicategory_pseudofunctor_coherence d).
  rewrite hleft.
  rewrite hright.
  reflexivity.
Qed.

Lemma transport_step_category_bicategory_pseudofunctor_coherence
    {A : Type} `{FrameworkStruct_category_bicategory_pseudofunctor_coherence A}
    (d : ContextData_category_bicategory_pseudofunctor_coherence A)
    (heta : eta_obj d = kappa_obj d) (htheta : theta_obj d = iota_obj d) :
    leftWhisker (primary_map_category_bicategory_pseudofunctor_coherence d) =
      comp (leftWhisker (leftWhisker (kappa_obj d)))
        (leftWhisker (rightWhisker (iota_obj d))).
Proof.
  assert (hstable := stability_step_category_bicategory_pseudofunctor_coherence d).
  rewrite hstable.
  rewrite heta.
  rewrite htheta.
  reflexivity.
Qed.

Lemma coherence_step_category_bicategory_pseudofunctor_coherence
    {A : Type} `{FrameworkStruct_category_bicategory_pseudofunctor_coherence A}
    (d : ContextData_category_bicategory_pseudofunctor_coherence A) :
    comp
      (primary_map_category_bicategory_pseudofunctor_coherence d)
      (comp
        (secondary_map_category_bicategory_pseudofunctor_coherence d)
        (leftWhisker (kappa_obj d)))
      = tertiary_map_category_bicategory_pseudofunctor_coherence d.
Proof.
  unfold tertiary_map_category_bicategory_pseudofunctor_coherence.
  transitivity
    (comp
      (comp
        (primary_map_category_bicategory_pseudofunctor_coherence d)
        (secondary_map_category_bicategory_pseudofunctor_coherence d))
      (leftWhisker (kappa_obj d))).
  - symmetry. apply comp_assoc.
  - reflexivity.
Qed.

Lemma iteration_step_category_bicategory_pseudofunctor_coherence
    {A : Type} `{FrameworkStruct_category_bicategory_pseudofunctor_coherence A}
    (d : ContextData_category_bicategory_pseudofunctor_coherence A) :
    rightWhisker (leftWhisker
      (tertiary_map_category_bicategory_pseudofunctor_coherence d)) =
    leftWhisker (rightWhisker
      (tertiary_map_category_bicategory_pseudofunctor_coherence d)).
Proof.
  assert (hswap := whisker_interchange
    (tertiary_map_category_bicategory_pseudofunctor_coherence d)).
  symmetry.
  exact hswap.
Qed.

Lemma main_result_category_bicategory_pseudofunctor_coherence
    {A : Type} `{FrameworkStruct_category_bicategory_pseudofunctor_coherence A}
    (d : ContextData_category_bicategory_pseudofunctor_coherence A) :
    rightWhisker (leftWhisker
      (comp
        (primary_map_category_bicategory_pseudofunctor_coherence d)
        (secondary_map_category_bicategory_pseudofunctor_coherence d)))
    =
    leftWhisker (rightWhisker
      (comp
        (primary_map_category_bicategory_pseudofunctor_coherence d)
        (secondary_map_category_bicategory_pseudofunctor_coherence d))).
Proof.
  assert (hcore := whisker_interchange
    (comp
      (primary_map_category_bicategory_pseudofunctor_coherence d)
      (secondary_map_category_bicategory_pseudofunctor_coherence d))).
  symmetry.
  exact hcore.
Qed.

End CategoryBicategoryPseudofunctorCoherenceLike.
