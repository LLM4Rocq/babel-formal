(*
BENCHMARK_ID: TINY_MATHLIB_BATCH07_PROBABILITY_SKOROKHOD_REPRESENTATION_REFINED_LIKE
PAIR_STEM: probability_skorokhod_representation_refined_like
MATH_DOMAIN: Probability
SOURCE_MATHLIB: Mathlib/Probability/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class FrameworkStruct_probability_skorokhod_representation_refined := {
  repr : nat -> nat;
  lift : nat -> nat;
  good : nat -> Prop;
  good_lift : forall n : nat, good (lift n);
  repr_lift : forall n : nat, repr (lift n) = repr n;
  lift_repr : forall n : nat, lift (repr n) = lift n;
  repr_idem : forall n : nat, repr (repr n) = repr n
}.

Record ContextData_probability_skorokhod_representation_refined
    `{FrameworkStruct_probability_skorokhod_representation_refined} := {
  n : nat;
  m : nat;
  hnm : repr n = repr m
}.

Definition primary_map_probability_skorokhod_representation_refined
    `{FrameworkStruct_probability_skorokhod_representation_refined}
    (ctx : ContextData_probability_skorokhod_representation_refined) : nat :=
  repr (lift (n ctx)).

Definition secondary_map_probability_skorokhod_representation_refined
    `{FrameworkStruct_probability_skorokhod_representation_refined}
    (ctx : ContextData_probability_skorokhod_representation_refined) : nat :=
  repr (m ctx).

Definition tertiary_map_probability_skorokhod_representation_refined
    `{FrameworkStruct_probability_skorokhod_representation_refined}
    (ctx : ContextData_probability_skorokhod_representation_refined) : Prop :=
  good (lift (repr (m ctx))).

Lemma stability_step_probability_skorokhod_representation_refined
    `{FrameworkStruct_probability_skorokhod_representation_refined}
    (ctx : ContextData_probability_skorokhod_representation_refined)
    (hneq : secondary_map_probability_skorokhod_representation_refined ctx <>
      primary_map_probability_skorokhod_representation_refined ctx -> False) :
    primary_map_probability_skorokhod_representation_refined ctx =
      secondary_map_probability_skorokhod_representation_refined ctx.
Proof.
  assert (hleft :
      primary_map_probability_skorokhod_representation_refined ctx =
      repr (n ctx)).
  {
    unfold primary_map_probability_skorokhod_representation_refined.
    exact (repr_lift _).
  }
  assert (hright :
      repr (n ctx) =
      secondary_map_probability_skorokhod_representation_refined ctx).
  {
    unfold secondary_map_probability_skorokhod_representation_refined.
    exact (hnm ctx).
  }
  rewrite hleft.
  exact hright.
Qed.

Lemma factorization_step_probability_skorokhod_representation_refined
    `{FrameworkStruct_probability_skorokhod_representation_refined}
    (ctx : ContextData_probability_skorokhod_representation_refined) :
    tertiary_map_probability_skorokhod_representation_refined ctx.
Proof.
  unfold tertiary_map_probability_skorokhod_representation_refined.
  exact (good_lift _).
Qed.

Lemma comparison_step_probability_skorokhod_representation_refined
    `{FrameworkStruct_probability_skorokhod_representation_refined}
    (ctx : ContextData_probability_skorokhod_representation_refined) :
    primary_map_probability_skorokhod_representation_refined ctx =
      secondary_map_probability_skorokhod_representation_refined ctx /\
      tertiary_map_probability_skorokhod_representation_refined ctx /\
      exists k : nat,
        repr k =
        secondary_map_probability_skorokhod_representation_refined ctx.
Proof.
  assert (hEq :
      primary_map_probability_skorokhod_representation_refined ctx =
      secondary_map_probability_skorokhod_representation_refined ctx).
  {
    assert (hleft :
        primary_map_probability_skorokhod_representation_refined ctx =
        repr (n ctx)).
    {
      unfold primary_map_probability_skorokhod_representation_refined.
      exact (repr_lift _).
    }
    assert (hright :
        repr (n ctx) =
        secondary_map_probability_skorokhod_representation_refined ctx).
    {
      unfold secondary_map_probability_skorokhod_representation_refined.
      exact (hnm ctx).
    }
    rewrite hleft.
    exact hright.
  }
  assert (hTer : tertiary_map_probability_skorokhod_representation_refined ctx).
  { apply factorization_step_probability_skorokhod_representation_refined. }
  split.
  - exact hEq.
  - split.
    + exact hTer.
    + exists (m ctx).
      unfold secondary_map_probability_skorokhod_representation_refined.
      reflexivity.
Qed.

Lemma transport_step_probability_skorokhod_representation_refined
    `{FrameworkStruct_probability_skorokhod_representation_refined}
    (ctx : ContextData_probability_skorokhod_representation_refined)
    (htr : forall k : nat,
      repr k =
        secondary_map_probability_skorokhod_representation_refined ctx ->
      good (lift k)) :
    tertiary_map_probability_skorokhod_representation_refined ctx.
Proof.
  unfold tertiary_map_probability_skorokhod_representation_refined.
  assert (hk : repr (m ctx) = secondary_map_probability_skorokhod_representation_refined ctx).
  {
    unfold secondary_map_probability_skorokhod_representation_refined.
    reflexivity.
  }
  assert (hgood : good (lift (m ctx))).
  { exact (htr _ hk). }
  rewrite (lift_repr (m ctx)).
  exact hgood.
Qed.

Lemma coherence_step_probability_skorokhod_representation_refined
    `{FrameworkStruct_probability_skorokhod_representation_refined}
    (ctx : ContextData_probability_skorokhod_representation_refined)
    (hbad : (tertiary_map_probability_skorokhod_representation_refined ctx -> False) -> False) :
    tertiary_map_probability_skorokhod_representation_refined ctx.
Proof.
  assert (hter : tertiary_map_probability_skorokhod_representation_refined ctx).
  { apply factorization_step_probability_skorokhod_representation_refined. }
  exact hter.
Qed.

Lemma iteration_step_probability_skorokhod_representation_refined
    `{FrameworkStruct_probability_skorokhod_representation_refined}
    (ctx : ContextData_probability_skorokhod_representation_refined) :
    exists k : nat,
      repr k =
        primary_map_probability_skorokhod_representation_refined ctx /\
      good (lift k).
Proof.
  exists (lift (n ctx)).
  split.
  - unfold primary_map_probability_skorokhod_representation_refined.
    reflexivity.
  - exact (good_lift _).
Qed.

Lemma main_result_probability_skorokhod_representation_refined
    `{FrameworkStruct_probability_skorokhod_representation_refined}
    (ctx : ContextData_probability_skorokhod_representation_refined) :
    primary_map_probability_skorokhod_representation_refined ctx =
      secondary_map_probability_skorokhod_representation_refined ctx /\
    exists k : nat,
      repr k =
        primary_map_probability_skorokhod_representation_refined ctx /\
      (tertiary_map_probability_skorokhod_representation_refined ctx /\
        good (lift k)).
Proof.
  assert (hEq :
      primary_map_probability_skorokhod_representation_refined ctx =
      secondary_map_probability_skorokhod_representation_refined ctx).
  { exact (proj1 (comparison_step_probability_skorokhod_representation_refined ctx)). }
  split.
  - exact hEq.
  - exists (lift (n ctx)).
    split.
    + unfold primary_map_probability_skorokhod_representation_refined.
      reflexivity.
    + split.
      * exact (factorization_step_probability_skorokhod_representation_refined ctx).
      * exact (good_lift _).
Qed.
