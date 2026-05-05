(*
BENCHMARK_ID: TINY_MATHLIB_BATCH07_PROBABILITY_MALLIAVIN_INTEGRATION_PARTS_ADVANCED_LIKE
PAIR_STEM: probability_malliavin_integration_parts_advanced_like
MATH_DOMAIN: Probability
SOURCE_MATHLIB: Mathlib/Probability/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class FrameworkStruct_probability_malliavin_integration_parts_advanced (Omega : Type) := {
  deriv : Omega -> nat;
  weight : Omega -> nat;
  pair : Omega -> Omega -> Omega;
  neutral : Omega;
  pair_neutral_left : forall x : Omega, pair neutral x = x;
  pair_neutral_right : forall x : Omega, pair x neutral = x;
  pair_score_swap : forall x y : Omega, deriv (pair x y) = deriv (pair y x);
  deriv_weight : forall x : Omega, deriv x = weight x;
  weight_pair : forall x y : Omega, weight (pair x y) = weight x + weight y
}.

Record ContextData_probability_malliavin_integration_parts_advanced
    (Omega : Type) `{FrameworkStruct_probability_malliavin_integration_parts_advanced Omega} := {
  x : Omega;
  y : Omega;
  hxy : pair x y = pair y x
}.

Definition primary_map_probability_malliavin_integration_parts_advanced
    {Omega : Type} `{FrameworkStruct_probability_malliavin_integration_parts_advanced Omega}
    (ctx : ContextData_probability_malliavin_integration_parts_advanced) : nat :=
  deriv (pair (x ctx) (y ctx)).

Definition secondary_map_probability_malliavin_integration_parts_advanced
    {Omega : Type} `{FrameworkStruct_probability_malliavin_integration_parts_advanced Omega}
    (ctx : ContextData_probability_malliavin_integration_parts_advanced) : nat :=
  deriv (pair (y ctx) (x ctx)).

Definition tertiary_map_probability_malliavin_integration_parts_advanced
    {Omega : Type} `{FrameworkStruct_probability_malliavin_integration_parts_advanced Omega}
    (ctx : ContextData_probability_malliavin_integration_parts_advanced) : nat :=
  weight (pair (pair (x ctx) (y ctx)) (pair (y ctx) (x ctx))).

Lemma stability_step_probability_malliavin_integration_parts_advanced
    {Omega : Type} `{FrameworkStruct_probability_malliavin_integration_parts_advanced Omega}
    (ctx : ContextData_probability_malliavin_integration_parts_advanced)
    (hneq : primary_map_probability_malliavin_integration_parts_advanced ctx <>
      secondary_map_probability_malliavin_integration_parts_advanced ctx) :
    False.
Proof.
  assert (hcomm : pair (x ctx) (y ctx) = pair (y ctx) (x ctx)).
  { exact (hxy ctx). }
  assert (hleft :
      primary_map_probability_malliavin_integration_parts_advanced ctx =
      deriv (pair (y ctx) (x ctx))).
  {
    unfold primary_map_probability_malliavin_integration_parts_advanced.
    rewrite hcomm.
    reflexivity.
  }
  assert (hright :
      deriv (pair (y ctx) (x ctx)) =
      secondary_map_probability_malliavin_integration_parts_advanced ctx).
  { reflexivity. }
  assert (hEq :
      primary_map_probability_malliavin_integration_parts_advanced ctx =
      secondary_map_probability_malliavin_integration_parts_advanced ctx).
  {
    rewrite hleft.
    exact hright.
  }
  exact (hneq hEq).
Qed.

Lemma factorization_step_probability_malliavin_integration_parts_advanced
    {Omega : Type} `{FrameworkStruct_probability_malliavin_integration_parts_advanced Omega}
    (ctx : ContextData_probability_malliavin_integration_parts_advanced) :
    tertiary_map_probability_malliavin_integration_parts_advanced ctx =
      primary_map_probability_malliavin_integration_parts_advanced ctx +
      secondary_map_probability_malliavin_integration_parts_advanced ctx.
Proof.
  assert (hw :
      weight (pair (pair (x ctx) (y ctx)) (pair (y ctx) (x ctx))) =
      weight (pair (x ctx) (y ctx)) + weight (pair (y ctx) (x ctx))).
  { exact (weight_pair _ _). }
  assert (hd1 :
      deriv (pair (x ctx) (y ctx)) =
      weight (pair (x ctx) (y ctx))).
  { exact (deriv_weight _). }
  assert (hd2 :
      deriv (pair (y ctx) (x ctx)) =
      weight (pair (y ctx) (x ctx))).
  { exact (deriv_weight _). }
  unfold tertiary_map_probability_malliavin_integration_parts_advanced.
  rewrite hw.
  rewrite <- hd1.
  rewrite <- hd2.
  reflexivity.
Qed.

Lemma comparison_step_probability_malliavin_integration_parts_advanced
    {Omega : Type} `{FrameworkStruct_probability_malliavin_integration_parts_advanced Omega}
    (ctx : ContextData_probability_malliavin_integration_parts_advanced) :
    primary_map_probability_malliavin_integration_parts_advanced ctx =
      secondary_map_probability_malliavin_integration_parts_advanced ctx /\
      exists z : Omega,
        deriv z =
        primary_map_probability_malliavin_integration_parts_advanced ctx.
Proof.
  split.
  - unfold primary_map_probability_malliavin_integration_parts_advanced.
    unfold secondary_map_probability_malliavin_integration_parts_advanced.
    exact (pair_score_swap (x ctx) (y ctx)).
  - exists (pair (x ctx) (y ctx)).
    reflexivity.
Qed.

Lemma transport_step_probability_malliavin_integration_parts_advanced
    {Omega : Type} `{FrameworkStruct_probability_malliavin_integration_parts_advanced Omega}
    (ctx : ContextData_probability_malliavin_integration_parts_advanced)
    (hLift : forall z : Omega,
      deriv z =
        primary_map_probability_malliavin_integration_parts_advanced ctx ->
      deriv z =
        secondary_map_probability_malliavin_integration_parts_advanced ctx) :
    exists z : Omega,
      deriv z =
      secondary_map_probability_malliavin_integration_parts_advanced ctx.
Proof.
  exists (pair (x ctx) (y ctx)).
  assert (hz :
      deriv (pair (x ctx) (y ctx)) =
      primary_map_probability_malliavin_integration_parts_advanced ctx).
  { reflexivity. }
  exact (hLift _ hz).
Qed.

Lemma coherence_step_probability_malliavin_integration_parts_advanced
    {Omega : Type} `{FrameworkStruct_probability_malliavin_integration_parts_advanced Omega}
    (ctx : ContextData_probability_malliavin_integration_parts_advanced)
    (hbad : (forall z : Omega,
      deriv z <>
        secondary_map_probability_malliavin_integration_parts_advanced ctx) -> False) :
    exists z : Omega,
      deriv z =
        secondary_map_probability_malliavin_integration_parts_advanced ctx.
Proof.
  assert (huse :
    (forall z : Omega,
      deriv z <> secondary_map_probability_malliavin_integration_parts_advanced ctx) -> False).
  { exact hbad. }
  assert (hfalse : False -> False).
  { intro hF. exact hF. }
  exists (pair (y ctx) (x ctx)).
  reflexivity.
Qed.

Lemma iteration_step_probability_malliavin_integration_parts_advanced
    {Omega : Type} `{FrameworkStruct_probability_malliavin_integration_parts_advanced Omega}
    (ctx : ContextData_probability_malliavin_integration_parts_advanced) :
    exists z : Omega,
      pair neutral z = z /\
      deriv z =
        primary_map_probability_malliavin_integration_parts_advanced ctx.
Proof.
  exists (pair (x ctx) (y ctx)).
  split.
  - exact (pair_neutral_left _).
  - reflexivity.
Qed.

Lemma main_result_probability_malliavin_integration_parts_advanced
    {Omega : Type} `{FrameworkStruct_probability_malliavin_integration_parts_advanced Omega}
    (ctx : ContextData_probability_malliavin_integration_parts_advanced) :
    primary_map_probability_malliavin_integration_parts_advanced ctx =
      secondary_map_probability_malliavin_integration_parts_advanced ctx /\
    exists z : Omega,
      deriv z =
        primary_map_probability_malliavin_integration_parts_advanced ctx /\
      tertiary_map_probability_malliavin_integration_parts_advanced ctx =
        deriv (pair z z).
Proof.
  split.
  - exact (proj1 (comparison_step_probability_malliavin_integration_parts_advanced ctx)).
  - exists (pair (x ctx) (y ctx)).
    split.
    + reflexivity.
    + assert (hpair :
        pair (pair (x ctx) (y ctx)) (pair (y ctx) (x ctx)) =
        pair (pair (x ctx) (y ctx)) (pair (x ctx) (y ctx))).
      { rewrite (hxy ctx). reflexivity. }
      assert (hrew :
        tertiary_map_probability_malliavin_integration_parts_advanced ctx =
        weight (pair (pair (x ctx) (y ctx)) (pair (x ctx) (y ctx)))).
      {
        unfold tertiary_map_probability_malliavin_integration_parts_advanced.
        rewrite hpair.
        reflexivity.
      }
      assert (hdw :
        deriv (pair (pair (x ctx) (y ctx)) (pair (x ctx) (y ctx))) =
        weight (pair (pair (x ctx) (y ctx)) (pair (x ctx) (y ctx)))).
      { exact (deriv_weight _). }
      rewrite hrew.
      symmetry.
      exact hdw.
Qed.
