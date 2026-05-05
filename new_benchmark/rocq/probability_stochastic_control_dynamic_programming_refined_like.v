(*
BENCHMARK_ID: TINY_MATHLIB_BATCH07_PROBABILITY_STOCHASTIC_CONTROL_DYNAMIC_PROGRAMMING_REFINED_LIKE
PAIR_STEM: probability_stochastic_control_dynamic_programming_refined_like
MATH_DOMAIN: Probability
SOURCE_MATHLIB: Mathlib/Probability/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class FrameworkStruct_probability_stochastic_control_dynamic_programming_refined (Omega : Type) := {
  cost : Omega -> nat;
  value : Omega -> nat;
  advance : Omega -> Omega;
  mix : Omega -> Omega -> Omega;
  origin : Omega;
  mix_origin_left : forall x : Omega, mix origin x = x;
  mix_origin_right : forall x : Omega, mix x origin = x;
  value_advance : forall x : Omega, value (advance x) = value x;
  cost_mix : forall x y : Omega, cost (mix x y) = cost x + cost y;
  bellman_like : forall x y : Omega, value (mix x y) = cost x + value y;
  value_mix_symm : forall x y : Omega, value (mix x y) = value (mix y x)
}.

Record ContextData_probability_stochastic_control_dynamic_programming_refined
    (Omega : Type) `{FrameworkStruct_probability_stochastic_control_dynamic_programming_refined Omega} := {
  state : Omega;
  policy : Omega;
  compat :
    advance (mix state policy) =
    mix (advance state) (advance policy)
}.

Definition primary_map_probability_stochastic_control_dynamic_programming_refined
    {Omega : Type} `{FrameworkStruct_probability_stochastic_control_dynamic_programming_refined Omega}
    (ctx : ContextData_probability_stochastic_control_dynamic_programming_refined) : nat :=
  value (mix (state ctx) (policy ctx)).

Definition secondary_map_probability_stochastic_control_dynamic_programming_refined
    {Omega : Type} `{FrameworkStruct_probability_stochastic_control_dynamic_programming_refined Omega}
    (ctx : ContextData_probability_stochastic_control_dynamic_programming_refined) : nat :=
  cost (mix (state ctx) (policy ctx)).

Definition tertiary_map_probability_stochastic_control_dynamic_programming_refined
    {Omega : Type} `{FrameworkStruct_probability_stochastic_control_dynamic_programming_refined Omega}
    (ctx : ContextData_probability_stochastic_control_dynamic_programming_refined) : nat :=
  value (advance (mix (state ctx) (advance (policy ctx)))).

Lemma stability_step_probability_stochastic_control_dynamic_programming_refined
    {Omega : Type} `{FrameworkStruct_probability_stochastic_control_dynamic_programming_refined Omega}
    (ctx : ContextData_probability_stochastic_control_dynamic_programming_refined)
    (hneq :
      primary_map_probability_stochastic_control_dynamic_programming_refined ctx <>
      value (mix (policy ctx) (state ctx))) :
    False.
Proof.
  assert (hsym :
      value (mix (state ctx) (policy ctx)) =
      value (mix (policy ctx) (state ctx))).
  { exact (value_mix_symm _ _). }
  assert (hEq :
      primary_map_probability_stochastic_control_dynamic_programming_refined ctx =
      value (mix (policy ctx) (state ctx))).
  {
    unfold primary_map_probability_stochastic_control_dynamic_programming_refined.
    exact hsym.
  }
  exact (hneq hEq).
Qed.

Lemma factorization_step_probability_stochastic_control_dynamic_programming_refined
    {Omega : Type} `{FrameworkStruct_probability_stochastic_control_dynamic_programming_refined Omega}
    (ctx : ContextData_probability_stochastic_control_dynamic_programming_refined) :
    primary_map_probability_stochastic_control_dynamic_programming_refined ctx =
      cost (state ctx) +
      value (policy ctx).
Proof.
  assert (hbell :
      value (mix (state ctx) (policy ctx)) =
      cost (state ctx) + value (policy ctx)).
  { exact (bellman_like _ _). }
  unfold primary_map_probability_stochastic_control_dynamic_programming_refined.
  rewrite hbell.
  reflexivity.
Qed.

Lemma comparison_step_probability_stochastic_control_dynamic_programming_refined
    {Omega : Type} `{FrameworkStruct_probability_stochastic_control_dynamic_programming_refined Omega}
    (ctx : ContextData_probability_stochastic_control_dynamic_programming_refined) :
    primary_map_probability_stochastic_control_dynamic_programming_refined ctx =
      value (mix (policy ctx) (state ctx)) /\
    exists z : Omega,
      value z =
        primary_map_probability_stochastic_control_dynamic_programming_refined ctx /\
      cost z =
        secondary_map_probability_stochastic_control_dynamic_programming_refined ctx.
Proof.
  split.
  - unfold primary_map_probability_stochastic_control_dynamic_programming_refined.
    exact (value_mix_symm _ _).
  - exists (mix (state ctx) (policy ctx)).
    split.
    + reflexivity.
    + unfold secondary_map_probability_stochastic_control_dynamic_programming_refined.
      reflexivity.
Qed.

Lemma transport_step_probability_stochastic_control_dynamic_programming_refined
    {Omega : Type} `{FrameworkStruct_probability_stochastic_control_dynamic_programming_refined Omega}
    (ctx : ContextData_probability_stochastic_control_dynamic_programming_refined)
    (htransport : forall z : Omega,
      value z =
        primary_map_probability_stochastic_control_dynamic_programming_refined ctx ->
      value (advance z) =
        tertiary_map_probability_stochastic_control_dynamic_programming_refined ctx) :
    exists z : Omega,
      value (advance z) =
        tertiary_map_probability_stochastic_control_dynamic_programming_refined ctx.
Proof.
  exists (mix (state ctx) (advance (policy ctx))).
  assert (hz :
      value (mix (state ctx) (advance (policy ctx))) =
      primary_map_probability_stochastic_control_dynamic_programming_refined ctx).
  {
    unfold primary_map_probability_stochastic_control_dynamic_programming_refined.
    assert (hv : value (advance (policy ctx)) = value (policy ctx)).
    { exact (value_advance _). }
    assert (hb1 :
      value (mix (state ctx) (advance (policy ctx))) =
      cost (state ctx) + value (advance (policy ctx))).
    { exact (bellman_like _ _). }
    assert (hb2 :
      value (mix (state ctx) (policy ctx)) =
      cost (state ctx) + value (policy ctx)).
    { exact (bellman_like _ _). }
    rewrite hb1.
    rewrite hv.
    rewrite hb2.
    reflexivity.
  }
  exact (htransport _ hz).
Qed.

Lemma coherence_step_probability_stochastic_control_dynamic_programming_refined
    {Omega : Type} `{FrameworkStruct_probability_stochastic_control_dynamic_programming_refined Omega}
    (ctx : ContextData_probability_stochastic_control_dynamic_programming_refined)
    (hmono : tertiary_map_probability_stochastic_control_dynamic_programming_refined ctx =
      value (mix (advance (state ctx)) (advance (policy ctx)))) :
    tertiary_map_probability_stochastic_control_dynamic_programming_refined ctx =
      primary_map_probability_stochastic_control_dynamic_programming_refined ctx.
Proof.
  unfold tertiary_map_probability_stochastic_control_dynamic_programming_refined in hmono.
  assert (hstep :
      value (advance (mix (state ctx) (advance (policy ctx)))) =
      value (mix (state ctx) (advance (policy ctx)))).
  { exact (value_advance _). }
  assert (hcompat :
      advance (mix (state ctx) (policy ctx)) =
      mix (advance (state ctx)) (advance (policy ctx))).
  { exact (compat ctx). }
  assert (hvalueCompat :
      value (advance (mix (state ctx) (policy ctx))) =
      value (mix (advance (state ctx)) (advance (policy ctx)))).
  { rewrite hcompat. reflexivity. }
  assert (hfix :
      value (mix (state ctx) (policy ctx)) =
      value (advance (mix (state ctx) (policy ctx)))).
  { symmetry. exact (value_advance _). }
  unfold primary_map_probability_stochastic_control_dynamic_programming_refined.
  transitivity (value (mix (advance (state ctx)) (advance (policy ctx)))).
  - exact hmono.
  - rewrite <- hvalueCompat.
    rewrite <- hfix.
    reflexivity.
Qed.

Lemma iteration_step_probability_stochastic_control_dynamic_programming_refined
    {Omega : Type} `{FrameworkStruct_probability_stochastic_control_dynamic_programming_refined Omega}
    (ctx : ContextData_probability_stochastic_control_dynamic_programming_refined) :
    exists z : Omega,
      value z =
        primary_map_probability_stochastic_control_dynamic_programming_refined ctx /\
      value (advance z) = value z /\
      mix z origin = z.
Proof.
  exists (mix (state ctx) (policy ctx)).
  split.
  - reflexivity.
  - split.
    + exact (value_advance _).
    + exact (mix_origin_right _).
Qed.

Lemma main_result_probability_stochastic_control_dynamic_programming_refined
    {Omega : Type} `{FrameworkStruct_probability_stochastic_control_dynamic_programming_refined Omega}
    (ctx : ContextData_probability_stochastic_control_dynamic_programming_refined) :
    primary_map_probability_stochastic_control_dynamic_programming_refined ctx =
      value (mix (policy ctx) (state ctx)) /\
    exists z : Omega,
      value z =
        primary_map_probability_stochastic_control_dynamic_programming_refined ctx /\
      tertiary_map_probability_stochastic_control_dynamic_programming_refined ctx =
        value (advance z).
Proof.
  split.
  - exact (proj1 (comparison_step_probability_stochastic_control_dynamic_programming_refined ctx)).
  - exists (mix (state ctx) (advance (policy ctx))).
    split.
    + unfold primary_map_probability_stochastic_control_dynamic_programming_refined.
      assert (hv : value (advance (policy ctx)) = value (policy ctx)).
      { exact (value_advance _). }
      assert (hb1 :
        value (mix (state ctx) (advance (policy ctx))) =
        cost (state ctx) + value (advance (policy ctx))).
      { exact (bellman_like _ _). }
      assert (hb2 :
        value (mix (state ctx) (policy ctx)) =
        cost (state ctx) + value (policy ctx)).
      { exact (bellman_like _ _). }
      rewrite hb1.
      rewrite hv.
      rewrite hb2.
      reflexivity.
    + unfold tertiary_map_probability_stochastic_control_dynamic_programming_refined.
      reflexivity.
Qed.
