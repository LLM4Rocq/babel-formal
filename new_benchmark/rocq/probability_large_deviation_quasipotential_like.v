(*
BENCHMARK_ID: TINY_MATHLIB_BATCH07_PROBABILITY_LARGE_DEVIATION_QUASIPOTENTIAL_LIKE
PAIR_STEM: probability_large_deviation_quasipotential_like
MATH_DOMAIN: Probability
SOURCE_MATHLIB: Mathlib/Probability/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class FrameworkStruct_probability_large_deviation_quasipotential := {
  rate : nat -> nat;
  quasi : nat -> nat;
  jump : nat -> nat;
  rate_jump : forall n : nat, rate (jump n) = rate n;
  quasi_rate : forall n : nat, quasi n = rate n;
  jump_idem : forall n : nat, jump (jump n) = jump n;
  jump_zero : jump 0 = 0
}.

Record ContextData_probability_large_deviation_quasipotential
    `{FrameworkStruct_probability_large_deviation_quasipotential} := {
  a : nat;
  b : nat;
  hab : jump a = b
}.

Definition primary_map_probability_large_deviation_quasipotential
    `{FrameworkStruct_probability_large_deviation_quasipotential}
    (ctx : ContextData_probability_large_deviation_quasipotential) : nat :=
  rate (jump (a ctx)).

Definition secondary_map_probability_large_deviation_quasipotential
    `{FrameworkStruct_probability_large_deviation_quasipotential}
    (ctx : ContextData_probability_large_deviation_quasipotential) : nat :=
  quasi (b ctx).

Definition tertiary_map_probability_large_deviation_quasipotential
    `{FrameworkStruct_probability_large_deviation_quasipotential}
    (ctx : ContextData_probability_large_deviation_quasipotential) : nat :=
  rate (jump (b ctx)).

Lemma stability_step_probability_large_deviation_quasipotential
    `{FrameworkStruct_probability_large_deviation_quasipotential}
    (ctx : ContextData_probability_large_deviation_quasipotential)
    (hneq : primary_map_probability_large_deviation_quasipotential ctx <>
      secondary_map_probability_large_deviation_quasipotential ctx)
    (hextra : a ctx = a ctx) :
    False.
Proof.
  assert (hkeep : a ctx = a ctx).
  { exact hextra. }
  assert (h1 :
      primary_map_probability_large_deviation_quasipotential ctx =
      rate (b ctx)).
  {
    unfold primary_map_probability_large_deviation_quasipotential.
    rewrite (hab ctx).
    reflexivity.
  }
  assert (h2 :
      secondary_map_probability_large_deviation_quasipotential ctx =
      rate (b ctx)).
  {
    unfold secondary_map_probability_large_deviation_quasipotential.
    rewrite quasi_rate.
    reflexivity.
  }
  assert (hEq :
      primary_map_probability_large_deviation_quasipotential ctx =
      secondary_map_probability_large_deviation_quasipotential ctx).
  {
    rewrite h1.
    symmetry.
    exact h2.
  }
  exact (hneq hEq).
Qed.

Lemma factorization_step_probability_large_deviation_quasipotential
    `{FrameworkStruct_probability_large_deviation_quasipotential}
    (ctx : ContextData_probability_large_deviation_quasipotential) :
    tertiary_map_probability_large_deviation_quasipotential ctx =
      rate (b ctx).
Proof.
  unfold tertiary_map_probability_large_deviation_quasipotential.
  exact (rate_jump _).
Qed.

Lemma comparison_step_probability_large_deviation_quasipotential
    `{FrameworkStruct_probability_large_deviation_quasipotential}
    (ctx : ContextData_probability_large_deviation_quasipotential) :
    (exists k : nat,
      rate k =
      secondary_map_probability_large_deviation_quasipotential ctx) /\
    primary_map_probability_large_deviation_quasipotential ctx =
      secondary_map_probability_large_deviation_quasipotential ctx.
Proof.
  assert (hEq :
      primary_map_probability_large_deviation_quasipotential ctx =
      secondary_map_probability_large_deviation_quasipotential ctx).
  {
    assert (h1 :
        primary_map_probability_large_deviation_quasipotential ctx =
        rate (b ctx)).
    {
      unfold primary_map_probability_large_deviation_quasipotential.
      rewrite (hab ctx).
      reflexivity.
    }
    assert (h2 :
        secondary_map_probability_large_deviation_quasipotential ctx =
        rate (b ctx)).
    {
      unfold secondary_map_probability_large_deviation_quasipotential.
      rewrite quasi_rate.
      reflexivity.
    }
    rewrite h1.
    symmetry.
    exact h2.
  }
  split.
  - exists (b ctx).
    unfold secondary_map_probability_large_deviation_quasipotential.
    rewrite quasi_rate.
    reflexivity.
  - exact hEq.
Qed.

Lemma transport_step_probability_large_deviation_quasipotential
    `{FrameworkStruct_probability_large_deviation_quasipotential}
    (ctx : ContextData_probability_large_deviation_quasipotential)
    (htransport : forall k : nat,
      rate k =
        secondary_map_probability_large_deviation_quasipotential ctx ->
      quasi k =
        secondary_map_probability_large_deviation_quasipotential ctx) :
    quasi (jump (a ctx)) =
      secondary_map_probability_large_deviation_quasipotential ctx.
Proof.
  assert (hk :
      rate (jump (a ctx)) =
      secondary_map_probability_large_deviation_quasipotential ctx).
  {
    unfold secondary_map_probability_large_deviation_quasipotential.
    rewrite quasi_rate.
    rewrite <- (hab ctx).
    reflexivity.
  }
  exact (htransport _ hk).
Qed.

Lemma coherence_step_probability_large_deviation_quasipotential
    `{FrameworkStruct_probability_large_deviation_quasipotential}
    (ctx : ContextData_probability_large_deviation_quasipotential)
    (hneg : (forall k : nat,
      quasi k <>
      secondary_map_probability_large_deviation_quasipotential ctx) -> False) :
    exists k : nat,
      quasi k =
      secondary_map_probability_large_deviation_quasipotential ctx /\ k = b ctx.
Proof.
  exists (b ctx).
  split.
  - unfold secondary_map_probability_large_deviation_quasipotential.
    reflexivity.
  - reflexivity.
Qed.

Lemma iteration_step_probability_large_deviation_quasipotential
    `{FrameworkStruct_probability_large_deviation_quasipotential}
    (ctx : ContextData_probability_large_deviation_quasipotential) :
    exists k : nat,
      jump k = k /\
      rate (jump k) = rate k.
Proof.
  exists (jump (a ctx)).
  split.
  - exact (jump_idem _).
  - exact (rate_jump _).
Qed.

Lemma main_result_probability_large_deviation_quasipotential
    `{FrameworkStruct_probability_large_deviation_quasipotential}
    (ctx : ContextData_probability_large_deviation_quasipotential) :
    primary_map_probability_large_deviation_quasipotential ctx =
      secondary_map_probability_large_deviation_quasipotential ctx /\
    exists k : nat,
      tertiary_map_probability_large_deviation_quasipotential ctx =
        rate (jump k) /\
      rate k =
        secondary_map_probability_large_deviation_quasipotential ctx /\
      k = b ctx.
Proof.
  assert (hEq :
      primary_map_probability_large_deviation_quasipotential ctx =
      secondary_map_probability_large_deviation_quasipotential ctx).
  { exact (proj2 (comparison_step_probability_large_deviation_quasipotential ctx)). }
  split.
  - exact hEq.
  - exists (b ctx).
    split.
    + unfold tertiary_map_probability_large_deviation_quasipotential.
      reflexivity.
    + split.
      * unfold secondary_map_probability_large_deviation_quasipotential.
        rewrite quasi_rate.
        reflexivity.
      * reflexivity.
Qed.
