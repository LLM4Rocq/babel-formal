(*
BENCHMARK_ID: TINY_MATHLIB_BATCH07_PROBABILITY_RANDOM_MATRIX_LOCAL_LAW_LIKE
PAIR_STEM: probability_random_matrix_local_law_like
MATH_DOMAIN: Probability
SOURCE_MATHLIB: Mathlib/Probability/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class FrameworkStruct_probability_random_matrix_local_law := {
  energy : nat -> nat;
  drift : nat -> nat;
  step : nat -> nat;
  monotone_energy : forall n m : nat, n = m -> energy n = energy m;
  drift_step : forall n : nat, drift (step n) = drift n;
  energy_drift : forall n : nat, energy n = drift n;
  step_idem : forall n : nat, step (step n) = step n
}.

Record ContextData_probability_random_matrix_local_law
    `{FrameworkStruct_probability_random_matrix_local_law} := {
  n : nat;
  m : nat;
  hnm : n = m
}.

Definition primary_map_probability_random_matrix_local_law
    `{FrameworkStruct_probability_random_matrix_local_law}
    (ctx : ContextData_probability_random_matrix_local_law) : nat :=
  energy (step (n ctx)).

Definition secondary_map_probability_random_matrix_local_law
    `{FrameworkStruct_probability_random_matrix_local_law}
    (ctx : ContextData_probability_random_matrix_local_law) : nat :=
  drift (step (m ctx)).

Definition tertiary_map_probability_random_matrix_local_law
    `{FrameworkStruct_probability_random_matrix_local_law}
    (ctx : ContextData_probability_random_matrix_local_law) : nat :=
  energy (step (step (n ctx))).

Lemma stability_step_probability_random_matrix_local_law
    `{FrameworkStruct_probability_random_matrix_local_law}
    (ctx : ContextData_probability_random_matrix_local_law)
    (hneq : primary_map_probability_random_matrix_local_law ctx <>
      secondary_map_probability_random_matrix_local_law ctx) :
    False.
Proof.
  assert (hstepEq : step (n ctx) = step (m ctx)).
  { rewrite (hnm ctx). reflexivity. }
  assert (hEnergy :
      energy (step (n ctx)) = energy (step (m ctx))).
  { apply monotone_energy. exact hstepEq. }
  assert (hBridge :
      energy (step (m ctx)) = drift (step (m ctx))).
  { apply energy_drift. }
  assert (hEq :
      primary_map_probability_random_matrix_local_law ctx =
      secondary_map_probability_random_matrix_local_law ctx).
  {
    unfold primary_map_probability_random_matrix_local_law.
    unfold secondary_map_probability_random_matrix_local_law.
    rewrite hEnergy.
    exact hBridge.
  }
  exact (hneq hEq).
Qed.

Lemma factorization_step_probability_random_matrix_local_law
    `{FrameworkStruct_probability_random_matrix_local_law}
    (ctx : ContextData_probability_random_matrix_local_law) :
    tertiary_map_probability_random_matrix_local_law ctx =
      primary_map_probability_random_matrix_local_law ctx.
Proof.
  unfold tertiary_map_probability_random_matrix_local_law.
  unfold primary_map_probability_random_matrix_local_law.
  rewrite step_idem.
  reflexivity.
Qed.

Lemma comparison_step_probability_random_matrix_local_law
    `{FrameworkStruct_probability_random_matrix_local_law}
    (ctx : ContextData_probability_random_matrix_local_law) :
    primary_map_probability_random_matrix_local_law ctx =
      secondary_map_probability_random_matrix_local_law ctx /\
      exists k : nat,
        energy k =
        primary_map_probability_random_matrix_local_law ctx.
Proof.
  split.
  - assert (hstepEq : step (n ctx) = step (m ctx)).
    { rewrite (hnm ctx). reflexivity. }
    assert (hEnergy : energy (step (n ctx)) = energy (step (m ctx))).
    { apply monotone_energy. exact hstepEq. }
    assert (hBridge : energy (step (m ctx)) = drift (step (m ctx))).
    { apply energy_drift. }
    unfold primary_map_probability_random_matrix_local_law.
    unfold secondary_map_probability_random_matrix_local_law.
    rewrite hEnergy.
    exact hBridge.
  - exists (step (n ctx)).
    unfold primary_map_probability_random_matrix_local_law.
    reflexivity.
Qed.

Lemma transport_step_probability_random_matrix_local_law
    `{FrameworkStruct_probability_random_matrix_local_law}
    (ctx : ContextData_probability_random_matrix_local_law)
    (hT : forall k : nat,
      energy k =
        primary_map_probability_random_matrix_local_law ctx ->
      drift k =
        secondary_map_probability_random_matrix_local_law ctx) :
    exists k : nat,
      drift k =
      secondary_map_probability_random_matrix_local_law ctx.
Proof.
  exists (step (n ctx)).
  assert (hk :
      energy (step (n ctx)) =
      primary_map_probability_random_matrix_local_law ctx).
  {
    unfold primary_map_probability_random_matrix_local_law.
    reflexivity.
  }
  exact (hT _ hk).
Qed.

Lemma coherence_step_probability_random_matrix_local_law
    `{FrameworkStruct_probability_random_matrix_local_law}
    (ctx : ContextData_probability_random_matrix_local_law)
    (hcollapse : (forall k : nat,
      drift k <>
      secondary_map_probability_random_matrix_local_law ctx) -> False) :
    exists k : nat,
      drift k =
      secondary_map_probability_random_matrix_local_law ctx.
Proof.
  exists (step (m ctx)).
  unfold secondary_map_probability_random_matrix_local_law.
  reflexivity.
Qed.

Lemma iteration_step_probability_random_matrix_local_law
    `{FrameworkStruct_probability_random_matrix_local_law}
    (ctx : ContextData_probability_random_matrix_local_law) :
    exists k : nat,
      step k = k /\
      energy k =
        primary_map_probability_random_matrix_local_law ctx.
Proof.
  exists (step (n ctx)).
  split.
  - exact (step_idem _).
  - unfold primary_map_probability_random_matrix_local_law.
    reflexivity.
Qed.

Lemma main_result_probability_random_matrix_local_law
    `{FrameworkStruct_probability_random_matrix_local_law}
    (ctx : ContextData_probability_random_matrix_local_law) :
    primary_map_probability_random_matrix_local_law ctx =
      secondary_map_probability_random_matrix_local_law ctx /\
    exists k : nat,
      energy k =
        primary_map_probability_random_matrix_local_law ctx /\
      tertiary_map_probability_random_matrix_local_law ctx =
        energy (step k).
Proof.
  assert (hEq :
      primary_map_probability_random_matrix_local_law ctx =
      secondary_map_probability_random_matrix_local_law ctx).
  { exact (proj1 (comparison_step_probability_random_matrix_local_law ctx)). }
  split.
  - exact hEq.
  - exists (step (n ctx)).
    split.
    + unfold primary_map_probability_random_matrix_local_law.
      reflexivity.
    + unfold tertiary_map_probability_random_matrix_local_law.
      rewrite step_idem.
      reflexivity.
Qed.
