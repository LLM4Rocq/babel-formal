(*
BENCHMARK_ID: TINY_MATHLIB_BATCH07_PROBABILITY_PERCOLATION_CRITICAL_WINDOW_LIKE
PAIR_STEM: probability_percolation_critical_window_like
MATH_DOMAIN: Probability
SOURCE_MATHLIB: Mathlib/Probability/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class FrameworkStruct_probability_percolation_critical_window := {
  open_event : nat -> Prop;
  close_event : nat -> Prop;
  window : nat -> nat;
  complement : forall n : nat, open_event n -> close_event (window n) -> False;
  window_idem : forall n : nat, window (window n) = window n;
  seed_open : open_event 0;
  propagate_open : forall n : nat, open_event n -> open_event (window n)
}.

Record ContextData_probability_percolation_critical_window
    `{FrameworkStruct_probability_percolation_critical_window} := {
  t : nat;
  u : nat;
  htu : window t = u;
  ht : open_event t
}.

Definition primary_map_probability_percolation_critical_window
    `{FrameworkStruct_probability_percolation_critical_window}
    (ctx : ContextData_probability_percolation_critical_window) : Prop :=
  open_event (t ctx).

Definition secondary_map_probability_percolation_critical_window
    `{FrameworkStruct_probability_percolation_critical_window}
    (ctx : ContextData_probability_percolation_critical_window) : Prop :=
  open_event (u ctx).

Definition tertiary_map_probability_percolation_critical_window
    `{FrameworkStruct_probability_percolation_critical_window}
    (ctx : ContextData_probability_percolation_critical_window) : Prop :=
  open_event (window (u ctx)).

Lemma stability_step_probability_percolation_critical_window
    `{FrameworkStruct_probability_percolation_critical_window}
    (ctx : ContextData_probability_percolation_critical_window)
    (hneg : primary_map_probability_percolation_critical_window ctx -> False) :
    False.
Proof.
  assert (hprim : primary_map_probability_percolation_critical_window ctx).
  {
    unfold primary_map_probability_percolation_critical_window.
    exact (ht ctx).
  }
  exact (hneg hprim).
Qed.

Lemma factorization_step_probability_percolation_critical_window
    `{FrameworkStruct_probability_percolation_critical_window}
    (ctx : ContextData_probability_percolation_critical_window)
    (hprim : primary_map_probability_percolation_critical_window ctx) :
    secondary_map_probability_percolation_critical_window ctx.
Proof.
  unfold secondary_map_probability_percolation_critical_window.
  assert (hwin : open_event (window (t ctx))).
  { apply propagate_open. exact hprim. }
  rewrite (htu ctx) in hwin.
  exact hwin.
Qed.

Lemma comparison_step_probability_percolation_critical_window
    `{FrameworkStruct_probability_percolation_critical_window}
    (ctx : ContextData_probability_percolation_critical_window) :
    primary_map_probability_percolation_critical_window ctx ->
      secondary_map_probability_percolation_critical_window ctx /\
      exists k : nat, open_event k.
Proof.
  intro hprim.
  assert (hsec : secondary_map_probability_percolation_critical_window ctx).
  { apply factorization_step_probability_percolation_critical_window. exact hprim. }
  split.
  - exact hsec.
  - exists (u ctx).
    unfold secondary_map_probability_percolation_critical_window in hsec.
    exact hsec.
Qed.

Lemma transport_step_probability_percolation_critical_window
    `{FrameworkStruct_probability_percolation_critical_window}
    (ctx : ContextData_probability_percolation_critical_window)
    (hclose : close_event (window (u ctx)))
    (hsec : secondary_map_probability_percolation_critical_window ctx) :
    False.
Proof.
  unfold secondary_map_probability_percolation_critical_window in hsec.
  exact (complement (u ctx) hsec hclose).
Qed.

Lemma coherence_step_probability_percolation_critical_window
    `{FrameworkStruct_probability_percolation_critical_window}
    (ctx : ContextData_probability_percolation_critical_window)
    (hprim : primary_map_probability_percolation_critical_window ctx) :
    tertiary_map_probability_percolation_critical_window ctx.
Proof.
  assert (hsec : secondary_map_probability_percolation_critical_window ctx).
  { apply factorization_step_probability_percolation_critical_window. exact hprim. }
  unfold tertiary_map_probability_percolation_critical_window.
  unfold secondary_map_probability_percolation_critical_window in hsec.
  exact (propagate_open _ hsec).
Qed.

Lemma iteration_step_probability_percolation_critical_window
    `{FrameworkStruct_probability_percolation_critical_window}
    (ctx : ContextData_probability_percolation_critical_window) :
    exists k : nat,
      window k = k /\
      open_event k.
Proof.
  exists (window (u ctx)).
  split.
  - exact (window_idem _).
  - assert (hsec : secondary_map_probability_percolation_critical_window ctx).
    { apply factorization_step_probability_percolation_critical_window. exact (ht ctx). }
    unfold secondary_map_probability_percolation_critical_window in hsec.
    exact (propagate_open _ hsec).
Qed.

Lemma main_result_probability_percolation_critical_window
    `{FrameworkStruct_probability_percolation_critical_window}
    (ctx : ContextData_probability_percolation_critical_window) :
    primary_map_probability_percolation_critical_window ctx ->
    exists k : nat,
      secondary_map_probability_percolation_critical_window ctx /\
      (open_event k /\ tertiary_map_probability_percolation_critical_window ctx).
Proof.
  intro hprim.
  assert (hsec : secondary_map_probability_percolation_critical_window ctx).
  { apply factorization_step_probability_percolation_critical_window. exact hprim. }
  assert (hter : tertiary_map_probability_percolation_critical_window ctx).
  { apply coherence_step_probability_percolation_critical_window. exact hprim. }
  exists (window (u ctx)).
  split.
  - exact hsec.
  - split.
    + unfold secondary_map_probability_percolation_critical_window in hsec.
      exact (propagate_open _ hsec).
    + exact hter.
Qed.
