(*
BENCHMARK_ID: TINY_MATHLIB_BATCH07_PROBABILITY_INTERACTING_PARTICLE_HYDRODYNAMIC_LIMIT_LIKE
PAIR_STEM: probability_interacting_particle_hydrodynamic_limit_like
MATH_DOMAIN: Probability
SOURCE_MATHLIB: Mathlib/Probability/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class FrameworkStruct_probability_interacting_particle_hydrodynamic_limit := {
  flux : nat -> nat;
  density : nat -> nat;
  push : nat -> nat;
  push_zero : push 0 = 0;
  flux_push : forall n : nat, flux (push n) = flux n;
  density_flux : forall n : nat, density n = flux n;
  push_succ : forall n : nat, push (n + 1) = push n + 1
}.

Record ContextData_probability_interacting_particle_hydrodynamic_limit
    `{FrameworkStruct_probability_interacting_particle_hydrodynamic_limit} := {
  i : nat;
  j : nat;
  hij : i = push j
}.

Definition primary_map_probability_interacting_particle_hydrodynamic_limit
    `{FrameworkStruct_probability_interacting_particle_hydrodynamic_limit}
    (ctx : ContextData_probability_interacting_particle_hydrodynamic_limit) : nat :=
  flux (i ctx).

Definition secondary_map_probability_interacting_particle_hydrodynamic_limit
    `{FrameworkStruct_probability_interacting_particle_hydrodynamic_limit}
    (ctx : ContextData_probability_interacting_particle_hydrodynamic_limit) : nat :=
  density (push (j ctx)).

Definition tertiary_map_probability_interacting_particle_hydrodynamic_limit
    `{FrameworkStruct_probability_interacting_particle_hydrodynamic_limit}
    (ctx : ContextData_probability_interacting_particle_hydrodynamic_limit) : nat :=
  flux (push (push (j ctx))).

Lemma stability_step_probability_interacting_particle_hydrodynamic_limit
    `{FrameworkStruct_probability_interacting_particle_hydrodynamic_limit}
    (ctx : ContextData_probability_interacting_particle_hydrodynamic_limit)
    (hneq : primary_map_probability_interacting_particle_hydrodynamic_limit ctx <>
      secondary_map_probability_interacting_particle_hydrodynamic_limit ctx) :
    False /\
    primary_map_probability_interacting_particle_hydrodynamic_limit ctx =
      secondary_map_probability_interacting_particle_hydrodynamic_limit ctx.
Proof.
  assert (hprim :
      primary_map_probability_interacting_particle_hydrodynamic_limit ctx =
      flux (push (j ctx))).
  {
    unfold primary_map_probability_interacting_particle_hydrodynamic_limit.
    rewrite (hij ctx).
    reflexivity.
  }
  assert (hsec :
      secondary_map_probability_interacting_particle_hydrodynamic_limit ctx =
      flux (push (j ctx))).
  {
    unfold secondary_map_probability_interacting_particle_hydrodynamic_limit.
    rewrite density_flux.
    reflexivity.
  }
  assert (hEq :
      primary_map_probability_interacting_particle_hydrodynamic_limit ctx =
      secondary_map_probability_interacting_particle_hydrodynamic_limit ctx).
  {
    rewrite hprim.
    symmetry.
    exact hsec.
  }
  split.
  - exact (hneq hEq).
  - exact hEq.
Qed.

Lemma factorization_step_probability_interacting_particle_hydrodynamic_limit
    `{FrameworkStruct_probability_interacting_particle_hydrodynamic_limit}
    (ctx : ContextData_probability_interacting_particle_hydrodynamic_limit) :
    tertiary_map_probability_interacting_particle_hydrodynamic_limit ctx =
      secondary_map_probability_interacting_particle_hydrodynamic_limit ctx.
Proof.
  unfold tertiary_map_probability_interacting_particle_hydrodynamic_limit.
  unfold secondary_map_probability_interacting_particle_hydrodynamic_limit.
  rewrite flux_push.
  rewrite density_flux.
  reflexivity.
Qed.

Lemma comparison_step_probability_interacting_particle_hydrodynamic_limit
    `{FrameworkStruct_probability_interacting_particle_hydrodynamic_limit}
    (ctx : ContextData_probability_interacting_particle_hydrodynamic_limit) :
    exists k : nat,
      flux k =
        secondary_map_probability_interacting_particle_hydrodynamic_limit ctx /\
      primary_map_probability_interacting_particle_hydrodynamic_limit ctx =
        secondary_map_probability_interacting_particle_hydrodynamic_limit ctx.
Proof.
  assert (hEq :
      primary_map_probability_interacting_particle_hydrodynamic_limit ctx =
      secondary_map_probability_interacting_particle_hydrodynamic_limit ctx).
  {
    assert (hprim :
        primary_map_probability_interacting_particle_hydrodynamic_limit ctx =
        flux (push (j ctx))).
    {
      unfold primary_map_probability_interacting_particle_hydrodynamic_limit.
      rewrite (hij ctx).
      reflexivity.
    }
    assert (hsec :
        secondary_map_probability_interacting_particle_hydrodynamic_limit ctx =
        flux (push (j ctx))).
    {
      unfold secondary_map_probability_interacting_particle_hydrodynamic_limit.
      rewrite density_flux.
      reflexivity.
    }
    rewrite hprim.
    symmetry.
    exact hsec.
  }
  exists (push (j ctx)).
  split.
  - unfold secondary_map_probability_interacting_particle_hydrodynamic_limit.
    rewrite density_flux.
    reflexivity.
  - exact hEq.
Qed.

Lemma transport_step_probability_interacting_particle_hydrodynamic_limit
    `{FrameworkStruct_probability_interacting_particle_hydrodynamic_limit}
    (ctx : ContextData_probability_interacting_particle_hydrodynamic_limit)
    (htrans : forall k : nat,
      flux k =
        secondary_map_probability_interacting_particle_hydrodynamic_limit ctx ->
      flux (push k) =
        secondary_map_probability_interacting_particle_hydrodynamic_limit ctx) :
    flux (push (push (j ctx))) =
      secondary_map_probability_interacting_particle_hydrodynamic_limit ctx.
Proof.
  assert (hk :
      flux (push (j ctx)) =
      secondary_map_probability_interacting_particle_hydrodynamic_limit ctx).
  {
    unfold secondary_map_probability_interacting_particle_hydrodynamic_limit.
    rewrite density_flux.
    reflexivity.
  }
  exact (htrans _ hk).
Qed.

Lemma coherence_step_probability_interacting_particle_hydrodynamic_limit
    `{FrameworkStruct_probability_interacting_particle_hydrodynamic_limit}
    (ctx : ContextData_probability_interacting_particle_hydrodynamic_limit)
    (hno : (forall k : nat,
      flux k <>
        secondary_map_probability_interacting_particle_hydrodynamic_limit ctx) -> False) :
    exists k : nat,
      flux k =
      secondary_map_probability_interacting_particle_hydrodynamic_limit ctx /\ True.
Proof.
  exists (push (j ctx)).
  split.
  - unfold secondary_map_probability_interacting_particle_hydrodynamic_limit.
    rewrite density_flux.
    reflexivity.
  - exact I.
Qed.

Lemma iteration_step_probability_interacting_particle_hydrodynamic_limit
    `{FrameworkStruct_probability_interacting_particle_hydrodynamic_limit}
    (ctx : ContextData_probability_interacting_particle_hydrodynamic_limit) :
    exists k : nat,
      push k = 0 /\
      flux (push k) = flux 0.
Proof.
  exists 0.
  split.
  - exact push_zero.
  - rewrite push_zero.
    reflexivity.
Qed.

Lemma main_result_probability_interacting_particle_hydrodynamic_limit
    `{FrameworkStruct_probability_interacting_particle_hydrodynamic_limit}
    (ctx : ContextData_probability_interacting_particle_hydrodynamic_limit) :
    exists k : nat,
      (flux k =
        secondary_map_probability_interacting_particle_hydrodynamic_limit ctx /\
      tertiary_map_probability_interacting_particle_hydrodynamic_limit ctx =
        flux (push k)) /\
      primary_map_probability_interacting_particle_hydrodynamic_limit ctx =
        secondary_map_probability_interacting_particle_hydrodynamic_limit ctx.
Proof.
  assert (hEq : primary_map_probability_interacting_particle_hydrodynamic_limit ctx =
      secondary_map_probability_interacting_particle_hydrodynamic_limit ctx).
  {
    destruct (comparison_step_probability_interacting_particle_hydrodynamic_limit ctx)
      as [k [hk1 hk2]].
    exact hk2.
  }
  exists (push (j ctx)).
  split.
  - split.
    + unfold secondary_map_probability_interacting_particle_hydrodynamic_limit.
      rewrite density_flux.
      reflexivity.
    + unfold tertiary_map_probability_interacting_particle_hydrodynamic_limit.
      reflexivity.
  - exact hEq.
Qed.
