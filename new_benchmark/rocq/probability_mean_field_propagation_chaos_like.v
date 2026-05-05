(*
BENCHMARK_ID: TINY_MATHLIB_BATCH07_PROBABILITY_MEAN_FIELD_PROPAGATION_CHAOS_LIKE
PAIR_STEM: probability_mean_field_propagation_chaos_like
MATH_DOMAIN: Probability
SOURCE_MATHLIB: Mathlib/Probability/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class FrameworkStruct_probability_mean_field_propagation_chaos (Omega : Type) := {
  state_val : Omega -> nat;
  evolve : Omega -> Omega;
  couple : Omega -> Omega -> Omega;
  root : Omega;
  good : Omega -> Prop;
  good_root : good root;
  good_evolve : forall x : Omega, good x -> good (evolve x);
  good_couple_left : forall x y : Omega, good (couple x y) -> good x;
  good_couple_right : forall x y : Omega, good (couple x y) -> good y;
  good_couple_intro : forall x y : Omega, good x -> good y -> good (couple x y)
}.

Record ContextData_probability_mean_field_propagation_chaos
    (Omega : Type) `{FrameworkStruct_probability_mean_field_propagation_chaos Omega} := {
  p : Omega;
  q : Omega;
  hp : good p;
  hq : good q
}.

Definition primary_map_probability_mean_field_propagation_chaos
    {Omega : Type} `{FrameworkStruct_probability_mean_field_propagation_chaos Omega}
    (ctx : ContextData_probability_mean_field_propagation_chaos) : Prop :=
  good (couple (p ctx) (q ctx)).

Definition secondary_map_probability_mean_field_propagation_chaos
    {Omega : Type} `{FrameworkStruct_probability_mean_field_propagation_chaos Omega}
    (ctx : ContextData_probability_mean_field_propagation_chaos) : Prop :=
  good (evolve (couple (q ctx) (p ctx))).

Definition tertiary_map_probability_mean_field_propagation_chaos
    {Omega : Type} `{FrameworkStruct_probability_mean_field_propagation_chaos Omega}
    (ctx : ContextData_probability_mean_field_propagation_chaos) : Prop :=
  good (couple (evolve (p ctx)) (evolve (q ctx))).

Lemma stability_step_probability_mean_field_propagation_chaos
    {Omega : Type} `{FrameworkStruct_probability_mean_field_propagation_chaos Omega}
    (ctx : ContextData_probability_mean_field_propagation_chaos)
    (hcontr : primary_map_probability_mean_field_propagation_chaos ctx -> False) :
    False.
Proof.
  assert (hp' : good (p ctx)).
  { exact (hp ctx). }
  assert (hq' : good (q ctx)).
  { exact (hq ctx). }
  assert (hpair :
      primary_map_probability_mean_field_propagation_chaos ctx).
  {
    unfold primary_map_probability_mean_field_propagation_chaos.
    exact (good_couple_intro _ _ hp' hq').
  }
  exact (hcontr hpair).
Qed.

Lemma factorization_step_probability_mean_field_propagation_chaos
    {Omega : Type} `{FrameworkStruct_probability_mean_field_propagation_chaos Omega}
    (ctx : ContextData_probability_mean_field_propagation_chaos)
    (hprim : primary_map_probability_mean_field_propagation_chaos ctx) :
    good (p ctx) /\ good (q ctx).
Proof.
  assert (hleft : good (p ctx)).
  {
    unfold primary_map_probability_mean_field_propagation_chaos in hprim.
    exact (good_couple_left _ _ hprim).
  }
  assert (hright : good (q ctx)).
  {
    unfold primary_map_probability_mean_field_propagation_chaos in hprim.
    exact (good_couple_right _ _ hprim).
  }
  split.
  - exact hleft.
  - exact hright.
Qed.

Lemma comparison_step_probability_mean_field_propagation_chaos
    {Omega : Type} `{FrameworkStruct_probability_mean_field_propagation_chaos Omega}
    (ctx : ContextData_probability_mean_field_propagation_chaos) :
    primary_map_probability_mean_field_propagation_chaos ctx /\
      exists z : Omega, good z.
Proof.
  assert (hprim :
      primary_map_probability_mean_field_propagation_chaos ctx).
  {
    unfold primary_map_probability_mean_field_propagation_chaos.
    exact (good_couple_intro _ _ (hp ctx) (hq ctx)).
  }
  split.
  - exact hprim.
  - exists (couple (p ctx) (q ctx)).
    exact hprim.
Qed.

Lemma transport_step_probability_mean_field_propagation_chaos
    {Omega : Type} `{FrameworkStruct_probability_mean_field_propagation_chaos Omega}
    (ctx : ContextData_probability_mean_field_propagation_chaos)
    (hstep : forall z : Omega, good z -> good (evolve z))
    (hprim : primary_map_probability_mean_field_propagation_chaos ctx) :
    secondary_map_probability_mean_field_propagation_chaos ctx.
Proof.
  unfold secondary_map_probability_mean_field_propagation_chaos.
  assert (hqp : good (couple (q ctx) (p ctx))).
  { exact (good_couple_intro _ _ (hq ctx) (hp ctx)). }
  exact (hstep _ hqp).
Qed.

Lemma coherence_step_probability_mean_field_propagation_chaos
    {Omega : Type} `{FrameworkStruct_probability_mean_field_propagation_chaos Omega}
    (ctx : ContextData_probability_mean_field_propagation_chaos)
    (hnot : secondary_map_probability_mean_field_propagation_chaos ctx -> False)
    (hstep : forall z : Omega, good z -> good (evolve z)) :
    False.
Proof.
  assert (hsec : secondary_map_probability_mean_field_propagation_chaos ctx).
  {
    apply (transport_step_probability_mean_field_propagation_chaos ctx hstep).
    exact (good_couple_intro _ _ (hp ctx) (hq ctx)).
  }
  exact (hnot hsec).
Qed.

Lemma iteration_step_probability_mean_field_propagation_chaos
    {Omega : Type} `{FrameworkStruct_probability_mean_field_propagation_chaos Omega}
    (ctx : ContextData_probability_mean_field_propagation_chaos) :
    exists z : Omega,
      good z /\
      (good (evolve z) ->
        secondary_map_probability_mean_field_propagation_chaos ctx).
Proof.
  exists (couple (q ctx) (p ctx)).
  split.
  - exact (good_couple_intro _ _ (hq ctx) (hp ctx)).
  - intro hz.
    assert (hzKeep : good (evolve (couple (q ctx) (p ctx)))).
    { exact hz. }
    unfold secondary_map_probability_mean_field_propagation_chaos.
    exact (good_evolve _ (good_couple_intro _ _ (hq ctx) (hp ctx))).
Qed.

Lemma main_result_probability_mean_field_propagation_chaos
    {Omega : Type} `{FrameworkStruct_probability_mean_field_propagation_chaos Omega}
    (ctx : ContextData_probability_mean_field_propagation_chaos) :
    primary_map_probability_mean_field_propagation_chaos ctx /\
    exists z : Omega,
      good z /\
      (good (evolve z) /\
        secondary_map_probability_mean_field_propagation_chaos ctx).
Proof.
  assert (hprim :
      primary_map_probability_mean_field_propagation_chaos ctx).
  {
    unfold primary_map_probability_mean_field_propagation_chaos.
    exact (good_couple_intro _ _ (hp ctx) (hq ctx)).
  }
  split.
  - exact hprim.
  - exists (couple (q ctx) (p ctx)).
    split.
    + exact (good_couple_intro _ _ (hq ctx) (hp ctx)).
    + split.
      * exact (good_evolve _ (good_couple_intro _ _ (hq ctx) (hp ctx))).
      * unfold secondary_map_probability_mean_field_propagation_chaos.
        exact (good_evolve _ (good_couple_intro _ _ (hq ctx) (hp ctx))).
Qed.
