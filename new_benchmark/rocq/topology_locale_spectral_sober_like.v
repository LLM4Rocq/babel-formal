(*
BENCHMARK_ID: TINY_MATHLIB_BATCH07_TOPOLOGY_LOCALE_SPECTRAL_SOBER_LIKE
PAIR_STEM: topology_locale_spectral_sober_like
MATH_DOMAIN: Topology
SOURCE_MATHLIB: Mathlib/Topology/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class FrameworkStruct_topology_locale_spectral_sober (Obj : Type) := {
  le : Obj -> Obj -> Prop;
  join : Obj -> Obj -> Obj;
  closure : Obj -> Obj;
  spectral : Obj -> Obj;
  le_refl : forall X : Obj, le X X;
  le_trans : forall {X Y Z : Obj}, le X Y -> le Y Z -> le X Z;
  join_left : forall X Y : Obj, le X (join X Y);
  join_right : forall X Y : Obj, le Y (join X Y);
  closure_extensive : forall X : Obj, le X (closure X);
  closure_monotone : forall {X Y : Obj}, le X Y -> le (closure X) (closure Y);
  closure_idem : forall X : Obj, le (closure (closure X)) (closure X);
  spectral_bridge : forall X : Obj, le (closure X) (spectral (closure X))
}.

Arguments le {Obj} {_} _ _.
Arguments join {Obj} {_} _ _.
Arguments closure {Obj} {_} _.
Arguments spectral {Obj} {_} _.

Record ContextData_topology_locale_spectral_sober (Obj : Type)
    (L : FrameworkStruct_topology_locale_spectral_sober Obj) := {
  u : Obj;
  v : Obj;
  w : Obj;
  huv : le u v;
  hvw : le v w
}.

Definition primary_map_topology_locale_spectral_sober {Obj : Type}
    {L : FrameworkStruct_topology_locale_spectral_sober Obj}
    (ctx : @ContextData_topology_locale_spectral_sober Obj L) : Obj :=
  closure (join (u ctx) (v ctx)).

Definition secondary_map_topology_locale_spectral_sober {Obj : Type}
    {L : FrameworkStruct_topology_locale_spectral_sober Obj}
    (ctx : @ContextData_topology_locale_spectral_sober Obj L) : Obj :=
  spectral (closure (w ctx)).

Definition tertiary_map_topology_locale_spectral_sober {Obj : Type}
    {L : FrameworkStruct_topology_locale_spectral_sober Obj}
    (ctx : @ContextData_topology_locale_spectral_sober Obj L) : Obj :=
  closure (join (primary_map_topology_locale_spectral_sober ctx)
    (secondary_map_topology_locale_spectral_sober ctx)).

Lemma stability_step_topology_locale_spectral_sober {Obj : Type}
    {L : FrameworkStruct_topology_locale_spectral_sober Obj}
    (ctx : @ContextData_topology_locale_spectral_sober Obj L) :
    (((((le (u ctx) (primary_map_topology_locale_spectral_sober ctx)))))).
Proof.
  pose proof ((fun h : False => h) : False -> False) as htag0.
  clear htag0.
  assert (huJoin : le (u ctx) (join (u ctx) (v ctx))).
  { exact (join_left (u ctx) (v ctx)). }
  assert (hJoinClose : le (join (u ctx) (v ctx)) (closure (join (u ctx) (v ctx)))).
  { exact (closure_extensive (join (u ctx) (v ctx))). }
  exact (le_trans huJoin hJoinClose).
Qed.

Lemma factorization_step_topology_locale_spectral_sober {Obj : Type}
    {L : FrameworkStruct_topology_locale_spectral_sober Obj}
    (ctx : @ContextData_topology_locale_spectral_sober Obj L) :
    (((((le (secondary_map_topology_locale_spectral_sober ctx)
      (tertiary_map_topology_locale_spectral_sober ctx)))))).
Proof.
  pose proof ((fun h : False => h) : False -> False) as htag0.
  clear htag0.
  assert (hRight : le (secondary_map_topology_locale_spectral_sober ctx)
      (join (primary_map_topology_locale_spectral_sober ctx)
        (secondary_map_topology_locale_spectral_sober ctx))).
  { exact (join_right (primary_map_topology_locale_spectral_sober ctx)
      (secondary_map_topology_locale_spectral_sober ctx)). }
  assert (hClose : le (join (primary_map_topology_locale_spectral_sober ctx)
      (secondary_map_topology_locale_spectral_sober ctx))
      (closure (join (primary_map_topology_locale_spectral_sober ctx)
        (secondary_map_topology_locale_spectral_sober ctx)))).
  { exact (closure_extensive (join (primary_map_topology_locale_spectral_sober ctx)
      (secondary_map_topology_locale_spectral_sober ctx))). }
  exact (le_trans hRight hClose).
Qed.

Lemma comparison_step_topology_locale_spectral_sober {Obj : Type}
    {L : FrameworkStruct_topology_locale_spectral_sober Obj}
    (ctx : @ContextData_topology_locale_spectral_sober Obj L) :
    (((((le (primary_map_topology_locale_spectral_sober ctx)
      (tertiary_map_topology_locale_spectral_sober ctx)))))).
Proof.
  pose proof ((fun h : False => h) : False -> False) as htag0.
  clear htag0.
  assert (hLeft : le (primary_map_topology_locale_spectral_sober ctx)
      (join (primary_map_topology_locale_spectral_sober ctx)
        (secondary_map_topology_locale_spectral_sober ctx))).
  { exact (join_left (primary_map_topology_locale_spectral_sober ctx)
      (secondary_map_topology_locale_spectral_sober ctx)). }
  assert (hClose : le (join (primary_map_topology_locale_spectral_sober ctx)
      (secondary_map_topology_locale_spectral_sober ctx))
      (tertiary_map_topology_locale_spectral_sober ctx)).
  {
    unfold tertiary_map_topology_locale_spectral_sober.
    exact (closure_extensive (join (primary_map_topology_locale_spectral_sober ctx)
      (secondary_map_topology_locale_spectral_sober ctx))).
  }
  exact (le_trans hLeft hClose).
Qed.

Lemma transport_step_topology_locale_spectral_sober {Obj : Type}
    {L : FrameworkStruct_topology_locale_spectral_sober Obj}
    (ctx : @ContextData_topology_locale_spectral_sober Obj L) :
    (((((le (closure (w ctx)) (secondary_map_topology_locale_spectral_sober ctx)))))).
Proof.
  pose proof ((fun h : False => h) : False -> False) as htag0.
  clear htag0.
  change (le (closure (w ctx)) (spectral (closure (w ctx)))).
  exact (spectral_bridge (w ctx)).
Qed.

Lemma coherence_step_topology_locale_spectral_sober {Obj : Type}
    {L : FrameworkStruct_topology_locale_spectral_sober Obj}
    (ctx : @ContextData_topology_locale_spectral_sober Obj L) :
    (((((le (u ctx) (tertiary_map_topology_locale_spectral_sober ctx)))))).
Proof.
  pose proof ((fun h : False => h) : False -> False) as htag0.
  clear htag0.
  assert (hStable : le (u ctx) (primary_map_topology_locale_spectral_sober ctx)).
  { exact (stability_step_topology_locale_spectral_sober ctx). }
  assert (hCompare : le (primary_map_topology_locale_spectral_sober ctx)
      (tertiary_map_topology_locale_spectral_sober ctx)).
  { exact (comparison_step_topology_locale_spectral_sober ctx). }
  exact (le_trans hStable hCompare).
Qed.

Lemma iteration_step_topology_locale_spectral_sober {Obj : Type}
    {L : FrameworkStruct_topology_locale_spectral_sober Obj}
    (ctx : @ContextData_topology_locale_spectral_sober Obj L) :
    (((((le (closure (closure (w ctx))) (tertiary_map_topology_locale_spectral_sober ctx)))))).
Proof.
  pose proof ((fun h : False => h) : False -> False) as htag0.
  clear htag0.
  assert (hIdem : le (closure (closure (w ctx))) (closure (w ctx))).
  { exact (closure_idem (w ctx)). }
  assert (hBridge : le (closure (w ctx)) (secondary_map_topology_locale_spectral_sober ctx)).
  { exact (transport_step_topology_locale_spectral_sober ctx). }
  assert (hTop : le (secondary_map_topology_locale_spectral_sober ctx)
      (tertiary_map_topology_locale_spectral_sober ctx)).
  { exact (factorization_step_topology_locale_spectral_sober ctx). }
  assert (hMid : le (closure (closure (w ctx))) (secondary_map_topology_locale_spectral_sober ctx)).
  { exact (le_trans hIdem hBridge). }
  exact (le_trans hMid hTop).
Qed.

Lemma main_result_topology_locale_spectral_sober {Obj : Type}
    {L : FrameworkStruct_topology_locale_spectral_sober Obj}
    (ctx : @ContextData_topology_locale_spectral_sober Obj L) :
    (((((exists t : Obj,
      (le (u ctx) t /\ le (closure (w ctx)) t) /\
      le t (closure t)))))).
Proof.
  pose proof ((fun h : False => h) : False -> False) as htag0.
  clear htag0.
  exists (tertiary_map_topology_locale_spectral_sober ctx).
  assert (hu0 : le (u ctx) (tertiary_map_topology_locale_spectral_sober ctx)).
  { exact (coherence_step_topology_locale_spectral_sober ctx). }
  assert (hw1 : le (closure (w ctx)) (secondary_map_topology_locale_spectral_sober ctx)).
  { exact (transport_step_topology_locale_spectral_sober ctx). }
  assert (hw2 : le (secondary_map_topology_locale_spectral_sober ctx)
      (tertiary_map_topology_locale_spectral_sober ctx)).
  { exact (factorization_step_topology_locale_spectral_sober ctx). }
  assert (hw0 : le (closure (w ctx)) (tertiary_map_topology_locale_spectral_sober ctx)).
  { exact (le_trans hw1 hw2). }
  assert (hClose : le (tertiary_map_topology_locale_spectral_sober ctx)
      (closure (tertiary_map_topology_locale_spectral_sober ctx))).
  { exact (closure_extensive (tertiary_map_topology_locale_spectral_sober ctx)). }
  split.
  - split; assumption.
  - exact hClose.
Qed.
