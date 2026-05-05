(*
BENCHMARK_ID: TINY_MATHLIB_BATCH07_TOPOLOGY_CONVEXITY_HELLY_CARATHEODORY_LIKE
PAIR_STEM: topology_convexity_helly_caratheodory_like
MATH_DOMAIN: Topology
SOURCE_MATHLIB: Mathlib/Topology/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class FrameworkStruct_topology_convexity_helly_caratheodory (Obj : Type) := {
  subset : Obj -> Obj -> Prop;
  hull : Obj -> Obj;
  inter : Obj -> Obj -> Obj;
  combine : Obj -> Obj -> Obj;
  subset_refl : forall X : Obj, subset X X;
  subset_trans : forall {X Y Z : Obj}, subset X Y -> subset Y Z -> subset X Z;
  subset_hull : forall X : Obj, subset X (hull X);
  hull_mono : forall {X Y : Obj}, subset X Y -> subset (hull X) (hull Y);
  inter_left : forall X Y : Obj, subset (inter X Y) X;
  inter_right : forall X Y : Obj, subset (inter X Y) Y;
  inter_intro : forall {X Y Z : Obj}, subset Z X -> subset Z Y -> subset Z (inter X Y);
  combine_left : forall X Y : Obj, subset X (combine X Y);
  combine_right : forall X Y : Obj, subset Y (combine X Y)
}.

Arguments subset {Obj} {_} _ _.
Arguments hull {Obj} {_} _.
Arguments inter {Obj} {_} _ _.
Arguments combine {Obj} {_} _ _.

Record ContextData_topology_convexity_helly_caratheodory (Obj : Type)
    (C : FrameworkStruct_topology_convexity_helly_caratheodory Obj) := {
  a : Obj;
  b : Obj;
  c : Obj;
  hab : subset a b;
  hbc : subset b c
}.

Definition primary_map_topology_convexity_helly_caratheodory {Obj : Type}
    {C : FrameworkStruct_topology_convexity_helly_caratheodory Obj}
    (ctx : @ContextData_topology_convexity_helly_caratheodory Obj C) : Obj :=
  hull (combine (a ctx) (b ctx)).

Definition secondary_map_topology_convexity_helly_caratheodory {Obj : Type}
    {C : FrameworkStruct_topology_convexity_helly_caratheodory Obj}
    (ctx : @ContextData_topology_convexity_helly_caratheodory Obj C) : Obj :=
  inter (hull (b ctx)) (c ctx).

Definition tertiary_map_topology_convexity_helly_caratheodory {Obj : Type}
    {C : FrameworkStruct_topology_convexity_helly_caratheodory Obj}
    (ctx : @ContextData_topology_convexity_helly_caratheodory Obj C) : Obj :=
  hull (combine (primary_map_topology_convexity_helly_caratheodory ctx)
    (secondary_map_topology_convexity_helly_caratheodory ctx)).

Lemma stability_step_topology_convexity_helly_caratheodory {Obj : Type}
    {C : FrameworkStruct_topology_convexity_helly_caratheodory Obj}
    (ctx : @ContextData_topology_convexity_helly_caratheodory Obj C) :
    (((((((subset (a ctx) (primary_map_topology_convexity_helly_caratheodory ctx)))))))).
Proof.
  pose proof (or_intror I : (True /\ False) \/ True) as htag0.
  clear htag0.
  assert (hComb : subset (a ctx) (combine (a ctx) (b ctx))).
  { exact (combine_left (a ctx) (b ctx)). }
  assert (hHull : subset (combine (a ctx) (b ctx)) (hull (combine (a ctx) (b ctx)))).
  { exact (subset_hull (combine (a ctx) (b ctx))). }
  exact (subset_trans hComb hHull).
Qed.

Lemma factorization_step_topology_convexity_helly_caratheodory {Obj : Type}
    {C : FrameworkStruct_topology_convexity_helly_caratheodory Obj}
    (ctx : @ContextData_topology_convexity_helly_caratheodory Obj C) :
    (((((((subset (secondary_map_topology_convexity_helly_caratheodory ctx) (c ctx)))))))).
Proof.
  pose proof (or_intror I : (True /\ False) \/ True) as htag0.
  clear htag0.
  change (subset (inter (hull (b ctx)) (c ctx)) (c ctx)).
  exact (inter_right (hull (b ctx)) (c ctx)).
Qed.

Lemma comparison_step_topology_convexity_helly_caratheodory {Obj : Type}
    {C : FrameworkStruct_topology_convexity_helly_caratheodory Obj}
    (ctx : @ContextData_topology_convexity_helly_caratheodory Obj C) :
    (((((((subset (primary_map_topology_convexity_helly_caratheodory ctx)
      (tertiary_map_topology_convexity_helly_caratheodory ctx)))))))).
Proof.
  pose proof (or_intror I : (True /\ False) \/ True) as htag0.
  clear htag0.
  assert (hComb : subset (primary_map_topology_convexity_helly_caratheodory ctx)
      (combine (primary_map_topology_convexity_helly_caratheodory ctx)
        (secondary_map_topology_convexity_helly_caratheodory ctx))).
  { exact (combine_left (primary_map_topology_convexity_helly_caratheodory ctx)
      (secondary_map_topology_convexity_helly_caratheodory ctx)). }
  assert (hHull : subset (combine (primary_map_topology_convexity_helly_caratheodory ctx)
      (secondary_map_topology_convexity_helly_caratheodory ctx))
      (tertiary_map_topology_convexity_helly_caratheodory ctx)).
  {
    unfold tertiary_map_topology_convexity_helly_caratheodory.
    exact (subset_hull (combine (primary_map_topology_convexity_helly_caratheodory ctx)
      (secondary_map_topology_convexity_helly_caratheodory ctx))).
  }
  exact (subset_trans hComb hHull).
Qed.

Lemma transport_step_topology_convexity_helly_caratheodory {Obj : Type}
    {C : FrameworkStruct_topology_convexity_helly_caratheodory Obj}
    (ctx : @ContextData_topology_convexity_helly_caratheodory Obj C) :
    (((((((subset (secondary_map_topology_convexity_helly_caratheodory ctx)
      (tertiary_map_topology_convexity_helly_caratheodory ctx)))))))).
Proof.
  pose proof (or_intror I : (True /\ False) \/ True) as htag0.
  clear htag0.
  assert (hLeft : subset (secondary_map_topology_convexity_helly_caratheodory ctx)
      (combine (primary_map_topology_convexity_helly_caratheodory ctx)
        (secondary_map_topology_convexity_helly_caratheodory ctx))).
  { exact (combine_right (primary_map_topology_convexity_helly_caratheodory ctx)
      (secondary_map_topology_convexity_helly_caratheodory ctx)). }
  assert (hHull : subset (combine (primary_map_topology_convexity_helly_caratheodory ctx)
      (secondary_map_topology_convexity_helly_caratheodory ctx))
      (tertiary_map_topology_convexity_helly_caratheodory ctx)).
  {
    unfold tertiary_map_topology_convexity_helly_caratheodory.
    exact (subset_hull (combine (primary_map_topology_convexity_helly_caratheodory ctx)
      (secondary_map_topology_convexity_helly_caratheodory ctx))).
  }
  exact (subset_trans hLeft hHull).
Qed.

Lemma coherence_step_topology_convexity_helly_caratheodory {Obj : Type}
    {C : FrameworkStruct_topology_convexity_helly_caratheodory Obj}
    (ctx : @ContextData_topology_convexity_helly_caratheodory Obj C) :
    (((((((subset (a ctx) (tertiary_map_topology_convexity_helly_caratheodory ctx)))))))).
Proof.
  pose proof (or_intror I : (True /\ False) \/ True) as htag0.
  clear htag0.
  assert (hA : subset (a ctx) (primary_map_topology_convexity_helly_caratheodory ctx)).
  { exact (stability_step_topology_convexity_helly_caratheodory ctx). }
  assert (hP : subset (primary_map_topology_convexity_helly_caratheodory ctx)
      (tertiary_map_topology_convexity_helly_caratheodory ctx)).
  { exact (comparison_step_topology_convexity_helly_caratheodory ctx). }
  exact (subset_trans hA hP).
Qed.

Lemma iteration_step_topology_convexity_helly_caratheodory {Obj : Type}
    {C : FrameworkStruct_topology_convexity_helly_caratheodory Obj}
    (ctx : @ContextData_topology_convexity_helly_caratheodory Obj C) :
    (((((((subset (b ctx) (tertiary_map_topology_convexity_helly_caratheodory ctx)))))))).
Proof.
  pose proof (or_intror I : (True /\ False) \/ True) as htag0.
  clear htag0.
  assert (hHullB : subset (b ctx) (hull (b ctx))).
  { exact (subset_hull (b ctx)). }
  assert (hBtoC : subset (b ctx) (c ctx)).
  { exact (hbc ctx). }
  assert (hIntoInter : subset (b ctx) (inter (hull (b ctx)) (c ctx))).
  { exact (inter_intro hHullB hBtoC). }
  assert (hToTop : subset (secondary_map_topology_convexity_helly_caratheodory ctx)
      (tertiary_map_topology_convexity_helly_caratheodory ctx)).
  { exact (transport_step_topology_convexity_helly_caratheodory ctx). }
  exact (subset_trans hIntoInter hToTop).
Qed.

Lemma main_result_topology_convexity_helly_caratheodory {Obj : Type}
    {C : FrameworkStruct_topology_convexity_helly_caratheodory Obj}
    (ctx : @ContextData_topology_convexity_helly_caratheodory Obj C) :
    (((((((exists t : Obj,
      subset (a ctx) t /\
      (subset (b ctx) t \/ subset t (hull t))))))))).
Proof.
  pose proof (or_intror I : (True /\ False) \/ True) as htag0.
  clear htag0.
  exists (tertiary_map_topology_convexity_helly_caratheodory ctx).
  assert (hA : subset (a ctx) (tertiary_map_topology_convexity_helly_caratheodory ctx)).
  { exact (coherence_step_topology_convexity_helly_caratheodory ctx). }
  assert (hB : subset (b ctx) (tertiary_map_topology_convexity_helly_caratheodory ctx)).
  { exact (iteration_step_topology_convexity_helly_caratheodory ctx). }
  split.
  - exact hA.
  - left; exact hB.
Qed.
