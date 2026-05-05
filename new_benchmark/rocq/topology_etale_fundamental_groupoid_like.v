(*
BENCHMARK_ID: TINY_MATHLIB_BATCH07_TOPOLOGY_ETALE_FUNDAMENTAL_GROUPOID_LIKE
PAIR_STEM: topology_etale_fundamental_groupoid_like
MATH_DOMAIN: Topology
SOURCE_MATHLIB: Mathlib/Topology/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class FrameworkStruct_topology_etale_fundamental_groupoid (Point : Type) := {
  path : Point -> Point -> Prop;
  lift : Point -> Point;
  concat : Point -> Point -> Point;
  basepoint : Point;
  path_refl : forall p : Point, path p p;
  path_symm : forall {p q : Point}, path p q -> path q p;
  path_trans : forall {p q r : Point}, path p q -> path q r -> path p r;
  path_lift : forall {p q : Point}, path p q -> path (lift p) (lift q);
  path_to_concat_left : forall p q : Point, path p (concat p q);
  path_to_concat_right : forall p q : Point, path q (concat p q);
  concat_base_left : forall p : Point, concat basepoint p = p
}.

Arguments path {Point} {_} _ _.
Arguments lift {Point} {_} _.
Arguments concat {Point} {_} _ _.

Record ContextData_topology_etale_fundamental_groupoid (Point : Type)
    (E : FrameworkStruct_topology_etale_fundamental_groupoid Point) := {
  p : Point;
  q : Point;
  r : Point;
  hpq : path p q;
  hqr : path q r
}.

Definition primary_map_topology_etale_fundamental_groupoid {Point : Type}
    {E : FrameworkStruct_topology_etale_fundamental_groupoid Point}
    (ctx : @ContextData_topology_etale_fundamental_groupoid Point E) : Point :=
  concat (p ctx) (q ctx).

Definition secondary_map_topology_etale_fundamental_groupoid {Point : Type}
    {E : FrameworkStruct_topology_etale_fundamental_groupoid Point}
    (ctx : @ContextData_topology_etale_fundamental_groupoid Point E) : Point :=
  lift (r ctx).

Definition tertiary_map_topology_etale_fundamental_groupoid {Point : Type}
    {E : FrameworkStruct_topology_etale_fundamental_groupoid Point}
    (ctx : @ContextData_topology_etale_fundamental_groupoid Point E) : Point :=
  concat (primary_map_topology_etale_fundamental_groupoid ctx)
    (secondary_map_topology_etale_fundamental_groupoid ctx).

Lemma stability_step_topology_etale_fundamental_groupoid {Point : Type}
    {E : FrameworkStruct_topology_etale_fundamental_groupoid Point}
    (ctx : @ContextData_topology_etale_fundamental_groupoid Point E) :
    ((((path (p ctx) (primary_map_topology_etale_fundamental_groupoid ctx))))).
Proof.
  pose proof (or_intror I : False \/ True) as htag0.
  clear htag0.
  change (path (p ctx) (concat (p ctx) (q ctx))).
  exact (path_to_concat_left (p ctx) (q ctx)).
Qed.

Lemma factorization_step_topology_etale_fundamental_groupoid {Point : Type}
    {E : FrameworkStruct_topology_etale_fundamental_groupoid Point}
    (ctx : @ContextData_topology_etale_fundamental_groupoid Point E) :
    ((((path (q ctx) (primary_map_topology_etale_fundamental_groupoid ctx))))).
Proof.
  pose proof (or_intror I : False \/ True) as htag0.
  clear htag0.
  change (path (q ctx) (concat (p ctx) (q ctx))).
  exact (path_to_concat_right (p ctx) (q ctx)).
Qed.

Lemma comparison_step_topology_etale_fundamental_groupoid {Point : Type}
    {E : FrameworkStruct_topology_etale_fundamental_groupoid Point}
    (ctx : @ContextData_topology_etale_fundamental_groupoid Point E) :
    ((((path (primary_map_topology_etale_fundamental_groupoid ctx)
      (tertiary_map_topology_etale_fundamental_groupoid ctx))))).
Proof.
  pose proof (or_intror I : False \/ True) as htag0.
  clear htag0.
  change (path (concat (p ctx) (q ctx))
      (concat (concat (p ctx) (q ctx)) (lift (r ctx)))).
  exact (path_to_concat_left (concat (p ctx) (q ctx)) (lift (r ctx))).
Qed.

Lemma transport_step_topology_etale_fundamental_groupoid {Point : Type}
    {E : FrameworkStruct_topology_etale_fundamental_groupoid Point}
    (ctx : @ContextData_topology_etale_fundamental_groupoid Point E) :
    ((((path (secondary_map_topology_etale_fundamental_groupoid ctx)
      (tertiary_map_topology_etale_fundamental_groupoid ctx))))).
Proof.
  pose proof (or_intror I : False \/ True) as htag0.
  clear htag0.
  change (path (lift (r ctx))
      (concat (concat (p ctx) (q ctx)) (lift (r ctx)))).
  exact (path_to_concat_right (concat (p ctx) (q ctx)) (lift (r ctx))).
Qed.

Lemma coherence_step_topology_etale_fundamental_groupoid {Point : Type}
    {E : FrameworkStruct_topology_etale_fundamental_groupoid Point}
    (ctx : @ContextData_topology_etale_fundamental_groupoid Point E) :
    ((((path (p ctx) (tertiary_map_topology_etale_fundamental_groupoid ctx))))).
Proof.
  pose proof (or_intror I : False \/ True) as htag0.
  clear htag0.
  assert (hp : path (p ctx) (primary_map_topology_etale_fundamental_groupoid ctx)).
  { exact (stability_step_topology_etale_fundamental_groupoid ctx). }
  assert (hq : path (primary_map_topology_etale_fundamental_groupoid ctx)
      (tertiary_map_topology_etale_fundamental_groupoid ctx)).
  { exact (comparison_step_topology_etale_fundamental_groupoid ctx). }
  exact (path_trans hp hq).
Qed.

Lemma iteration_step_topology_etale_fundamental_groupoid {Point : Type}
    {E : FrameworkStruct_topology_etale_fundamental_groupoid Point}
    (ctx : @ContextData_topology_etale_fundamental_groupoid Point E) :
    ((((path (q ctx) (tertiary_map_topology_etale_fundamental_groupoid ctx))))).
Proof.
  pose proof (or_intror I : False \/ True) as htag0.
  clear htag0.
  assert (hp : path (q ctx) (primary_map_topology_etale_fundamental_groupoid ctx)).
  { exact (factorization_step_topology_etale_fundamental_groupoid ctx). }
  assert (hq : path (primary_map_topology_etale_fundamental_groupoid ctx)
      (tertiary_map_topology_etale_fundamental_groupoid ctx)).
  { exact (comparison_step_topology_etale_fundamental_groupoid ctx). }
  exact (path_trans hp hq).
Qed.

Lemma main_result_topology_etale_fundamental_groupoid {Point : Type}
    {E : FrameworkStruct_topology_etale_fundamental_groupoid Point}
    (ctx : @ContextData_topology_etale_fundamental_groupoid Point E) :
    (((((path (p ctx) (tertiary_map_topology_etale_fundamental_groupoid ctx) /\
      path (q ctx) (tertiary_map_topology_etale_fundamental_groupoid ctx)) /\
    exists t : Point,
      path (tertiary_map_topology_etale_fundamental_groupoid ctx) t /\
      path t (tertiary_map_topology_etale_fundamental_groupoid ctx))))).
Proof.
  pose proof (or_intror I : False \/ True) as htag0.
  clear htag0.
  assert (hp : path (p ctx) (tertiary_map_topology_etale_fundamental_groupoid ctx)).
  { exact (coherence_step_topology_etale_fundamental_groupoid ctx). }
  assert (hq : path (q ctx) (tertiary_map_topology_etale_fundamental_groupoid ctx)).
  { exact (iteration_step_topology_etale_fundamental_groupoid ctx). }
  split.
  - split; assumption.
  - exists (tertiary_map_topology_etale_fundamental_groupoid ctx).
    assert (hrr : path (tertiary_map_topology_etale_fundamental_groupoid ctx)
        (tertiary_map_topology_etale_fundamental_groupoid ctx)).
    { exact (path_refl (tertiary_map_topology_etale_fundamental_groupoid ctx)). }
    split; exact hrr.
Qed.
