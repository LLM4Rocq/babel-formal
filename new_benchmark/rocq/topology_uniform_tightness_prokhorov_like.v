(*
BENCHMARK_ID: TINY_MATHLIB_BATCH07_TOPOLOGY_UNIFORM_TIGHTNESS_PROKHOROV_LIKE
PAIR_STEM: topology_uniform_tightness_prokhorov_like
MATH_DOMAIN: Topology
SOURCE_MATHLIB: Mathlib/Topology/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class FrameworkStruct_topology_uniform_tightness_prokhorov (Obj : Type) := {
  good : Obj -> Prop;
  bound : Obj -> nat;
  refine : Obj -> Obj;
  tighten : Obj -> Obj;
  merge : Obj -> Obj -> Obj;
  good_refine : forall {X : Obj}, good X -> good (refine X);
  good_tighten : forall {X : Obj}, good X -> good (tighten X);
  good_merge : forall {X Y : Obj}, good X -> good Y -> good (merge X Y);
  bound_refine : forall X : Obj, bound (refine X) <= bound X + 1;
  bound_tighten : forall X : Obj, bound (tighten X) <= bound X;
  bound_merge : forall X Y : Obj, bound (merge X Y) <= bound X + bound Y;
  nat_le_trans : forall {m n k : nat}, m <= n -> n <= k -> m <= k;
  add_left_mono : forall {a b c : nat}, a <= b -> c + a <= c + b;
  tightness_axiom : forall {X : Obj}, good X -> bound X <= bound (refine X) + bound (tighten X)
}.

Arguments good {Obj} {_} _.
Arguments bound {Obj} {_} _.
Arguments refine {Obj} {_} _.
Arguments tighten {Obj} {_} _.
Arguments merge {Obj} {_} _ _.

Record ContextData_topology_uniform_tightness_prokhorov (Obj : Type)
    (U : FrameworkStruct_topology_uniform_tightness_prokhorov Obj) := {
  mu : Obj;
  nu : Obj;
  hmu : good mu;
  hnu : good nu
}.

Definition primary_map_topology_uniform_tightness_prokhorov {Obj : Type}
    {U : FrameworkStruct_topology_uniform_tightness_prokhorov Obj}
    (ctx : @ContextData_topology_uniform_tightness_prokhorov Obj U) : Obj :=
  merge (refine (mu ctx)) (tighten (nu ctx)).

Definition secondary_map_topology_uniform_tightness_prokhorov {Obj : Type}
    {U : FrameworkStruct_topology_uniform_tightness_prokhorov Obj}
    (ctx : @ContextData_topology_uniform_tightness_prokhorov Obj U) : Obj :=
  tighten (merge (mu ctx) (nu ctx)).

Definition tertiary_map_topology_uniform_tightness_prokhorov {Obj : Type}
    {U : FrameworkStruct_topology_uniform_tightness_prokhorov Obj}
    (ctx : @ContextData_topology_uniform_tightness_prokhorov Obj U) : Obj :=
  merge (primary_map_topology_uniform_tightness_prokhorov ctx)
    (secondary_map_topology_uniform_tightness_prokhorov ctx).

Lemma stability_step_topology_uniform_tightness_prokhorov {Obj : Type}
    {U : FrameworkStruct_topology_uniform_tightness_prokhorov Obj}
    (ctx : @ContextData_topology_uniform_tightness_prokhorov Obj U) :
    ((((((good (primary_map_topology_uniform_tightness_prokhorov ctx))))))).
Proof.
  pose proof ((fun h : True => h) : True -> True) as htag0.
  clear htag0.
  assert (hRefined : good (refine (mu ctx))).
  { exact (good_refine (hmu ctx)). }
  assert (hTight : good (tighten (nu ctx))).
  { exact (good_tighten (hnu ctx)). }
  change (good (merge (refine (mu ctx)) (tighten (nu ctx)))).
  exact (good_merge hRefined hTight).
Qed.

Lemma factorization_step_topology_uniform_tightness_prokhorov {Obj : Type}
    {U : FrameworkStruct_topology_uniform_tightness_prokhorov Obj}
    (ctx : @ContextData_topology_uniform_tightness_prokhorov Obj U) :
    ((((((bound (secondary_map_topology_uniform_tightness_prokhorov ctx) <=
      bound (mu ctx) + bound (nu ctx))))))).
Proof.
  pose proof ((fun h : True => h) : True -> True) as htag0.
  clear htag0.
  assert (hTight : bound (tighten (merge (mu ctx) (nu ctx))) <= bound (merge (mu ctx) (nu ctx))).
  { exact (bound_tighten (merge (mu ctx) (nu ctx))). }
  assert (hMerge : bound (merge (mu ctx) (nu ctx)) <= bound (mu ctx) + bound (nu ctx)).
  { exact (bound_merge (mu ctx) (nu ctx)). }
  exact (nat_le_trans hTight hMerge).
Qed.

Lemma comparison_step_topology_uniform_tightness_prokhorov {Obj : Type}
    {U : FrameworkStruct_topology_uniform_tightness_prokhorov Obj}
    (ctx : @ContextData_topology_uniform_tightness_prokhorov Obj U) :
    ((((((bound (primary_map_topology_uniform_tightness_prokhorov ctx) <=
      bound (refine (mu ctx)) + bound (tighten (nu ctx)))))))).
Proof.
  pose proof ((fun h : True => h) : True -> True) as htag0.
  clear htag0.
  change (bound (merge (refine (mu ctx)) (tighten (nu ctx))) <=
    bound (refine (mu ctx)) + bound (tighten (nu ctx))).
  exact (bound_merge (refine (mu ctx)) (tighten (nu ctx))).
Qed.

Lemma transport_step_topology_uniform_tightness_prokhorov {Obj : Type}
    {U : FrameworkStruct_topology_uniform_tightness_prokhorov Obj}
    (ctx : @ContextData_topology_uniform_tightness_prokhorov Obj U) :
    ((((((good (secondary_map_topology_uniform_tightness_prokhorov ctx))))))).
Proof.
  pose proof ((fun h : True => h) : True -> True) as htag0.
  clear htag0.
  assert (hMerged : good (merge (mu ctx) (nu ctx))).
  { exact (good_merge (hmu ctx) (hnu ctx)). }
  change (good (tighten (merge (mu ctx) (nu ctx)))).
  exact (good_tighten hMerged).
Qed.

Lemma coherence_step_topology_uniform_tightness_prokhorov {Obj : Type}
    {U : FrameworkStruct_topology_uniform_tightness_prokhorov Obj}
    (ctx : @ContextData_topology_uniform_tightness_prokhorov Obj U) :
    ((((((good (tertiary_map_topology_uniform_tightness_prokhorov ctx))))))).
Proof.
  pose proof ((fun h : True => h) : True -> True) as htag0.
  clear htag0.
  assert (hPrimary : good (primary_map_topology_uniform_tightness_prokhorov ctx)).
  { exact (stability_step_topology_uniform_tightness_prokhorov ctx). }
  assert (hSecondary : good (secondary_map_topology_uniform_tightness_prokhorov ctx)).
  { exact (transport_step_topology_uniform_tightness_prokhorov ctx). }
  exact (good_merge hPrimary hSecondary).
Qed.

Lemma iteration_step_topology_uniform_tightness_prokhorov {Obj : Type}
    {U : FrameworkStruct_topology_uniform_tightness_prokhorov Obj}
    (ctx : @ContextData_topology_uniform_tightness_prokhorov Obj U) :
    ((((((bound (tertiary_map_topology_uniform_tightness_prokhorov ctx) <=
      bound (primary_map_topology_uniform_tightness_prokhorov ctx) +
      bound (secondary_map_topology_uniform_tightness_prokhorov ctx))))))).
Proof.
  pose proof ((fun h : True => h) : True -> True) as htag0.
  clear htag0.
  change (bound (merge (primary_map_topology_uniform_tightness_prokhorov ctx)
      (secondary_map_topology_uniform_tightness_prokhorov ctx)) <=
    bound (primary_map_topology_uniform_tightness_prokhorov ctx) +
      bound (secondary_map_topology_uniform_tightness_prokhorov ctx)).
  exact (bound_merge (primary_map_topology_uniform_tightness_prokhorov ctx)
    (secondary_map_topology_uniform_tightness_prokhorov ctx)).
Qed.

Lemma main_result_topology_uniform_tightness_prokhorov {Obj : Type}
    {U : FrameworkStruct_topology_uniform_tightness_prokhorov Obj}
    (ctx : @ContextData_topology_uniform_tightness_prokhorov Obj U) :
    (((((((exists t : Obj,
      good t /\
      bound t <= bound (primary_map_topology_uniform_tightness_prokhorov ctx) +
        (bound (mu ctx) + bound (nu ctx))) /\
    good (tertiary_map_topology_uniform_tightness_prokhorov ctx))))))).
Proof.
  pose proof ((fun h : True => h) : True -> True) as htag0.
  clear htag0.
  assert (hGoodT : good (tertiary_map_topology_uniform_tightness_prokhorov ctx)).
  { exact (coherence_step_topology_uniform_tightness_prokhorov ctx). }
  assert (hIter : bound (tertiary_map_topology_uniform_tightness_prokhorov ctx) <=
      bound (primary_map_topology_uniform_tightness_prokhorov ctx) +
      bound (secondary_map_topology_uniform_tightness_prokhorov ctx)).
  { exact (iteration_step_topology_uniform_tightness_prokhorov ctx). }
  assert (hSec : bound (secondary_map_topology_uniform_tightness_prokhorov ctx) <=
      bound (mu ctx) + bound (nu ctx)).
  { exact (factorization_step_topology_uniform_tightness_prokhorov ctx). }
  assert (hAdd : bound (primary_map_topology_uniform_tightness_prokhorov ctx) +
      bound (secondary_map_topology_uniform_tightness_prokhorov ctx) <=
      bound (primary_map_topology_uniform_tightness_prokhorov ctx) +
        (bound (mu ctx) + bound (nu ctx))).
  { exact (add_left_mono hSec). }
  assert (hBound : bound (tertiary_map_topology_uniform_tightness_prokhorov ctx) <=
      bound (primary_map_topology_uniform_tightness_prokhorov ctx) +
        (bound (mu ctx) + bound (nu ctx))).
  { exact (nat_le_trans hIter hAdd). }
  split.
  - exists (tertiary_map_topology_uniform_tightness_prokhorov ctx).
    split.
    + exact hGoodT.
    + exact hBound.
  - exact hGoodT.
Qed.
