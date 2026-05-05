(*
BENCHMARK_ID: TINY_MATHLIB_BATCH07_TOPOLOGY_SHEAF_HYPERDESCENT_STACK_LIKE
PAIR_STEM: topology_sheaf_hyperdescent_stack_like
MATH_DOMAIN: Topology
SOURCE_MATHLIB: Mathlib/Topology/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class FrameworkStruct_topology_sheaf_hyperdescent_stack (Obj : Type) := {
  base : Obj -> Prop;
  refine : Obj -> Obj;
  glue : Obj -> Obj -> Obj;
  descent : Obj -> Obj;
  base_refine : forall {X : Obj}, base X -> base (refine X);
  base_glue : forall {X Y : Obj}, base X -> base Y -> base (glue X Y);
  base_descent : forall {X : Obj}, base X -> base (descent X);
  descent_refine : forall X : Obj, descent (refine X) = refine (descent X);
  glue_refine : forall X Y : Obj, refine (glue X Y) = glue (refine X) (refine Y);
  hyper_axiom : forall {X : Obj}, base X -> base (descent (descent X));
  locality_axiom : forall {X Y : Obj}, base (glue X Y) -> base X /\ base Y
}.

Arguments base {Obj} {_} _.
Arguments refine {Obj} {_} _.
Arguments glue {Obj} {_} _ _.
Arguments descent {Obj} {_} _.

Record ContextData_topology_sheaf_hyperdescent_stack (Obj : Type)
    (T : FrameworkStruct_topology_sheaf_hyperdescent_stack Obj) := {
  u : Obj;
  v : Obj;
  hu : base u;
  hv : base v
}.

Definition primary_map_topology_sheaf_hyperdescent_stack {Obj : Type}
    {T : FrameworkStruct_topology_sheaf_hyperdescent_stack Obj}
    (ctx : @ContextData_topology_sheaf_hyperdescent_stack Obj T) : Obj :=
  glue (refine (u ctx)) (descent (v ctx)).

Definition secondary_map_topology_sheaf_hyperdescent_stack {Obj : Type}
    {T : FrameworkStruct_topology_sheaf_hyperdescent_stack Obj}
    (ctx : @ContextData_topology_sheaf_hyperdescent_stack Obj T) : Obj :=
  descent (glue (u ctx) (v ctx)).

Definition tertiary_map_topology_sheaf_hyperdescent_stack {Obj : Type}
    {T : FrameworkStruct_topology_sheaf_hyperdescent_stack Obj}
    (ctx : @ContextData_topology_sheaf_hyperdescent_stack Obj T) : Obj :=
  glue (primary_map_topology_sheaf_hyperdescent_stack ctx)
    (secondary_map_topology_sheaf_hyperdescent_stack ctx).

Lemma stability_step_topology_sheaf_hyperdescent_stack {Obj : Type}
    {T : FrameworkStruct_topology_sheaf_hyperdescent_stack Obj}
    (ctx : @ContextData_topology_sheaf_hyperdescent_stack Obj T) :
    (((base (primary_map_topology_sheaf_hyperdescent_stack ctx)))).
Proof.
  pose proof (or_introl I : True \/ False) as htag0.
  clear htag0.
  assert (hRefined : base (refine (u ctx))).
  { exact (base_refine (hu ctx)). }
  assert (hDescended : base (descent (v ctx))).
  { exact (base_descent (hv ctx)). }
  change (base (glue (refine (u ctx)) (descent (v ctx)))).
  exact (base_glue hRefined hDescended).
Qed.

Lemma factorization_step_topology_sheaf_hyperdescent_stack {Obj : Type}
    {T : FrameworkStruct_topology_sheaf_hyperdescent_stack Obj}
    (ctx : @ContextData_topology_sheaf_hyperdescent_stack Obj T) :
    (((base (secondary_map_topology_sheaf_hyperdescent_stack ctx)))).
Proof.
  pose proof (or_introl I : True \/ False) as htag0.
  clear htag0.
  assert (hGlued : base (glue (u ctx) (v ctx))).
  { exact (base_glue (hu ctx) (hv ctx)). }
  change (base (descent (glue (u ctx) (v ctx)))).
  exact (base_descent hGlued).
Qed.

Lemma comparison_step_topology_sheaf_hyperdescent_stack {Obj : Type}
    {T : FrameworkStruct_topology_sheaf_hyperdescent_stack Obj}
    (ctx : @ContextData_topology_sheaf_hyperdescent_stack Obj T) :
    (((base (tertiary_map_topology_sheaf_hyperdescent_stack ctx)))).
Proof.
  pose proof (or_introl I : True \/ False) as htag0.
  clear htag0.
  assert (hPrimary : base (primary_map_topology_sheaf_hyperdescent_stack ctx)).
  { exact (stability_step_topology_sheaf_hyperdescent_stack ctx). }
  assert (hSecondary : base (secondary_map_topology_sheaf_hyperdescent_stack ctx)).
  { exact (factorization_step_topology_sheaf_hyperdescent_stack ctx). }
  exact (base_glue hPrimary hSecondary).
Qed.

Lemma transport_step_topology_sheaf_hyperdescent_stack {Obj : Type}
    {T : FrameworkStruct_topology_sheaf_hyperdescent_stack Obj}
    (ctx : @ContextData_topology_sheaf_hyperdescent_stack Obj T) :
    (((base (tertiary_map_topology_sheaf_hyperdescent_stack ctx) ->
      base (primary_map_topology_sheaf_hyperdescent_stack ctx) /\
      base (secondary_map_topology_sheaf_hyperdescent_stack ctx)))).
Proof.
  pose proof (or_introl I : True \/ False) as htag0.
  clear htag0.
  intro hTertiary.
  exact (locality_axiom hTertiary).
Qed.

Lemma coherence_step_topology_sheaf_hyperdescent_stack {Obj : Type}
    {T : FrameworkStruct_topology_sheaf_hyperdescent_stack Obj}
    (ctx : @ContextData_topology_sheaf_hyperdescent_stack Obj T) :
    (((base (descent (tertiary_map_topology_sheaf_hyperdescent_stack ctx))))).
Proof.
  pose proof (or_introl I : True \/ False) as htag0.
  clear htag0.
  assert (hTertiary : base (tertiary_map_topology_sheaf_hyperdescent_stack ctx)).
  { exact (comparison_step_topology_sheaf_hyperdescent_stack ctx). }
  exact (base_descent hTertiary).
Qed.

Lemma iteration_step_topology_sheaf_hyperdescent_stack {Obj : Type}
    {T : FrameworkStruct_topology_sheaf_hyperdescent_stack Obj}
    (ctx : @ContextData_topology_sheaf_hyperdescent_stack Obj T) :
    (((base (descent (descent (tertiary_map_topology_sheaf_hyperdescent_stack ctx)))))).
Proof.
  pose proof (or_introl I : True \/ False) as htag0.
  clear htag0.
  assert (hTertiary : base (tertiary_map_topology_sheaf_hyperdescent_stack ctx)).
  { exact (comparison_step_topology_sheaf_hyperdescent_stack ctx). }
  exact (hyper_axiom hTertiary).
Qed.

Lemma main_result_topology_sheaf_hyperdescent_stack {Obj : Type}
    {T : FrameworkStruct_topology_sheaf_hyperdescent_stack Obj}
    (ctx : @ContextData_topology_sheaf_hyperdescent_stack Obj T) :
    (((exists w : Obj,
      base w /\
      (base (tertiary_map_topology_sheaf_hyperdescent_stack ctx) -> base (descent w))))).
Proof.
  pose proof (or_introl I : True \/ False) as htag0.
  clear htag0.
  exists (tertiary_map_topology_sheaf_hyperdescent_stack ctx).
  assert (hBase : base (tertiary_map_topology_sheaf_hyperdescent_stack ctx)).
  { exact (comparison_step_topology_sheaf_hyperdescent_stack ctx). }
  split.
  - exact hBase.
  - intro hInput.
    exact (base_descent hInput).
Qed.
