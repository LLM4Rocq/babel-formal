(*
BENCHMARK_ID: TINY_MATHLIB_BATCH07_TOPOLOGY_MORSE_THEORY_GRADIENT_FLOW_LIKE
PAIR_STEM: topology_morse_theory_gradient_flow_like
MATH_DOMAIN: Topology
SOURCE_MATHLIB: Mathlib/Topology/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class FrameworkStruct_topology_morse_theory_gradient_flow (Obj : Type) := {
  desc : Obj -> Obj -> Prop;
  flow : Obj -> Obj;
  crit : Obj -> Obj;
  basin : Obj -> Obj -> Obj;
  desc_refl : forall X : Obj, desc X X;
  desc_trans : forall {X Y Z : Obj}, desc X Y -> desc Y Z -> desc X Z;
  flow_desc : forall X : Obj, desc (flow X) X;
  flow_monotone : forall {X Y : Obj}, desc X Y -> desc (flow X) (flow Y);
  crit_desc : forall X : Obj, desc X (crit X);
  basin_left : forall X Y : Obj, desc X (basin X Y);
  basin_right : forall X Y : Obj, desc Y (basin X Y);
  crit_flow : forall X : Obj, desc (flow (crit X)) (crit X)
}.

Arguments desc {Obj} {_} _ _.
Arguments flow {Obj} {_} _.
Arguments crit {Obj} {_} _.
Arguments basin {Obj} {_} _ _.

Record ContextData_topology_morse_theory_gradient_flow (Obj : Type)
    (M : FrameworkStruct_topology_morse_theory_gradient_flow Obj) := {
  p : Obj;
  q : Obj;
  r : Obj;
  hpq : desc p q;
  hqr : desc q r
}.

Definition primary_map_topology_morse_theory_gradient_flow {Obj : Type}
    {M : FrameworkStruct_topology_morse_theory_gradient_flow Obj}
    (ctx : @ContextData_topology_morse_theory_gradient_flow Obj M) : Obj :=
  basin (flow (p ctx)) (crit (q ctx)).

Definition secondary_map_topology_morse_theory_gradient_flow {Obj : Type}
    {M : FrameworkStruct_topology_morse_theory_gradient_flow Obj}
    (ctx : @ContextData_topology_morse_theory_gradient_flow Obj M) : Obj :=
  basin (flow (q ctx)) (crit (r ctx)).

Definition tertiary_map_topology_morse_theory_gradient_flow {Obj : Type}
    {M : FrameworkStruct_topology_morse_theory_gradient_flow Obj}
    (ctx : @ContextData_topology_morse_theory_gradient_flow Obj M) : Obj :=
  basin (primary_map_topology_morse_theory_gradient_flow ctx)
    (secondary_map_topology_morse_theory_gradient_flow ctx).

Lemma stability_step_topology_morse_theory_gradient_flow {Obj : Type}
    {M : FrameworkStruct_topology_morse_theory_gradient_flow Obj}
    (ctx : @ContextData_topology_morse_theory_gradient_flow Obj M) :
    ((((((((desc (flow (p ctx)) (primary_map_topology_morse_theory_gradient_flow ctx))))))))).
Proof.
  pose proof (conj I (or_introl I) : True /\ (True \/ False)) as htag0.
  clear htag0.
  change (desc (flow (p ctx)) (basin (flow (p ctx)) (crit (q ctx)))).
  exact (basin_left (flow (p ctx)) (crit (q ctx))).
Qed.

Lemma factorization_step_topology_morse_theory_gradient_flow {Obj : Type}
    {M : FrameworkStruct_topology_morse_theory_gradient_flow Obj}
    (ctx : @ContextData_topology_morse_theory_gradient_flow Obj M) :
    ((((((((desc (crit (q ctx)) (primary_map_topology_morse_theory_gradient_flow ctx))))))))).
Proof.
  pose proof (conj I (or_introl I) : True /\ (True \/ False)) as htag0.
  clear htag0.
  change (desc (crit (q ctx)) (basin (flow (p ctx)) (crit (q ctx)))).
  exact (basin_right (flow (p ctx)) (crit (q ctx))).
Qed.

Lemma comparison_step_topology_morse_theory_gradient_flow {Obj : Type}
    {M : FrameworkStruct_topology_morse_theory_gradient_flow Obj}
    (ctx : @ContextData_topology_morse_theory_gradient_flow Obj M) :
    ((((((((desc (primary_map_topology_morse_theory_gradient_flow ctx)
      (tertiary_map_topology_morse_theory_gradient_flow ctx))))))))).
Proof.
  pose proof (conj I (or_introl I) : True /\ (True \/ False)) as htag0.
  clear htag0.
  change (desc (basin (flow (p ctx)) (crit (q ctx)))
      (basin (basin (flow (p ctx)) (crit (q ctx)))
        (basin (flow (q ctx)) (crit (r ctx))))).
  exact (basin_left (basin (flow (p ctx)) (crit (q ctx)))
    (basin (flow (q ctx)) (crit (r ctx)))).
Qed.

Lemma transport_step_topology_morse_theory_gradient_flow {Obj : Type}
    {M : FrameworkStruct_topology_morse_theory_gradient_flow Obj}
    (ctx : @ContextData_topology_morse_theory_gradient_flow Obj M) :
    ((((((((desc (secondary_map_topology_morse_theory_gradient_flow ctx)
      (tertiary_map_topology_morse_theory_gradient_flow ctx))))))))).
Proof.
  pose proof (conj I (or_introl I) : True /\ (True \/ False)) as htag0.
  clear htag0.
  change (desc (basin (flow (q ctx)) (crit (r ctx)))
      (basin (basin (flow (p ctx)) (crit (q ctx)))
        (basin (flow (q ctx)) (crit (r ctx))))).
  exact (basin_right (basin (flow (p ctx)) (crit (q ctx)))
    (basin (flow (q ctx)) (crit (r ctx)))).
Qed.

Lemma coherence_step_topology_morse_theory_gradient_flow {Obj : Type}
    {M : FrameworkStruct_topology_morse_theory_gradient_flow Obj}
    (ctx : @ContextData_topology_morse_theory_gradient_flow Obj M) :
    ((((((((desc (flow (p ctx)) (tertiary_map_topology_morse_theory_gradient_flow ctx))))))))).
Proof.
  pose proof (conj I (or_introl I) : True /\ (True \/ False)) as htag0.
  clear htag0.
  assert (hP : desc (flow (p ctx)) (primary_map_topology_morse_theory_gradient_flow ctx)).
  { exact (stability_step_topology_morse_theory_gradient_flow ctx). }
  assert (hTop : desc (primary_map_topology_morse_theory_gradient_flow ctx)
      (tertiary_map_topology_morse_theory_gradient_flow ctx)).
  { exact (comparison_step_topology_morse_theory_gradient_flow ctx). }
  exact (desc_trans hP hTop).
Qed.

Lemma iteration_step_topology_morse_theory_gradient_flow {Obj : Type}
    {M : FrameworkStruct_topology_morse_theory_gradient_flow Obj}
    (ctx : @ContextData_topology_morse_theory_gradient_flow Obj M) :
    ((((((((desc (crit (q ctx)) (tertiary_map_topology_morse_theory_gradient_flow ctx))))))))).
Proof.
  pose proof (conj I (or_introl I) : True /\ (True \/ False)) as htag0.
  clear htag0.
  assert (hQ : desc (crit (q ctx)) (primary_map_topology_morse_theory_gradient_flow ctx)).
  { exact (factorization_step_topology_morse_theory_gradient_flow ctx). }
  assert (hTop : desc (primary_map_topology_morse_theory_gradient_flow ctx)
      (tertiary_map_topology_morse_theory_gradient_flow ctx)).
  { exact (comparison_step_topology_morse_theory_gradient_flow ctx). }
  exact (desc_trans hQ hTop).
Qed.

Lemma main_result_topology_morse_theory_gradient_flow {Obj : Type}
    {M : FrameworkStruct_topology_morse_theory_gradient_flow Obj}
    (ctx : @ContextData_topology_morse_theory_gradient_flow Obj M) :
    ((((((((exists m : Obj,
      desc (flow (p ctx)) m /\
      desc (crit (q ctx)) m /\
      (desc m (basin m (crit m)) /\ desc (flow (crit m)) (crit m)))))))))).
Proof.
  pose proof (conj I (or_introl I) : True /\ (True \/ False)) as htag0.
  clear htag0.
  exists (tertiary_map_topology_morse_theory_gradient_flow ctx).
  assert (hFlow : desc (flow (p ctx)) (tertiary_map_topology_morse_theory_gradient_flow ctx)).
  { exact (coherence_step_topology_morse_theory_gradient_flow ctx). }
  assert (hCrit : desc (crit (q ctx)) (tertiary_map_topology_morse_theory_gradient_flow ctx)).
  { exact (iteration_step_topology_morse_theory_gradient_flow ctx). }
  assert (hBasin : desc (tertiary_map_topology_morse_theory_gradient_flow ctx)
      (basin (tertiary_map_topology_morse_theory_gradient_flow ctx)
        (crit (tertiary_map_topology_morse_theory_gradient_flow ctx)))).
  { exact (basin_left _ _). }
  assert (hFlowCrit : desc (flow (crit (tertiary_map_topology_morse_theory_gradient_flow ctx)))
      (crit (tertiary_map_topology_morse_theory_gradient_flow ctx))).
  { exact (crit_flow (tertiary_map_topology_morse_theory_gradient_flow ctx)). }
  split.
  - exact hFlow.
  - split.
    + exact hCrit.
    + split; assumption.
Qed.
