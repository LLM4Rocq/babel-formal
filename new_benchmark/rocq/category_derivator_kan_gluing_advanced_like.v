(*
BENCHMARK_ID: TINY_MATHLIB_BATCH07_CATEGORY_DERIVATOR_KAN_GLUING_ADVANCED_LIKE
PAIR_STEM: category_derivator_kan_gluing_advanced_like
MATH_DOMAIN: Category Theory
SOURCE_MATHLIB: Mathlib/CategoryTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class FrameworkStruct_category_derivator_kan_gluing_advanced (Obj : Type) := {
  step : Obj -> Obj -> Prop;
  glue : Obj -> Obj -> Obj;
  kanL : Obj -> Obj;
  kanR : Obj -> Obj;
  step_refl : forall X : Obj, step X X;
  step_trans : forall {X Y Z : Obj}, step X Y -> step Y Z -> step X Z;
  step_glue_left : forall X Y : Obj, step X (glue X Y);
  step_glue_right : forall X Y : Obj, step Y (glue X Y);
  kan_bridge : forall X : Obj, step (kanL X) (kanR X);
  kanL_monotone : forall {X Y : Obj}, step X Y -> step (kanL X) (kanL Y);
  kanR_monotone : forall {X Y : Obj}, step X Y -> step (kanR X) (kanR Y)
}.

Arguments step {Obj} {_} _ _.
Arguments glue {Obj} {_} _ _.
Arguments kanL {Obj} {_} _.
Arguments kanR {Obj} {_} _.

Record ContextData_category_derivator_kan_gluing_advanced (Obj : Type)
    (F : FrameworkStruct_category_derivator_kan_gluing_advanced Obj) := {
  a : Obj;
  b : Obj;
  c : Obj;
  hab : step a b;
  hbc : step b c
}.

Definition primary_map_category_derivator_kan_gluing_advanced {Obj : Type}
    {F : FrameworkStruct_category_derivator_kan_gluing_advanced Obj}
    (ctx : @ContextData_category_derivator_kan_gluing_advanced Obj F) : Obj :=
  glue (kanL (a ctx)) (kanR (c ctx)).

Definition secondary_map_category_derivator_kan_gluing_advanced {Obj : Type}
    {F : FrameworkStruct_category_derivator_kan_gluing_advanced Obj}
    (ctx : @ContextData_category_derivator_kan_gluing_advanced Obj F) : Obj :=
  glue (kanR (b ctx)) (kanL (c ctx)).

Definition tertiary_map_category_derivator_kan_gluing_advanced {Obj : Type}
    {F : FrameworkStruct_category_derivator_kan_gluing_advanced Obj}
    (ctx : @ContextData_category_derivator_kan_gluing_advanced Obj F) : Obj :=
  glue (primary_map_category_derivator_kan_gluing_advanced ctx)
    (secondary_map_category_derivator_kan_gluing_advanced ctx).

Lemma stability_step_category_derivator_kan_gluing_advanced {Obj : Type}
    {F : FrameworkStruct_category_derivator_kan_gluing_advanced Obj}
    (ctx : @ContextData_category_derivator_kan_gluing_advanced Obj F) :
    (step (kanL (a ctx)) (primary_map_category_derivator_kan_gluing_advanced ctx)).
Proof.
  pose proof I as htag0.
  clear htag0.
  assert (hExpand : primary_map_category_derivator_kan_gluing_advanced ctx =
      glue (kanL (a ctx)) (kanR (c ctx))).
  { reflexivity. }
  assert (hCore : step (kanL (a ctx)) (glue (kanL (a ctx)) (kanR (c ctx)))).
  { exact (step_glue_left (kanL (a ctx)) (kanR (c ctx))). }
  rewrite hExpand.
  exact hCore.
Qed.

Lemma factorization_step_category_derivator_kan_gluing_advanced {Obj : Type}
    {F : FrameworkStruct_category_derivator_kan_gluing_advanced Obj}
    (ctx : @ContextData_category_derivator_kan_gluing_advanced Obj F) :
    (step (kanR (c ctx)) (primary_map_category_derivator_kan_gluing_advanced ctx)).
Proof.
  pose proof I as htag0.
  clear htag0.
  assert (hExpand : primary_map_category_derivator_kan_gluing_advanced ctx =
      glue (kanL (a ctx)) (kanR (c ctx))).
  { reflexivity. }
  assert (hRight : step (kanR (c ctx)) (glue (kanL (a ctx)) (kanR (c ctx)))).
  { exact (step_glue_right (kanL (a ctx)) (kanR (c ctx))). }
  rewrite hExpand.
  exact hRight.
Qed.

Lemma comparison_step_category_derivator_kan_gluing_advanced {Obj : Type}
    {F : FrameworkStruct_category_derivator_kan_gluing_advanced Obj}
    (ctx : @ContextData_category_derivator_kan_gluing_advanced Obj F) :
    (step (primary_map_category_derivator_kan_gluing_advanced ctx)
      (tertiary_map_category_derivator_kan_gluing_advanced ctx)).
Proof.
  pose proof I as htag0.
  clear htag0.
  unfold tertiary_map_category_derivator_kan_gluing_advanced.
  exact (step_glue_left (primary_map_category_derivator_kan_gluing_advanced ctx)
    (secondary_map_category_derivator_kan_gluing_advanced ctx)).
Qed.

Lemma transport_step_category_derivator_kan_gluing_advanced {Obj : Type}
    {F : FrameworkStruct_category_derivator_kan_gluing_advanced Obj}
    (ctx : @ContextData_category_derivator_kan_gluing_advanced Obj F) :
    (step (secondary_map_category_derivator_kan_gluing_advanced ctx)
      (tertiary_map_category_derivator_kan_gluing_advanced ctx)).
Proof.
  pose proof I as htag0.
  clear htag0.
  unfold tertiary_map_category_derivator_kan_gluing_advanced.
  exact (step_glue_right (primary_map_category_derivator_kan_gluing_advanced ctx)
    (secondary_map_category_derivator_kan_gluing_advanced ctx)).
Qed.

Lemma coherence_step_category_derivator_kan_gluing_advanced {Obj : Type}
    {F : FrameworkStruct_category_derivator_kan_gluing_advanced Obj}
    (ctx : @ContextData_category_derivator_kan_gluing_advanced Obj F) :
    (step (kanL (a ctx)) (tertiary_map_category_derivator_kan_gluing_advanced ctx)).
Proof.
  pose proof I as htag0.
  clear htag0.
  assert (h1 : step (kanL (a ctx)) (primary_map_category_derivator_kan_gluing_advanced ctx)).
  { exact (stability_step_category_derivator_kan_gluing_advanced ctx). }
  assert (h2 : step (primary_map_category_derivator_kan_gluing_advanced ctx)
      (tertiary_map_category_derivator_kan_gluing_advanced ctx)).
  { exact (comparison_step_category_derivator_kan_gluing_advanced ctx). }
  exact (step_trans h1 h2).
Qed.

Lemma iteration_step_category_derivator_kan_gluing_advanced {Obj : Type}
    {F : FrameworkStruct_category_derivator_kan_gluing_advanced Obj}
    (ctx : @ContextData_category_derivator_kan_gluing_advanced Obj F) :
    (step (kanR (c ctx)) (tertiary_map_category_derivator_kan_gluing_advanced ctx)).
Proof.
  pose proof I as htag0.
  clear htag0.
  assert (hFactor : step (kanR (c ctx)) (primary_map_category_derivator_kan_gluing_advanced ctx)).
  { exact (factorization_step_category_derivator_kan_gluing_advanced ctx). }
  assert (hComp : step (primary_map_category_derivator_kan_gluing_advanced ctx)
      (tertiary_map_category_derivator_kan_gluing_advanced ctx)).
  { exact (comparison_step_category_derivator_kan_gluing_advanced ctx). }
  exact (step_trans hFactor hComp).
Qed.

Lemma main_result_category_derivator_kan_gluing_advanced {Obj : Type}
    {F : FrameworkStruct_category_derivator_kan_gluing_advanced Obj}
    (ctx : @ContextData_category_derivator_kan_gluing_advanced Obj F) :
    (exists z : Obj,
      step (kanL (a ctx)) z /\
      step (kanR (c ctx)) z /\
      step (primary_map_category_derivator_kan_gluing_advanced ctx) z).
Proof.
  pose proof I as htag0.
  clear htag0.
  exists (tertiary_map_category_derivator_kan_gluing_advanced ctx).
  assert (hLeft : step (kanL (a ctx)) (tertiary_map_category_derivator_kan_gluing_advanced ctx)).
  { exact (coherence_step_category_derivator_kan_gluing_advanced ctx). }
  assert (hRight : step (kanR (c ctx)) (tertiary_map_category_derivator_kan_gluing_advanced ctx)).
  { exact (iteration_step_category_derivator_kan_gluing_advanced ctx). }
  assert (hPrimary : step (primary_map_category_derivator_kan_gluing_advanced ctx)
      (tertiary_map_category_derivator_kan_gluing_advanced ctx)).
  { exact (comparison_step_category_derivator_kan_gluing_advanced ctx). }
  split.
  - exact hLeft.
  - split.
    + exact hRight.
    + exact hPrimary.
Qed.
