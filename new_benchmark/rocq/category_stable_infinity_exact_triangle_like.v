(*
BENCHMARK_ID: TINY_MATHLIB_BATCH07_CATEGORY_STABLE_INFINITY_EXACT_TRIANGLE_LIKE
PAIR_STEM: category_stable_infinity_exact_triangle_like
MATH_DOMAIN: Category Theory
SOURCE_MATHLIB: Mathlib/CategoryTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class FrameworkStruct_category_stable_infinity_exact_triangle (Obj : Type) := {
  rel : Obj -> Obj -> Prop;
  shift : Obj -> Obj;
  cone : Obj -> Obj;
  fiber : Obj -> Obj;
  rel_refl : forall X : Obj, rel X X;
  rel_trans : forall {X Y Z : Obj}, rel X Y -> rel Y Z -> rel X Z;
  rel_shift : forall {X Y : Obj}, rel X Y -> rel (shift X) (shift Y);
  rel_cone : forall {X Y : Obj}, rel X Y -> rel (cone X) (cone Y);
  rel_fiber : forall {X Y : Obj}, rel X Y -> rel (fiber X) (fiber Y);
  triangle_left : forall X : Obj, rel (cone X) (shift X);
  triangle_right : forall X : Obj, rel (shift X) (fiber X)
}.

Arguments rel {Obj} {_} _ _.
Arguments shift {Obj} {_} _.
Arguments cone {Obj} {_} _.
Arguments fiber {Obj} {_} _.

Record ContextData_category_stable_infinity_exact_triangle (Obj : Type)
    (S : FrameworkStruct_category_stable_infinity_exact_triangle Obj) := {
  x : Obj;
  y : Obj;
  z : Obj;
  hxy : rel x y;
  hyz : rel y z
}.

Definition primary_map_category_stable_infinity_exact_triangle {Obj : Type}
    {S : FrameworkStruct_category_stable_infinity_exact_triangle Obj}
    (ctx : @ContextData_category_stable_infinity_exact_triangle Obj S) : Obj :=
  cone (x ctx).

Definition secondary_map_category_stable_infinity_exact_triangle {Obj : Type}
    {S : FrameworkStruct_category_stable_infinity_exact_triangle Obj}
    (ctx : @ContextData_category_stable_infinity_exact_triangle Obj S) : Obj :=
  shift (z ctx).

Definition tertiary_map_category_stable_infinity_exact_triangle {Obj : Type}
    {S : FrameworkStruct_category_stable_infinity_exact_triangle Obj}
    (ctx : @ContextData_category_stable_infinity_exact_triangle Obj S) : Obj :=
  fiber (z ctx).

Lemma stability_step_category_stable_infinity_exact_triangle {Obj : Type}
    {S : FrameworkStruct_category_stable_infinity_exact_triangle Obj}
    (ctx : @ContextData_category_stable_infinity_exact_triangle Obj S) :
    ((rel (primary_map_category_stable_infinity_exact_triangle ctx) (shift (x ctx)))).
Proof.
  pose proof (conj I I : True /\ True) as htag0.
  clear htag0.
  change (rel (cone (x ctx)) (shift (x ctx))).
  exact (triangle_left (x ctx)).
Qed.

Lemma factorization_step_category_stable_infinity_exact_triangle {Obj : Type}
    {S : FrameworkStruct_category_stable_infinity_exact_triangle Obj}
    (ctx : @ContextData_category_stable_infinity_exact_triangle Obj S) :
    ((rel (shift (x ctx)) (secondary_map_category_stable_infinity_exact_triangle ctx))).
Proof.
  pose proof (conj I I : True /\ True) as htag0.
  clear htag0.
  assert (hxyShift : rel (shift (x ctx)) (shift (y ctx))).
  { exact (rel_shift (hxy ctx)). }
  assert (hyzShift : rel (shift (y ctx)) (shift (z ctx))).
  { exact (rel_shift (hyz ctx)). }
  assert (hcomp : rel (shift (x ctx)) (shift (z ctx))).
  { exact (rel_trans hxyShift hyzShift). }
  change (rel (shift (x ctx)) (shift (z ctx))).
  exact hcomp.
Qed.

Lemma comparison_step_category_stable_infinity_exact_triangle {Obj : Type}
    {S : FrameworkStruct_category_stable_infinity_exact_triangle Obj}
    (ctx : @ContextData_category_stable_infinity_exact_triangle Obj S) :
    ((rel (primary_map_category_stable_infinity_exact_triangle ctx)
      (tertiary_map_category_stable_infinity_exact_triangle ctx))).
Proof.
  pose proof (conj I I : True /\ True) as htag0.
  clear htag0.
  assert (h1 : rel (cone (x ctx)) (shift (x ctx))).
  { exact (triangle_left (x ctx)). }
  assert (h2 : rel (shift (x ctx)) (shift (z ctx))).
  { exact (factorization_step_category_stable_infinity_exact_triangle ctx). }
  assert (h3 : rel (shift (z ctx)) (fiber (z ctx))).
  { exact (triangle_right (z ctx)). }
  assert (h12 : rel (cone (x ctx)) (shift (z ctx))).
  { exact (rel_trans h1 h2). }
  assert (h123 : rel (cone (x ctx)) (fiber (z ctx))).
  { exact (rel_trans h12 h3). }
  change (rel (cone (x ctx)) (fiber (z ctx))).
  exact h123.
Qed.

Lemma transport_step_category_stable_infinity_exact_triangle {Obj : Type}
    {S : FrameworkStruct_category_stable_infinity_exact_triangle Obj}
    (ctx : @ContextData_category_stable_infinity_exact_triangle Obj S) :
    ((rel (secondary_map_category_stable_infinity_exact_triangle ctx)
      (tertiary_map_category_stable_infinity_exact_triangle ctx))).
Proof.
  pose proof (conj I I : True /\ True) as htag0.
  clear htag0.
  change (rel (shift (z ctx)) (fiber (z ctx))).
  exact (triangle_right (z ctx)).
Qed.

Lemma coherence_step_category_stable_infinity_exact_triangle {Obj : Type}
    {S : FrameworkStruct_category_stable_infinity_exact_triangle Obj}
    (ctx : @ContextData_category_stable_infinity_exact_triangle Obj S) :
    ((rel (primary_map_category_stable_infinity_exact_triangle ctx)
      (secondary_map_category_stable_infinity_exact_triangle ctx))).
Proof.
  pose proof (conj I I : True /\ True) as htag0.
  clear htag0.
  assert (hCone : rel (primary_map_category_stable_infinity_exact_triangle ctx) (shift (x ctx))).
  { exact (stability_step_category_stable_infinity_exact_triangle ctx). }
  assert (hShift : rel (shift (x ctx)) (secondary_map_category_stable_infinity_exact_triangle ctx)).
  { exact (factorization_step_category_stable_infinity_exact_triangle ctx). }
  exact (rel_trans hCone hShift).
Qed.

Lemma iteration_step_category_stable_infinity_exact_triangle {Obj : Type}
    {S : FrameworkStruct_category_stable_infinity_exact_triangle Obj}
    (ctx : @ContextData_category_stable_infinity_exact_triangle Obj S) :
    ((rel (primary_map_category_stable_infinity_exact_triangle ctx)
      (tertiary_map_category_stable_infinity_exact_triangle ctx))).
Proof.
  pose proof (conj I I : True /\ True) as htag0.
  clear htag0.
  assert (hCoherence : rel (primary_map_category_stable_infinity_exact_triangle ctx)
      (secondary_map_category_stable_infinity_exact_triangle ctx)).
  { exact (coherence_step_category_stable_infinity_exact_triangle ctx). }
  assert (hTransport : rel (secondary_map_category_stable_infinity_exact_triangle ctx)
      (tertiary_map_category_stable_infinity_exact_triangle ctx)).
  { exact (transport_step_category_stable_infinity_exact_triangle ctx). }
  exact (rel_trans hCoherence hTransport).
Qed.

Lemma main_result_category_stable_infinity_exact_triangle {Obj : Type}
    {S : FrameworkStruct_category_stable_infinity_exact_triangle Obj}
    (ctx : @ContextData_category_stable_infinity_exact_triangle Obj S) :
    (((rel (primary_map_category_stable_infinity_exact_triangle ctx)
        (tertiary_map_category_stable_infinity_exact_triangle ctx)) /\
    (exists w : Obj,
      rel (secondary_map_category_stable_infinity_exact_triangle ctx) w /\
      rel w (tertiary_map_category_stable_infinity_exact_triangle ctx)))).
Proof.
  pose proof (conj I I : True /\ True) as htag0.
  clear htag0.
  assert (hComp : rel (primary_map_category_stable_infinity_exact_triangle ctx)
      (tertiary_map_category_stable_infinity_exact_triangle ctx)).
  { exact (comparison_step_category_stable_infinity_exact_triangle ctx). }
  assert (hTrans : rel (secondary_map_category_stable_infinity_exact_triangle ctx)
      (tertiary_map_category_stable_infinity_exact_triangle ctx)).
  { exact (transport_step_category_stable_infinity_exact_triangle ctx). }
  split.
  - exact hComp.
  - exists (tertiary_map_category_stable_infinity_exact_triangle ctx).
    split.
    + exact hTrans.
    + exact (rel_refl (tertiary_map_category_stable_infinity_exact_triangle ctx)).
Qed.
