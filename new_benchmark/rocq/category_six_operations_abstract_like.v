(*
BENCHMARK_ID: TINY_MATHLIB_BATCH04_CATEGORY_SIX_OPERATIONS_ABSTRACT_LIKE
PAIR_STEM: category_six_operations_abstract_like
MATH_DOMAIN: Category Theory / Sheaf Theory
SOURCE_MATHLIB: Mathlib/CategoryTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class TriangulatedCategoryLike (Obj : Type) := {
  Hom : Obj -> Obj -> Type;
  id : forall {X : Obj}, Hom X X;
  comp : forall {X Y Z : Obj}, Hom X Y -> Hom Y Z -> Hom X Z;
  comp_assoc :
    forall {W X Y Z : Obj} (f : Hom W X) (g : Hom X Y) (h : Hom Y Z),
      comp (comp f g) h = comp f (comp g h);
  id_comp : forall {X Y : Obj} (f : Hom X Y), comp id f = f;
  comp_id : forall {X Y : Obj} (f : Hom X Y), comp f id = f
}.

Arguments Hom {Obj} _ _ _.
Arguments id {Obj} _ {X}.
Arguments comp {Obj} _ {X Y Z} _ _.

Record FunctorLike (C : Type) (TC : TriangulatedCategoryLike C) := {
  obj : C -> C;
  map : forall {X Y : C}, Hom TC X Y -> Hom TC X Y;
  map_id : forall X : C, map (id TC (X := X)) = id TC;
  map_comp : forall {X Y Z : C} (f : Hom TC X Y) (g : Hom TC Y Z),
      map (comp TC f g) = comp TC (map f) (map g)
}.

Definition PullbackLike {C : Type}
    {TC : TriangulatedCategoryLike C}
    (F : @FunctorLike C TC) : @FunctorLike C TC :=
  F.

Definition PushforwardLike {C : Type}
    {TC : TriangulatedCategoryLike C}
    (F : @FunctorLike C TC) : @FunctorLike C TC :=
  F.

Definition ExceptionalPushforwardLike {C : Type}
    {TC : TriangulatedCategoryLike C}
    (F : @FunctorLike C TC) : @FunctorLike C TC :=
  F.

Definition ExceptionalPullbackLike {C : Type}
    {TC : TriangulatedCategoryLike C}
    (F : @FunctorLike C TC) : @FunctorLike C TC :=
  F.

Lemma base_change_like {C : Type}
    {TC : TriangulatedCategoryLike C}
    (fstar : @FunctorLike C TC) (gstar : @FunctorLike C TC)
    (hbase : forall {X Y Z : C} (u : Hom TC X Y) (v : Hom TC Y Z),
      map fstar (comp TC u v) = comp TC (map gstar u) (map gstar v))
    {X Y Z : C} (u : Hom TC X Y) (v : Hom TC Y Z) :
    map (PullbackLike fstar) (comp TC u v) =
      comp TC (map (PushforwardLike gstar) u) (map (PushforwardLike gstar) v).
Proof.
  assert (hRaw : map fstar (comp TC u v) = comp TC (map gstar u) (map gstar v)).
  { exact (hbase _ _ _ u v). }
  assert (hLeft : map (PullbackLike fstar) (comp TC u v) = map fstar (comp TC u v)).
  { reflexivity. }
  assert (hRightU : map (PushforwardLike gstar) u = map gstar u).
  { reflexivity. }
  assert (hRightV : map (PushforwardLike gstar) v = map gstar v).
  { reflexivity. }
  assert (hRight : comp TC (map gstar u) (map gstar v) =
      comp TC (map (PushforwardLike gstar) u) (map (PushforwardLike gstar) v)).
  {
    rewrite hRightU.
    rewrite hRightV.
    reflexivity.
  }
  rewrite hLeft.
  rewrite hRaw.
  exact hRight.
Qed.

Lemma projection_formula_like {C : Type}
    {TC : TriangulatedCategoryLike C}
    (fstar : @FunctorLike C TC) (fbang : @FunctorLike C TC)
    (hproj : forall {X Y Z : C} (u : Hom TC X Y) (v : Hom TC Y Z),
      map fbang (comp TC u v) = comp TC (map fstar u) (map fbang v))
    {W X Y Z : C} (u : Hom TC W X) (v : Hom TC X Y) (w : Hom TC Y Z) :
    map (ExceptionalPushforwardLike fbang) (comp TC (comp TC u v) w) =
      comp TC (map (PushforwardLike fstar) u)
        (comp TC (map (PushforwardLike fstar) v) (map (ExceptionalPushforwardLike fbang) w)).
Proof.
  assert (hAssocC : comp TC (comp TC u v) w = comp TC u (comp TC v w)).
  { apply comp_assoc. }
  assert (hOuter : map fbang (comp TC u (comp TC v w)) =
      comp TC (map fstar u) (map fbang (comp TC v w))).
  { exact (hproj _ _ _ u (comp TC v w)). }
  assert (hInner : map fbang (comp TC v w) = comp TC (map fstar v) (map fbang w)).
  { exact (hproj _ _ _ v w). }
  unfold ExceptionalPushforwardLike, PushforwardLike.
  rewrite hAssocC.
  rewrite hOuter.
  rewrite hInner.
  reflexivity.
Qed.

Lemma localization_triangle_like {C : Type}
    {TC : TriangulatedCategoryLike C}
    (fstar : @FunctorLike C TC) (fshriek : @FunctorLike C TC)
    (hloc : forall {X Y : C} (u : Hom TC X Y), map fshriek u = map fstar u)
    {X Y Z : C} (u : Hom TC X Y) (v : Hom TC Y Z) :
    map (ExceptionalPullbackLike fshriek)
      (comp TC (comp TC u (id TC (X := Y))) v) =
      comp TC (map (PushforwardLike fstar) u) (map (PushforwardLike fstar) v).
Proof.
  assert (hOuter : map fshriek (comp TC (comp TC u (id TC (X := Y))) v) =
      comp TC (map fshriek (comp TC u (id TC (X := Y)))) (map fshriek v)).
  { exact (map_comp fshriek (comp TC u (id TC (X := Y))) v). }
  assert (hInner : map fshriek (comp TC u (id TC (X := Y))) =
      comp TC (map fshriek u) (map fshriek (id TC (X := Y)))).
  { exact (map_comp fshriek u (id TC (X := Y))). }
  assert (hMapId : map fshriek (id TC (X := Y)) = id TC).
  { exact (map_id fshriek Y). }
  assert (hCompId : comp TC (map fshriek u) (id TC) = map fshriek u).
  { apply comp_id. }
  assert (hLocU : map fshriek u = map fstar u).
  { exact (hloc _ _ u). }
  assert (hLocV : map fshriek v = map fstar v).
  { exact (hloc _ _ v). }
  unfold ExceptionalPullbackLike, PushforwardLike.
  rewrite hOuter.
  rewrite hInner.
  rewrite hMapId.
  rewrite hCompId.
  rewrite hLocU.
  rewrite hLocV.
  reflexivity.
Qed.

Lemma duality_exchange_like {C : Type}
    {TC : TriangulatedCategoryLike C}
    (fstar : @FunctorLike C TC)
    (fbang : @FunctorLike C TC)
    (fsharp : @FunctorLike C TC)
    (hbang : forall {X Y : C} (u : Hom TC X Y), map fbang u = map fstar u)
    (hsharp : forall {X Y : C} (u : Hom TC X Y), map fsharp u = map fstar u)
    {X Y Z : C} (u : Hom TC X Y) (v : Hom TC Y Z) :
    map (ExceptionalPullbackLike fsharp) (comp TC u v) =
      map (ExceptionalPushforwardLike fbang) (comp TC u v).
Proof.
  assert (hSharpComp : map fsharp (comp TC u v) = comp TC (map fsharp u) (map fsharp v)).
  { exact (map_comp fsharp u v). }
  assert (hBangComp : map fbang (comp TC u v) = comp TC (map fbang u) (map fbang v)).
  { exact (map_comp fbang u v). }
  assert (hSharpU : map fsharp u = map fstar u).
  { exact (hsharp _ _ u). }
  assert (hSharpV : map fsharp v = map fstar v).
  { exact (hsharp _ _ v). }
  assert (hBangU : map fbang u = map fstar u).
  { exact (hbang _ _ u). }
  assert (hBangV : map fbang v = map fstar v).
  { exact (hbang _ _ v). }
  unfold ExceptionalPullbackLike, ExceptionalPushforwardLike.
  rewrite hSharpComp.
  rewrite hSharpU.
  rewrite hSharpV.
  rewrite <- hBangU.
  rewrite <- hBangV.
  rewrite <- hBangComp.
  reflexivity.
Qed.

Lemma compact_generation_transfer {C : Type}
    {TC : TriangulatedCategoryLike C}
    (fstar : @FunctorLike C TC)
    (fbang : @FunctorLike C TC)
    (hcompare : forall {X Y : C} (u : Hom TC X Y), map fstar u = map fbang u)
    {X Y : C} (u : Hom TC X Y) :
    map (PushforwardLike fstar)
      (comp TC (comp TC (id TC (X := X)) u) (id TC (X := Y))) =
      map (ExceptionalPushforwardLike fbang) u.
Proof.
  assert (hCompLeft : comp TC (id TC (X := X)) u = u).
  { apply id_comp. }
  assert (hCompRight : comp TC u (id TC (X := Y)) = u).
  { apply comp_id. }
  assert (hMapEq : map fstar u = map fbang u).
  { exact (hcompare _ _ u). }
  unfold PushforwardLike, ExceptionalPushforwardLike.
  rewrite hCompLeft.
  rewrite hCompRight.
  rewrite hMapEq.
  reflexivity.
Qed.

Lemma six_ops_coherence_like {C : Type}
    {TC : TriangulatedCategoryLike C}
    (fpull : @FunctorLike C TC)
    (fstar : @FunctorLike C TC)
    (fbang : @FunctorLike C TC)
    (fsharp : @FunctorLike C TC)
    (hpull : forall {X Y : C} (u : Hom TC X Y), map fpull u = map fstar u)
    (hbang : forall {X Y : C} (u : Hom TC X Y), map fbang u = map fstar u)
    (hsharp : forall {X Y : C} (u : Hom TC X Y), map fsharp u = map fstar u)
    {W X Y Z : C} (u : Hom TC W X) (v : Hom TC X Y) (w : Hom TC Y Z) :
    map (PullbackLike fpull) (comp TC u (comp TC v w)) =
      comp TC (map (ExceptionalPullbackLike fsharp) u)
        (comp TC (map (ExceptionalPushforwardLike fbang) v)
          (map (ExceptionalPushforwardLike fbang) w)).
Proof.
  assert (hPull : map fpull (comp TC u (comp TC v w)) = map fstar (comp TC u (comp TC v w))).
  { exact (hpull _ _ (comp TC u (comp TC v w))). }
  assert (hStarOuter : map fstar (comp TC u (comp TC v w)) =
      comp TC (map fstar u) (map fstar (comp TC v w))).
  { exact (map_comp fstar u (comp TC v w)). }
  assert (hStarInner : map fstar (comp TC v w) = comp TC (map fstar v) (map fstar w)).
  { exact (map_comp fstar v w). }
  assert (hSharpU : map fsharp u = map fstar u).
  { exact (hsharp _ _ u). }
  assert (hBangV : map fbang v = map fstar v).
  { exact (hbang _ _ v). }
  assert (hBangW : map fbang w = map fstar w).
  { exact (hbang _ _ w). }
  unfold PullbackLike, ExceptionalPullbackLike, ExceptionalPushforwardLike.
  rewrite hPull.
  rewrite hStarOuter.
  rewrite hStarInner.
  rewrite <- hSharpU.
  rewrite <- hBangV.
  rewrite <- hBangW.
  reflexivity.
Qed.
