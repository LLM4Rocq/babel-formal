/-
BENCHMARK_ID: TINY_MATHLIB_BATCH04_CATEGORY_SIX_OPERATIONS_ABSTRACT_LIKE
PAIR_STEM: category_six_operations_abstract_like
MATH_DOMAIN: Category Theory / Sheaf Theory
SOURCE_MATHLIB: Mathlib/CategoryTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u v

class TriangulatedCategoryLike (Obj : Type u) where
  Hom : Obj -> Obj -> Type v
  id : {X : Obj} -> Hom X X
  comp : {X Y Z : Obj} -> Hom X Y -> Hom Y Z -> Hom X Z
  comp_assoc :
    forall {W X Y Z : Obj} (f : Hom W X) (g : Hom X Y) (h : Hom Y Z),
      comp (comp f g) h = comp f (comp g h)
  id_comp : forall {X Y : Obj} (f : Hom X Y), comp id f = f
  comp_id : forall {X Y : Obj} (f : Hom X Y), comp f id = f

infixr:10 " ~> " => TriangulatedCategoryLike.Hom
infixr:80 " >>> " => TriangulatedCategoryLike.comp

structure FunctorLike (C : Type u) [TriangulatedCategoryLike C] where
  obj : C -> C
  map : {X Y : C} -> (X ~> Y) -> (X ~> Y)
  map_id : forall X : C, map (TriangulatedCategoryLike.id (X := X)) = TriangulatedCategoryLike.id
  map_comp : forall {X Y Z : C} (f : X ~> Y) (g : Y ~> Z), map (f >>> g) = map f >>> map g

def PullbackLike {C : Type u} [TriangulatedCategoryLike C]
    (F : FunctorLike C) : FunctorLike C :=
  F

def PushforwardLike {C : Type u} [TriangulatedCategoryLike C]
    (F : FunctorLike C) : FunctorLike C :=
  F

def ExceptionalPushforwardLike {C : Type u} [TriangulatedCategoryLike C]
    (F : FunctorLike C) : FunctorLike C :=
  F

def ExceptionalPullbackLike {C : Type u} [TriangulatedCategoryLike C]
    (F : FunctorLike C) : FunctorLike C :=
  F

theorem base_change_like {C : Type u}
    [TriangulatedCategoryLike C]
    (fstar : FunctorLike C) (gstar : FunctorLike C)
    (hbase : forall {X Y Z : C} (u : X ~> Y) (v : Y ~> Z),
      fstar.map (u >>> v) = gstar.map u >>> gstar.map v)
    {X Y Z : C} (u : X ~> Y) (v : Y ~> Z) :
    (PullbackLike fstar).map (u >>> v) =
      (PushforwardLike gstar).map u >>> (PushforwardLike gstar).map v := by
  have hRaw : fstar.map (u >>> v) = gstar.map u >>> gstar.map v := hbase u v
  have hLeft : (PullbackLike fstar).map (u >>> v) = fstar.map (u >>> v) := by
    rfl
  have hRightU : (PushforwardLike gstar).map u = gstar.map u := by
    rfl
  have hRightV : (PushforwardLike gstar).map v = gstar.map v := by
    rfl
  have hRight :
      gstar.map u >>> gstar.map v =
        (PushforwardLike gstar).map u >>> (PushforwardLike gstar).map v := by
    rw [hRightU, hRightV]
  calc
    (PullbackLike fstar).map (u >>> v) = fstar.map (u >>> v) := hLeft
    _ = gstar.map u >>> gstar.map v := hRaw
    _ = (PushforwardLike gstar).map u >>> (PushforwardLike gstar).map v := hRight

theorem projection_formula_like {C : Type u}
    [TriangulatedCategoryLike C]
    (fstar : FunctorLike C) (fbang : FunctorLike C)
    (hproj : forall {X Y Z : C} (u : X ~> Y) (v : Y ~> Z),
      fbang.map (u >>> v) = fstar.map u >>> fbang.map v)
    {W X Y Z : C} (u : W ~> X) (v : X ~> Y) (w : Y ~> Z) :
    (ExceptionalPushforwardLike fbang).map ((u >>> v) >>> w) =
      (PushforwardLike fstar).map u >>>
        ((PushforwardLike fstar).map v >>> (ExceptionalPushforwardLike fbang).map w) := by
  have hAssocC : (u >>> v) >>> w = u >>> (v >>> w) :=
    TriangulatedCategoryLike.comp_assoc u v w
  have hOuter : fbang.map (u >>> (v >>> w)) = fstar.map u >>> fbang.map (v >>> w) :=
    hproj u (v >>> w)
  have hInner : fbang.map (v >>> w) = fstar.map v >>> fbang.map w := hproj v w
  calc
    (ExceptionalPushforwardLike fbang).map ((u >>> v) >>> w)
        = fbang.map ((u >>> v) >>> w) := by
          rfl
    _ = fbang.map (u >>> (v >>> w)) := by
          rw [hAssocC]
    _ = fstar.map u >>> fbang.map (v >>> w) := hOuter
    _ = fstar.map u >>> (fstar.map v >>> fbang.map w) := by
          rw [hInner]
    _ = (PushforwardLike fstar).map u >>>
          ((PushforwardLike fstar).map v >>> (ExceptionalPushforwardLike fbang).map w) := by
          rfl

theorem localization_triangle_like {C : Type u}
    [TriangulatedCategoryLike C]
    (fstar : FunctorLike C) (fshriek : FunctorLike C)
    (hloc : forall {X Y : C} (u : X ~> Y), fshriek.map u = fstar.map u)
    {X Y Z : C} (u : X ~> Y) (v : Y ~> Z) :
    (ExceptionalPullbackLike fshriek).map
        ((u >>> (TriangulatedCategoryLike.id (X := Y))) >>> v) =
      (PushforwardLike fstar).map u >>> (PushforwardLike fstar).map v := by
  have hOuter :
      fshriek.map ((u >>> (TriangulatedCategoryLike.id (X := Y))) >>> v) =
        fshriek.map (u >>> (TriangulatedCategoryLike.id (X := Y))) >>> fshriek.map v :=
    fshriek.map_comp (u >>> (TriangulatedCategoryLike.id (X := Y))) v
  have hInner :
      fshriek.map (u >>> (TriangulatedCategoryLike.id (X := Y))) =
        fshriek.map u >>> fshriek.map (TriangulatedCategoryLike.id (X := Y)) :=
    fshriek.map_comp u (TriangulatedCategoryLike.id (X := Y))
  have hMapId :
      fshriek.map (TriangulatedCategoryLike.id (X := Y)) = TriangulatedCategoryLike.id :=
    fshriek.map_id Y
  have hCompId :
      fshriek.map u >>> TriangulatedCategoryLike.id = fshriek.map u :=
    TriangulatedCategoryLike.comp_id (fshriek.map u)
  have hLocU : fshriek.map u = fstar.map u := hloc u
  have hLocV : fshriek.map v = fstar.map v := hloc v
  calc
    (ExceptionalPullbackLike fshriek).map
        ((u >>> (TriangulatedCategoryLike.id (X := Y))) >>> v)
        = fshriek.map ((u >>> (TriangulatedCategoryLike.id (X := Y))) >>> v) := by
          rfl
    _ = fshriek.map (u >>> (TriangulatedCategoryLike.id (X := Y))) >>> fshriek.map v := hOuter
    _ = (fshriek.map u >>> fshriek.map (TriangulatedCategoryLike.id (X := Y))) >>> fshriek.map v := by
          rw [hInner]
    _ = (fshriek.map u >>> TriangulatedCategoryLike.id) >>> fshriek.map v := by
          rw [hMapId]
    _ = fshriek.map u >>> fshriek.map v := by
          rw [hCompId]
    _ = fstar.map u >>> fshriek.map v := by
          rw [hLocU]
    _ = fstar.map u >>> fstar.map v := by
          rw [hLocV]
    _ = (PushforwardLike fstar).map u >>> (PushforwardLike fstar).map v := by
          rfl

theorem duality_exchange_like {C : Type u}
    [TriangulatedCategoryLike C]
    (fstar : FunctorLike C) (fbang : FunctorLike C) (fsharp : FunctorLike C)
    (hbang : forall {X Y : C} (u : X ~> Y), fbang.map u = fstar.map u)
    (hsharp : forall {X Y : C} (u : X ~> Y), fsharp.map u = fstar.map u)
    {X Y Z : C} (u : X ~> Y) (v : Y ~> Z) :
    (ExceptionalPullbackLike fsharp).map (u >>> v) =
      (ExceptionalPushforwardLike fbang).map (u >>> v) := by
  have hSharpComp : fsharp.map (u >>> v) = fsharp.map u >>> fsharp.map v :=
    fsharp.map_comp u v
  have hBangComp : fbang.map (u >>> v) = fbang.map u >>> fbang.map v :=
    fbang.map_comp u v
  have hSharpU : fsharp.map u = fstar.map u := hsharp u
  have hSharpV : fsharp.map v = fstar.map v := hsharp v
  have hBangU : fbang.map u = fstar.map u := hbang u
  have hBangV : fbang.map v = fstar.map v := hbang v
  calc
    (ExceptionalPullbackLike fsharp).map (u >>> v)
        = fsharp.map (u >>> v) := by
          rfl
    _ = fsharp.map u >>> fsharp.map v := hSharpComp
    _ = fstar.map u >>> fsharp.map v := by
          rw [hSharpU]
    _ = fstar.map u >>> fstar.map v := by
          rw [hSharpV]
    _ = fbang.map u >>> fstar.map v := by
          rw [hBangU]
    _ = fbang.map u >>> fbang.map v := by
          rw [hBangV]
    _ = fbang.map (u >>> v) := by
          exact Eq.symm hBangComp
    _ = (ExceptionalPushforwardLike fbang).map (u >>> v) := by
          rfl

theorem compact_generation_transfer {C : Type u}
    [TriangulatedCategoryLike C]
    (fstar : FunctorLike C) (fbang : FunctorLike C)
    (hcompare : forall {X Y : C} (u : X ~> Y), fstar.map u = fbang.map u)
    {X Y : C} (u : X ~> Y) :
    (PushforwardLike fstar).map
        (((TriangulatedCategoryLike.id (X := X)) >>> u) >>> (TriangulatedCategoryLike.id (X := Y))) =
      (ExceptionalPushforwardLike fbang).map u := by
  have hCompLeft : (TriangulatedCategoryLike.id (X := X)) >>> u = u :=
    TriangulatedCategoryLike.id_comp u
  have hCompRight : u >>> (TriangulatedCategoryLike.id (X := Y)) = u :=
    TriangulatedCategoryLike.comp_id u
  have hMapEq : fstar.map u = fbang.map u := hcompare u
  calc
    (PushforwardLike fstar).map
        (((TriangulatedCategoryLike.id (X := X)) >>> u) >>> (TriangulatedCategoryLike.id (X := Y)))
        = fstar.map (((TriangulatedCategoryLike.id (X := X)) >>> u) >>> (TriangulatedCategoryLike.id (X := Y))) := by
          rfl
    _ = fstar.map (u >>> (TriangulatedCategoryLike.id (X := Y))) := by
          rw [hCompLeft]
    _ = fstar.map u := by
          rw [hCompRight]
    _ = fbang.map u := hMapEq
    _ = (ExceptionalPushforwardLike fbang).map u := by
          rfl

theorem six_ops_coherence_like {C : Type u}
    [TriangulatedCategoryLike C]
    (fpull : FunctorLike C) (fstar : FunctorLike C)
    (fbang : FunctorLike C) (fsharp : FunctorLike C)
    (hpull : forall {X Y : C} (u : X ~> Y), fpull.map u = fstar.map u)
    (hbang : forall {X Y : C} (u : X ~> Y), fbang.map u = fstar.map u)
    (hsharp : forall {X Y : C} (u : X ~> Y), fsharp.map u = fstar.map u)
    {W X Y Z : C} (u : W ~> X) (v : X ~> Y) (w : Y ~> Z) :
    (PullbackLike fpull).map (u >>> (v >>> w)) =
      (ExceptionalPullbackLike fsharp).map u >>>
        ((ExceptionalPushforwardLike fbang).map v >>> (ExceptionalPushforwardLike fbang).map w) := by
  have hPull : fpull.map (u >>> (v >>> w)) = fstar.map (u >>> (v >>> w)) := hpull (u >>> (v >>> w))
  have hStarOuter : fstar.map (u >>> (v >>> w)) = fstar.map u >>> fstar.map (v >>> w) :=
    fstar.map_comp u (v >>> w)
  have hStarInner : fstar.map (v >>> w) = fstar.map v >>> fstar.map w :=
    fstar.map_comp v w
  have hSharpU : fsharp.map u = fstar.map u := hsharp u
  have hBangV : fbang.map v = fstar.map v := hbang v
  have hBangW : fbang.map w = fstar.map w := hbang w
  calc
    (PullbackLike fpull).map (u >>> (v >>> w))
        = fpull.map (u >>> (v >>> w)) := by
          rfl
    _ = fstar.map (u >>> (v >>> w)) := hPull
    _ = fstar.map u >>> fstar.map (v >>> w) := hStarOuter
    _ = fstar.map u >>> (fstar.map v >>> fstar.map w) := by
          rw [hStarInner]
    _ = fsharp.map u >>> (fstar.map v >>> fstar.map w) := by
          rw [hSharpU]
    _ = fsharp.map u >>> (fbang.map v >>> fstar.map w) := by
          rw [hBangV]
    _ = fsharp.map u >>> (fbang.map v >>> fbang.map w) := by
          rw [hBangW]
    _ = (ExceptionalPullbackLike fsharp).map u >>>
          ((ExceptionalPushforwardLike fbang).map v >>> (ExceptionalPushforwardLike fbang).map w) := by
          rfl
