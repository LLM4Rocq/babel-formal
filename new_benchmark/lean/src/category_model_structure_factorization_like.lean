/-
BENCHMARK_ID: TINY_MATHLIB_BATCH05_CATEGORY_MODEL_STRUCTURE_FACTORIZATION_LIKE
PAIR_STEM: category_model_structure_factorization_like
MATH_DOMAIN: Category Theory
SOURCE_MATHLIB: Mathlib/CategoryTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u v w

class CatStruct_model_structure_factorization (C : Type u) where
  Hom : C -> C -> Type v
  id : {X : C} -> Hom X X
  comp : {X Y Z : C} -> Hom X Y -> Hom Y Z -> Hom X Z
  comp_assoc :
    forall {W X Y Z : C} (f : Hom W X) (g : Hom X Y) (h : Hom Y Z),
      comp (comp f g) h = comp f (comp g h)
  id_comp : forall {X Y : C} (f : Hom X Y), comp id f = f
  comp_id : forall {X Y : C} (f : Hom X Y), comp f id = f

infixr:10 " ⟶ " => CatStruct_model_structure_factorization.Hom
infixr:80 " ≫ " => CatStruct_model_structure_factorization.comp

structure Functor_model_structure_factorization
    (C : Type u) (D : Type w)
    [CatStruct_model_structure_factorization C]
    [CatStruct_model_structure_factorization D] where
  obj : C -> D
  map : {X Y : C} -> (X ⟶ Y) -> (obj X ⟶ obj Y)
  map_id : forall X : C, map (CatStruct_model_structure_factorization.id (X := X)) =
    CatStruct_model_structure_factorization.id
  map_comp : forall {X Y Z : C} (f : X ⟶ Y) (g : Y ⟶ Z),
      map (f ≫ g) = map f ≫ map g

structure NatIso_model_structure_factorization
    {C : Type u} {D : Type w}
    [CatStruct_model_structure_factorization C]
    [CatStruct_model_structure_factorization D]
    (F G : Functor_model_structure_factorization C D) where
  hom : forall X : C, F.obj X ⟶ G.obj X
  inv : forall X : C, G.obj X ⟶ F.obj X
  left_inv : forall X : C, hom X ≫ inv X = CatStruct_model_structure_factorization.id
  right_inv : forall X : C, inv X ≫ hom X = CatStruct_model_structure_factorization.id

def whisker_model_structure_factorization
    {C : Type u} {D : Type w}
    [CatStruct_model_structure_factorization C]
    [CatStruct_model_structure_factorization D]
    {F G H : Functor_model_structure_factorization C D}
    (α : NatIso_model_structure_factorization F G)
    (β : NatIso_model_structure_factorization G H) :
    NatIso_model_structure_factorization F H where
  hom X := α.hom X ≫ β.hom X
  inv X := β.inv X ≫ α.inv X
  left_inv X := by
    calc
      (α.hom X ≫ β.hom X) ≫ (β.inv X ≫ α.inv X)
          = α.hom X ≫ (β.hom X ≫ (β.inv X ≫ α.inv X)) :=
            CatStruct_model_structure_factorization.comp_assoc _ _ _
      _ = α.hom X ≫ ((β.hom X ≫ β.inv X) ≫ α.inv X) := by
            rw [CatStruct_model_structure_factorization.comp_assoc]
      _ = α.hom X ≫ (CatStruct_model_structure_factorization.id ≫ α.inv X) := by
            rw [β.left_inv X]
      _ = α.hom X ≫ α.inv X := by
            rw [CatStruct_model_structure_factorization.id_comp]
      _ = CatStruct_model_structure_factorization.id := α.left_inv X
  right_inv X := by
    calc
      (β.inv X ≫ α.inv X) ≫ (α.hom X ≫ β.hom X)
          = β.inv X ≫ (α.inv X ≫ (α.hom X ≫ β.hom X)) :=
            CatStruct_model_structure_factorization.comp_assoc _ _ _
      _ = β.inv X ≫ ((α.inv X ≫ α.hom X) ≫ β.hom X) := by
            rw [CatStruct_model_structure_factorization.comp_assoc]
      _ = β.inv X ≫ (CatStruct_model_structure_factorization.id ≫ β.hom X) := by
            rw [α.right_inv X]
      _ = β.inv X ≫ β.hom X := by
            rw [CatStruct_model_structure_factorization.id_comp]
      _ = CatStruct_model_structure_factorization.id := β.right_inv X

def compose_model_structure_factorization
    {C : Type u} {D : Type v} {E : Type w}
    [CatStruct_model_structure_factorization C]
    [CatStruct_model_structure_factorization D]
    [CatStruct_model_structure_factorization E]
    (F : Functor_model_structure_factorization C D)
    (G : Functor_model_structure_factorization D E) :
    Functor_model_structure_factorization C E where
  obj X := G.obj (F.obj X)
  map := fun {X Y} f => G.map (F.map f)
  map_id X := by
    have hF : F.map (CatStruct_model_structure_factorization.id (X := X)) =
        CatStruct_model_structure_factorization.id := F.map_id X
    have hG : G.map (CatStruct_model_structure_factorization.id (X := F.obj X)) =
        CatStruct_model_structure_factorization.id := G.map_id (F.obj X)
    rw [hF]
    exact hG
  map_comp := by
    intro X Y Z f g
    have hF := F.map_comp f g
    have hG := G.map_comp (F.map f) (F.map g)
    rw [hF]
    exact hG

theorem whisker_assoc_model_structure_factorization
    {C : Type u} {D : Type w}
    [CatStruct_model_structure_factorization C]
    [CatStruct_model_structure_factorization D]
    {F G H K : Functor_model_structure_factorization C D}
    (α : NatIso_model_structure_factorization F G)
    (β : NatIso_model_structure_factorization G H)
    (γ : NatIso_model_structure_factorization H K)
    (X : C) :
    (whisker_model_structure_factorization
      (whisker_model_structure_factorization α β) γ).hom X =
      α.hom X ≫ (β.hom X ≫ γ.hom X) := by
  calc
    (whisker_model_structure_factorization
      (whisker_model_structure_factorization α β) γ).hom X
        = (α.hom X ≫ β.hom X) ≫ γ.hom X := by
            rfl
    _ = α.hom X ≫ (β.hom X ≫ γ.hom X) :=
          CatStruct_model_structure_factorization.comp_assoc _ _ _

theorem unit_whisker_model_structure_factorization
    {C : Type u} {D : Type w}
    [CatStruct_model_structure_factorization C]
    [CatStruct_model_structure_factorization D]
    {F G : Functor_model_structure_factorization C D}
    (ι : NatIso_model_structure_factorization F F)
    (α : NatIso_model_structure_factorization F G)
    (hι : forall X : C, ι.hom X = CatStruct_model_structure_factorization.id)
    (X : C) :
    (whisker_model_structure_factorization ι α).hom X = α.hom X := by
  calc
    (whisker_model_structure_factorization ι α).hom X
        = ι.hom X ≫ α.hom X := by
            rfl
    _ = CatStruct_model_structure_factorization.id ≫ α.hom X := by
          rw [hι X]
    _ = α.hom X :=
          CatStruct_model_structure_factorization.id_comp _

theorem counit_whisker_model_structure_factorization
    {C : Type u} {D : Type w}
    [CatStruct_model_structure_factorization C]
    [CatStruct_model_structure_factorization D]
    {F G : Functor_model_structure_factorization C D}
    (α : NatIso_model_structure_factorization F G)
    (ι : NatIso_model_structure_factorization G G)
    (hι : forall X : C, ι.hom X = CatStruct_model_structure_factorization.id)
    (X : C) :
    (whisker_model_structure_factorization α ι).hom X = α.hom X := by
  calc
    (whisker_model_structure_factorization α ι).hom X
        = α.hom X ≫ ι.hom X := by
            rfl
    _ = α.hom X ≫ CatStruct_model_structure_factorization.id := by
          rw [hι X]
    _ = α.hom X :=
          CatStruct_model_structure_factorization.comp_id _

theorem pasting_coherence_model_structure_factorization
    {C : Type u} {D : Type w}
    [CatStruct_model_structure_factorization C]
    [CatStruct_model_structure_factorization D]
    {F G H K L : Functor_model_structure_factorization C D}
    (α : NatIso_model_structure_factorization F G)
    (β : NatIso_model_structure_factorization G H)
    (γ : NatIso_model_structure_factorization H K)
    (δ : NatIso_model_structure_factorization K L)
    (X : C) :
    ((whisker_model_structure_factorization
      (whisker_model_structure_factorization
        (whisker_model_structure_factorization α β) γ) δ).hom X)
      = α.hom X ≫ (β.hom X ≫ (γ.hom X ≫ δ.hom X)) := by
  have hinner : ((α.hom X ≫ β.hom X) ≫ γ.hom X) =
      α.hom X ≫ (β.hom X ≫ γ.hom X) :=
    CatStruct_model_structure_factorization.comp_assoc _ _ _
  calc
    (whisker_model_structure_factorization
      (whisker_model_structure_factorization
        (whisker_model_structure_factorization α β) γ) δ).hom X
      = (((α.hom X ≫ β.hom X) ≫ γ.hom X) ≫ δ.hom X) := by
          rfl
    _ = ((α.hom X ≫ (β.hom X ≫ γ.hom X)) ≫ δ.hom X) := by
          rw [hinner]
    _ = (α.hom X ≫ ((β.hom X ≫ γ.hom X) ≫ δ.hom X)) :=
          CatStruct_model_structure_factorization.comp_assoc _ _ _
    _ = α.hom X ≫ (β.hom X ≫ (γ.hom X ≫ δ.hom X)) := by
          rw [CatStruct_model_structure_factorization.comp_assoc]

theorem comparison_full_model_structure_factorization
    {C : Type u} {D : Type w}
    [CatStruct_model_structure_factorization C]
    [CatStruct_model_structure_factorization D]
    {F G : Functor_model_structure_factorization C D}
    (α β : NatIso_model_structure_factorization F G)
    (hhom : forall X : C, α.hom X = β.hom X)
    (hinv : forall X : C, α.inv X = β.inv X) :
    forall X : C,
      (α.hom X ≫ α.inv X = β.hom X ≫ β.inv X) /\
      (α.inv X ≫ α.hom X = β.inv X ≫ β.hom X) := by
  intro X
  constructor
  · calc
      α.hom X ≫ α.inv X = β.hom X ≫ α.inv X := by
            rw [hhom X]
      _ = β.hom X ≫ β.inv X := by
            rw [hinv X]
  · calc
      α.inv X ≫ α.hom X = β.inv X ≫ α.hom X := by
            rw [hinv X]
      _ = β.inv X ≫ β.hom X := by
            rw [hhom X]

theorem comparison_faithful_model_structure_factorization
    {C : Type u} {D : Type w}
    [CatStruct_model_structure_factorization C]
    [CatStruct_model_structure_factorization D]
    {F G H : Functor_model_structure_factorization C D}
    (α β : NatIso_model_structure_factorization F G)
    (γ : NatIso_model_structure_factorization G H)
    (hwhisk : forall X : C,
      (whisker_model_structure_factorization α γ).hom X =
      (whisker_model_structure_factorization β γ).hom X)
    (hcancel : forall (X : C) (f g : F.obj X ⟶ G.obj X),
      f ≫ γ.hom X = g ≫ γ.hom X -> f = g) :
    forall X : C, α.hom X = β.hom X := by
  intro X
  have hraw : α.hom X ≫ γ.hom X = β.hom X ≫ γ.hom X := by
    simpa using hwhisk X
  have hstep : α.hom X = β.hom X :=
    hcancel X (α.hom X) (β.hom X) hraw
  exact hstep

theorem equivalence_core_model_structure_factorization
    {C : Type u} {D : Type w}
    [CatStruct_model_structure_factorization C]
    [CatStruct_model_structure_factorization D]
    {F G H : Functor_model_structure_factorization C D}
    (α β : NatIso_model_structure_factorization F G)
    (γ : NatIso_model_structure_factorization G H)
    (hwhisk : forall X : C,
      (whisker_model_structure_factorization α γ).hom X =
      (whisker_model_structure_factorization β γ).hom X)
    (hcancel : forall (X : C) (f g : F.obj X ⟶ G.obj X),
      f ≫ γ.hom X = g ≫ γ.hom X -> f = g)
    (hinv : forall X : C, α.inv X = β.inv X) :
    forall X : C, α.hom X ≫ α.inv X = β.hom X ≫ β.inv X := by
  have hhom : forall X : C, α.hom X = β.hom X :=
    comparison_faithful_model_structure_factorization α β γ hwhisk hcancel
  have hcmp := comparison_full_model_structure_factorization α β hhom hinv
  intro X
  exact (hcmp X).1
