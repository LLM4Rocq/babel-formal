/-
BENCHMARK_ID: TINY_MATHLIB_BATCH04_CATEGORY_TANNAKIAN_DUALITY_AXIOMATIC
PAIR_STEM: category_tannakian_duality_axiomatic
MATH_DOMAIN: Category Theory / Representation Theory
SOURCE_MATHLIB: Mathlib/CategoryTheory/Monoidal
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u v w

class TensorCategoryLike (Obj : Type u) where
  Hom : Obj → Obj → Type v
  id : {X : Obj} → Hom X X
  comp : {X Y Z : Obj} → Hom X Y → Hom Y Z → Hom X Z
  comp_assoc :
    ∀ {W X Y Z : Obj} (f : Hom W X) (g : Hom X Y) (h : Hom Y Z),
      comp (comp f g) h = comp f (comp g h)
  id_comp : ∀ {X Y : Obj} (f : Hom X Y), comp id f = f
  comp_id : ∀ {X Y : Obj} (f : Hom X Y), comp f id = f
  tensorObj : Obj → Obj → Obj
  tensorHom :
    {X1 X2 Y1 Y2 : Obj} →
      Hom X1 Y1 → Hom X2 Y2 → Hom (tensorObj X1 X2) (tensorObj Y1 Y2)
  tensor_id : ∀ X Y : Obj, tensorHom (id (X := X)) (id (X := Y)) = id
  tensor_comp :
    ∀ {A1 A2 B1 B2 C1 C2 : Obj}
      (f1 : Hom A1 B1) (g1 : Hom B1 C1)
      (f2 : Hom A2 B2) (g2 : Hom B2 C2),
      tensorHom (comp f1 g1) (comp f2 g2) = comp (tensorHom f1 f2) (tensorHom g1 g2)

infixr:10 " ⟶ " => TensorCategoryLike.Hom
infixr:80 " ≫ " => TensorCategoryLike.comp

structure FiberFunctorLike (C : Type u) [TensorCategoryLike C] where
  FObj : C → Type w
  map : {X Y : C} → (X ⟶ Y) → FObj X → FObj Y
  map_id : ∀ (X : C) (x : FObj X), map (TensorCategoryLike.id (X := X)) x = x
  map_comp :
    ∀ {X Y Z : C} (f : X ⟶ Y) (g : Y ⟶ Z) (x : FObj X),
      map (f ≫ g) x = map g (map f x)
  reflects_eq :
    ∀ {X Y : C} (f g : X ⟶ Y), (∀ x : FObj X, map f x = map g x) → f = g
  realizes :
    ∀ {X Y : C} (t : FObj X → FObj Y),
      ∃ f : X ⟶ Y, ∀ x : FObj X, map f x = t x

def EndCoalgebraLike {C : Type u} [TensorCategoryLike C]
    (ω : FiberFunctorLike C) : Prop :=
  ∀ {X Y : C} (f g : X ⟶ Y),
    (∀ x : ω.FObj X, ω.map f x = ω.map g x) → f = g

def RepresentationLike {C : Type u} [TensorCategoryLike C]
    (ω : FiberFunctorLike C) : Prop :=
  ∀ {X Y : C} (t : ω.FObj X → ω.FObj Y),
    ∃ f : X ⟶ Y, ∀ x : ω.FObj X, ω.map f x = t x

def ReconstructionLike {C : Type u} [TensorCategoryLike C]
    (ω : FiberFunctorLike C) : Prop :=
  EndCoalgebraLike ω ∧ RepresentationLike ω ∧ (∀ X : C, ∃ x : ω.FObj X, True)

def ComparisonLike {C : Type u} [TensorCategoryLike C]
    (ω : FiberFunctorLike C) : Prop :=
  ReconstructionLike ω ∧
    (∀ {X Y : C} (f g : X ⟶ Y),
      (∀ x : ω.FObj X, ω.map f x = ω.map g x) ↔ f = g)

theorem fiber_reflects_iso {C : Type u} [TensorCategoryLike C]
    (ω : FiberFunctorLike C) (hCmp : ComparisonLike ω)
    {X Y : C} (f g : X ⟶ Y)
    (hpt : ∀ x : ω.FObj X, ω.map f x = ω.map g x) :
    f = g := by
  have hiff : (∀ x : ω.FObj X, ω.map f x = ω.map g x) ↔ f = g := hCmp.2 f g
  have hForward : (∀ x : ω.FObj X, ω.map f x = ω.map g x) → f = g := hiff.1
  have hEq : f = g := hForward hpt
  have hRecFaithful : EndCoalgebraLike ω := hCmp.1.1
  have hEq' : f = g := hRecFaithful f g hpt
  have _ : f = g := hEq'
  exact hEq

theorem reconstruction_faithful {C : Type u} [TensorCategoryLike C]
    (ω : FiberFunctorLike C)
    (hRec : ReconstructionLike ω) :
    EndCoalgebraLike ω := by
  have hRep : RepresentationLike ω := hRec.2.1
  have hNonempty : ∀ X : C, ∃ x : ω.FObj X, True := hRec.2.2
  have _ : RepresentationLike ω := hRep
  have _ : ∀ X : C, ∃ x : ω.FObj X, True := hNonempty
  intro X Y f g hfg
  exact hRec.1 f g hfg

theorem reconstruction_full {C : Type u} [TensorCategoryLike C]
    (ω : FiberFunctorLike C)
    (hRec : ReconstructionLike ω)
    {X Y : C} (t : ω.FObj X → ω.FObj Y) :
    ∃ f : X ⟶ Y, ∀ x : ω.FObj X, ω.map f x = t x := by
  have hRep : RepresentationLike ω := hRec.2.1
  have hWitness : ∃ f : X ⟶ Y, ∀ x : ω.FObj X, ω.map f x = t x := hRep t
  rcases hWitness with ⟨f, hf⟩
  have _ : ∀ x : ω.FObj X, ω.map f x = t x := hf
  exact ⟨f, hf⟩

theorem tannaka_unit_like {C : Type u} [TensorCategoryLike C]
    (ω : FiberFunctorLike C)
    (hCmp : ComparisonLike ω)
    {X Y : C} (f : X ⟶ Y) :
    ∃ u : X ⟶ Y, (∀ x : ω.FObj X, ω.map u x = ω.map f x) ∧ u = f := by
  have hRec : ReconstructionLike ω := hCmp.1
  have hFull : ∃ u : X ⟶ Y, ∀ x : ω.FObj X, ω.map u x = ω.map f x :=
    reconstruction_full ω hRec (fun x => ω.map f x)
  rcases hFull with ⟨u, hu⟩
  have hiff : (∀ x : ω.FObj X, ω.map u x = ω.map f x) ↔ u = f := hCmp.2 u f
  have huEq : u = f := hiff.1 hu
  have huMap : ∀ x : ω.FObj X, ω.map u x = ω.map f x := hu
  exact ⟨u, huMap, huEq⟩

theorem tannaka_counit_like {C : Type u} [TensorCategoryLike C]
    (ω : FiberFunctorLike C)
    (hCmp : ComparisonLike ω)
    {X Y : C} (f : X ⟶ Y) :
    ∃ v : X ⟶ Y, v = f ∧ (∀ x : ω.FObj X, ω.map f x = ω.map v x) := by
  rcases tannaka_unit_like ω hCmp f with ⟨u, huMap, huEq⟩
  refine ⟨u, huEq, ?_⟩
  intro x
  have hstep : ω.map u x = ω.map f x := huMap x
  calc
    ω.map f x = ω.map u x := by symm; exact hstep
    _ = ω.map u x := rfl

theorem tannaka_equivalence_like {C : Type u} [TensorCategoryLike C]
    (ω : FiberFunctorLike C)
    (hCmp : ComparisonLike ω) :
    (∀ {X Y : C} (f : X ⟶ Y), ∃ v : X ⟶ Y, v = f) ∧
      (∀ {X Y : C} (f g : X ⟶ Y),
        (∀ x : ω.FObj X, ω.map f x = ω.map g x) ↔ f = g) := by
  refine ⟨?_, ?_⟩
  · intro X Y f
    rcases tannaka_counit_like ω hCmp f with ⟨v, hvEq, hvMap⟩
    have _ : ∀ x : ω.FObj X, ω.map f x = ω.map v x := hvMap
    exact ⟨v, hvEq⟩
  · intro X Y f g
    have hiff : (∀ x : ω.FObj X, ω.map f x = ω.map g x) ↔ f = g := hCmp.2 f g
    exact hiff
