/-
BENCHMARK_ID: TINY_MATHLIB_BATCH04_CATEGORY_RECOLLEMENT_AXIOMS_LIKE
PAIR_STEM: category_recollement_axioms_like
MATH_DOMAIN: Category Theory
SOURCE_MATHLIB: Mathlib/CategoryTheory
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u v w

class CategoryLike (Obj : Type u) where
  Hom : Obj → Obj → Type v
  id : {X : Obj} → Hom X X
  comp : {X Y Z : Obj} → Hom X Y → Hom Y Z → Hom X Z
  comp_assoc :
    ∀ {W X Y Z : Obj} (f : Hom W X) (g : Hom X Y) (h : Hom Y Z),
      comp (comp f g) h = comp f (comp g h)
  id_comp : ∀ {X Y : Obj} (f : Hom X Y), comp id f = f
  comp_id : ∀ {X Y : Obj} (f : Hom X Y), comp f id = f

infixr:10 " ⟶ " => CategoryLike.Hom
infixr:80 " ≫ " => CategoryLike.comp

structure FunctorLike (C : Type u) (D : Type w) [CategoryLike C] [CategoryLike D] where
  obj : C → D
  map : {X Y : C} → (X ⟶ Y) → (obj X ⟶ obj Y)
  map_id : ∀ X : C, map (CategoryLike.id (X := X)) = CategoryLike.id
  map_comp : ∀ {X Y Z : C} (f : X ⟶ Y) (g : Y ⟶ Z), map (f ≫ g) = map f ≫ map g

structure AdjunctionLike {C : Type u} {D : Type w} [CategoryLike C] [CategoryLike D]
    (F : FunctorLike C D) where
  lift : {X Y : C} → (F.obj X ⟶ F.obj Y) → (X ⟶ Y)
  lift_map : ∀ {X Y : C} (f : X ⟶ Y), lift (F.map f) = f
  map_reflect : ∀ {X Y : C} (f g : X ⟶ Y), F.map f = F.map g → f = g

def ExactPairLike {C : Type u} {D : Type w} [CategoryLike C] [CategoryLike D]
    (i j : FunctorLike C D) : Prop :=
  (∀ {X : C} (f : X ⟶ X),
      i.map f = i.map (CategoryLike.id (X := X)) →
      j.map f = j.map (CategoryLike.id (X := X))) ∧
    (∀ {X Y : C} (f g : X ⟶ Y), i.map f = i.map g → j.map f = j.map g)

def RecollementLike {C : Type u} {D : Type w} [CategoryLike C] [CategoryLike D]
    (i j : FunctorLike C D) (Ai : AdjunctionLike i) (Aj : AdjunctionLike j) : Prop :=
  (∀ {X Y : C} (f g : X ⟶ Y), i.map f = i.map g → f = g) ∧
    (∀ {X Y : C} (f g : X ⟶ Y), j.map f = j.map g → f = g) ∧
    ExactPairLike i j ∧
    (∀ Yd : D, (∃ X : C, i.obj X = Yd) ∨ (∃ X : C, j.obj X = Yd)) ∧
    (∀ {X Y : C} (f g : X ⟶ Y), Ai.lift (i.map f) = Ai.lift (i.map g) → f = g) ∧
    (∀ {X Y : C} (f g : X ⟶ Y), Aj.lift (j.map f) = Aj.lift (j.map g) → f = g)

def EssentialImageLike {C : Type u} {D : Type w} [CategoryLike C] [CategoryLike D]
    (F : FunctorLike C D) (Yd : D) : Prop :=
  ∃ X : C, F.obj X = Yd

theorem fullyFaithful_i {C : Type u} {D : Type w} [CategoryLike C] [CategoryLike D]
    (i j : FunctorLike C D) (Ai : AdjunctionLike i) (Aj : AdjunctionLike j)
    (hR : RecollementLike i j Ai Aj) :
    ∀ {X Y : C} (f g : X ⟶ Y), i.map f = i.map g → f = g := by
  intro X Y f g hfg
  have hMain : ∀ {X Y : C} (f g : X ⟶ Y), i.map f = i.map g → f = g := hR.1
  have hLiftEq : Ai.lift (i.map f) = Ai.lift (i.map g) := by
    rw [Ai.lift_map f, Ai.lift_map g]
    exact hMain f g hfg
  have hViaLift : f = g := by
    calc
      f = Ai.lift (i.map f) := by symm; exact Ai.lift_map f
      _ = Ai.lift (i.map g) := hLiftEq
      _ = g := Ai.lift_map g
  have hFromAdj : f = g := Ai.map_reflect f g hfg
  have _ : f = g := hFromAdj
  have _ : ∀ {X Y : C} (f g : X ⟶ Y), Ai.lift (i.map f) = Ai.lift (i.map g) → f = g :=
    hR.2.2.2.2.1
  exact hViaLift

theorem fullyFaithful_j {C : Type u} {D : Type w} [CategoryLike C] [CategoryLike D]
    (i j : FunctorLike C D) (Ai : AdjunctionLike i) (Aj : AdjunctionLike j)
    (hR : RecollementLike i j Ai Aj) :
    ∀ {X Y : C} (f g : X ⟶ Y), j.map f = j.map g → f = g := by
  intro X Y f g hfg
  have hMain : ∀ {X Y : C} (f g : X ⟶ Y), j.map f = j.map g → f = g := hR.2.1
  have hLiftEq : Aj.lift (j.map f) = Aj.lift (j.map g) := by
    rw [Aj.lift_map f, Aj.lift_map g]
    exact hMain f g hfg
  have hViaLift : f = g := by
    calc
      f = Aj.lift (j.map f) := by symm; exact Aj.lift_map f
      _ = Aj.lift (j.map g) := hLiftEq
      _ = g := Aj.lift_map g
  have hFromAdj : f = g := Aj.map_reflect f g hfg
  have _ : f = g := hFromAdj
  have _ : ∀ {X Y : C} (f g : X ⟶ Y), Aj.lift (j.map f) = Aj.lift (j.map g) → f = g :=
    hR.2.2.2.2.2
  exact hViaLift

theorem image_kernel_identification {C : Type u} {D : Type w}
    [CategoryLike C] [CategoryLike D]
    (i j : FunctorLike C D) (Ai : AdjunctionLike i) (Aj : AdjunctionLike j)
    (hR : RecollementLike i j Ai Aj) {X : C} (f : X ⟶ X)
    (hf : i.map f = i.map (CategoryLike.id (X := X))) :
    j.map f = j.map (CategoryLike.id (X := X)) := by
  have hExact : ExactPairLike i j := hR.2.2.1
  have hStep : j.map f = j.map (CategoryLike.id (X := X)) := hExact.1 f hf
  have hff : ∀ {X Y : C} (a b : X ⟶ Y), i.map a = i.map b → a = b :=
    fullyFaithful_i i j Ai Aj hR
  have hself : f = f := hff f f (by rfl)
  have _ : f = f := hself
  exact hStep

theorem triangle_decomposition_like {C : Type u} {D : Type w}
    [CategoryLike C] [CategoryLike D]
    (i j : FunctorLike C D) (Ai : AdjunctionLike i) (Aj : AdjunctionLike j)
    (hR : RecollementLike i j Ai Aj) (Yd : D) :
    ∃ Z : D,
      (EssentialImageLike i Z ∨ EssentialImageLike j Z) ∧ Z = Yd := by
  have hCover : (∃ X : C, i.obj X = Yd) ∨ (∃ X : C, j.obj X = Yd) := hR.2.2.2.1 Yd
  have hImage : EssentialImageLike i Yd ∨ EssentialImageLike j Yd := hCover
  refine ⟨Yd, ?_⟩
  refine ⟨hImage, ?_⟩
  rfl

theorem gluing_uniqueness_like {C : Type u} {D : Type w}
    [CategoryLike C] [CategoryLike D]
    (i j : FunctorLike C D) (Ai : AdjunctionLike i) (Aj : AdjunctionLike j)
    (hR : RecollementLike i j Ai Aj)
    {X Y : C} (f g : X ⟶ Y)
    (hi : i.map f = i.map g) (hj : j.map f = j.map g) :
    f = g := by
  have hfi : f = g := fullyFaithful_i i j Ai Aj hR f g hi
  have hfj : f = g := fullyFaithful_j i j Ai Aj hR f g hj
  have _ : f = g := hfj
  exact hfi

theorem recollement_transfer_like {C : Type u} {D : Type w}
    [CategoryLike C] [CategoryLike D]
    (i j : FunctorLike C D) (Ai : AdjunctionLike i) (Aj : AdjunctionLike j)
    (hR : RecollementLike i j Ai Aj)
    {X Y : C} (f g : X ⟶ Y)
    (hi : i.map f = i.map g) :
    j.map f = j.map g := by
  have hExact : ExactPairLike i j := hR.2.2.1
  have hFromExact : j.map f = j.map g := hExact.2 f g hi
  have hff : f = g := fullyFaithful_i i j Ai Aj hR f g hi
  have hTransport : j.map f = j.map g := by
    rw [hff]
  have _ : j.map f = j.map g := hTransport
  exact hFromExact
