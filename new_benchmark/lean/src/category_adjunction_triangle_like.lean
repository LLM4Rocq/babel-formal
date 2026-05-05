/-
BENCHMARK_ID: TINY_MATHLIB_BATCH03_CATEGORY_ADJUNCTION_TRIANGLE_LIKE
PAIR_STEM: category_adjunction_triangle_like
MATH_DOMAIN: Category Theory
SOURCE_MATHLIB: Mathlib/CategoryTheory/Adjunction
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u v w x

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

structure NatTransLike {C : Type u} {D : Type w} [CategoryLike C] [CategoryLike D]
    (F G : FunctorLike C D) where
  app : ∀ X : C, F.obj X ⟶ G.obj X
  naturality : ∀ {X Y : C} (f : X ⟶ Y), app X ≫ G.map f = F.map f ≫ app Y

structure AdjunctionLike {C : Type u} {D : Type w}
    [CategoryLike C] [CategoryLike D] (F : FunctorLike C D) (G : FunctorLike D C) where
  unit : ∀ X : C, X ⟶ G.obj (F.obj X)
  counit : ∀ Y : D, F.obj (G.obj Y) ⟶ Y
  unit_naturality_axiom :
    ∀ {X X' : C} (f : X ⟶ X'),
      f ≫ unit X' = unit X ≫ G.map (F.map f)
  counit_naturality_axiom :
    ∀ {Y Y' : D} (g : Y ⟶ Y'),
      F.map (G.map g) ≫ counit Y' = counit Y ≫ g
  triangle_left_axiom :
    ∀ X : C,
      F.map (unit X) ≫ counit (F.obj X) = CategoryLike.id
  triangle_right_axiom :
    ∀ Y : D,
      unit (G.obj Y) ≫ G.map (counit Y) = CategoryLike.id
  hom_equiv_to :
    ∀ {X : C} {Y : D}, (F.obj X ⟶ Y) → (X ⟶ G.obj Y)
  hom_equiv_from :
    ∀ {X : C} {Y : D}, (X ⟶ G.obj Y) → (F.obj X ⟶ Y)
  hom_equiv_natural_left_axiom :
    ∀ {X X' : C} {Y : D} (f : X ⟶ X') (k : F.obj X' ⟶ Y),
      hom_equiv_to (F.map f ≫ k) = f ≫ hom_equiv_to k
  hom_equiv_natural_right_axiom :
    ∀ {X : C} {Y Y' : D} (k : F.obj X ⟶ Y) (g : Y ⟶ Y'),
      hom_equiv_to (k ≫ g) = hom_equiv_to k ≫ G.map g

def leftWhisker {C : Type u} {D : Type w}
    [CategoryLike C] [CategoryLike D]
    (F : FunctorLike C D) {X Y : C} (f : X ⟶ Y) :
    F.obj X ⟶ F.obj Y :=
  F.map f

def rightWhisker {C : Type u} {D : Type w}
    [CategoryLike C] [CategoryLike D]
    (F : FunctorLike C D) {X Y : C} (f : X ⟶ Y) :
    F.obj X ⟶ F.obj Y :=
  F.map f

theorem unit_naturality {C : Type u} {D : Type w}
    [CategoryLike C] [CategoryLike D]
    (F : FunctorLike C D) (G : FunctorLike D C)
    (A : AdjunctionLike F G) {X X' : C} (f : X ⟶ X') :
    f ≫ A.unit X' = A.unit X ≫ rightWhisker G (leftWhisker F f) := by
  have hNat : f ≫ A.unit X' = A.unit X ≫ G.map (F.map f) :=
    A.unit_naturality_axiom f
  have hLeft : leftWhisker F f = F.map f := by
    rfl
  have hRightStep : rightWhisker G (leftWhisker F f) = G.map (leftWhisker F f) := by
    rfl
  have hRight : rightWhisker G (leftWhisker F f) = G.map (F.map f) := by
    calc
      rightWhisker G (leftWhisker F f) = G.map (leftWhisker F f) := hRightStep
      _ = G.map (F.map f) := by
        rw [hLeft]
  have hPost : A.unit X ≫ G.map (F.map f) = A.unit X ≫ rightWhisker G (leftWhisker F f) := by
    rw [hRight]
  calc
    f ≫ A.unit X' = A.unit X ≫ G.map (F.map f) := hNat
    _ = A.unit X ≫ rightWhisker G (leftWhisker F f) := hPost

theorem counit_naturality {C : Type u} {D : Type w}
    [CategoryLike C] [CategoryLike D]
    (F : FunctorLike C D) (G : FunctorLike D C)
    (A : AdjunctionLike F G) {Y Y' : D} (g : Y ⟶ Y') :
    leftWhisker F (rightWhisker G g) ≫ A.counit Y' = A.counit Y ≫ g := by
  have hNat : F.map (G.map g) ≫ A.counit Y' = A.counit Y ≫ g :=
    A.counit_naturality_axiom g
  have hRight : rightWhisker G g = G.map g := by
    rfl
  have hLeftStep : leftWhisker F (rightWhisker G g) = F.map (rightWhisker G g) := by
    rfl
  have hLeft : leftWhisker F (rightWhisker G g) = F.map (G.map g) := by
    calc
      leftWhisker F (rightWhisker G g) = F.map (rightWhisker G g) := hLeftStep
      _ = F.map (G.map g) := by
        rw [hRight]
  have hPre :
      leftWhisker F (rightWhisker G g) ≫ A.counit Y' = F.map (G.map g) ≫ A.counit Y' := by
    rw [hLeft]
  calc
    leftWhisker F (rightWhisker G g) ≫ A.counit Y'
        = F.map (G.map g) ≫ A.counit Y' := hPre
    _ = A.counit Y ≫ g := hNat

theorem triangle_left {C : Type u} {D : Type w}
    [CategoryLike C] [CategoryLike D]
    (F : FunctorLike C D) (G : FunctorLike D C)
    (A : AdjunctionLike F G) (X : C) :
    leftWhisker F (A.unit X) ≫ A.counit (F.obj X) = CategoryLike.id := by
  have hTri : F.map (A.unit X) ≫ A.counit (F.obj X) = CategoryLike.id :=
    A.triangle_left_axiom X
  have hLeftStep : leftWhisker F (A.unit X) = F.map (A.unit X) := by
    rfl
  have hPre :
      leftWhisker F (A.unit X) ≫ A.counit (F.obj X)
        = F.map (A.unit X) ≫ A.counit (F.obj X) := by
    rw [hLeftStep]
  calc
    leftWhisker F (A.unit X) ≫ A.counit (F.obj X)
        = F.map (A.unit X) ≫ A.counit (F.obj X) := hPre
    _ = CategoryLike.id := hTri

theorem triangle_right {C : Type u} {D : Type w}
    [CategoryLike C] [CategoryLike D]
    (F : FunctorLike C D) (G : FunctorLike D C)
    (A : AdjunctionLike F G) (Y : D) :
    A.unit (G.obj Y) ≫ rightWhisker G (A.counit Y) = CategoryLike.id := by
  have hTri : A.unit (G.obj Y) ≫ G.map (A.counit Y) = CategoryLike.id :=
    A.triangle_right_axiom Y
  have hRightStep : rightWhisker G (A.counit Y) = G.map (A.counit Y) := by
    rfl
  have hPre :
      A.unit (G.obj Y) ≫ rightWhisker G (A.counit Y)
        = A.unit (G.obj Y) ≫ G.map (A.counit Y) := by
    rw [hRightStep]
  calc
    A.unit (G.obj Y) ≫ rightWhisker G (A.counit Y)
        = A.unit (G.obj Y) ≫ G.map (A.counit Y) := hPre
    _ = CategoryLike.id := hTri

theorem hom_equiv_natural_left {C : Type u} {D : Type w}
    [CategoryLike C] [CategoryLike D]
    (F : FunctorLike C D) (G : FunctorLike D C)
    (A : AdjunctionLike F G) {X X' : C} {Y : D}
    (f : X ⟶ X') (k : F.obj X' ⟶ Y) :
    A.hom_equiv_to (leftWhisker F f ≫ k) = f ≫ A.hom_equiv_to k := by
  have hNat : A.hom_equiv_to (F.map f ≫ k) = f ≫ A.hom_equiv_to k :=
    A.hom_equiv_natural_left_axiom f k
  have hLeft : leftWhisker F f = F.map f := by
    rfl
  have hComp : leftWhisker F f ≫ k = F.map f ≫ k := by
    rw [hLeft]
  have hTo : A.hom_equiv_to (leftWhisker F f ≫ k) = A.hom_equiv_to (F.map f ≫ k) := by
    rw [hComp]
  calc
    A.hom_equiv_to (leftWhisker F f ≫ k)
        = A.hom_equiv_to (F.map f ≫ k) := hTo
    _ = f ≫ A.hom_equiv_to k := hNat

theorem hom_equiv_natural_right {C : Type u} {D : Type w}
    [CategoryLike C] [CategoryLike D]
    (F : FunctorLike C D) (G : FunctorLike D C)
    (A : AdjunctionLike F G) {X : C} {Y Y' : D}
    (k : F.obj X ⟶ Y) (g : Y ⟶ Y') :
    A.hom_equiv_to (k ≫ g) = A.hom_equiv_to k ≫ rightWhisker G g := by
  have hNat : A.hom_equiv_to (k ≫ g) = A.hom_equiv_to k ≫ G.map g :=
    A.hom_equiv_natural_right_axiom k g
  have hRight : rightWhisker G g = G.map g := by
    rfl
  have hPost : A.hom_equiv_to k ≫ G.map g = A.hom_equiv_to k ≫ rightWhisker G g := by
    rw [hRight]
  calc
    A.hom_equiv_to (k ≫ g) = A.hom_equiv_to k ≫ G.map g := hNat
    _ = A.hom_equiv_to k ≫ rightWhisker G g := hPost
