/-
BENCHMARK_ID: TINY_MATHLIB_BATCH03_CATEGORY_YONEDA_EMBEDDING_LIKE
PAIR_STEM: category_yoneda_embedding_like
MATH_DOMAIN: Category Theory
SOURCE_MATHLIB: Mathlib/CategoryTheory/Yoneda
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u v

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

def HomFunctorLike {C : Type u} [CategoryLike C] (A : C) : C → Type v :=
  fun X => A ⟶ X

def YonedaObjLike {C : Type u} [CategoryLike C] (A : C) : C → Type v :=
  HomFunctorLike A

def YonedaMapLike {C : Type u} [CategoryLike C] {A B : C}
    (f : A ⟶ B) :
    ∀ X : C, HomFunctorLike B X → HomFunctorLike A X :=
  fun _ u => f ≫ u

theorem yoneda_map_id {C : Type u} [CategoryLike C] (A : C) :
    ∀ X : C, ∀ u : HomFunctorLike A X,
      YonedaMapLike (CategoryLike.id (X := A)) X u = u := by
  intro X u
  have hDef : YonedaMapLike (CategoryLike.id (X := A)) X u = CategoryLike.id (X := A) ≫ u := by
    rfl
  have hId : CategoryLike.id (X := A) ≫ u = u :=
    CategoryLike.id_comp u
  exact Eq.trans hDef hId

theorem yoneda_map_comp {C : Type u} [CategoryLike C]
    {A B D : C} (f : A ⟶ B) (g : B ⟶ D) :
    ∀ X : C, ∀ u : HomFunctorLike D X,
      YonedaMapLike (f ≫ g) X u = YonedaMapLike f X (YonedaMapLike g X u) := by
  intro X u
  have hLeft :
      YonedaMapLike (f ≫ g) X u = (f ≫ g) ≫ u := by
    rfl
  have hInner :
      YonedaMapLike g X u = g ≫ u := by
    rfl
  have hRight :
      YonedaMapLike f X (YonedaMapLike g X u) = f ≫ (YonedaMapLike g X u) := by
    rfl
  have hAssoc : (f ≫ g) ≫ u = f ≫ (g ≫ u) :=
    CategoryLike.comp_assoc f g u
  have hStep3 : f ≫ (g ≫ u) = f ≫ (YonedaMapLike g X u) := by
    rw [hInner]
  have hStep4 : f ≫ (YonedaMapLike g X u) = YonedaMapLike f X (YonedaMapLike g X u) := by
    symm
    exact hRight
  exact Eq.trans hLeft (Eq.trans hAssoc (Eq.trans hStep3 hStep4))

def YonedaFaithfulLike {C : Type u} [CategoryLike C] : Prop :=
  ∀ {A B : C} (f g : A ⟶ B),
    (∀ X : C, YonedaMapLike f X = YonedaMapLike g X) → f = g

theorem yoneda_faithful_cancel {C : Type u} [CategoryLike C]
    (hFaithful : YonedaFaithfulLike (C := C)) {A B : C}
    (f g : A ⟶ B)
    (hEq : ∀ X : C, YonedaMapLike f X = YonedaMapLike g X) :
    f = g := by
  have hPointwise : ∀ X : C, YonedaMapLike f X = YonedaMapLike g X := by
    intro X
    exact hEq X
  have hAtB : YonedaMapLike f B = YonedaMapLike g B := hPointwise B
  have hAtId :
      YonedaMapLike f B (CategoryLike.id (X := B))
        = YonedaMapLike g B (CategoryLike.id (X := B)) := by
    exact congrArg (fun q => q (CategoryLike.id (X := B))) hAtB
  have hComp :
      f ≫ CategoryLike.id (X := B) = g ≫ CategoryLike.id (X := B) := by
    simpa [YonedaMapLike] using hAtId
  have hById : f = g := by
    calc
      f = f ≫ CategoryLike.id (X := B) := by
        symm
        exact CategoryLike.comp_id f
      _ = g ≫ CategoryLike.id (X := B) := hComp
      _ = g := by
        exact CategoryLike.comp_id g
  have hByFaithful : f = g := hFaithful f g hPointwise
  calc
    f = g := hByFaithful
    _ = f := by
      symm
      exact hById
    _ = g := hById

theorem yoneda_ext {C : Type u} [CategoryLike C]
    {A B : C} (f g : A ⟶ B)
    (hEq : ∀ X : C, ∀ u : HomFunctorLike B X,
      YonedaMapLike f X u = YonedaMapLike g X u) :
    f = g := by
  have hEval :
      YonedaMapLike f B (CategoryLike.id (X := B))
        = YonedaMapLike g B (CategoryLike.id (X := B)) :=
    hEq B (CategoryLike.id (X := B))
  have hAtId : f ≫ CategoryLike.id (X := B) = g ≫ CategoryLike.id (X := B) := by
    simpa [YonedaMapLike] using hEval
  have hLeftId : f = f ≫ CategoryLike.id (X := B) := by
    symm
    exact CategoryLike.comp_id f
  have hRightId : g ≫ CategoryLike.id (X := B) = g := by
    exact CategoryLike.comp_id g
  calc
    f = f ≫ CategoryLike.id (X := B) := hLeftId
    _ = g ≫ CategoryLike.id (X := B) := hAtId
    _ = g := hRightId

theorem yoneda_full_lift {C : Type u} [CategoryLike C]
    {A B : C}
    (τ : ∀ X : C, HomFunctorLike B X → HomFunctorLike A X)
    (hNat : ∀ {X Y : C} (k : X ⟶ Y) (u : HomFunctorLike B X),
      τ Y (u ≫ k) = τ X u ≫ k) :
    ∃ f : A ⟶ B, f = τ B (CategoryLike.id (X := B)) := by
  have hNatId :
      τ B (CategoryLike.id (X := B) ≫ CategoryLike.id (X := B))
        = τ B (CategoryLike.id (X := B)) ≫ CategoryLike.id (X := B) := by
    exact hNat (k := CategoryLike.id (X := B)) (u := CategoryLike.id (X := B))
  have hLeft :
      τ B (CategoryLike.id (X := B) ≫ CategoryLike.id (X := B))
        = τ B (CategoryLike.id (X := B)) := by
    rw [CategoryLike.id_comp (CategoryLike.id (X := B))]
  have hRight :
      τ B (CategoryLike.id (X := B)) ≫ CategoryLike.id (X := B)
        = τ B (CategoryLike.id (X := B)) := by
    exact CategoryLike.comp_id (τ B (CategoryLike.id (X := B)))
  have hStable : τ B (CategoryLike.id (X := B)) = τ B (CategoryLike.id (X := B)) := by
    have hStep1 :
        τ B (CategoryLike.id (X := B))
          = τ B (CategoryLike.id (X := B) ≫ CategoryLike.id (X := B)) := by
      symm
      exact hLeft
    exact Eq.trans hStep1 (Eq.trans hNatId hRight)
  refine ⟨τ B (CategoryLike.id (X := B)), ?_⟩
  exact hStable

theorem yoneda_full_spec {C : Type u} [CategoryLike C]
    {A B : C}
    (τ : ∀ X : C, HomFunctorLike B X → HomFunctorLike A X)
    (hNat : ∀ {X Y : C} (k : X ⟶ Y) (u : HomFunctorLike B X),
      τ Y (u ≫ k) = τ X u ≫ k)
    (f : A ⟶ B)
    (hf : f = τ B (CategoryLike.id (X := B))) :
    ∀ X : C, ∀ u : HomFunctorLike B X,
      τ X u = YonedaMapLike f X u := by
  intro X u
  have hNatStep : τ X (CategoryLike.id (X := B) ≫ u)
      = τ B (CategoryLike.id (X := B)) ≫ u :=
    hNat (k := u) (u := CategoryLike.id (X := B))
  have hId : CategoryLike.id (X := B) ≫ u = u :=
    CategoryLike.id_comp u
  have hf' : τ B (CategoryLike.id (X := B)) = f := by
    symm
    exact hf
  have hEq1 : τ X u = τ X (CategoryLike.id (X := B) ≫ u) := by
    rw [hId]
  have hEq2 : τ X u = τ B (CategoryLike.id (X := B)) ≫ u := by
    exact hEq1.trans hNatStep
  have hEq3 : τ X u = f ≫ u := by
    calc
      τ X u = τ B (CategoryLike.id (X := B)) ≫ u := hEq2
      _ = f ≫ u := by
        rw [hf']
  simpa [YonedaMapLike] using hEq3

theorem yoneda_full_unique {C : Type u} [CategoryLike C]
    {A B : C}
    (τ : ∀ X : C, HomFunctorLike B X → HomFunctorLike A X)
    (hNat : ∀ {X Y : C} (k : X ⟶ Y) (u : HomFunctorLike B X),
      τ Y (u ≫ k) = τ X u ≫ k)
    (f g : A ⟶ B)
    (hF : ∀ X : C, ∀ u : HomFunctorLike B X,
      τ X u = YonedaMapLike f X u)
    (hG : ∀ X : C, ∀ u : HomFunctorLike B X,
      τ X u = YonedaMapLike g X u) :
    f = g := by
  have hNatId :
      τ B (CategoryLike.id (X := B) ≫ CategoryLike.id (X := B))
        = τ B (CategoryLike.id (X := B)) ≫ CategoryLike.id (X := B) := by
    exact hNat (k := CategoryLike.id (X := B)) (u := CategoryLike.id (X := B))
  have hNatConsistency : τ B (CategoryLike.id (X := B)) = τ B (CategoryLike.id (X := B)) := by
    have hStep1 :
        τ B (CategoryLike.id (X := B))
          = τ B (CategoryLike.id (X := B) ≫ CategoryLike.id (X := B)) := by
      rw [CategoryLike.id_comp (CategoryLike.id (X := B))]
    have hStep3 :
        τ B (CategoryLike.id (X := B)) ≫ CategoryLike.id (X := B)
          = τ B (CategoryLike.id (X := B)) := by
      exact CategoryLike.comp_id (τ B (CategoryLike.id (X := B)))
    exact Eq.trans hStep1 (Eq.trans hNatId hStep3)
  have hFId : τ B (CategoryLike.id (X := B)) = f ≫ CategoryLike.id (X := B) := by
    simpa [YonedaMapLike] using hF B (CategoryLike.id (X := B))
  have hGId : τ B (CategoryLike.id (X := B)) = g ≫ CategoryLike.id (X := B) := by
    simpa [YonedaMapLike] using hG B (CategoryLike.id (X := B))
  have hComp : f ≫ CategoryLike.id (X := B) = g ≫ CategoryLike.id (X := B) := by
    have hComp1 : f ≫ CategoryLike.id (X := B) = τ B (CategoryLike.id (X := B)) := by
      symm
      exact hFId
    have hComp2 : τ B (CategoryLike.id (X := B)) = g ≫ CategoryLike.id (X := B) := hGId
    exact Eq.trans hComp1 (Eq.trans hNatConsistency hComp2)
  have hLeftId : f = f ≫ CategoryLike.id (X := B) := by
    symm
    exact CategoryLike.comp_id f
  have hRightId : g ≫ CategoryLike.id (X := B) = g := by
    exact CategoryLike.comp_id g
  calc
    f = f ≫ CategoryLike.id (X := B) := hLeftId
    _ = g ≫ CategoryLike.id (X := B) := hComp
    _ = g := hRightId
