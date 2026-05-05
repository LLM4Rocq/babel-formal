/-
BENCHMARK_ID: TINY_MATHLIB_BATCH04_CATEGORY_KAN_EXTENSION_PASTING_LIKE
PAIR_STEM: category_kan_extension_pasting_like
MATH_DOMAIN: Category Theory
SOURCE_MATHLIB: Mathlib/CategoryTheory/KanExtension
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

structure NatTransLike {C : Type u} {D : Type w} [CategoryLike C] [CategoryLike D]
    (F G : FunctorLike C D) where
  app : ∀ X : C, F.obj X ⟶ G.obj X
  naturality : ∀ {X Y : C} (f : X ⟶ Y), app X ≫ G.map f = F.map f ≫ app Y

def LanLike {C : Type u} {D : Type w} [CategoryLike C] [CategoryLike D]
    (J F L : FunctorLike C D) (η : NatTransLike F L) : Prop :=
  ∀ (X : FunctorLike C D) (τ : NatTransLike F X),
    ∃ σ : NatTransLike L X,
      (∀ Z : C, η.app Z ≫ σ.app Z = τ.app Z) ∧
      (∀ ψ : NatTransLike L X, (∀ Z : C, η.app Z ≫ ψ.app Z = τ.app Z) → ψ = σ)

def RanLike {C : Type u} {D : Type w} [CategoryLike C] [CategoryLike D]
    (J F R : FunctorLike C D) (ρ : NatTransLike R F) : Prop :=
  ∀ (X : FunctorLike C D) (τ : NatTransLike X F),
    ∃ σ : NatTransLike X R,
      (∀ Z : C, σ.app Z ≫ ρ.app Z = τ.app Z) ∧
      (∀ ψ : NatTransLike X R, (∀ Z : C, ψ.app Z ≫ ρ.app Z = τ.app Z) → ψ = σ)

def WhiskerLike {C : Type u} {D : Type w} [CategoryLike C] [CategoryLike D]
    {F G H : FunctorLike C D} (α : NatTransLike F G) (β : NatTransLike G H) :
    NatTransLike F H where
  app X := α.app X ≫ β.app X
  naturality {X} {Y} f := by
    calc
      (α.app X ≫ β.app X) ≫ H.map f = α.app X ≫ (β.app X ≫ H.map f) :=
        CategoryLike.comp_assoc _ _ _
      _ = α.app X ≫ (G.map f ≫ β.app Y) := by
        rw [β.naturality f]
      _ = (α.app X ≫ G.map f) ≫ β.app Y := by
        symm
        exact CategoryLike.comp_assoc _ _ _
      _ = (F.map f ≫ α.app Y) ≫ β.app Y := by
        rw [α.naturality f]
      _ = F.map f ≫ (α.app Y ≫ β.app Y) :=
        CategoryLike.comp_assoc _ _ _

theorem lan_universal_factor {C : Type u} {D : Type w} [CategoryLike C] [CategoryLike D]
    (J F L : FunctorLike C D) (η : NatTransLike F L)
    (hLan : LanLike J F L η) (X : FunctorLike C D) (τ : NatTransLike F X) :
    ∃ σ : NatTransLike L X, ∀ Z : C, η.app Z ≫ σ.app Z = τ.app Z := by
  rcases hLan X τ with ⟨σ, hσ, huniq⟩
  have _ : ∀ Z : C, η.app Z ≫ σ.app Z = τ.app Z := hσ
  have _ : ∀ ψ : NatTransLike L X,
      (∀ Z : C, η.app Z ≫ ψ.app Z = τ.app Z) → ψ = σ := huniq
  exact ⟨σ, hσ⟩

theorem lan_universal_unique {C : Type u} {D : Type w} [CategoryLike C] [CategoryLike D]
    (J F L : FunctorLike C D) (η : NatTransLike F L)
    (hLan : LanLike J F L η) (X : FunctorLike C D) (τ : NatTransLike F X)
    (σ₁ σ₂ : NatTransLike L X)
    (hσ₁ : ∀ Z : C, η.app Z ≫ σ₁.app Z = τ.app Z)
    (hσ₂ : ∀ Z : C, η.app Z ≫ σ₂.app Z = τ.app Z) :
    σ₁ = σ₂ := by
  rcases hLan X τ with ⟨σ, hσ, huniq⟩
  have hs₁ : σ₁ = σ := huniq σ₁ hσ₁
  have hs₂ : σ₂ = σ := huniq σ₂ hσ₂
  calc
    σ₁ = σ := hs₁
    _ = σ₂ := by
      symm
      exact hs₂

theorem ran_universal_factor {C : Type u} {D : Type w} [CategoryLike C] [CategoryLike D]
    (J F R : FunctorLike C D) (ρ : NatTransLike R F)
    (hRan : RanLike J F R ρ) (X : FunctorLike C D) (τ : NatTransLike X F) :
    ∃ σ : NatTransLike X R, ∀ Z : C, σ.app Z ≫ ρ.app Z = τ.app Z := by
  rcases hRan X τ with ⟨σ, hσ, huniq⟩
  have _ : ∀ Z : C, σ.app Z ≫ ρ.app Z = τ.app Z := hσ
  have _ : ∀ ψ : NatTransLike X R,
      (∀ Z : C, ψ.app Z ≫ ρ.app Z = τ.app Z) → ψ = σ := huniq
  exact ⟨σ, hσ⟩

theorem ran_universal_unique {C : Type u} {D : Type w} [CategoryLike C] [CategoryLike D]
    (J F R : FunctorLike C D) (ρ : NatTransLike R F)
    (hRan : RanLike J F R ρ) (X : FunctorLike C D) (τ : NatTransLike X F)
    (σ₁ σ₂ : NatTransLike X R)
    (hσ₁ : ∀ Z : C, σ₁.app Z ≫ ρ.app Z = τ.app Z)
    (hσ₂ : ∀ Z : C, σ₂.app Z ≫ ρ.app Z = τ.app Z) :
    σ₁ = σ₂ := by
  rcases hRan X τ with ⟨σ, hσ, huniq⟩
  have hs₁ : σ₁ = σ := huniq σ₁ hσ₁
  have hs₂ : σ₂ = σ := huniq σ₂ hσ₂
  calc
    σ₁ = σ := hs₁
    _ = σ₂ := by
      symm
      exact hs₂

theorem lan_pasting_like {C : Type u} {D : Type w} [CategoryLike C] [CategoryLike D]
    (J F L M : FunctorLike C D)
    (η : NatTransLike F L) (θ : NatTransLike L M)
    (hLan₁ : LanLike J F L η) (hLan₂ : LanLike J L M θ)
    (X : FunctorLike C D) (τ : NatTransLike F X) :
    ∃ σ : NatTransLike M X,
      ∀ Z : C, (WhiskerLike η θ).app Z ≫ σ.app Z = τ.app Z := by
  rcases hLan₁ X τ with ⟨μ, hμ, hμuniq⟩
  rcases hLan₂ X μ with ⟨σ, hσ, hσuniq⟩
  have hcompose : ∀ Z : C, (WhiskerLike η θ).app Z ≫ σ.app Z = τ.app Z := by
    intro Z
    calc
      (WhiskerLike η θ).app Z ≫ σ.app Z = (η.app Z ≫ θ.app Z) ≫ σ.app Z := by
        rfl
      _ = η.app Z ≫ (θ.app Z ≫ σ.app Z) :=
        CategoryLike.comp_assoc _ _ _
      _ = η.app Z ≫ μ.app Z := by
        rw [hσ Z]
      _ = τ.app Z := hμ Z
  have _ : ∀ Z : C, θ.app Z ≫ σ.app Z = μ.app Z := hσ
  have _ : ∀ ν : NatTransLike M X,
      (∀ Z : C, θ.app Z ≫ ν.app Z = μ.app Z) → ν = σ := hσuniq
  have _ : ∀ ν : NatTransLike L X,
      (∀ Z : C, η.app Z ≫ ν.app Z = τ.app Z) → ν = μ := hμuniq
  exact ⟨σ, hcompose⟩

theorem ran_pasting_like {C : Type u} {D : Type w} [CategoryLike C] [CategoryLike D]
    (J F R S : FunctorLike C D)
    (ρ : NatTransLike R F) (θ : NatTransLike S R)
    (hRan₁ : RanLike J F R ρ) (hRan₂ : RanLike J R S θ)
    (X : FunctorLike C D) (τ : NatTransLike X F) :
    ∃ σ : NatTransLike X S,
      ∀ Z : C, σ.app Z ≫ (WhiskerLike θ ρ).app Z = τ.app Z := by
  rcases hRan₁ X τ with ⟨μ, hμ, hμuniq⟩
  rcases hRan₂ X μ with ⟨σ, hσ, hσuniq⟩
  have hcompose : ∀ Z : C, σ.app Z ≫ (WhiskerLike θ ρ).app Z = τ.app Z := by
    intro Z
    calc
      σ.app Z ≫ (WhiskerLike θ ρ).app Z = σ.app Z ≫ (θ.app Z ≫ ρ.app Z) := by
        rfl
      _ = (σ.app Z ≫ θ.app Z) ≫ ρ.app Z := by
        symm
        exact CategoryLike.comp_assoc _ _ _
      _ = μ.app Z ≫ ρ.app Z := by
        rw [hσ Z]
      _ = τ.app Z := hμ Z
  have _ : ∀ Z : C, σ.app Z ≫ θ.app Z = μ.app Z := hσ
  have _ : ∀ ν : NatTransLike X S,
      (∀ Z : C, ν.app Z ≫ θ.app Z = μ.app Z) → ν = σ := hσuniq
  have _ : ∀ ν : NatTransLike X R,
      (∀ Z : C, ν.app Z ≫ ρ.app Z = τ.app Z) → ν = μ := hμuniq
  exact ⟨σ, hcompose⟩
