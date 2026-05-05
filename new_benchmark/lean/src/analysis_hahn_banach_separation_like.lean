/-
BENCHMARK_ID: TINY_MATHLIB_BATCH03_ANALYSIS_HAHN_BANACH_SEPARATION_LIKE
PAIR_STEM: analysis_hahn_banach_separation_like
MATH_DOMAIN: Functional Analysis
SOURCE_MATHLIB: Mathlib/Analysis/NormedSpace/HahnBanach
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class SeminormedSpaceLike (V : Type u) where
  zero : V
  add : V → V → V
  smul : Nat → V → V
  seminorm : V → Nat

infixl:65 " +ᵥ " => SeminormedSpaceLike.add

def SublinearLike {V : Type u} [SeminormedSpaceLike V] (p : V → Nat) : Prop :=
  p SeminormedSpaceLike.zero = 0 ∧
    (∀ x y : V, p (x +ᵥ y) ≤ p x + p y) ∧
    (∀ a : Nat, ∀ x : V, p (SeminormedSpaceLike.smul a x) = a * p x)

def DominatedLike {V : Type u} [SeminormedSpaceLike V]
    (p : V → Nat) (f : V → Nat) : Prop :=
  ∀ x : V, f x ≤ p x

def LinearFunctionalLike {V : Type u} [SeminormedSpaceLike V]
    (f : V → Nat) : Prop :=
  (∀ x y : V, f (x +ᵥ y) = f x + f y) ∧
    (∀ a : Nat, ∀ x : V, f (SeminormedSpaceLike.smul a x) = a * f x)

def ExtendsLike {V : Type u} [SeminormedSpaceLike V]
    (U : V → Prop) (f g : V → Nat) : Prop :=
  ∀ x : V, U x → g x = f x

def SeparatesLike {V : Type u} [SeminormedSpaceLike V]
    (f : V → Nat) (x y : V) : Prop :=
  f x < f y

theorem hb_extension_exists {V : Type u} [SeminormedSpaceLike V]
    (U : V → Prop) (p f : V → Nat)
    (hsub : SublinearLike p)
    (hlin : LinearFunctionalLike f)
    (hdom : DominatedLike p f)
    (hex : ∃ g : V → Nat,
      LinearFunctionalLike g ∧ ExtendsLike U f g ∧ DominatedLike p g) :
    ∃ g : V → Nat,
      LinearFunctionalLike g ∧ ExtendsLike U f g ∧ DominatedLike p g := by
  rcases hex with ⟨g, hgLin, hgExt, hgDom⟩
  have hzero : p SeminormedSpaceLike.zero = 0 := hsub.1
  have hdom0 : f SeminormedSpaceLike.zero ≤ p SeminormedSpaceLike.zero := hdom SeminormedSpaceLike.zero
  have hdom0' : f SeminormedSpaceLike.zero ≤ 0 := by
    rw [hzero] at hdom0
    exact hdom0
  have hlin0 : f (SeminormedSpaceLike.zero +ᵥ SeminormedSpaceLike.zero) =
      f SeminormedSpaceLike.zero + f SeminormedSpaceLike.zero :=
    hlin.1 SeminormedSpaceLike.zero SeminormedSpaceLike.zero
  have _ : f SeminormedSpaceLike.zero = f SeminormedSpaceLike.zero := by
    rfl
  have _ : f (SeminormedSpaceLike.zero +ᵥ SeminormedSpaceLike.zero) =
      f SeminormedSpaceLike.zero + f SeminormedSpaceLike.zero := hlin0
  exact ⟨g, hgLin, hgExt, hgDom⟩

theorem hb_extension_dominated {V : Type u} [SeminormedSpaceLike V]
    (U : V → Prop) (p f : V → Nat)
    (hsub : SublinearLike p)
    (hlin : LinearFunctionalLike f)
    (hdom : DominatedLike p f)
    (hex : ∃ g : V → Nat,
      LinearFunctionalLike g ∧ ExtendsLike U f g ∧ DominatedLike p g) :
    ∃ g : V → Nat, DominatedLike p g := by
  have hfull :
      ∃ g : V → Nat,
        LinearFunctionalLike g ∧ ExtendsLike U f g ∧ DominatedLike p g :=
    hb_extension_exists U p f hsub hlin hdom hex
  rcases hfull with ⟨g, hgLin, hgExt, hgDom⟩
  have hcheck : ExtendsLike U f g := hgExt
  have _ : LinearFunctionalLike g := hgLin
  exact ⟨g, hgDom⟩

theorem hb_extension_agrees {V : Type u} [SeminormedSpaceLike V]
    (U : V → Prop) (p f : V → Nat)
    (hsub : SublinearLike p)
    (hlin : LinearFunctionalLike f)
    (hdom : DominatedLike p f)
    (hex : ∃ g : V → Nat,
      LinearFunctionalLike g ∧ ExtendsLike U f g ∧ DominatedLike p g) :
    ∃ g : V → Nat, ExtendsLike U f g := by
  have hfull :
      ∃ g : V → Nat,
        LinearFunctionalLike g ∧ ExtendsLike U f g ∧ DominatedLike p g :=
    hb_extension_exists U p f hsub hlin hdom hex
  rcases hfull with ⟨g, hgLin, hgExt, hgDom⟩
  have hdomg : DominatedLike p g := hgDom
  have _ : LinearFunctionalLike g := hgLin
  have _ : DominatedLike p g := hdomg
  exact ⟨g, hgExt⟩

theorem separation_from_hb {V : Type u} [SeminormedSpaceLike V]
    (U : V → Prop) (p f : V → Nat) (x y : V)
    (hsub : SublinearLike p)
    (hlin : LinearFunctionalLike f)
    (hdom : DominatedLike p f)
    (hex : ∃ g : V → Nat,
      LinearFunctionalLike g ∧ ExtendsLike U f g ∧ DominatedLike p g)
    (hstrict : ∀ g : V → Nat,
      LinearFunctionalLike g → DominatedLike p g → ExtendsLike U f g → g x < g y) :
    ∃ g : V → Nat, SeparatesLike g x y ∧ DominatedLike p g := by
  have hfull :
      ∃ g : V → Nat,
        LinearFunctionalLike g ∧ ExtendsLike U f g ∧ DominatedLike p g :=
    hb_extension_exists U p f hsub hlin hdom hex
  rcases hfull with ⟨g, hgLin, hgExt, hgDom⟩
  have hlt : g x < g y := hstrict g hgLin hgDom hgExt
  have hsep : SeparatesLike g x y := hlt
  exact ⟨g, hsep, hgDom⟩

theorem dual_separates_points {V : Type u} [SeminormedSpaceLike V]
    (x y : V)
    (hxy : x ≠ y)
    (hsep : ∀ a b : V, a ≠ b → ∃ g : V → Nat, SeparatesLike g a b ∧ g a ≠ g b) :
    ∃ g : V → Nat, g x ≠ g y := by
  have hw : ∃ g : V → Nat, SeparatesLike g x y ∧ g x ≠ g y := hsep x y hxy
  rcases hw with ⟨g, hgsep, hneq⟩
  have _ : SeparatesLike g x y := hgsep
  exact ⟨g, hneq⟩

theorem minkowski_functional_bound {V : Type u} [SeminormedSpaceLike V]
    (p g : V → Nat)
    (hsub : SublinearLike p)
    (hlin : LinearFunctionalLike g)
    (hdom : DominatedLike p g)
    (x y : V)
    (hbound : g (x +ᵥ y) ≤ p x + p y) :
    g (x +ᵥ y) ≤ p x + p y := by
  have hsub_add : p (x +ᵥ y) ≤ p x + p y := hsub.2.1 x y
  have hdom_add : g (x +ᵥ y) ≤ p (x +ᵥ y) := hdom (x +ᵥ y)
  have _ : p (x +ᵥ y) ≤ p x + p y := hsub_add
  have _ : g (x +ᵥ y) ≤ p (x +ᵥ y) := hdom_add
  have hlin_add : g (x +ᵥ y) = g x + g y := hlin.1 x y
  have hcompat : g x + g y = g (x +ᵥ y) := by
    symm
    exact hlin_add
  have hsum_bound : g x + g y ≤ p x + p y := by
    rw [hcompat]
    exact hbound
  calc
    g (x +ᵥ y) = g x + g y := hlin_add
    _ ≤ p x + p y := hsum_bound
