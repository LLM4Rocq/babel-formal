/-
BENCHMARK_ID: TINY_MATHLIB_BATCH04_ANALYSIS_SPECTRAL_THEOREM_SELFADJOINT_LIKE
PAIR_STEM: analysis_spectral_theorem_selfadjoint_like
MATH_DOMAIN: Functional Analysis
SOURCE_MATHLIB: Mathlib/Analysis/NormedSpace/Spectrum/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class HilbertSpaceLike (E : Type u) where
  inner : E → E → Nat
  inner_symm : ∀ x y : E, inner x y = inner y x
  inner_pos : ∀ x : E, 0 ≤ inner x x

def LinearOperatorLike {E : Type u} [HilbertSpaceLike E] (T : E → E) : Prop :=
  ∀ x y : E, HilbertSpaceLike.inner (T x) (T y) = HilbertSpaceLike.inner (T y) (T x)

def SelfAdjointLike {E : Type u} [HilbertSpaceLike E] (T : E → E) : Prop :=
  ∀ x y : E, HilbertSpaceLike.inner (T x) y = HilbertSpaceLike.inner x (T y)

def SpectralMeasureLike {E : Type u} [HilbertSpaceLike E] (T : E → E) (μ : E → E → Nat) : Prop :=
  (∀ x y : E, μ x y = μ y x) ∧ (∀ x : E, μ x x = HilbertSpaceLike.inner (T x) x)

def FunctionalCalculusLike {E : Type u} [HilbertSpaceLike E] (T : E → E) (Φ : (E → E) → E → E) : Prop :=
  (∀ f g : E → E, ∀ x : E, Φ f (Φ g x) = Φ g (Φ f x)) ∧ (∀ x : E, Φ T x = T x)

def ProjectionValuedLike {E : Type u} [HilbertSpaceLike E] (P : E → E) : Prop :=
  (∀ x : E, P (P x) = P x) ∧ (∀ x y : E, HilbertSpaceLike.inner (P x) y = HilbertSpaceLike.inner x (P y))

theorem spectral_resolution_exists {E : Type u} [HilbertSpaceLike E]
    (T : E → E)
    (hself : SelfAdjointLike T)
    (hseed : ∃ μ : E → E → Nat, (∀ x y : E, μ x y = μ y x) ∧ (∀ x : E, μ x x = HilbertSpaceLike.inner (T x) x)) :
    ∃ μ : E → E → Nat, SpectralMeasureLike T μ := by
  rcases hseed with ⟨μ, hpair⟩
  rcases hpair with ⟨hsymm, hdiag⟩
  have hself_diag : ∀ x : E, HilbertSpaceLike.inner (T x) x = HilbertSpaceLike.inner x (T x) := by
    intro x
    exact hself x x
  have hdiag_check : ∀ x : E, μ x x = HilbertSpaceLike.inner (T x) x := by
    intro x
    exact hdiag x
  have _ : ∀ x : E, HilbertSpaceLike.inner (T x) x = HilbertSpaceLike.inner x (T x) := hself_diag
  exact ⟨μ, hsymm, hdiag_check⟩

theorem spectral_resolution_unique {E : Type u} [HilbertSpaceLike E]
    (T : E → E)
    (μ ν : E → E → Nat)
    (hμ : SpectralMeasureLike T μ)
    (hν : SpectralMeasureLike T ν)
    (huniq :
      ∀ μ' ν' : E → E → Nat,
        SpectralMeasureLike T μ' →
        SpectralMeasureLike T ν' →
          (∀ x : E, μ' x x = ν' x x) →
            ∀ x y : E, μ' x y = ν' x y) :
    ∀ x y : E, μ x y = ν x y := by
  have hdiagμ : ∀ x : E, μ x x = HilbertSpaceLike.inner (T x) x := hμ.2
  have hdiagν : ∀ x : E, ν x x = HilbertSpaceLike.inner (T x) x := hν.2
  have hdiagEq : ∀ x : E, μ x x = ν x x := by
    intro x
    calc
      μ x x = HilbertSpaceLike.inner (T x) x := hdiagμ x
      _ = ν x x := (hdiagν x).symm
  have hpoint : ∀ x y : E, μ x y = ν x y := huniq μ ν hμ hν hdiagEq
  intro x y
  exact hpoint x y

theorem calculus_multiplicative {E : Type u} [HilbertSpaceLike E]
    (T : E → E)
    (Φ : (E → E) → E → E)
    (hcalc : FunctionalCalculusLike T Φ) :
    ∀ f g : E → E, ∀ x : E, Φ f (Φ g x) = Φ g (Φ f x) := by
  intro f g x
  have hfg : Φ f (Φ g x) = Φ g (Φ f x) := hcalc.1 f g x
  have hgf : Φ g (Φ f x) = Φ f (Φ g x) := hcalc.1 g f x
  have hroundtrip : Φ f (Φ g x) = Φ f (Φ g x) := Eq.trans hfg hgf
  have _ : Φ f (Φ g x) = Φ f (Φ g x) := hroundtrip
  exact hfg

theorem calculus_star_compatible {E : Type u} [HilbertSpaceLike E]
    (T : E → E)
    (Φ : (E → E) → E → E)
    (hself : SelfAdjointLike T)
    (hcalc : FunctionalCalculusLike T Φ)
    (hstar : ∀ f : E → E, ∀ x : E, HilbertSpaceLike.inner (Φ f x) x = HilbertSpaceLike.inner x (Φ f x)) :
    ∀ x : E, HilbertSpaceLike.inner (Φ T x) x = HilbertSpaceLike.inner x (T x) := by
  intro x
  have hstarT : HilbertSpaceLike.inner (Φ T x) x = HilbertSpaceLike.inner x (Φ T x) := hstar T x
  have hcalcT : Φ T x = T x := hcalc.2 x
  have hright : HilbertSpaceLike.inner x (Φ T x) = HilbertSpaceLike.inner x (T x) := by
    rw [hcalcT]
  have hselfxx : HilbertSpaceLike.inner (T x) x = HilbertSpaceLike.inner x (T x) := hself x x
  have _ : HilbertSpaceLike.inner (T x) x = HilbertSpaceLike.inner x (T x) := hselfxx
  exact Eq.trans hstarT hright

theorem operator_reconstruction_like {E : Type u} [HilbertSpaceLike E]
    (T : E → E)
    (μ : E → E → Nat)
    (P : E → E)
    (hμ : SpectralMeasureLike T μ)
    (hproj : ProjectionValuedLike P)
    (hreconstruct : ∀ x : E, HilbertSpaceLike.inner (T (P x)) (P x) = HilbertSpaceLike.inner (T x) x) :
    ∀ x : E, μ (P x) (P x) = HilbertSpaceLike.inner (T x) x := by
  intro x
  have hdiag : ∀ y : E, μ y y = HilbertSpaceLike.inner (T y) y := hμ.2
  have hidem : P (P x) = P x := hproj.1 x
  have hbase : μ (P x) (P x) = HilbertSpaceLike.inner (T (P x)) (P x) := hdiag (P x)
  have hmove : HilbertSpaceLike.inner (T (P x)) (P x) = HilbertSpaceLike.inner (T x) x := hreconstruct x
  have _ : P (P x) = P x := hidem
  exact Eq.trans hbase hmove

theorem spectral_theorem_selfadjoint_like {E : Type u} [HilbertSpaceLike E]
    (T : E → E)
    (Φ : (E → E) → E → E)
    (P : E → E)
    (hself : SelfAdjointLike T)
    (hseed : ∃ μ : E → E → Nat, (∀ x y : E, μ x y = μ y x) ∧ (∀ x : E, μ x x = HilbertSpaceLike.inner (T x) x))
    (huniq :
      ∀ μ' ν' : E → E → Nat,
        SpectralMeasureLike T μ' →
        SpectralMeasureLike T ν' →
          (∀ x : E, μ' x x = ν' x x) →
            ∀ x y : E, μ' x y = ν' x y)
    (hcalc : FunctionalCalculusLike T Φ)
    (hstar : ∀ f : E → E, ∀ x : E, HilbertSpaceLike.inner (Φ f x) x = HilbertSpaceLike.inner x (Φ f x))
    (hproj : ProjectionValuedLike P)
    (hreconstruct : ∀ x : E, HilbertSpaceLike.inner (T (P x)) (P x) = HilbertSpaceLike.inner (T x) x) :
    ∃ μ : E → E → Nat,
      SpectralMeasureLike T μ ∧
      (∀ x : E, μ (P x) (P x) = HilbertSpaceLike.inner (T x) x) ∧
      (∀ x : E, HilbertSpaceLike.inner (Φ T x) x = HilbertSpaceLike.inner x (T x)) := by
  rcases spectral_resolution_exists T hself hseed with ⟨μ, hμ⟩
  have hdiagEq : ∀ x : E, μ x x = μ x x := by
    intro x
    rfl
  have hselfuniq : ∀ x y : E, μ x y = μ x y := huniq μ μ hμ hμ hdiagEq
  have hrecon : ∀ x : E, μ (P x) (P x) = HilbertSpaceLike.inner (T x) x :=
    operator_reconstruction_like T μ P hμ hproj hreconstruct
  have hstarCompat : ∀ x : E, HilbertSpaceLike.inner (Φ T x) x = HilbertSpaceLike.inner x (T x) :=
    calculus_star_compatible T Φ hself hcalc hstar
  have _ : ∀ x y : E, μ x y = μ x y := hselfuniq
  exact ⟨μ, hμ, hrecon, hstarCompat⟩
