/-
BENCHMARK_ID: TINY_MATHLIB_BATCH04_ANALYSIS_RIESZ_REPRESENTATION_REGULAR_LIKE
PAIR_STEM: analysis_riesz_representation_regular_like
MATH_DOMAIN: Functional Analysis / Measure Theory
SOURCE_MATHLIB: Mathlib/MeasureTheory/Integral/RieszMarkov
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class LocCompactSpaceLike (X : Type u) where
  empty : X → Prop
  univ : X → Prop
  compact : (X → Prop) → Prop
  openSet : (X → Prop) → Prop
  integral : ((X → Prop) → Nat) → (X → Nat) → Nat

def ContinuousCompactSupportLike {X : Type u} [LocCompactSpaceLike X]
    (f : X → Nat) : Prop :=
  ∃ K : X → Prop,
    LocCompactSpaceLike.compact K ∧
      (∀ x : X, f x = 0 ∨ K x)

def PositiveFunctionalLike {X : Type u} [LocCompactSpaceLike X]
    (L : (X → Nat) → Nat) : Prop :=
  (∀ f : X → Nat, 0 ≤ L f) ∧
    (∀ f g : X → Nat, L (fun x => f x + g x) = L f + L g)

def RadonMeasureLike {X : Type u} [LocCompactSpaceLike X]
    (μ : (X → Prop) → Nat) : Prop :=
  (∀ K : X → Prop,
    LocCompactSpaceLike.compact K →
      μ K ≤ μ (LocCompactSpaceLike.univ)) ∧
    μ (LocCompactSpaceLike.empty) = 0

def RepresentationLike {X : Type u} [LocCompactSpaceLike X]
    (L : (X → Nat) → Nat) (μ : (X → Prop) → Nat) : Prop :=
  ∀ f : X → Nat,
    ContinuousCompactSupportLike f →
      L f = LocCompactSpaceLike.integral μ f

def RegularLike {X : Type u} [LocCompactSpaceLike X]
    (μ : (X → Prop) → Nat) : Prop :=
  (∀ U : X → Prop,
    LocCompactSpaceLike.openSet U →
      ∃ K : X → Prop, LocCompactSpaceLike.compact K ∧ μ K ≤ μ U) ∧
    (∀ K : X → Prop,
      LocCompactSpaceLike.compact K →
        ∃ U : X → Prop, LocCompactSpaceLike.openSet U ∧ μ K ≤ μ U)

theorem representation_exists_like {X : Type u} [LocCompactSpaceLike X]
    (L : (X → Nat) → Nat)
    (hpos : PositiveFunctionalLike L)
    (hex : ∃ μ : (X → Prop) → Nat,
      RadonMeasureLike μ ∧ RepresentationLike L μ ∧ RegularLike μ) :
    ∃ μ : (X → Prop) → Nat,
      RadonMeasureLike μ ∧ RepresentationLike L μ ∧ RegularLike μ := by
  rcases hex with ⟨μ, hμRadon, hμRepr, hμReg⟩
  have hpos0 : 0 ≤ L (fun _ : X => 0) := hpos.1 (fun _ : X => 0)
  have hcheck : RepresentationLike L μ := hμRepr
  have _ : 0 ≤ L (fun _ : X => 0) := hpos0
  have _ : RepresentationLike L μ := hcheck
  exact ⟨μ, hμRadon, hμRepr, hμReg⟩

theorem representation_unique_like {X : Type u} [LocCompactSpaceLike X]
    (L : (X → Nat) → Nat)
    (μ ν : (X → Prop) → Nat)
    (hμ : RepresentationLike L μ)
    (hν : RepresentationLike L ν)
    (hall : ∀ f : X → Nat, ContinuousCompactSupportLike f)
    (hsep : ∀ μ₁ ν₁ : (X → Prop) → Nat,
      (∀ f : X → Nat,
        LocCompactSpaceLike.integral μ₁ f = LocCompactSpaceLike.integral ν₁ f) →
      μ₁ = ν₁) :
    μ = ν := by
  apply hsep μ ν
  intro f
  have hcf : ContinuousCompactSupportLike f := hall f
  have hμf : L f = LocCompactSpaceLike.integral μ f := hμ f hcf
  have hνf : L f = LocCompactSpaceLike.integral ν f := hν f hcf
  calc
    LocCompactSpaceLike.integral μ f = L f := by
      symm
      exact hμf
    _ = LocCompactSpaceLike.integral ν f := hνf

theorem positivity_transfer_like {X : Type u} [LocCompactSpaceLike X]
    (L : (X → Nat) → Nat)
    (μ : (X → Prop) → Nat)
    (hrepr : RepresentationLike L μ)
    (hmono : ∀ μ₁ : (X → Prop) → Nat, ∀ f : X → Nat,
      0 ≤ LocCompactSpaceLike.integral μ₁ f)
    (hadd : ∀ μ₁ : (X → Prop) → Nat, ∀ f g : X → Nat,
      LocCompactSpaceLike.integral μ₁ (fun x => f x + g x) =
        LocCompactSpaceLike.integral μ₁ f + LocCompactSpaceLike.integral μ₁ g)
    (hall : ∀ f : X → Nat, ContinuousCompactSupportLike f) :
    PositiveFunctionalLike L := by
  constructor
  · intro f
    have hcf : ContinuousCompactSupportLike f := hall f
    have hreprf : L f = LocCompactSpaceLike.integral μ f := hrepr f hcf
    have hμf : 0 ≤ LocCompactSpaceLike.integral μ f := hmono μ f
    rw [hreprf]
    exact hμf
  · intro f g
    have hcf : ContinuousCompactSupportLike f := hall f
    have hcg : ContinuousCompactSupportLike g := hall g
    have hsum : ContinuousCompactSupportLike (fun x => f x + g x) := hall (fun x => f x + g x)
    have hrepr_sum : L (fun x => f x + g x) = LocCompactSpaceLike.integral μ (fun x => f x + g x) :=
      hrepr (fun x => f x + g x) hsum
    have hrepr_f : L f = LocCompactSpaceLike.integral μ f := hrepr f hcf
    have hrepr_g : L g = LocCompactSpaceLike.integral μ g := hrepr g hcg
    calc
      L (fun x => f x + g x)
          = LocCompactSpaceLike.integral μ (fun x => f x + g x) := hrepr_sum
      _ = LocCompactSpaceLike.integral μ f + LocCompactSpaceLike.integral μ g :=
        hadd μ f g
      _ = L f + L g := by
        rw [hrepr_f, hrepr_g]

theorem regularity_inner_like {X : Type u} [LocCompactSpaceLike X]
    (μ : (X → Prop) → Nat)
    (hreg : RegularLike μ)
    (U : X → Prop)
    (hU : LocCompactSpaceLike.openSet U) :
    ∃ K : X → Prop, LocCompactSpaceLike.compact K ∧ μ K ≤ μ U := by
  have hinner := hreg.1 U hU
  rcases hinner with ⟨K, hKc, hKle⟩
  have _ : LocCompactSpaceLike.compact K := hKc
  exact ⟨K, hKc, hKle⟩

theorem regularity_outer_like {X : Type u} [LocCompactSpaceLike X]
    (μ : (X → Prop) → Nat)
    (hreg : RegularLike μ)
    (K : X → Prop)
    (hK : LocCompactSpaceLike.compact K) :
    ∃ U : X → Prop, LocCompactSpaceLike.openSet U ∧ μ K ≤ μ U := by
  have houter := hreg.2 K hK
  rcases houter with ⟨U, hUo, hKle⟩
  have _ : LocCompactSpaceLike.openSet U := hUo
  exact ⟨U, hUo, hKle⟩

theorem riesz_markov_theorem_like {X : Type u} [LocCompactSpaceLike X]
    (L : (X → Nat) → Nat)
    (hpos : PositiveFunctionalLike L)
    (hex : ∃ μ : (X → Prop) → Nat,
      RadonMeasureLike μ ∧ RepresentationLike L μ ∧ RegularLike μ)
    (hall : ∀ f : X → Nat, ContinuousCompactSupportLike f)
    (hsep : ∀ μ₁ ν₁ : (X → Prop) → Nat,
      (∀ f : X → Nat,
        LocCompactSpaceLike.integral μ₁ f = LocCompactSpaceLike.integral ν₁ f) →
      μ₁ = ν₁)
    (hmono : ∀ μ₁ : (X → Prop) → Nat, ∀ f : X → Nat,
      0 ≤ LocCompactSpaceLike.integral μ₁ f)
    (hadd : ∀ μ₁ : (X → Prop) → Nat, ∀ f g : X → Nat,
      LocCompactSpaceLike.integral μ₁ (fun x => f x + g x) =
        LocCompactSpaceLike.integral μ₁ f + LocCompactSpaceLike.integral μ₁ g) :
    ∃ μ : (X → Prop) → Nat,
      RepresentationLike L μ ∧
      RegularLike μ ∧
      (∀ ν : (X → Prop) → Nat,
        RepresentationLike L ν → RegularLike ν → ν = μ) ∧
      PositiveFunctionalLike L := by
  rcases representation_exists_like L hpos hex with ⟨μ, hμRadon, hμRepr, hμReg⟩
  have hposL : PositiveFunctionalLike L :=
    positivity_transfer_like L μ hμRepr hmono hadd hall
  refine ⟨μ, hμRepr, hμReg, ?_, hposL⟩
  intro ν hνRepr hνReg
  have _ : RegularLike ν := hνReg
  have huniq : μ = ν := representation_unique_like L μ ν hμRepr hνRepr hall hsep
  exact huniq.symm
