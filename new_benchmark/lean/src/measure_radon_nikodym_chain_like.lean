/-
BENCHMARK_ID: TINY_MATHLIB_BATCH03_MEASURE_RN_CHAIN_LIKE
PAIR_STEM: measure_radon_nikodym_chain_like
MATH_DOMAIN: Measure Theory
SOURCE_MATHLIB: Mathlib/MeasureTheory/Decomposition/RadonNikodym
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u v

class MeasurableSpaceLike (α : Type u) where
  measurable : (α → Prop) → Prop

class MeasureLike (α : Type u) (M : Type v) [MeasurableSpaceLike α] where
  integral : M → (α → Nat) → Nat

def AbsolutelyContinuous {α : Type u} {M : Type v}
    [MeasurableSpaceLike α] [MeasureLike α M]
    (μ ν : M) : Prop :=
  ∀ f : α → Nat,
    MeasureLike.integral (α := α) (M := M) ν f = 0 →
      MeasureLike.integral (α := α) (M := M) μ f = 0

def RNDerivative {α : Type u} {M : Type v}
    [MeasurableSpaceLike α] [MeasureLike α M]
    (μ ν : M) (f : α → Nat) : Prop :=
  ∀ g : α → Nat,
    MeasureLike.integral (α := α) (M := M) μ g =
      MeasureLike.integral (α := α) (M := M) ν (fun x => g x * f x)

def IntegrableLike {α : Type u} {M : Type v}
    [MeasurableSpaceLike α] [MeasureLike α M]
    (ν : M) (f : α → Nat) : Prop :=
  ∃ n : Nat, MeasureLike.integral (α := α) (M := M) ν f = n

def IntegralLike {α : Type u} {M : Type v}
    [MeasurableSpaceLike α] [MeasureLike α M]
    (ν : M) (f : α → Nat) : Nat :=
  MeasureLike.integral (α := α) (M := M) ν f

theorem rn_spec {α : Type u} {M : Type v}
    [MeasurableSpaceLike α] [MeasureLike α M]
    (μ ν : M) (f : α → Nat)
    (hder : RNDerivative μ ν f) :
    IntegrableLike ν f ∧
      (∀ g : α → Nat, IntegralLike μ g = IntegralLike ν (fun x => g x * f x)) := by
  constructor
  · refine ⟨IntegralLike ν f, ?_⟩
    rfl
  · intro g
    have hraw :
        MeasureLike.integral (α := α) (M := M) μ g =
          MeasureLike.integral (α := α) (M := M) ν (fun x => g x * f x) :=
      hder g
    exact hraw

theorem rn_unique {α : Type u} {M : Type v}
    [MeasurableSpaceLike α] [MeasureLike α M]
    (μ ν : M) (f g : α → Nat)
    (hf : RNDerivative μ ν f) (hg : RNDerivative μ ν g)
    (hsep : ∀ f0 g0 : α → Nat,
      (∀ h : α → Nat,
        IntegralLike ν (fun x => h x * f0 x) = IntegralLike ν (fun x => h x * g0 x)) →
      f0 = g0) :
    f = g := by
  apply hsep f g
  intro h
  have hf_h : IntegralLike μ h = IntegralLike ν (fun x => h x * f x) := hf h
  have hg_h : IntegralLike μ h = IntegralLike ν (fun x => h x * g x) := hg h
  calc
    IntegralLike ν (fun x => h x * f x)
        = IntegralLike μ h := by
          symm
          exact hf_h
    _ = IntegralLike ν (fun x => h x * g x) := hg_h

theorem rn_linear_combo {α : Type u} {M : Type v}
    [MeasurableSpaceLike α] [MeasureLike α M]
    (μ₁ μ₂ μsum ν : M) (f g : α → Nat)
    (hf : RNDerivative μ₁ ν f) (hg : RNDerivative μ₂ ν g)
    (hsum : ∀ h : α → Nat,
      IntegralLike μsum h = IntegralLike μ₁ h + IntegralLike μ₂ h)
    (hmul_add : ∀ h f0 g0 : α → Nat,
      IntegralLike ν (fun x => h x * (f0 x + g0 x)) =
        IntegralLike ν (fun x => h x * f0 x) + IntegralLike ν (fun x => h x * g0 x)) :
    RNDerivative μsum ν (fun x => f x + g x) := by
  intro h
  have hsum_h : IntegralLike μsum h = IntegralLike μ₁ h + IntegralLike μ₂ h := hsum h
  have hf_h : IntegralLike μ₁ h = IntegralLike ν (fun x => h x * f x) := hf h
  have hg_h : IntegralLike μ₂ h = IntegralLike ν (fun x => h x * g x) := hg h
  calc
    IntegralLike μsum h
        = IntegralLike μ₁ h + IntegralLike μ₂ h := hsum_h
    _ = IntegralLike ν (fun x => h x * f x) + IntegralLike ν (fun x => h x * g x) := by
      rw [hf_h, hg_h]
    _ = IntegralLike ν (fun x => h x * (f x + g x)) := by
      symm
      exact hmul_add h f g

theorem rn_chain_rule_like {α : Type u} {M : Type v}
    [MeasurableSpaceLike α] [MeasureLike α M]
    (μ ν ρ : M) (f g : α → Nat)
    (hμν : RNDerivative μ ν f) (hνρ : RNDerivative ν ρ g)
    (hmul_assoc : ∀ h f0 g0 : α → Nat,
      IntegralLike ρ (fun x => h x * (f0 x * g0 x)) =
        IntegralLike ρ (fun x => (h x * f0 x) * g0 x)) :
    RNDerivative μ ρ (fun x => f x * g x) := by
  intro h
  have hμ : IntegralLike μ h = IntegralLike ν (fun x => h x * f x) := hμν h
  have hν : IntegralLike ν (fun x => h x * f x) =
      IntegralLike ρ (fun x => (h x * f x) * g x) :=
    hνρ (fun x => h x * f x)
  calc
    IntegralLike μ h
        = IntegralLike ν (fun x => h x * f x) := hμ
    _ = IntegralLike ρ (fun x => (h x * f x) * g x) := hν
    _ = IntegralLike ρ (fun x => h x * (f x * g x)) := by
      symm
      exact hmul_assoc h f g

theorem rn_restrict_like {α : Type u} {M : Type v}
    [MeasurableSpaceLike α] [MeasureLike α M]
    (μ μr ν : M) (r f : α → Nat)
    (hf : RNDerivative μ ν f)
    (hrestrict : ∀ h : α → Nat,
      IntegralLike μr h = IntegralLike μ (fun x => r x * h x))
    (hswap : ∀ h f0 : α → Nat,
      IntegralLike ν (fun x => (r x * h x) * f0 x) =
        IntegralLike ν (fun x => h x * (r x * f0 x))) :
    RNDerivative μr ν (fun x => r x * f x) := by
  intro h
  have hμr : IntegralLike μr h = IntegralLike μ (fun x => r x * h x) := hrestrict h
  have hμ : IntegralLike μ (fun x => r x * h x) =
      IntegralLike ν (fun x => (r x * h x) * f x) :=
    hf (fun x => r x * h x)
  have hν : IntegralLike ν (fun x => (r x * h x) * f x) =
      IntegralLike ν (fun x => h x * (r x * f x)) :=
    hswap h f
  calc
    IntegralLike μr h
        = IntegralLike μ (fun x => r x * h x) := hμr
    _ = IntegralLike ν (fun x => (r x * h x) * f x) := hμ
    _ = IntegralLike ν (fun x => h x * (r x * f x)) := hν

theorem rn_zero_of_singular_like {α : Type u} {M : Type v}
    [MeasurableSpaceLike α] [MeasureLike α M]
    (μ ν : M)
    (hsing : ∀ h : α → Nat, IntegralLike μ h = 0)
    (hzero : ∀ h : α → Nat, IntegralLike ν (fun x => h x * 0) = 0) :
    RNDerivative μ ν (fun _ : α => 0) := by
  intro h
  have hμ0 : IntegralLike μ h = 0 := hsing h
  have hν0 : IntegralLike ν (fun x => h x * 0) = 0 := hzero h
  calc
    IntegralLike μ h = 0 := hμ0
    _ = IntegralLike ν (fun x => h x * 0) := by
      symm
      exact hν0
