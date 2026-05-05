/-
BENCHMARK_ID: TINY_MATHLIB_BATCH05_MEASURE_CONCENTRATION_LEVY_FAMILY_LIKE
PAIR_STEM: measure_concentration_levy_family_like
MATH_DOMAIN: Measure Theory
SOURCE_MATHLIB: Mathlib/MeasureTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class MeasureStruct_concentration_levy_family (Ω : Type u) where
  density : Ω → Nat
  kernel : Ω → Ω → Nat
  entropy : (Ω → Nat) → Nat
  integral : (Ω → Nat) → Nat
  nat_le_trans : ∀ a b c : Nat, a ≤ b → b ≤ c → a ≤ c
  nat_add_mono : ∀ a b c d : Nat, a ≤ b → c ≤ d → a + c ≤ b + d
  nat_le_refl : ∀ a : Nat, a ≤ a
  density_nonneg_axiom : ∀ x : Ω, 0 ≤ density x
  integral_mono_axiom :
    ∀ f g : Ω → Nat,
      (∀ x : Ω, f x ≤ g x) →
        integral f ≤ integral g
  kernel_identity_axiom :
    ∀ x : Ω,
      kernel x x = density x
  chain_rule_axiom :
    ∀ f g : Ω → Nat,
      entropy f + integral g ≤ entropy (fun x : Ω => f x + g x) + integral f
  dual_variational_axiom :
    ∀ f : Ω → Nat,
      entropy f ≤ integral f + integral (fun x : Ω => density x)
  concentration_axiom :
    ∀ f : Ω → Nat,
      ∀ n : Nat,
        integral f ≤ n →
          entropy f ≤ n + integral (fun x : Ω => density x)
  decomposition_axiom :
    ∀ f g : Ω → Nat,
      integral (fun x : Ω => f x + g x) = integral f + integral g

def DensityFn_concentration_levy_family (Ω : Type u) : Type u :=
  Ω → Nat

def KernelMap_concentration_levy_family (Ω : Type u) : Type u :=
  Ω → Ω → Nat

def EntropyLike_concentration_levy_family
    {Ω : Type u} [h : MeasureStruct_concentration_levy_family Ω]
    (f : DensityFn_concentration_levy_family Ω) : Nat :=
  h.entropy f

def IntegralForm_concentration_levy_family
    {Ω : Type u} [h : MeasureStruct_concentration_levy_family Ω]
    (f : DensityFn_concentration_levy_family Ω) : Nat :=
  h.integral f

theorem density_nonneg_concentration_levy_family
    {Ω : Type u} [h : MeasureStruct_concentration_levy_family Ω]
    (x : Ω) :
    (∀ n : Nat, n = h.density x → 0 ≤ n) ∧
    (h.kernel x x = h.density x → 0 ≤ h.kernel x x) := by
  constructor
  · intro n hn
    rw [hn]
    exact h.density_nonneg_axiom x
  · intro hDiag
    rw [hDiag]
    exact h.density_nonneg_axiom x

theorem integral_mono_concentration_levy_family
    {Ω : Type u} [h : MeasureStruct_concentration_levy_family Ω]
    (f g : DensityFn_concentration_levy_family Ω)
    (hfg : ∀ x : Ω, f x ≤ g x) :
    (∀ n : Nat,
      n = IntegralForm_concentration_levy_family f →
        n ≤ IntegralForm_concentration_levy_family g) ∧
    IntegralForm_concentration_levy_family (fun x : Ω => f x + f x) ≤
      IntegralForm_concentration_levy_family (fun x : Ω => g x + g x) := by
  have hMono : h.integral f ≤ h.integral g := h.integral_mono_axiom f g hfg
  have hFF : h.integral (fun x : Ω => f x + f x) = h.integral f + h.integral f :=
    h.decomposition_axiom f f
  have hGG : h.integral (fun x : Ω => g x + g x) = h.integral g + h.integral g :=
    h.decomposition_axiom g g
  have hLift : h.integral f + h.integral f ≤ h.integral g + h.integral g :=
    h.nat_add_mono (h.integral f) (h.integral g) (h.integral f) (h.integral g) hMono hMono
  constructor
  · intro n hn
    rw [hn]
    exact hMono
  · calc
      h.integral (fun x : Ω => f x + f x)
          = h.integral f + h.integral f := hFF
      _ ≤ h.integral g + h.integral g := hLift
      _ = h.integral (fun x : Ω => g x + g x) := by
            exact Eq.symm hGG

theorem transport_identity_concentration_levy_family
    {Ω : Type u} [h : MeasureStruct_concentration_levy_family Ω]
    (x : Ω) :
    ∀ y : Ω, y = x → h.kernel y y = h.density y := by
  intro y hy
  rw [hy]
  exact h.kernel_identity_axiom x

theorem chain_rule_measure_concentration_levy_family
    {Ω : Type u} [h : MeasureStruct_concentration_levy_family Ω]
    (f g : DensityFn_concentration_levy_family Ω) :
    EntropyLike_concentration_levy_family f +
        IntegralForm_concentration_levy_family g ≤
      EntropyLike_concentration_levy_family (fun x : Ω => f x + g x) +
        IntegralForm_concentration_levy_family f ∧
    (EntropyLike_concentration_levy_family f +
        IntegralForm_concentration_levy_family g) +
        IntegralForm_concentration_levy_family g ≤
      (EntropyLike_concentration_levy_family (fun x : Ω => f x + g x) +
        IntegralForm_concentration_levy_family f) +
          IntegralForm_concentration_levy_family g := by
  have hChain :
      h.entropy f + h.integral g ≤ h.entropy (fun x : Ω => f x + g x) + h.integral f :=
    h.chain_rule_axiom f g
  have hReflG : h.integral g ≤ h.integral g := h.nat_le_refl (h.integral g)
  have hLift :
      (h.entropy f + h.integral g) + h.integral g ≤
        (h.entropy (fun x : Ω => f x + g x) + h.integral f) + h.integral g :=
    h.nat_add_mono
      (h.entropy f + h.integral g)
      (h.entropy (fun x : Ω => f x + g x) + h.integral f)
      (h.integral g) (h.integral g)
      hChain hReflG
  constructor
  · exact hChain
  · exact hLift

theorem dual_variational_bound_concentration_levy_family
    {Ω : Type u} [h : MeasureStruct_concentration_levy_family Ω]
    (f : DensityFn_concentration_levy_family Ω) :
    EntropyLike_concentration_levy_family f ≤
      IntegralForm_concentration_levy_family f +
        IntegralForm_concentration_levy_family (fun x : Ω => h.density x) ∧
    EntropyLike_concentration_levy_family f +
        IntegralForm_concentration_levy_family f ≤
      (IntegralForm_concentration_levy_family f +
        IntegralForm_concentration_levy_family (fun x : Ω => h.density x)) +
          IntegralForm_concentration_levy_family f := by
  have hDual :
      h.entropy f ≤ h.integral f + h.integral (fun x : Ω => h.density x) :=
    h.dual_variational_axiom f
  have hReflF : h.integral f ≤ h.integral f := h.nat_le_refl (h.integral f)
  have hLift :
      h.entropy f + h.integral f ≤
        (h.integral f + h.integral (fun x : Ω => h.density x)) + h.integral f :=
    h.nat_add_mono (h.entropy f) (h.integral f + h.integral (fun x : Ω => h.density x))
      (h.integral f) (h.integral f) hDual hReflF
  constructor
  · exact hDual
  · exact hLift

theorem concentration_step_concentration_levy_family
    {Ω : Type u} [h : MeasureStruct_concentration_levy_family Ω]
    (f : DensityFn_concentration_levy_family Ω)
    (n : Nat)
    (hBound : IntegralForm_concentration_levy_family f ≤ n) :
    EntropyLike_concentration_levy_family f ≤
      n + IntegralForm_concentration_levy_family (fun x : Ω => h.density x) ∧
    EntropyLike_concentration_levy_family f +
        IntegralForm_concentration_levy_family f ≤
      (n + IntegralForm_concentration_levy_family (fun x : Ω => h.density x)) +
        IntegralForm_concentration_levy_family f := by
  have hConc : h.entropy f ≤ n + h.integral (fun x : Ω => h.density x) :=
    h.concentration_axiom f n hBound
  have hReflF : h.integral f ≤ h.integral f := h.nat_le_refl (h.integral f)
  have hLift :
      h.entropy f + h.integral f ≤
        (n + h.integral (fun x : Ω => h.density x)) + h.integral f :=
    h.nat_add_mono (h.entropy f) (n + h.integral (fun x : Ω => h.density x))
      (h.integral f) (h.integral f) hConc hReflF
  constructor
  · exact hConc
  · exact hLift

theorem decomposition_formula_concentration_levy_family
    {Ω : Type u} [h : MeasureStruct_concentration_levy_family Ω]
    (f g : DensityFn_concentration_levy_family Ω) :
    ∃ s t : Nat,
      s = IntegralForm_concentration_levy_family (fun x : Ω => f x + g x) ∧
      t = IntegralForm_concentration_levy_family (fun x : Ω => g x + f x) ∧
      s = IntegralForm_concentration_levy_family f +
            IntegralForm_concentration_levy_family g ∧
      t = IntegralForm_concentration_levy_family g +
            IntegralForm_concentration_levy_family f := by
  have hFG : h.integral (fun x : Ω => f x + g x) = h.integral f + h.integral g :=
    h.decomposition_axiom f g
  have hGF : h.integral (fun x : Ω => g x + f x) = h.integral g + h.integral f :=
    h.decomposition_axiom g f
  refine ⟨h.integral (fun x : Ω => f x + g x), h.integral (fun x : Ω => g x + f x), ?_⟩
  constructor
  · rfl
  constructor
  · rfl
  constructor
  · exact hFG
  · exact hGF
