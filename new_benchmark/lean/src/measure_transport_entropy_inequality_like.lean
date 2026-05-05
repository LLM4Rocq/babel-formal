/-
BENCHMARK_ID: TINY_MATHLIB_BATCH05_MEASURE_TRANSPORT_ENTROPY_INEQUALITY_LIKE
PAIR_STEM: measure_transport_entropy_inequality_like
MATH_DOMAIN: Measure Theory
SOURCE_MATHLIB: Mathlib/MeasureTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class MeasureStruct_transport_entropy_inequality (Ω : Type u) where
  density : Ω → Nat
  kernel : Ω → Ω → Nat
  entropy : (Ω → Nat) → Nat
  integral : (Ω → Nat) → Nat
  density_nonneg_axiom : ∀ x : Ω, 0 ≤ density x
  integral_mono_axiom :
    ∀ f g : Ω → Nat,
      (∀ x : Ω, f x ≤ g x) →
        integral f ≤ integral g
  kernel_identity_axiom :
    ∀ x : Ω, kernel x x = density x
  chain_rule_axiom :
    ∀ f : Ω → Nat,
      entropy f + integral f = integral (fun x : Ω => f x + density x)
  dual_variational_axiom :
    ∀ f : Ω → Nat,
      entropy f ≤ integral f
  concentration_axiom :
    ∀ f : Ω → Nat,
      ∀ n : Nat,
        integral f ≤ n →
          entropy f ≤ n
  decomposition_axiom :
    ∀ f g : Ω → Nat,
      integral (fun x : Ω => f x + g x) = integral f + integral g

def DensityFn_transport_entropy_inequality
    (Ω : Type u) : Type u :=
  Ω → Nat

def KernelMap_transport_entropy_inequality
    (Ω : Type u) : Type u :=
  Ω → Ω → Nat

def EntropyLike_transport_entropy_inequality
    {Ω : Type u} [h : MeasureStruct_transport_entropy_inequality Ω]
    (f : DensityFn_transport_entropy_inequality Ω) : Nat :=
  h.entropy f

def IntegralForm_transport_entropy_inequality
    {Ω : Type u} [h : MeasureStruct_transport_entropy_inequality Ω]
    (f : DensityFn_transport_entropy_inequality Ω) : Nat :=
  h.integral f

theorem density_nonneg_transport_entropy_inequality
    {Ω : Type u} [h : MeasureStruct_transport_entropy_inequality Ω]
    (x : Ω) :
    0 ≤ h.density x ∧ h.kernel x x = h.density x := by
  have hNonneg : 0 ≤ h.density x := h.density_nonneg_axiom x
  have hDiag : h.kernel x x = h.density x := h.kernel_identity_axiom x
  constructor
  · exact hNonneg
  · exact hDiag

theorem integral_mono_transport_entropy_inequality
    {Ω : Type u} [h : MeasureStruct_transport_entropy_inequality Ω]
    (f g : DensityFn_transport_entropy_inequality Ω)
    (hfg : ∀ x : Ω, f x ≤ g x) :
    IntegralForm_transport_entropy_inequality f ≤
      IntegralForm_transport_entropy_inequality g ∧
    EntropyLike_transport_entropy_inequality f ≤
      IntegralForm_transport_entropy_inequality g := by
  have hMono : h.integral f ≤ h.integral g :=
    h.integral_mono_axiom f g hfg
  have hConc : h.entropy f ≤ h.integral g :=
    h.concentration_axiom f (h.integral g) hMono
  constructor
  · exact hMono
  · exact hConc

theorem transport_identity_transport_entropy_inequality
    {Ω : Type u} [h : MeasureStruct_transport_entropy_inequality Ω]
    (x : Ω) :
    h.kernel x x = h.density x := by
  exact h.kernel_identity_axiom x

theorem chain_rule_measure_transport_entropy_inequality
    {Ω : Type u} [h : MeasureStruct_transport_entropy_inequality Ω]
    (f : DensityFn_transport_entropy_inequality Ω) :
    EntropyLike_transport_entropy_inequality f +
        IntegralForm_transport_entropy_inequality f =
      IntegralForm_transport_entropy_inequality (fun x : Ω => f x + h.density x) ∧
    EntropyLike_transport_entropy_inequality f ≤
      IntegralForm_transport_entropy_inequality f := by
  have hChain :
      h.entropy f + h.integral f = h.integral (fun x : Ω => f x + h.density x) :=
    h.chain_rule_axiom f
  have hDual : h.entropy f ≤ h.integral f := h.dual_variational_axiom f
  constructor
  · exact hChain
  · exact hDual

theorem dual_variational_bound_transport_entropy_inequality
    {Ω : Type u} [h : MeasureStruct_transport_entropy_inequality Ω]
    (f : DensityFn_transport_entropy_inequality Ω) :
    EntropyLike_transport_entropy_inequality f ≤
      IntegralForm_transport_entropy_inequality f ∧
    EntropyLike_transport_entropy_inequality (fun x : Ω => f x + h.density x) ≤
      IntegralForm_transport_entropy_inequality (fun x : Ω => f x + h.density x) := by
  have hMain : h.entropy f ≤ h.integral f := h.dual_variational_axiom f
  have hShift :
      h.entropy (fun x : Ω => f x + h.density x) ≤
        h.integral (fun x : Ω => f x + h.density x) :=
    h.dual_variational_axiom (fun x : Ω => f x + h.density x)
  constructor
  · exact hMain
  · exact hShift

theorem concentration_step_transport_entropy_inequality
    {Ω : Type u} [h : MeasureStruct_transport_entropy_inequality Ω]
    (f : DensityFn_transport_entropy_inequality Ω)
    (n : Nat)
    (hBound : IntegralForm_transport_entropy_inequality f ≤ n) :
    EntropyLike_transport_entropy_inequality f ≤ n ∧
    EntropyLike_transport_entropy_inequality f ≤
      IntegralForm_transport_entropy_inequality f := by
  have hConc : h.entropy f ≤ n := h.concentration_axiom f n hBound
  have hDual : h.entropy f ≤ h.integral f := h.dual_variational_axiom f
  constructor
  · exact hConc
  · exact hDual

theorem decomposition_formula_transport_entropy_inequality
    {Ω : Type u} [h : MeasureStruct_transport_entropy_inequality Ω]
    (f g : DensityFn_transport_entropy_inequality Ω) :
    IntegralForm_transport_entropy_inequality (fun x : Ω => f x + g x) =
      IntegralForm_transport_entropy_inequality f +
        IntegralForm_transport_entropy_inequality g ∧
    IntegralForm_transport_entropy_inequality (fun x : Ω => g x + f x) =
      IntegralForm_transport_entropy_inequality g +
        IntegralForm_transport_entropy_inequality f := by
  have hFG : h.integral (fun x : Ω => f x + g x) = h.integral f + h.integral g :=
    h.decomposition_axiom f g
  have hGF : h.integral (fun x : Ω => g x + f x) = h.integral g + h.integral f :=
    h.decomposition_axiom g f
  constructor
  · exact hFG
  · exact hGF
