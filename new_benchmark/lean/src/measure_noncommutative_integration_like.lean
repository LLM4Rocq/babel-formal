/-
BENCHMARK_ID: TINY_MATHLIB_BATCH05_MEASURE_NONCOMMUTATIVE_INTEGRATION_LIKE
PAIR_STEM: measure_noncommutative_integration_like
MATH_DOMAIN: Measure Theory
SOURCE_MATHLIB: Mathlib/MeasureTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class MeasureStruct_noncommutative_integration (Ω : Type u) where
  density : Ω → Nat
  kernel : Ω → Ω → Nat
  entropy : (Ω → Nat) → Nat
  integral : (Ω → Nat) → Nat
  density_nonneg_axiom : ∀ x : Ω, 0 ≤ density x
  integral_mono_axiom :
    ∀ f g : Ω → Nat,
      (∀ x : Ω, f x ≤ g x) →
        integral f ≤ integral g
  integral_congr_axiom :
    ∀ f g : Ω → Nat,
      (∀ x : Ω, f x = g x) →
        integral f = integral g
  nat_le_trans_axiom :
    ∀ a b c : Nat,
      a ≤ b →
      b ≤ c →
        a ≤ c
  nat_add_mono_right_axiom :
    ∀ a b c : Nat,
      a ≤ b →
      a + c ≤ b + c
  integral_add_axiom :
    ∀ f g : Ω → Nat,
      integral (fun x : Ω => f x + g x) = integral f + integral g
  integral_zero_axiom :
    integral (fun _ : Ω => 0) = 0
  kernel_diag_axiom :
    ∀ x : Ω, kernel x x = density x
  entropy_le_integral_axiom :
    ∀ f : Ω → Nat, entropy f ≤ integral f
  entropy_subadd_axiom :
    ∀ f g : Ω → Nat,
      entropy (fun x : Ω => f x + g x) ≤ entropy f + integral g
  transport_axiom :
    ∀ f : Ω → Nat,
      integral (fun x : Ω => kernel x x + f x) =
        integral (fun x : Ω => kernel x x) + integral f

def DensityFn_noncommutative_integration
    (Ω : Type u) : Type u :=
  Ω → Nat

def KernelMap_noncommutative_integration
    (Ω : Type u) : Type u :=
  Ω → Ω → Nat

def EntropyLike_noncommutative_integration
    {Ω : Type u} [h : MeasureStruct_noncommutative_integration Ω]
    (f : DensityFn_noncommutative_integration Ω) : Nat :=
  h.entropy f

def IntegralForm_noncommutative_integration
    {Ω : Type u} [h : MeasureStruct_noncommutative_integration Ω]
    (f : DensityFn_noncommutative_integration Ω) : Nat :=
  h.integral f

theorem density_nonneg_noncommutative_integration
    {Ω : Type u} [h : MeasureStruct_noncommutative_integration Ω]
    (x : Ω) :
    0 ≤ h.density x ∧ IntegralForm_noncommutative_integration (fun _ : Ω => 0) = 0 := by
  have hNonneg : 0 ≤ h.density x := h.density_nonneg_axiom x
  have hZero : h.integral (fun _ : Ω => 0) = 0 := h.integral_zero_axiom
  constructor
  · exact hNonneg
  · exact hZero

theorem integral_mono_noncommutative_integration
    {Ω : Type u} [h : MeasureStruct_noncommutative_integration Ω]
    (f g : DensityFn_noncommutative_integration Ω)
    (hfg : ∀ x : Ω, f x ≤ g x) :
    IntegralForm_noncommutative_integration f ≤
      IntegralForm_noncommutative_integration g ∧
    IntegralForm_noncommutative_integration f +
      IntegralForm_noncommutative_integration g ≤
        IntegralForm_noncommutative_integration g +
          IntegralForm_noncommutative_integration g := by
  have hMono : h.integral f ≤ h.integral g := h.integral_mono_axiom f g hfg
  have hLift : h.integral f + h.integral g ≤ h.integral g + h.integral g :=
    h.nat_add_mono_right_axiom (h.integral f) (h.integral g) (h.integral g) hMono
  constructor
  · exact hMono
  · exact hLift

theorem transport_identity_noncommutative_integration
    {Ω : Type u} [h : MeasureStruct_noncommutative_integration Ω] :
    IntegralForm_noncommutative_integration (fun x : Ω => h.kernel x x) =
      IntegralForm_noncommutative_integration (fun x : Ω => h.density x) := by
  apply h.integral_congr_axiom
  intro x
  exact h.kernel_diag_axiom x

theorem chain_rule_measure_noncommutative_integration
    {Ω : Type u} [h : MeasureStruct_noncommutative_integration Ω]
    (f g : DensityFn_noncommutative_integration Ω) :
    IntegralForm_noncommutative_integration (fun x : Ω => f x + g x) =
      IntegralForm_noncommutative_integration f +
        IntegralForm_noncommutative_integration g ∧
    IntegralForm_noncommutative_integration (fun x : Ω => h.kernel x x + f x) =
      IntegralForm_noncommutative_integration (fun x : Ω => h.kernel x x) +
        IntegralForm_noncommutative_integration f := by
  have hAdd :
      h.integral (fun x : Ω => f x + g x) = h.integral f + h.integral g :=
    h.integral_add_axiom f g
  have hTransport :
      h.integral (fun x : Ω => h.kernel x x + f x) =
        h.integral (fun x : Ω => h.kernel x x) + h.integral f :=
    h.transport_axiom f
  constructor
  · exact hAdd
  · exact hTransport

theorem dual_variational_bound_noncommutative_integration
    {Ω : Type u} [h : MeasureStruct_noncommutative_integration Ω]
    (f : DensityFn_noncommutative_integration Ω) :
    EntropyLike_noncommutative_integration f ≤
      IntegralForm_noncommutative_integration f ∧
    EntropyLike_noncommutative_integration (fun x : Ω => f x + 0) ≤
      IntegralForm_noncommutative_integration (fun x : Ω => f x + 0) := by
  have hMain : h.entropy f ≤ h.integral f := h.entropy_le_integral_axiom f
  have hShift : h.entropy (fun x : Ω => f x + 0) ≤ h.integral (fun x : Ω => f x + 0) :=
    h.entropy_le_integral_axiom (fun x : Ω => f x + 0)
  constructor
  · exact hMain
  · exact hShift

theorem concentration_step_noncommutative_integration
    {Ω : Type u} [h : MeasureStruct_noncommutative_integration Ω]
    (f g : DensityFn_noncommutative_integration Ω)
    (hfg : ∀ x : Ω, f x ≤ g x) :
    EntropyLike_noncommutative_integration f ≤
      IntegralForm_noncommutative_integration g ∧
    EntropyLike_noncommutative_integration f +
      IntegralForm_noncommutative_integration g ≤
        IntegralForm_noncommutative_integration g +
          IntegralForm_noncommutative_integration g := by
  have hEntropyToInt : h.entropy f ≤ h.integral f := h.entropy_le_integral_axiom f
  have hIntMono : h.integral f ≤ h.integral g := h.integral_mono_axiom f g hfg
  have hMain : h.entropy f ≤ h.integral g :=
    h.nat_le_trans_axiom (h.entropy f) (h.integral f) (h.integral g) hEntropyToInt hIntMono
  have hLift : h.entropy f + h.integral g ≤ h.integral g + h.integral g :=
    h.nat_add_mono_right_axiom (h.entropy f) (h.integral g) (h.integral g) hMain
  constructor
  · exact hMain
  · exact hLift

theorem decomposition_formula_noncommutative_integration
    {Ω : Type u} [h : MeasureStruct_noncommutative_integration Ω]
    (f g : DensityFn_noncommutative_integration Ω)
    (hf : ∀ x : Ω, f x ≤ h.kernel x x) :
    EntropyLike_noncommutative_integration (fun x : Ω => f x + g x) ≤
      IntegralForm_noncommutative_integration (fun x : Ω => h.kernel x x) +
        IntegralForm_noncommutative_integration g ∧
    IntegralForm_noncommutative_integration (fun x : Ω => h.kernel x x + f x) =
      IntegralForm_noncommutative_integration (fun x : Ω => h.kernel x x) +
        IntegralForm_noncommutative_integration f := by
  have hSubadd :
      h.entropy (fun x : Ω => f x + g x) ≤ h.entropy f + h.integral g :=
    h.entropy_subadd_axiom f g
  have hDual : h.entropy f ≤ h.integral f := h.entropy_le_integral_axiom f
  have hMonoDiag : h.integral f ≤ h.integral (fun x : Ω => h.kernel x x) :=
    h.integral_mono_axiom f (fun x : Ω => h.kernel x x) hf
  have hEntropyToDiag : h.entropy f ≤ h.integral (fun x : Ω => h.kernel x x) :=
    h.nat_le_trans_axiom (h.entropy f) (h.integral f)
      (h.integral (fun x : Ω => h.kernel x x)) hDual hMonoDiag
  have hLift : h.entropy f + h.integral g ≤
      h.integral (fun x : Ω => h.kernel x x) + h.integral g :=
    h.nat_add_mono_right_axiom (h.entropy f)
      (h.integral (fun x : Ω => h.kernel x x)) (h.integral g) hEntropyToDiag
  have hMain : h.entropy (fun x : Ω => f x + g x) ≤
      h.integral (fun x : Ω => h.kernel x x) + h.integral g :=
    h.nat_le_trans_axiom
      (h.entropy (fun x : Ω => f x + g x))
      (h.entropy f + h.integral g)
      (h.integral (fun x : Ω => h.kernel x x) + h.integral g)
      hSubadd hLift
  have hTransport :
      h.integral (fun x : Ω => h.kernel x x + f x) =
        h.integral (fun x : Ω => h.kernel x x) + h.integral f :=
    h.transport_axiom f
  constructor
  · exact hMain
  · exact hTransport
