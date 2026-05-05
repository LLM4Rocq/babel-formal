/-
BENCHMARK_ID: TINY_MATHLIB_BATCH05_MEASURE_LOG_SOBOLEV_AXIOMATIC_LIKE
PAIR_STEM: measure_log_sobolev_axiomatic_like
MATH_DOMAIN: Measure Theory
SOURCE_MATHLIB: Mathlib/MeasureTheory/Integral
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class MeasureStruct_log_sobolev (Omega : Type u) where
  density : Omega -> Nat
  kernel : Omega -> Omega -> Nat
  entropy : (Omega -> Nat) -> Nat
  integral : (Omega -> Nat) -> Nat
  density_nonneg_axiom : forall x : Omega, 0 <= density x
  integral_mono_axiom :
    forall f g : Omega -> Nat,
      (forall x : Omega, f x <= g x) ->
      integral f <= integral g
  transport_identity_axiom :
    forall x : Omega, kernel x x = density x
  chain_rule_axiom :
    forall f : Omega -> Nat,
      entropy f + integral f = integral (fun x : Omega => f x + density x)
  dual_variational_axiom :
    forall f : Omega -> Nat,
      entropy f <= integral f
  concentration_axiom :
    forall f : Omega -> Nat,
      forall n : Nat,
        integral f <= n ->
        entropy f <= n
  decomposition_axiom :
    forall f g : Omega -> Nat,
      integral (fun x : Omega => f x + g x) = integral f + integral g

def DensityFn_log_sobolev
    (Omega : Type u) : Type u :=
  Omega -> Nat

def KernelMap_log_sobolev
    (Omega : Type u) : Type u :=
  Omega -> Omega -> Nat

def EntropyLike_log_sobolev
    {Omega : Type u} [h : MeasureStruct_log_sobolev Omega]
    (f : DensityFn_log_sobolev Omega) : Nat :=
  h.entropy f

def IntegralForm_log_sobolev
    {Omega : Type u} [h : MeasureStruct_log_sobolev Omega]
    (f : DensityFn_log_sobolev Omega) : Nat :=
  h.integral f

theorem density_nonneg_log_sobolev
    {Omega : Type u} [h : MeasureStruct_log_sobolev Omega]
    (x : Omega) :
    ∃ n : Nat, h.kernel x x = n ∧ 0 <= n := by
  refine ⟨h.density x, ?_⟩
  constructor
  · exact h.transport_identity_axiom x
  · exact h.density_nonneg_axiom x

theorem integral_mono_log_sobolev
    {Omega : Type u} [h : MeasureStruct_log_sobolev Omega]
    (f g : DensityFn_log_sobolev Omega)
    (hfg : forall x : Omega, f x <= g x) :
    IntegralForm_log_sobolev f <= IntegralForm_log_sobolev g ∧
    IntegralForm_log_sobolev (fun x : Omega => f x + g x) =
      IntegralForm_log_sobolev f + IntegralForm_log_sobolev g := by
  have hMono : h.integral f <= h.integral g := h.integral_mono_axiom f g hfg
  have hDecomp :
      h.integral (fun x : Omega => f x + g x) = h.integral f + h.integral g :=
    h.decomposition_axiom f g
  constructor
  · exact hMono
  · exact hDecomp

theorem transport_identity_log_sobolev
    {Omega : Type u} [h : MeasureStruct_log_sobolev Omega]
    (x : Omega) :
    h.kernel x x = h.density x ∧
    EntropyLike_log_sobolev (fun _ : Omega => 0) <=
      IntegralForm_log_sobolev (fun _ : Omega => 0) := by
  have hDiag : h.kernel x x = h.density x := h.transport_identity_axiom x
  have hZero : h.entropy (fun _ : Omega => 0) <= h.integral (fun _ : Omega => 0) :=
    h.dual_variational_axiom (fun _ : Omega => 0)
  constructor
  · exact hDiag
  · exact hZero

theorem chain_rule_measure_log_sobolev
    {Omega : Type u} [h : MeasureStruct_log_sobolev Omega]
    (f : DensityFn_log_sobolev Omega) :
    IntegralForm_log_sobolev (fun x : Omega => f x + h.density x) =
      EntropyLike_log_sobolev f + IntegralForm_log_sobolev f := by
  have hChain : h.entropy f + h.integral f = h.integral (fun x : Omega => f x + h.density x) :=
    h.chain_rule_axiom f
  exact Eq.symm hChain

theorem dual_variational_bound_log_sobolev
    {Omega : Type u} [h : MeasureStruct_log_sobolev Omega]
    (f : DensityFn_log_sobolev Omega) :
    EntropyLike_log_sobolev (fun x : Omega => h.kernel x x) <=
      IntegralForm_log_sobolev (fun x : Omega => h.kernel x x) := by
  exact h.dual_variational_axiom (fun x : Omega => h.kernel x x)

theorem concentration_step_log_sobolev
    {Omega : Type u} [h : MeasureStruct_log_sobolev Omega]
    (f : DensityFn_log_sobolev Omega)
    (n : Nat)
    (hBound : IntegralForm_log_sobolev f <= n) :
    ∃ m : Nat, m = n ∧ EntropyLike_log_sobolev f <= m := by
  have hConc : h.entropy f <= n := h.concentration_axiom f n hBound
  refine ⟨n, rfl, ?_⟩
  exact hConc

theorem decomposition_formula_log_sobolev
    {Omega : Type u} [h : MeasureStruct_log_sobolev Omega]
    (f g : DensityFn_log_sobolev Omega) :
    IntegralForm_log_sobolev (fun x : Omega => f x + g x) =
      IntegralForm_log_sobolev f + IntegralForm_log_sobolev g := by
  exact h.decomposition_axiom f g
