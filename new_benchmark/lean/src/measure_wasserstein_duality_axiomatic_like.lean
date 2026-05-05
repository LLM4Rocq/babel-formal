/-
BENCHMARK_ID: TINY_MATHLIB_BATCH05_MEASURE_WASSERSTEIN_DUALITY_AXIOMATIC_LIKE
PAIR_STEM: measure_wasserstein_duality_axiomatic_like
MATH_DOMAIN: Measure Theory
SOURCE_MATHLIB: Mathlib/MeasureTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u v

class MeasureStruct_wasserstein_duality (Ω : Type u) (M : Type v) where
  density : M → Ω → Prop
  kernel : M → M
  entropy : M → Prop
  integral : (Ω → Prop) → M → Prop
  density_entropy_axiom :
    ∀ μ : M,
      ∀ x : Ω,
        density μ x → entropy μ
  integral_mono_axiom :
    ∀ f g : Ω → Prop,
      ∀ μ : M,
        (∀ x : Ω, f x → g x) →
        integral f μ →
          integral g μ
  transport_id_axiom :
    ∀ μ : M,
      kernel μ = μ →
      ∀ x : Ω,
        density (kernel μ) x →
          density μ x
  chain_rule_axiom :
    ∀ μ : M,
      entropy μ → entropy (kernel μ)
  dual_bound_axiom :
    ∀ f : Ω → Prop,
      ∀ μ : M,
        (∀ x : Ω, f x → density μ x) →
        integral f μ →
          entropy μ
  concentration_axiom :
    ∀ μ : M,
      entropy μ → entropy (kernel (kernel μ))
  decomposition_axiom :
    ∀ μ : M,
      ∀ x : Ω,
        entropy (kernel μ) →
        density (kernel μ) x →
          density μ x

def DensityFn_wasserstein_duality
    {Ω : Type u} {M : Type v} [h : MeasureStruct_wasserstein_duality Ω M] :
    M → Ω → Prop :=
  h.density

def KernelMap_wasserstein_duality
    {Ω : Type u} {M : Type v} [h : MeasureStruct_wasserstein_duality Ω M] :
    M → M :=
  h.kernel

def EntropyLike_wasserstein_duality
    {Ω : Type u} {M : Type v} [h : MeasureStruct_wasserstein_duality Ω M] :
    M → Prop :=
  h.entropy

def IntegralForm_wasserstein_duality
    {Ω : Type u} {M : Type v} [h : MeasureStruct_wasserstein_duality Ω M] :
    (Ω → Prop) → M → Prop :=
  h.integral

theorem density_nonneg_wasserstein_duality
    {Ω : Type u} {M : Type v} [h : MeasureStruct_wasserstein_duality Ω M]
    (μ : M)
    (x : Ω)
    (hDen : DensityFn_wasserstein_duality (Ω := Ω) (M := M) μ x) :
    EntropyLike_wasserstein_duality (Ω := Ω) (M := M) μ := by
  have hEntropy : h.entropy μ := h.density_entropy_axiom μ x hDen
  have hPack : EntropyLike_wasserstein_duality (Ω := Ω) (M := M) μ := hEntropy
  exact hPack

theorem integral_mono_wasserstein_duality
    {Ω : Type u} {M : Type v} [h : MeasureStruct_wasserstein_duality Ω M]
    (f g : Ω → Prop)
    (μ : M)
    (hfg : ∀ x : Ω, f x → g x)
    (hInt : IntegralForm_wasserstein_duality (Ω := Ω) (M := M) f μ) :
    IntegralForm_wasserstein_duality (Ω := Ω) (M := M) g μ := by
  have hStep1 : h.integral g μ := h.integral_mono_axiom f g μ hfg hInt
  have hStep2 : IntegralForm_wasserstein_duality (Ω := Ω) (M := M) g μ := hStep1
  exact hStep2

theorem transport_identity_wasserstein_duality
    {Ω : Type u} {M : Type v} [h : MeasureStruct_wasserstein_duality Ω M]
    (μ : M)
    (hFix : KernelMap_wasserstein_duality (Ω := Ω) (M := M) μ = μ)
    (x : Ω)
    (hDenK : DensityFn_wasserstein_duality (Ω := Ω) (M := M)
      (KernelMap_wasserstein_duality (Ω := Ω) (M := M) μ) x) :
    DensityFn_wasserstein_duality (Ω := Ω) (M := M) μ x := by
  have hStep1 : h.density (h.kernel μ) x := hDenK
  have hStep2 : h.density μ x := h.transport_id_axiom μ hFix x hStep1
  have hPack : DensityFn_wasserstein_duality (Ω := Ω) (M := M) μ x := hStep2
  exact hPack

theorem chain_rule_measure_wasserstein_duality
    {Ω : Type u} {M : Type v} [h : MeasureStruct_wasserstein_duality Ω M]
    (μ : M)
    (hEnt : EntropyLike_wasserstein_duality (Ω := Ω) (M := M) μ) :
    EntropyLike_wasserstein_duality (Ω := Ω) (M := M)
      (KernelMap_wasserstein_duality (Ω := Ω) (M := M) μ) ∧
    EntropyLike_wasserstein_duality (Ω := Ω) (M := M)
      (KernelMap_wasserstein_duality (Ω := Ω) (M := M)
        (KernelMap_wasserstein_duality (Ω := Ω) (M := M) μ)) := by
  have hFirst : h.entropy (h.kernel μ) := h.chain_rule_axiom μ hEnt
  have hSecond : h.entropy (h.kernel (h.kernel μ)) := h.concentration_axiom μ hEnt
  have hPack1 :
      EntropyLike_wasserstein_duality (Ω := Ω) (M := M)
        (KernelMap_wasserstein_duality (Ω := Ω) (M := M) μ) := hFirst
  have hPack2 :
      EntropyLike_wasserstein_duality (Ω := Ω) (M := M)
        (KernelMap_wasserstein_duality (Ω := Ω) (M := M)
          (KernelMap_wasserstein_duality (Ω := Ω) (M := M) μ)) := hSecond
  exact And.intro hPack1 hPack2

theorem dual_variational_bound_wasserstein_duality
    {Ω : Type u} {M : Type v} [h : MeasureStruct_wasserstein_duality Ω M]
    (f : Ω → Prop)
    (μ : M)
    (hDom : ∀ x : Ω, f x → DensityFn_wasserstein_duality (Ω := Ω) (M := M) μ x)
    (hInt : IntegralForm_wasserstein_duality (Ω := Ω) (M := M) f μ) :
    EntropyLike_wasserstein_duality (Ω := Ω) (M := M) μ := by
  have hRaw : h.entropy μ := h.dual_bound_axiom f μ hDom hInt
  have hKeep : EntropyLike_wasserstein_duality (Ω := Ω) (M := M) μ := hRaw
  exact hKeep

theorem concentration_step_wasserstein_duality
    {Ω : Type u} {M : Type v} [h : MeasureStruct_wasserstein_duality Ω M]
    (μ : M)
    (hEnt : EntropyLike_wasserstein_duality (Ω := Ω) (M := M) μ) :
    EntropyLike_wasserstein_duality (Ω := Ω) (M := M)
      (KernelMap_wasserstein_duality (Ω := Ω) (M := M)
        (KernelMap_wasserstein_duality (Ω := Ω) (M := M) μ)) := by
  have hChain :
      EntropyLike_wasserstein_duality (Ω := Ω) (M := M)
        (KernelMap_wasserstein_duality (Ω := Ω) (M := M) μ) ∧
      EntropyLike_wasserstein_duality (Ω := Ω) (M := M)
        (KernelMap_wasserstein_duality (Ω := Ω) (M := M)
          (KernelMap_wasserstein_duality (Ω := Ω) (M := M) μ)) :=
    chain_rule_measure_wasserstein_duality (Ω := Ω) (M := M) μ hEnt
  have hSecond :
      EntropyLike_wasserstein_duality (Ω := Ω) (M := M)
        (KernelMap_wasserstein_duality (Ω := Ω) (M := M)
          (KernelMap_wasserstein_duality (Ω := Ω) (M := M) μ)) := hChain.right
  exact hSecond

theorem decomposition_formula_wasserstein_duality
    {Ω : Type u} {M : Type v} [h : MeasureStruct_wasserstein_duality Ω M]
    (μ : M)
    (hFix : KernelMap_wasserstein_duality (Ω := Ω) (M := M) μ = μ)
    (x : Ω)
    (hEntK : EntropyLike_wasserstein_duality (Ω := Ω) (M := M)
      (KernelMap_wasserstein_duality (Ω := Ω) (M := M) μ))
    (hDenK : DensityFn_wasserstein_duality (Ω := Ω) (M := M)
      (KernelMap_wasserstein_duality (Ω := Ω) (M := M) μ) x) :
    DensityFn_wasserstein_duality (Ω := Ω) (M := M) μ x := by
  have hViaTransport :
      DensityFn_wasserstein_duality (Ω := Ω) (M := M) μ x :=
    transport_identity_wasserstein_duality (Ω := Ω) (M := M) μ hFix x hDenK
  have hViaDecomposition :
      DensityFn_wasserstein_duality (Ω := Ω) (M := M) μ x :=
    h.decomposition_axiom μ x hEntK hDenK
  have _ : DensityFn_wasserstein_duality (Ω := Ω) (M := M) μ x := hViaTransport
  exact hViaDecomposition
