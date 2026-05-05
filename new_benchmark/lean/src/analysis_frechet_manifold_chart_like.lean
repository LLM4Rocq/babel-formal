/-
BENCHMARK_ID: TINY_MATHLIB_BATCH05_ANALYSIS_FRECHET_MANIFOLD_CHART_LIKE
PAIR_STEM: analysis_frechet_manifold_chart_like
MATH_DOMAIN: Analysis
SOURCE_MATHLIB: Mathlib/Analysis/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class AnalysisStruct_frechet_manifold_chart (M : Type u) where
  norm : M → Nat
  add : M → M → M
  chart : M → M
  transform : M → M → M
  generator : Nat → M → M
  transform_bound :
    ∀ x y : M,
      norm (transform x y) ≤ norm x + norm y
  chart_idem :
    ∀ x : M,
      chart (chart x) = chart x
  chart_bound :
    ∀ x : M,
      norm (chart x) ≤ norm x
  nat_add_mono :
    ∀ a b c d : Nat,
      a ≤ b →
      c ≤ d →
        a + c ≤ b + d
  nat_le_trans :
    ∀ a b c : Nat,
      a ≤ b →
      b ≤ c →
        a ≤ c
  nat_add_comm :
    ∀ a b : Nat,
      a + b = b + a
  nat_le_refl :
    ∀ a : Nat,
      a ≤ a
  generator_step :
    ∀ n : Nat,
      ∀ x : M,
        norm (generator (n + 1) x) ≤ norm (generator n x) + norm x
  generator_zero :
    ∀ x : M,
      generator 0 x = x
  semigroup_axiom :
    ∀ m n : Nat,
      ∀ x : M,
        generator (m + n) x = generator m (generator n x)
  regularity_axiom :
    ∀ x : M,
      norm (generator 1 (chart x)) ≤ norm (chart x)

def NormCtrl_frechet_manifold_chart
    {M : Type u} [h : AnalysisStruct_frechet_manifold_chart M] : Prop :=
  ∀ x : M, h.norm (h.chart x) ≤ h.norm x

def LocalChart_frechet_manifold_chart
    (M : Type u) : Type u :=
  M → M

def Transform_frechet_manifold_chart
    (M : Type u) : Type u :=
  M → M → M

def Generator_frechet_manifold_chart
    (M : Type u) : Type u :=
  Nat → M → M

theorem local_estimate_frechet_manifold_chart
    {M : Type u} [h : AnalysisStruct_frechet_manifold_chart M] :
    NormCtrl_frechet_manifold_chart (M := M) := by
  intro x
  have hLocal : h.norm (h.chart x) ≤ h.norm x := h.chart_bound x
  have hStable : h.norm (h.chart (h.chart x)) ≤ h.norm (h.chart x) :=
    h.chart_bound (h.chart x)
  have _ : h.norm (h.chart (h.chart x)) ≤ h.norm x :=
    h.nat_le_trans _ _ _ hStable hLocal
  exact hLocal

theorem patching_estimate_frechet_manifold_chart
    {M : Type u} [h : AnalysisStruct_frechet_manifold_chart M]
    (x y : M) :
    h.norm (h.transform (h.chart x) y) ≤ h.norm x + h.norm y := by
  have hTrans : h.norm (h.transform (h.chart x) y) ≤ h.norm (h.chart x) + h.norm y :=
    h.transform_bound (h.chart x) y
  have hx : h.norm (h.chart x) ≤ h.norm x := h.chart_bound x
  have hy : h.norm y ≤ h.norm y := h.nat_le_refl (h.norm y)
  have hLift : h.norm (h.chart x) + h.norm y ≤ h.norm x + h.norm y :=
    h.nat_add_mono _ _ _ _ hx hy
  exact h.nat_le_trans _ _ _ hTrans hLift

theorem transform_isometry_frechet_manifold_chart
    {M : Type u} [h : AnalysisStruct_frechet_manifold_chart M]
    (x y : M)
    (hIso : h.norm (h.transform x y) = h.norm x + h.norm y) :
    h.norm (h.transform x y) = h.norm y + h.norm x ∧
    h.norm (h.chart (h.chart x)) ≤ h.norm x := by
  have hComm : h.norm x + h.norm y = h.norm y + h.norm x :=
    h.nat_add_comm (h.norm x) (h.norm y)
  have hEq : h.norm (h.transform x y) = h.norm y + h.norm x := by
    calc
      h.norm (h.transform x y) = h.norm x + h.norm y := hIso
      _ = h.norm y + h.norm x := hComm
  have hChart1 : h.norm (h.chart (h.chart x)) ≤ h.norm (h.chart x) :=
    h.chart_bound (h.chart x)
  have hChart2 : h.norm (h.chart x) ≤ h.norm x := h.chart_bound x
  have hChart : h.norm (h.chart (h.chart x)) ≤ h.norm x :=
    h.nat_le_trans _ _ _ hChart1 hChart2
  exact And.intro hEq hChart

theorem decomposition_bound_frechet_manifold_chart
    {M : Type u} [h : AnalysisStruct_frechet_manifold_chart M]
    (x : M) :
    h.norm (h.generator 1 x) ≤ h.norm x + h.norm x := by
  have hStep : h.norm (h.generator (0 + 1) x) ≤ h.norm (h.generator 0 x) + h.norm x :=
    h.generator_step 0 x
  have hZero : h.generator 0 x = x := h.generator_zero x
  have hRewrite : h.norm (h.generator 1 x) ≤ h.norm x + h.norm x := by
    calc
      h.norm (h.generator 1 x) = h.norm (h.generator (0 + 1) x) := by rfl
      _ ≤ h.norm (h.generator 0 x) + h.norm x := hStep
      _ = h.norm x + h.norm x := by rw [hZero]
  exact hRewrite

theorem symbol_composition_frechet_manifold_chart
    {M : Type u} [h : AnalysisStruct_frechet_manifold_chart M]
    (x y : M) :
    h.norm (h.chart (h.transform (h.chart x) (h.chart y))) ≤ h.norm x + h.norm y := by
  have hPatch : h.norm (h.transform (h.chart x) (h.chart y)) ≤ h.norm x + h.norm y := by
    have hFirst : h.norm (h.transform (h.chart x) (h.chart y)) ≤ h.norm (h.chart x) + h.norm (h.chart y) :=
      h.transform_bound (h.chart x) (h.chart y)
    have hx : h.norm (h.chart x) ≤ h.norm x := h.chart_bound x
    have hy : h.norm (h.chart y) ≤ h.norm y := h.chart_bound y
    have hAdd : h.norm (h.chart x) + h.norm (h.chart y) ≤ h.norm x + h.norm y :=
      h.nat_add_mono _ _ _ _ hx hy
    exact h.nat_le_trans _ _ _ hFirst hAdd
  have hChart : h.norm (h.chart (h.transform (h.chart x) (h.chart y))) ≤ h.norm (h.transform (h.chart x) (h.chart y)) :=
    h.chart_bound (h.transform (h.chart x) (h.chart y))
  exact h.nat_le_trans _ _ _ hChart hPatch

theorem semigroup_generation_frechet_manifold_chart
    {M : Type u} [h : AnalysisStruct_frechet_manifold_chart M]
    (m n : Nat)
    (x : M) :
    h.generator (m + n) x = h.generator m (h.generator n x) ∧
    h.norm (h.chart (h.generator (m + n) x)) ≤ h.norm (h.generator m (h.generator n x)) := by
  have hSemi : h.generator (m + n) x = h.generator m (h.generator n x) :=
    h.semigroup_axiom m n x
  have hChartRaw : h.norm (h.chart (h.generator m (h.generator n x))) ≤ h.norm (h.generator m (h.generator n x)) :=
    h.chart_bound (h.generator m (h.generator n x))
  have hChart : h.norm (h.chart (h.generator (m + n) x)) ≤ h.norm (h.generator m (h.generator n x)) := by
    calc
      h.norm (h.chart (h.generator (m + n) x))
          = h.norm (h.chart (h.generator m (h.generator n x))) := by rw [hSemi]
      _ ≤ h.norm (h.generator m (h.generator n x)) := hChartRaw
  exact And.intro hSemi hChart

theorem regularity_upgrade_frechet_manifold_chart
    {M : Type u} [h : AnalysisStruct_frechet_manifold_chart M]
    (x : M) :
    h.norm (h.generator 1 (h.chart x)) ≤ h.norm x ∧
    h.norm (h.chart (h.generator 1 (h.chart x))) ≤ h.norm x := by
  have hReg : h.norm (h.generator 1 (h.chart x)) ≤ h.norm (h.chart x) :=
    h.regularity_axiom x
  have hChart : h.norm (h.chart x) ≤ h.norm x := h.chart_bound x
  have hFirst : h.norm (h.generator 1 (h.chart x)) ≤ h.norm x :=
    h.nat_le_trans _ _ _ hReg hChart
  have hOuter : h.norm (h.chart (h.generator 1 (h.chart x))) ≤ h.norm (h.generator 1 (h.chart x)) :=
    h.chart_bound (h.generator 1 (h.chart x))
  have hSecond : h.norm (h.chart (h.generator 1 (h.chart x))) ≤ h.norm x :=
    h.nat_le_trans _ _ _ hOuter hFirst
  exact And.intro hFirst hSecond
