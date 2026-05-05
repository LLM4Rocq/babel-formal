/-
BENCHMARK_ID: TINY_MATHLIB_BATCH05_ANALYSIS_FOURIER_PLANCHEREL_AXIOMATIC_LIKE
PAIR_STEM: analysis_fourier_plancherel_axiomatic_like
MATH_DOMAIN: Analysis
SOURCE_MATHLIB: Mathlib/Analysis/Fourier
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 17
-/

universe u

class AnalysisStruct_fourier_plancherel (V : Type u) where
  norm : V → Nat
  chart : V → V
  transform : V → V
  generator : V → V
  nat_le_trans : ∀ a b c : Nat, a ≤ b → b ≤ c → a ≤ c
  nat_add_mono : ∀ a b c d : Nat, a ≤ b → c ≤ d → a + c ≤ b + d
  nat_le_refl : ∀ a : Nat, a ≤ a
  nat_add_comm : ∀ a b : Nat, a + b = b + a
  nat_le_add_right : ∀ a b : Nat, a ≤ a + b
  local_estimate_axiom :
    ∀ x : V,
      norm (transform x) ≤ norm (chart x) + norm x
  patching_estimate_axiom :
    ∀ x y : V,
      norm (chart x) ≤ norm (chart y) + norm x + norm y
  isometry_axiom :
    ∀ x : V,
      norm (transform x) = norm x
  decomposition_axiom :
    ∀ x y : V,
      norm (transform x) + norm (transform y) = norm x + norm y
  symbol_composition_axiom :
    ∀ x : V,
      transform (generator x) = generator (transform x)
  semigroup_axiom :
    ∀ x : V,
      norm (generator x) ≤ norm x + norm (transform x)
  regularity_axiom :
    ∀ x : V,
      norm (chart (generator x)) ≤ norm (generator x) + norm (chart x)

def NormCtrl_fourier_plancherel
    (V : Type u) : Type u :=
  V → Nat

def LocalChart_fourier_plancherel
    (V : Type u) : Type u :=
  V → V

def Transform_fourier_plancherel
    (V : Type u) : Type u :=
  V → V

def Generator_fourier_plancherel
    (V : Type u) : Type u :=
  V → V

theorem local_estimate_fourier_plancherel
    {V : Type u} [h : AnalysisStruct_fourier_plancherel V]
    (x : V) :
    h.norm (h.transform x) ≤ h.norm (h.chart x) + h.norm x ∧
      h.norm (h.chart x) ≤ h.norm (h.chart x) := by
  have hLoc : h.norm (h.transform x) ≤ h.norm (h.chart x) + h.norm x :=
    h.local_estimate_axiom x
  have hDiag : h.norm (h.chart x) ≤ h.norm (h.chart x) := h.nat_le_refl (h.norm (h.chart x))
  exact And.intro hLoc hDiag

theorem patching_estimate_fourier_plancherel
    {V : Type u} [h : AnalysisStruct_fourier_plancherel V]
    (x y : V) :
    h.norm (h.chart x) ≤ h.norm (h.chart y) + h.norm x + h.norm y ∧
    h.norm (h.transform y) = h.norm y := by
  have hPatch : h.norm (h.chart x) ≤ h.norm (h.chart y) + h.norm x + h.norm y :=
    h.patching_estimate_axiom x y
  have hIsoY : h.norm (h.transform y) = h.norm y := h.isometry_axiom y
  exact And.intro hPatch hIsoY

theorem transform_isometry_fourier_plancherel
    {V : Type u} [h : AnalysisStruct_fourier_plancherel V]
    (x : V) :
    h.norm (h.transform x) = h.norm x ∧
    h.norm (h.transform x) ≤ h.norm x + h.norm x := by
  have hIso : h.norm (h.transform x) = h.norm x := h.isometry_axiom x
  have hBound : h.norm x ≤ h.norm x + h.norm x := by
    exact h.nat_le_add_right (h.norm x) (h.norm x)
  have hFinal : h.norm (h.transform x) ≤ h.norm x + h.norm x := by
    rw [hIso]
    exact hBound
  exact And.intro hIso hFinal

theorem decomposition_bound_fourier_plancherel
    {V : Type u} [h : AnalysisStruct_fourier_plancherel V]
    (x y : V) :
    h.norm (h.transform x) + h.norm (h.transform y) = h.norm x + h.norm y ∧
    h.norm (h.transform x) + h.norm (h.transform y) = h.norm y + h.norm x := by
  have hDec : h.norm (h.transform x) + h.norm (h.transform y) = h.norm x + h.norm y :=
    h.decomposition_axiom x y
  have hComm : h.norm x + h.norm y = h.norm y + h.norm x :=
    h.nat_add_comm (h.norm x) (h.norm y)
  have hSwap : h.norm (h.transform x) + h.norm (h.transform y) = h.norm y + h.norm x := by
    calc
      h.norm (h.transform x) + h.norm (h.transform y) = h.norm x + h.norm y := hDec
      _ = h.norm y + h.norm x := hComm
  exact And.intro hDec hSwap

theorem symbol_composition_fourier_plancherel
    {V : Type u} [h : AnalysisStruct_fourier_plancherel V]
    (x : V) :
    h.transform (h.generator x) = h.generator (h.transform x) ∧
    h.norm (h.generator x) ≤ h.norm x + h.norm (h.transform x) := by
  have hEq : h.transform (h.generator x) = h.generator (h.transform x) :=
    h.symbol_composition_axiom x
  have hBound : h.norm (h.generator x) ≤ h.norm x + h.norm (h.transform x) :=
    h.semigroup_axiom x
  exact And.intro hEq hBound

theorem semigroup_generation_fourier_plancherel
    {V : Type u} [h : AnalysisStruct_fourier_plancherel V]
    (x : V) :
    h.norm (h.generator x) ≤ h.norm x + h.norm x := by
  have hBase : h.norm (h.generator x) ≤ h.norm x + h.norm (h.transform x) :=
    h.semigroup_axiom x
  have hIso : h.norm (h.transform x) = h.norm x := h.isometry_axiom x
  have hRewrite : h.norm (h.generator x) ≤ h.norm x + h.norm x := by
    rw [hIso] at hBase
    exact hBase
  exact hRewrite

theorem regularity_upgrade_fourier_plancherel
    {V : Type u} [h : AnalysisStruct_fourier_plancherel V]
    (x : V) :
    h.norm (h.chart (h.generator x)) ≤ (h.norm x + h.norm x) + h.norm (h.chart x) := by
  have hReg : h.norm (h.chart (h.generator x)) ≤ h.norm (h.generator x) + h.norm (h.chart x) :=
    h.regularity_axiom x
  have hGen : h.norm (h.generator x) ≤ h.norm x + h.norm x :=
    semigroup_generation_fourier_plancherel (V := V) x
  have hLift : h.norm (h.generator x) + h.norm (h.chart x) ≤
      (h.norm x + h.norm x) + h.norm (h.chart x) :=
    h.nat_add_mono _ _ _ _ hGen (h.nat_le_refl (h.norm (h.chart x)))
  exact h.nat_le_trans _ _ _ hReg hLift

theorem chart_generator_patching_bound_fourier_plancherel
    {V : Type u} [h : AnalysisStruct_fourier_plancherel V]
    (x y : V) :
    h.norm (h.chart (h.generator x)) ≤
      (h.norm (h.chart y) + (h.norm x + h.norm x)) + h.norm y := by
  have hPatch :
      h.norm (h.chart (h.generator x)) ≤
        h.norm (h.chart y) + h.norm (h.generator x) + h.norm y :=
    h.patching_estimate_axiom (h.generator x) y
  have hGen : h.norm (h.generator x) ≤ h.norm x + h.norm x :=
    semigroup_generation_fourier_plancherel (V := V) x
  have hHead :
      h.norm (h.chart y) + h.norm (h.generator x) ≤
        h.norm (h.chart y) + (h.norm x + h.norm x) :=
    h.nat_add_mono _ _ _ _ (h.nat_le_refl (h.norm (h.chart y))) hGen
  have hTail :
      (h.norm (h.chart y) + h.norm (h.generator x)) + h.norm y ≤
        (h.norm (h.chart y) + (h.norm x + h.norm x)) + h.norm y :=
    h.nat_add_mono _ _ _ _ hHead (h.nat_le_refl (h.norm y))
  exact h.nat_le_trans _ _ _ hPatch hTail

theorem symbol_generator_isometry_bound_fourier_plancherel
    {V : Type u} [h : AnalysisStruct_fourier_plancherel V]
    (x : V) :
    h.transform (h.generator x) = h.generator (h.transform x) ∧
    h.norm (h.transform (h.generator x)) ≤ h.norm x + h.norm x := by
  have hSym : h.transform (h.generator x) = h.generator (h.transform x) :=
    h.symbol_composition_axiom x
  have hIsoGen :
      h.norm (h.transform (h.generator x)) = h.norm (h.generator x) :=
    h.isometry_axiom (h.generator x)
  have hGen : h.norm (h.generator x) ≤ h.norm x + h.norm x :=
    semigroup_generation_fourier_plancherel (V := V) x
  have hBound : h.norm (h.transform (h.generator x)) ≤ h.norm x + h.norm x := by
    rw [hIsoGen]
    exact hGen
  exact And.intro hSym hBound

theorem decomposition_generator_mix_bound_fourier_plancherel
    {V : Type u} [h : AnalysisStruct_fourier_plancherel V]
    (x y : V) :
    h.norm (h.transform (h.generator x)) + h.norm (h.transform y) ≤
      (h.norm x + h.norm x) + h.norm y := by
  have hDec :
      h.norm (h.transform (h.generator x)) + h.norm (h.transform y) =
        h.norm (h.generator x) + h.norm y :=
    h.decomposition_axiom (h.generator x) y
  have hGen : h.norm (h.generator x) ≤ h.norm x + h.norm x :=
    semigroup_generation_fourier_plancherel (V := V) x
  have hLift :
      h.norm (h.generator x) + h.norm y ≤
        (h.norm x + h.norm x) + h.norm y :=
    h.nat_add_mono _ _ _ _ hGen (h.nat_le_refl (h.norm y))
  have hFinal :
      h.norm (h.transform (h.generator x)) + h.norm (h.transform y) ≤
        (h.norm x + h.norm x) + h.norm y := by
    rw [hDec]
    exact hLift
  exact hFinal

theorem semigroup_transform_input_bound_fourier_plancherel
    {V : Type u} [h : AnalysisStruct_fourier_plancherel V]
    (x : V) :
    h.norm (h.generator (h.transform x)) ≤ h.norm x + h.norm x := by
  have hGenT :
      h.norm (h.generator (h.transform x)) ≤
        h.norm (h.transform x) + h.norm (h.transform x) :=
    semigroup_generation_fourier_plancherel (V := V) (h.transform x)
  have hIso : h.norm (h.transform x) = h.norm x := h.isometry_axiom x
  have hFinal : h.norm (h.generator (h.transform x)) ≤ h.norm x + h.norm x := by
    rw [hIso] at hGenT
    exact hGenT
  exact hFinal

theorem double_transform_local_patching_bound_fourier_plancherel
    {V : Type u} [h : AnalysisStruct_fourier_plancherel V]
    (x y : V) :
    h.norm (h.transform (h.transform x)) ≤
      (h.norm (h.chart y) + h.norm x + h.norm y) + h.norm x := by
  have hLocal :
      h.norm (h.transform (h.transform x)) ≤
        h.norm (h.chart (h.transform x)) + h.norm (h.transform x) :=
    h.local_estimate_axiom (h.transform x)
  have hPatch :
      h.norm (h.chart (h.transform x)) ≤
        h.norm (h.chart y) + h.norm (h.transform x) + h.norm y :=
    h.patching_estimate_axiom (h.transform x) y
  have hIso : h.norm (h.transform x) = h.norm x := h.isometry_axiom x
  have hPatch' :
      h.norm (h.chart (h.transform x)) ≤
        h.norm (h.chart y) + h.norm x + h.norm y := by
    rw [hIso] at hPatch
    exact hPatch
  have hLift :
      h.norm (h.chart (h.transform x)) + h.norm (h.transform x) ≤
        (h.norm (h.chart y) + h.norm x + h.norm y) + h.norm (h.transform x) :=
    h.nat_add_mono _ _ _ _ hPatch' (h.nat_le_refl (h.norm (h.transform x)))
  have hIsoLe : h.norm (h.transform x) ≤ h.norm x := by
    rw [hIso]
    exact h.nat_le_refl (h.norm x)
  have hTail :
      (h.norm (h.chart y) + h.norm x + h.norm y) + h.norm (h.transform x) ≤
        (h.norm (h.chart y) + h.norm x + h.norm y) + h.norm x :=
    h.nat_add_mono _ _ _ _
      (h.nat_le_refl (h.norm (h.chart y) + h.norm x + h.norm y)) hIsoLe
  have hMid :
      h.norm (h.chart (h.transform x)) + h.norm (h.transform x) ≤
        (h.norm (h.chart y) + h.norm x + h.norm y) + h.norm x :=
    h.nat_le_trans _ _ _ hLift hTail
  exact h.nat_le_trans _ _ _ hLocal hMid
