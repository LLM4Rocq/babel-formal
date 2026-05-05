/-
BENCHMARK_ID: TINY_MATHLIB_BATCH05_ANALYSIS_SEMIGROUP_GENERATOR_HILLE_YOSIDA_LIKE
PAIR_STEM: analysis_semigroup_generator_hille_yosida_like
MATH_DOMAIN: Analysis
SOURCE_MATHLIB: Mathlib/Analysis/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class AnalysisStruct_semigroup_generator_hille (E : Type u) where
  zero : E
  add : E → E → E
  norm : E → Nat
  chart : E → E
  transform : Nat → E → E
  generator : E → E
  norm_add_le : ∀ x y : E, norm (add x y) ≤ norm x + norm y
  chart_bound : ∀ x : E, norm (chart x) ≤ norm x
  chart_idem : ∀ x : E, chart (chart x) = chart x
  transform_zero : ∀ n : Nat, transform n zero = zero
  transform_add : ∀ n : Nat, ∀ x y : E, transform n (add x y) = add (transform n x) (transform n y)
  semigroup_axiom : ∀ m n : Nat, ∀ x : E, transform (m + n) x = transform m (transform n x)
  generator_def : ∀ x : E, generator x = transform 1 x
  isometry_one : ∀ x : E, norm (transform 1 x) = norm x
  nat_le_trans_axiom :
    ∀ a b c : Nat,
      a ≤ b →
      b ≤ c →
        a ≤ c
  nat_le_refl_axiom :
    ∀ a : Nat, a ≤ a
  step_bound : ∀ n : Nat, ∀ x : E, norm (transform (n + 1) x) ≤ norm (transform n x) + norm x
  generator_regularity : ∀ x : E, norm (generator (chart x)) ≤ norm (chart x)

infixl:65 " +ₕ " => AnalysisStruct_semigroup_generator_hille.add

def NormCtrl_semigroup_generator_hille
    (E : Type u) : Type u :=
  E → Nat

def LocalChart_semigroup_generator_hille
    (E : Type u) : Type u :=
  E → E

def Transform_semigroup_generator_hille
    (E : Type u) : Type u :=
  Nat → E → E

def Generator_semigroup_generator_hille
    (E : Type u) : Type u :=
  E → E

theorem local_estimate_semigroup_generator_hille
    {E : Type u} [h : AnalysisStruct_semigroup_generator_hille E]
    (x : E) :
    h.norm (h.chart x) ≤ h.norm x := by
  have hRaw : h.norm (h.chart x) ≤ h.norm x := h.chart_bound x
  exact hRaw

theorem patching_estimate_semigroup_generator_hille
    {E : Type u} [h : AnalysisStruct_semigroup_generator_hille E]
    (x y : E) :
    h.norm (h.transform 1 (x +ₕ y)) ≤ h.norm x + h.norm y := by
  have hExpand : h.transform 1 (x +ₕ y) = h.add (h.transform 1 x) (h.transform 1 y) :=
    h.transform_add 1 x y
  have hNormAdd :
      h.norm (h.add (h.transform 1 x) (h.transform 1 y)) ≤
        h.norm (h.transform 1 x) + h.norm (h.transform 1 y) :=
    h.norm_add_le (h.transform 1 x) (h.transform 1 y)
  have hIsoX : h.norm (h.transform 1 x) = h.norm x := h.isometry_one x
  have hIsoY : h.norm (h.transform 1 y) = h.norm y := h.isometry_one y
  calc
    h.norm (h.transform 1 (x +ₕ y)) = h.norm (h.add (h.transform 1 x) (h.transform 1 y)) := by
      rw [hExpand]
    _ ≤ h.norm (h.transform 1 x) + h.norm (h.transform 1 y) := hNormAdd
    _ = h.norm x + h.norm y := by rw [hIsoX, hIsoY]

theorem transform_isometry_semigroup_generator_hille
    {E : Type u} [h : AnalysisStruct_semigroup_generator_hille E]
    (x : E) :
    h.norm (h.transform 1 x) = h.norm x := by
  exact h.isometry_one x

theorem decomposition_bound_semigroup_generator_hille
    {E : Type u} [h : AnalysisStruct_semigroup_generator_hille E]
    (x y : E) :
    h.norm (h.generator (x +ₕ y)) ≤ h.norm x + h.norm y := by
  have hGen : h.generator (x +ₕ y) = h.transform 1 (x +ₕ y) := h.generator_def (x +ₕ y)
  calc
    h.norm (h.generator (x +ₕ y)) = h.norm (h.transform 1 (x +ₕ y)) := by rw [hGen]
    _ ≤ h.norm x + h.norm y := patching_estimate_semigroup_generator_hille x y

theorem symbol_composition_semigroup_generator_hille
    {E : Type u} [h : AnalysisStruct_semigroup_generator_hille E]
    (m n : Nat)
    (x : E) :
    h.transform (m + n) (h.chart x) = h.transform m (h.transform n (h.chart x)) := by
  have hSemi : h.transform (m + n) (h.chart x) = h.transform m (h.transform n (h.chart x)) :=
    h.semigroup_axiom m n (h.chart x)
  exact hSemi

theorem semigroup_generation_semigroup_generator_hille
    {E : Type u} [h : AnalysisStruct_semigroup_generator_hille E]
    (x : E) :
    h.norm (h.generator x) ≤ h.norm x := by
  have hDef : h.generator x = h.transform 1 x := h.generator_def x
  have hIso : h.norm (h.transform 1 x) = h.norm x := h.isometry_one x
  have hEq : h.norm (h.generator x) = h.norm x := by
    calc
      h.norm (h.generator x) = h.norm (h.transform 1 x) := by rw [hDef]
      _ = h.norm x := hIso
  rw [hEq]
  exact h.nat_le_refl_axiom (h.norm x)

theorem regularity_upgrade_semigroup_generator_hille
    {E : Type u} [h : AnalysisStruct_semigroup_generator_hille E]
    (x : E) :
    h.norm (h.generator (h.chart x)) ≤ h.norm x := by
  have hReg : h.norm (h.generator (h.chart x)) ≤ h.norm (h.chart x) := h.generator_regularity x
  have hChart : h.norm (h.chart x) ≤ h.norm x := h.chart_bound x
  exact h.nat_le_trans_axiom (h.norm (h.generator (h.chart x))) (h.norm (h.chart x)) (h.norm x) hReg hChart
