/-
BENCHMARK_ID: TINY_MATHLIB_BATCH07_ANALYSIS_NONARCHIMEDEAN_OPEN_MAPPING_LIKE
PAIR_STEM: analysis_nonarchimedean_open_mapping_like
MATH_DOMAIN: Analysis
SOURCE_MATHLIB: Mathlib/Analysis/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class FrameworkStruct_analysis_nonarchimedean_open_mapping (E : Type u) where
  seminorm : E → Nat
  add : E → E → E
  zero : E
  map : E → E
  radius : E → Nat
  le_trans_nat : ∀ a b c : Nat, a ≤ b → b ≤ c → a ≤ c
  add_le_add_left_nat : ∀ a b c : Nat, a ≤ b → c + a ≤ c + b
  add_le_add_right_nat : ∀ a b c : Nat, a ≤ b → a + c ≤ b + c
  le_add_left_nat : ∀ a b : Nat, b ≤ a + b
  le_add_right_nat : ∀ a b : Nat, a ≤ a + b
  map_bound : ∀ x : E, seminorm (map x) ≤ seminorm x + radius x
  map_contract : ∀ x : E, seminorm (map x) ≤ seminorm x
  add_bound : ∀ x y : E, seminorm (add x y) ≤ seminorm x + seminorm y
  zero_norm : seminorm zero = 0
  add_zero_right : ∀ x : E, add x zero = x
  add_zero_left : ∀ x : E, add zero x = x
  open_surj : ∀ x : E, ∃ y : E, map y = x
  preimage_bound : ∀ x y : E, map y = x → seminorm y ≤ seminorm x + radius x
  radius_control : ∀ x : E, radius (map x) ≤ radius x + radius x

structure ContextData_analysis_nonarchimedean_open_mapping
    (E : Type u) [s : FrameworkStruct_analysis_nonarchimedean_open_mapping E] where
  anchor : E
  target : E
  anchor_small : s.seminorm anchor ≤ s.seminorm target + s.radius target
  target_large : s.seminorm target ≤ s.seminorm target + s.seminorm target

def primary_map_analysis_nonarchimedean_open_mapping
    {E : Type u} [s : FrameworkStruct_analysis_nonarchimedean_open_mapping E]
    (d : ContextData_analysis_nonarchimedean_open_mapping E) : E :=
  s.add (s.map d.anchor) d.target

def secondary_map_analysis_nonarchimedean_open_mapping
    {E : Type u} [s : FrameworkStruct_analysis_nonarchimedean_open_mapping E]
    (d : ContextData_analysis_nonarchimedean_open_mapping E) : Nat :=
  s.seminorm (primary_map_analysis_nonarchimedean_open_mapping (E := E) d) + s.radius d.target

def tertiary_map_analysis_nonarchimedean_open_mapping
    {E : Type u} [s : FrameworkStruct_analysis_nonarchimedean_open_mapping E]
    (d : ContextData_analysis_nonarchimedean_open_mapping E) : Prop :=
  s.radius d.target ≤ secondary_map_analysis_nonarchimedean_open_mapping (E := E) d

theorem stability_step_analysis_nonarchimedean_open_mapping
    {E : Type u} [s : FrameworkStruct_analysis_nonarchimedean_open_mapping E]
    (d : ContextData_analysis_nonarchimedean_open_mapping E) :
    (s.seminorm (s.map d.anchor) ≤ s.seminorm d.anchor + s.radius d.anchor ∧
    s.radius (s.map d.anchor) ≤ s.radius d.anchor + s.radius d.anchor) := by
  have hMap : s.seminorm (s.map d.anchor) ≤ s.seminorm d.anchor + s.radius d.anchor :=
    s.map_bound d.anchor
  have hRad : s.radius (s.map d.anchor) ≤ s.radius d.anchor + s.radius d.anchor :=
    s.radius_control d.anchor
  constructor
  · exact hMap
  · exact hRad

theorem factorization_step_analysis_nonarchimedean_open_mapping
    {E : Type u} [s : FrameworkStruct_analysis_nonarchimedean_open_mapping E]
    (d : ContextData_analysis_nonarchimedean_open_mapping E)
    (w : E)
    (hw : s.map w = d.target) :
    (s.seminorm w ≤ s.seminorm d.target + s.radius d.target ∧
    s.seminorm (s.map w) ≤ s.seminorm w) := by
  have hPre : s.seminorm w ≤ s.seminorm d.target + s.radius d.target :=
    s.preimage_bound d.target w hw
  have hCon : s.seminorm (s.map w) ≤ s.seminorm w :=
    s.map_contract w
  constructor
  · exact hPre
  · exact hCon

theorem comparison_step_analysis_nonarchimedean_open_mapping
    {E : Type u} [s : FrameworkStruct_analysis_nonarchimedean_open_mapping E]
    (d : ContextData_analysis_nonarchimedean_open_mapping E) :
    (s.seminorm (primary_map_analysis_nonarchimedean_open_mapping (E := E) d)
      ≤ s.seminorm (s.map d.anchor) + s.seminorm d.target ∧
    tertiary_map_analysis_nonarchimedean_open_mapping (E := E) d) := by
  have hPrimary :
      s.seminorm (primary_map_analysis_nonarchimedean_open_mapping (E := E) d)
        ≤ s.seminorm (s.map d.anchor) + s.seminorm d.target := by
    simpa [primary_map_analysis_nonarchimedean_open_mapping] using s.add_bound (s.map d.anchor) d.target
  have hRadius :
      s.radius d.target ≤
        s.seminorm (primary_map_analysis_nonarchimedean_open_mapping (E := E) d) + s.radius d.target := by
    exact s.le_add_left_nat (s.seminorm (primary_map_analysis_nonarchimedean_open_mapping (E := E) d))
      (s.radius d.target)
  have hTertiary : tertiary_map_analysis_nonarchimedean_open_mapping (E := E) d := by
    simpa [tertiary_map_analysis_nonarchimedean_open_mapping, secondary_map_analysis_nonarchimedean_open_mapping] using hRadius
  constructor
  · exact hPrimary
  · exact hTertiary

theorem transport_step_analysis_nonarchimedean_open_mapping
    {E : Type u} [s : FrameworkStruct_analysis_nonarchimedean_open_mapping E]
    (d : ContextData_analysis_nonarchimedean_open_mapping E) :
    (s.seminorm (primary_map_analysis_nonarchimedean_open_mapping (E := E) d)
      ≤ secondary_map_analysis_nonarchimedean_open_mapping (E := E) d ∧
    s.seminorm d.target ≤
      secondary_map_analysis_nonarchimedean_open_mapping (E := E) d + s.seminorm d.target ∧
    tertiary_map_analysis_nonarchimedean_open_mapping (E := E) d) := by
  have hComp := comparison_step_analysis_nonarchimedean_open_mapping (E := E) d
  rcases hComp with ⟨hPrimary, hTer⟩
  have hFirst :
      s.seminorm (primary_map_analysis_nonarchimedean_open_mapping (E := E) d)
        ≤ secondary_map_analysis_nonarchimedean_open_mapping (E := E) d := by
    have hLift :
        s.seminorm (primary_map_analysis_nonarchimedean_open_mapping (E := E) d)
          ≤ s.seminorm (primary_map_analysis_nonarchimedean_open_mapping (E := E) d) + s.radius d.target :=
      s.le_add_right_nat _ _
    simpa [secondary_map_analysis_nonarchimedean_open_mapping] using hLift
  have hSecond :
      s.seminorm d.target ≤
        secondary_map_analysis_nonarchimedean_open_mapping (E := E) d + s.seminorm d.target :=
    s.le_add_left_nat (secondary_map_analysis_nonarchimedean_open_mapping (E := E) d) (s.seminorm d.target)
  exact ⟨hFirst, hSecond, hTer⟩

theorem coherence_step_analysis_nonarchimedean_open_mapping
    {E : Type u} [s : FrameworkStruct_analysis_nonarchimedean_open_mapping E]
    (d : ContextData_analysis_nonarchimedean_open_mapping E) :
    (s.seminorm d.target ≤ secondary_map_analysis_nonarchimedean_open_mapping (E := E) d + s.seminorm d.target ∧
    tertiary_map_analysis_nonarchimedean_open_mapping (E := E) d) := by
  have hLeft :
      s.seminorm d.target ≤ secondary_map_analysis_nonarchimedean_open_mapping (E := E) d + s.seminorm d.target :=
    s.le_add_left_nat (secondary_map_analysis_nonarchimedean_open_mapping (E := E) d) (s.seminorm d.target)
  have hComp := comparison_step_analysis_nonarchimedean_open_mapping (E := E) d
  rcases hComp with ⟨_hPrimary, hTer⟩
  exact ⟨hLeft, hTer⟩

theorem iteration_step_analysis_nonarchimedean_open_mapping
    {E : Type u} [s : FrameworkStruct_analysis_nonarchimedean_open_mapping E]
    (d : ContextData_analysis_nonarchimedean_open_mapping E) :
    (∃ w : E,
      s.map w = d.target ∧
      s.seminorm w ≤ s.seminorm d.target + s.radius d.target ∧
      s.seminorm (s.map w) ≤ s.seminorm w) := by
  rcases s.open_surj d.target with ⟨w, hw⟩
  have hFact := factorization_step_analysis_nonarchimedean_open_mapping (E := E) d w hw
  rcases hFact with ⟨hNorm, hCon⟩
  refine ⟨w, hw, hNorm, hCon⟩

theorem main_result_analysis_nonarchimedean_open_mapping
    {E : Type u} [s : FrameworkStruct_analysis_nonarchimedean_open_mapping E]
    (d : ContextData_analysis_nonarchimedean_open_mapping E) :
    (∃ w : E,
      s.map w = d.target ∧
      s.seminorm w ≤ (s.seminorm d.target + s.radius d.target) + s.radius d.target ∧
      tertiary_map_analysis_nonarchimedean_open_mapping (E := E) d) := by
  rcases iteration_step_analysis_nonarchimedean_open_mapping (E := E) d with
      ⟨w, hwMap, hwNorm, hwCon⟩
  have hLift :
      s.seminorm w ≤ (s.seminorm d.target + s.radius d.target) + s.radius d.target :=
    s.le_trans_nat _ _ _ hwNorm (s.le_add_right_nat _ _)
  have hCoh := coherence_step_analysis_nonarchimedean_open_mapping (E := E) d
  rcases hCoh with ⟨hTarget, hTer⟩
  have _ : s.seminorm (s.map w) ≤ s.seminorm w := hwCon
  have _ : s.seminorm d.target ≤ secondary_map_analysis_nonarchimedean_open_mapping (E := E) d + s.seminorm d.target :=
    hTarget
  refine ⟨w, hwMap, hLift, hTer⟩
