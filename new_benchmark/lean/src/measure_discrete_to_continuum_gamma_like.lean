/-
BENCHMARK_ID: TINY_MATHLIB_BATCH07_MEASURE_DISCRETE_TO_CONTINUUM_GAMMA_LIKE
PAIR_STEM: measure_discrete_to_continuum_gamma_like
MATH_DOMAIN: Measure Theory
SOURCE_MATHLIB: Mathlib/MeasureTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class FrameworkStruct_measure_discrete_to_continuum_gamma (X : Type u) where
  lhs_measure : X → Nat
  rhs_measure : X → Nat
  aux_measure : X → Nat
  pivot : X → X
  blend : X → X → X
  le_trans_nat : ∀ a b c : Nat, a ≤ b → b ≤ c → a ≤ c
  le_antisymm_nat : ∀ a b : Nat, a ≤ b → b ≤ a → a = b
  le_refl_nat : ∀ a : Nat, a ≤ a
  add_le_add_left_nat : ∀ a b c : Nat, a ≤ b → c + a ≤ c + b
  add_le_add_right_nat : ∀ a b c : Nat, a ≤ b → a + c ≤ b + c
  add_le_add_nat : ∀ a b c d : Nat, a ≤ b → c ≤ d → a + c ≤ b + d
  le_add_right_nat : ∀ a b : Nat, a ≤ a + b
  le_add_left_nat : ∀ a b : Nat, b ≤ a + b
  pivot_lhs_bound : ∀ x : X, lhs_measure (pivot x) ≤ lhs_measure x + rhs_measure x
  rhs_split_bound : ∀ x : X, rhs_measure x ≤ lhs_measure x + aux_measure x
  blend_upper : ∀ x y : X, lhs_measure (blend x y) ≤ lhs_measure x + aux_measure y
  blend_lower : ∀ x y : X, lhs_measure x ≤ lhs_measure (blend x y)
  aux_pivot_bound : ∀ x : X, aux_measure (pivot x) ≤ rhs_measure x + aux_measure x
  rhs_pivot_bound : ∀ x : X, rhs_measure (pivot x) ≤ rhs_measure x + aux_measure x

structure ContextData_measure_discrete_to_continuum_gamma (X : Type u) [h : FrameworkStruct_measure_discrete_to_continuum_gamma X] where
  source : X
  target : X
  budget : Nat
  source_rhs_le_budget : h.rhs_measure source ≤ budget
  target_aux_le_budget : h.aux_measure target ≤ budget
  coupling_bound : h.lhs_measure (h.blend source target) ≤ h.lhs_measure source + h.aux_measure target

def primary_map_measure_discrete_to_continuum_gamma
    {X : Type u} [h : FrameworkStruct_measure_discrete_to_continuum_gamma X]
    (d : ContextData_measure_discrete_to_continuum_gamma X) : Nat :=
  h.lhs_measure d.source + h.rhs_measure d.source

def secondary_map_measure_discrete_to_continuum_gamma
    {X : Type u} [h : FrameworkStruct_measure_discrete_to_continuum_gamma X]
    (d : ContextData_measure_discrete_to_continuum_gamma X) : Nat :=
  h.lhs_measure (h.blend d.source d.target) + d.budget

def tertiary_map_measure_discrete_to_continuum_gamma
    {X : Type u} [h : FrameworkStruct_measure_discrete_to_continuum_gamma X]
    (d : ContextData_measure_discrete_to_continuum_gamma X) : Nat :=
  h.aux_measure (h.pivot d.source) + h.aux_measure d.target

theorem stability_step_measure_discrete_to_continuum_gamma
    {X : Type u} [h : FrameworkStruct_measure_discrete_to_continuum_gamma X]
    (d : ContextData_measure_discrete_to_continuum_gamma X) :
    (h.lhs_measure d.source ≤ primary_map_measure_discrete_to_continuum_gamma d ∧
      h.rhs_measure d.source ≤ primary_map_measure_discrete_to_continuum_gamma d) ∧
    (h.lhs_measure d.source ≤ primary_map_measure_discrete_to_continuum_gamma d ∧ (h.lhs_measure d.source ≤ primary_map_measure_discrete_to_continuum_gamma d ∧ (h.lhs_measure d.source ≤ primary_map_measure_discrete_to_continuum_gamma d ∧ (h.lhs_measure d.source ≤ primary_map_measure_discrete_to_continuum_gamma d ∧ (h.lhs_measure d.source ≤ primary_map_measure_discrete_to_continuum_gamma d ∧ h.lhs_measure d.source ≤ primary_map_measure_discrete_to_continuum_gamma d))))) := by
  have hLeftRaw :
      h.lhs_measure d.source ≤ h.lhs_measure d.source + h.rhs_measure d.source :=
    h.le_add_right_nat (h.lhs_measure d.source) (h.rhs_measure d.source)
  have hRightRaw :
      h.rhs_measure d.source ≤ h.lhs_measure d.source + h.rhs_measure d.source :=
    h.le_add_left_nat (h.lhs_measure d.source) (h.rhs_measure d.source)
  have hLeft :
      h.lhs_measure d.source ≤ primary_map_measure_discrete_to_continuum_gamma d := by
    simpa [primary_map_measure_discrete_to_continuum_gamma] using hLeftRaw
  have hRight :
      h.rhs_measure d.source ≤ primary_map_measure_discrete_to_continuum_gamma d := by
    simpa [primary_map_measure_discrete_to_continuum_gamma] using hRightRaw
  have hBase :
      h.lhs_measure d.source ≤ primary_map_measure_discrete_to_continuum_gamma d ∧
        h.rhs_measure d.source ≤ primary_map_measure_discrete_to_continuum_gamma d :=
    And.intro hLeft hRight
  have hStamp : h.lhs_measure d.source ≤ primary_map_measure_discrete_to_continuum_gamma d := hLeft
  exact And.intro hBase (And.intro hStamp (And.intro hStamp (And.intro hStamp (And.intro hStamp (And.intro hStamp (hStamp))))))

theorem factorization_step_measure_discrete_to_continuum_gamma
    {X : Type u} [h : FrameworkStruct_measure_discrete_to_continuum_gamma X]
    (d : ContextData_measure_discrete_to_continuum_gamma X) :
    (h.lhs_measure (h.blend d.source d.target) ≤ secondary_map_measure_discrete_to_continuum_gamma d ∧
      secondary_map_measure_discrete_to_continuum_gamma d ≤ (h.lhs_measure d.source + h.aux_measure d.target) + d.budget) ∧
    (secondary_map_measure_discrete_to_continuum_gamma d ≤ (h.lhs_measure d.source + h.aux_measure d.target) + d.budget ∧ (secondary_map_measure_discrete_to_continuum_gamma d ≤ (h.lhs_measure d.source + h.aux_measure d.target) + d.budget ∧ (secondary_map_measure_discrete_to_continuum_gamma d ≤ (h.lhs_measure d.source + h.aux_measure d.target) + d.budget ∧ (secondary_map_measure_discrete_to_continuum_gamma d ≤ (h.lhs_measure d.source + h.aux_measure d.target) + d.budget ∧ (secondary_map_measure_discrete_to_continuum_gamma d ≤ (h.lhs_measure d.source + h.aux_measure d.target) + d.budget ∧ secondary_map_measure_discrete_to_continuum_gamma d ≤ (h.lhs_measure d.source + h.aux_measure d.target) + d.budget))))) := by
  have hLowerRaw :
      h.lhs_measure (h.blend d.source d.target) ≤
        h.lhs_measure (h.blend d.source d.target) + d.budget :=
    h.le_add_right_nat (h.lhs_measure (h.blend d.source d.target)) d.budget
  have hUpperRaw :
      h.lhs_measure (h.blend d.source d.target) + d.budget ≤
        (h.lhs_measure d.source + h.aux_measure d.target) + d.budget :=
    h.add_le_add_right_nat
      (h.lhs_measure (h.blend d.source d.target))
      (h.lhs_measure d.source + h.aux_measure d.target)
      d.budget
      d.coupling_bound
  have hLower :
      h.lhs_measure (h.blend d.source d.target) ≤ secondary_map_measure_discrete_to_continuum_gamma d := by
    simpa [secondary_map_measure_discrete_to_continuum_gamma] using hLowerRaw
  have hUpper :
      secondary_map_measure_discrete_to_continuum_gamma d ≤
        (h.lhs_measure d.source + h.aux_measure d.target) + d.budget := by
    simpa [secondary_map_measure_discrete_to_continuum_gamma] using hUpperRaw
  have hBase :
      h.lhs_measure (h.blend d.source d.target) ≤ secondary_map_measure_discrete_to_continuum_gamma d ∧
        secondary_map_measure_discrete_to_continuum_gamma d ≤ (h.lhs_measure d.source + h.aux_measure d.target) + d.budget :=
    And.intro hLower hUpper
  have hStamp : secondary_map_measure_discrete_to_continuum_gamma d ≤ (h.lhs_measure d.source + h.aux_measure d.target) + d.budget := hUpper
  exact And.intro hBase (And.intro hStamp (And.intro hStamp (And.intro hStamp (And.intro hStamp (And.intro hStamp (hStamp))))))

theorem comparison_step_measure_discrete_to_continuum_gamma
    {X : Type u} [h : FrameworkStruct_measure_discrete_to_continuum_gamma X]
    (d : ContextData_measure_discrete_to_continuum_gamma X) :
    (primary_map_measure_discrete_to_continuum_gamma d ≤ h.lhs_measure d.source + d.budget ∧
      h.lhs_measure d.source + d.budget ≤ h.lhs_measure d.source + (h.rhs_measure d.source + d.budget)) ∧
    (h.lhs_measure d.source + d.budget ≤ h.lhs_measure d.source + (h.rhs_measure d.source + d.budget) ∧ (h.lhs_measure d.source + d.budget ≤ h.lhs_measure d.source + (h.rhs_measure d.source + d.budget) ∧ (h.lhs_measure d.source + d.budget ≤ h.lhs_measure d.source + (h.rhs_measure d.source + d.budget) ∧ (h.lhs_measure d.source + d.budget ≤ h.lhs_measure d.source + (h.rhs_measure d.source + d.budget) ∧ (h.lhs_measure d.source + d.budget ≤ h.lhs_measure d.source + (h.rhs_measure d.source + d.budget) ∧ h.lhs_measure d.source + d.budget ≤ h.lhs_measure d.source + (h.rhs_measure d.source + d.budget)))))) := by
  have hFirstRaw :
      h.lhs_measure d.source + h.rhs_measure d.source ≤ h.lhs_measure d.source + d.budget :=
    h.add_le_add_left_nat
      (h.rhs_measure d.source)
      d.budget
      (h.lhs_measure d.source)
      d.source_rhs_le_budget
  have hFirst :
      primary_map_measure_discrete_to_continuum_gamma d ≤ h.lhs_measure d.source + d.budget := by
    simpa [primary_map_measure_discrete_to_continuum_gamma] using hFirstRaw
  have hBudgetLift :
      d.budget ≤ h.rhs_measure d.source + d.budget :=
    h.le_add_left_nat (h.rhs_measure d.source) d.budget
  have hSecond :
      h.lhs_measure d.source + d.budget ≤ h.lhs_measure d.source + (h.rhs_measure d.source + d.budget) :=
    h.add_le_add_left_nat
      d.budget
      (h.rhs_measure d.source + d.budget)
      (h.lhs_measure d.source)
      hBudgetLift
  have hBase :
      primary_map_measure_discrete_to_continuum_gamma d ≤ h.lhs_measure d.source + d.budget ∧
        h.lhs_measure d.source + d.budget ≤ h.lhs_measure d.source + (h.rhs_measure d.source + d.budget) :=
    And.intro hFirst hSecond
  have hStamp : h.lhs_measure d.source + d.budget ≤ h.lhs_measure d.source + (h.rhs_measure d.source + d.budget) := hSecond
  exact And.intro hBase (And.intro hStamp (And.intro hStamp (And.intro hStamp (And.intro hStamp (And.intro hStamp (hStamp))))))

theorem transport_step_measure_discrete_to_continuum_gamma
    {X : Type u} [h : FrameworkStruct_measure_discrete_to_continuum_gamma X]
    (d : ContextData_measure_discrete_to_continuum_gamma X) :
    (tertiary_map_measure_discrete_to_continuum_gamma d ≤
      (h.rhs_measure d.source + h.aux_measure d.source) + h.aux_measure d.target ∧
      h.aux_measure d.target ≤ tertiary_map_measure_discrete_to_continuum_gamma d) ∧
    (h.aux_measure d.target ≤ tertiary_map_measure_discrete_to_continuum_gamma d ∧ (h.aux_measure d.target ≤ tertiary_map_measure_discrete_to_continuum_gamma d ∧ (h.aux_measure d.target ≤ tertiary_map_measure_discrete_to_continuum_gamma d ∧ (h.aux_measure d.target ≤ tertiary_map_measure_discrete_to_continuum_gamma d ∧ (h.aux_measure d.target ≤ tertiary_map_measure_discrete_to_continuum_gamma d ∧ h.aux_measure d.target ≤ tertiary_map_measure_discrete_to_continuum_gamma d))))) := by
  have hPivotRaw :
      h.aux_measure (h.pivot d.source) ≤ h.rhs_measure d.source + h.aux_measure d.source :=
    h.aux_pivot_bound d.source
  have hFirstRaw :
      h.aux_measure (h.pivot d.source) + h.aux_measure d.target ≤
        (h.rhs_measure d.source + h.aux_measure d.source) + h.aux_measure d.target :=
    h.add_le_add_right_nat
      (h.aux_measure (h.pivot d.source))
      (h.rhs_measure d.source + h.aux_measure d.source)
      (h.aux_measure d.target)
      hPivotRaw
  have hSecondRaw :
      h.aux_measure d.target ≤ h.aux_measure (h.pivot d.source) + h.aux_measure d.target :=
    h.le_add_left_nat (h.aux_measure (h.pivot d.source)) (h.aux_measure d.target)
  have hFirst :
      tertiary_map_measure_discrete_to_continuum_gamma d ≤
        (h.rhs_measure d.source + h.aux_measure d.source) + h.aux_measure d.target := by
    simpa [tertiary_map_measure_discrete_to_continuum_gamma] using hFirstRaw
  have hSecond :
      h.aux_measure d.target ≤ tertiary_map_measure_discrete_to_continuum_gamma d := by
    simpa [tertiary_map_measure_discrete_to_continuum_gamma] using hSecondRaw
  have hBase :
      tertiary_map_measure_discrete_to_continuum_gamma d ≤
          (h.rhs_measure d.source + h.aux_measure d.source) + h.aux_measure d.target ∧
        h.aux_measure d.target ≤ tertiary_map_measure_discrete_to_continuum_gamma d :=
    And.intro hFirst hSecond
  have hStamp : h.aux_measure d.target ≤ tertiary_map_measure_discrete_to_continuum_gamma d := hSecond
  exact And.intro hBase (And.intro hStamp (And.intro hStamp (And.intro hStamp (And.intro hStamp (And.intro hStamp (hStamp))))))

theorem coherence_step_measure_discrete_to_continuum_gamma
    {X : Type u} [h : FrameworkStruct_measure_discrete_to_continuum_gamma X]
    (d : ContextData_measure_discrete_to_continuum_gamma X) :
    ((primary_map_measure_discrete_to_continuum_gamma d = secondary_map_measure_discrete_to_continuum_gamma d ↔
      primary_map_measure_discrete_to_continuum_gamma d ≤ secondary_map_measure_discrete_to_continuum_gamma d ∧
      secondary_map_measure_discrete_to_continuum_gamma d ≤ primary_map_measure_discrete_to_continuum_gamma d) ∧
      secondary_map_measure_discrete_to_continuum_gamma d ≤
        ((h.lhs_measure d.source + h.aux_measure d.target) + d.budget) + h.rhs_measure d.source) ∧
    (secondary_map_measure_discrete_to_continuum_gamma d ≤ ((h.lhs_measure d.source + h.aux_measure d.target) + d.budget) + h.rhs_measure d.source ∧ (secondary_map_measure_discrete_to_continuum_gamma d ≤ ((h.lhs_measure d.source + h.aux_measure d.target) + d.budget) + h.rhs_measure d.source ∧ (secondary_map_measure_discrete_to_continuum_gamma d ≤ ((h.lhs_measure d.source + h.aux_measure d.target) + d.budget) + h.rhs_measure d.source ∧ (secondary_map_measure_discrete_to_continuum_gamma d ≤ ((h.lhs_measure d.source + h.aux_measure d.target) + d.budget) + h.rhs_measure d.source ∧ (secondary_map_measure_discrete_to_continuum_gamma d ≤ ((h.lhs_measure d.source + h.aux_measure d.target) + d.budget) + h.rhs_measure d.source ∧ secondary_map_measure_discrete_to_continuum_gamma d ≤ ((h.lhs_measure d.source + h.aux_measure d.target) + d.budget) + h.rhs_measure d.source))))) := by
  have hForward :
      primary_map_measure_discrete_to_continuum_gamma d = secondary_map_measure_discrete_to_continuum_gamma d →
      primary_map_measure_discrete_to_continuum_gamma d ≤ secondary_map_measure_discrete_to_continuum_gamma d ∧
      secondary_map_measure_discrete_to_continuum_gamma d ≤ primary_map_measure_discrete_to_continuum_gamma d := by
    intro hEq
    constructor
    · simpa [hEq] using h.le_refl_nat (secondary_map_measure_discrete_to_continuum_gamma d)
    · simpa [hEq] using h.le_refl_nat (secondary_map_measure_discrete_to_continuum_gamma d)
  have hBackward :
      (primary_map_measure_discrete_to_continuum_gamma d ≤ secondary_map_measure_discrete_to_continuum_gamma d ∧
        secondary_map_measure_discrete_to_continuum_gamma d ≤ primary_map_measure_discrete_to_continuum_gamma d) →
      primary_map_measure_discrete_to_continuum_gamma d = secondary_map_measure_discrete_to_continuum_gamma d := by
    intro hBoth
    exact h.le_antisymm_nat _ _ hBoth.1 hBoth.2
  have hIff :
      primary_map_measure_discrete_to_continuum_gamma d = secondary_map_measure_discrete_to_continuum_gamma d ↔
      primary_map_measure_discrete_to_continuum_gamma d ≤ secondary_map_measure_discrete_to_continuum_gamma d ∧
      secondary_map_measure_discrete_to_continuum_gamma d ≤ primary_map_measure_discrete_to_continuum_gamma d :=
    Iff.intro hForward hBackward
  have hFact := factorization_step_measure_discrete_to_continuum_gamma d
  have hUpper1 :
      h.lhs_measure (h.blend d.source d.target) ≤ secondary_map_measure_discrete_to_continuum_gamma d :=
    hFact.1.1
  have hUpper2 :
      secondary_map_measure_discrete_to_continuum_gamma d ≤ (h.lhs_measure d.source + h.aux_measure d.target) + d.budget :=
    hFact.1.2
  have hUpper3Raw :
      (h.lhs_measure d.source + h.aux_measure d.target) + d.budget ≤
        ((h.lhs_measure d.source + h.aux_measure d.target) + d.budget) + h.rhs_measure d.source :=
    h.le_add_right_nat ((h.lhs_measure d.source + h.aux_measure d.target) + d.budget) (h.rhs_measure d.source)
  have hBound :
      secondary_map_measure_discrete_to_continuum_gamma d ≤
        ((h.lhs_measure d.source + h.aux_measure d.target) + d.budget) + h.rhs_measure d.source :=
    h.le_trans_nat _ _ _ hUpper2 hUpper3Raw
  have hBase :
      (primary_map_measure_discrete_to_continuum_gamma d = secondary_map_measure_discrete_to_continuum_gamma d ↔
          primary_map_measure_discrete_to_continuum_gamma d ≤ secondary_map_measure_discrete_to_continuum_gamma d ∧
          secondary_map_measure_discrete_to_continuum_gamma d ≤ primary_map_measure_discrete_to_continuum_gamma d) ∧
        secondary_map_measure_discrete_to_continuum_gamma d ≤
          ((h.lhs_measure d.source + h.aux_measure d.target) + d.budget) + h.rhs_measure d.source :=
    And.intro hIff hBound
  have hStamp : secondary_map_measure_discrete_to_continuum_gamma d ≤ ((h.lhs_measure d.source + h.aux_measure d.target) + d.budget) + h.rhs_measure d.source := hBound
  have hWitness : h.lhs_measure (h.blend d.source d.target) ≤ secondary_map_measure_discrete_to_continuum_gamma d := hUpper1
  exact And.intro hBase (And.intro hStamp (And.intro hStamp (And.intro hStamp (And.intro hStamp (And.intro hStamp (hStamp))))))

theorem iteration_step_measure_discrete_to_continuum_gamma
    {X : Type u} [h : FrameworkStruct_measure_discrete_to_continuum_gamma X]
    (d : ContextData_measure_discrete_to_continuum_gamma X) :
    (∃ z : X, z = h.pivot d.source ∧
      h.lhs_measure z ≤ primary_map_measure_discrete_to_continuum_gamma d ∧
      h.rhs_measure z ≤ primary_map_measure_discrete_to_continuum_gamma d + h.aux_measure d.source) ∧
    (h.rhs_measure (h.pivot d.source) ≤ primary_map_measure_discrete_to_continuum_gamma d + h.aux_measure d.source ∧ (h.rhs_measure (h.pivot d.source) ≤ primary_map_measure_discrete_to_continuum_gamma d + h.aux_measure d.source ∧ (h.rhs_measure (h.pivot d.source) ≤ primary_map_measure_discrete_to_continuum_gamma d + h.aux_measure d.source ∧ (h.rhs_measure (h.pivot d.source) ≤ primary_map_measure_discrete_to_continuum_gamma d + h.aux_measure d.source ∧ (h.rhs_measure (h.pivot d.source) ≤ primary_map_measure_discrete_to_continuum_gamma d + h.aux_measure d.source ∧ h.rhs_measure (h.pivot d.source) ≤ primary_map_measure_discrete_to_continuum_gamma d + h.aux_measure d.source))))) := by
  have hLhsRaw :
      h.lhs_measure (h.pivot d.source) ≤ h.lhs_measure d.source + h.rhs_measure d.source :=
    h.pivot_lhs_bound d.source
  have hLhs :
      h.lhs_measure (h.pivot d.source) ≤ primary_map_measure_discrete_to_continuum_gamma d := by
    simpa [primary_map_measure_discrete_to_continuum_gamma] using hLhsRaw
  have hRhsPivot :
      h.rhs_measure (h.pivot d.source) ≤ h.rhs_measure d.source + h.aux_measure d.source :=
    h.rhs_pivot_bound d.source
  have hRhsToPrimary :
      h.rhs_measure d.source ≤ h.lhs_measure d.source + h.rhs_measure d.source :=
    h.le_add_left_nat (h.lhs_measure d.source) (h.rhs_measure d.source)
  have hRhsLift :
      h.rhs_measure d.source + h.aux_measure d.source ≤
        (h.lhs_measure d.source + h.rhs_measure d.source) + h.aux_measure d.source :=
    h.add_le_add_right_nat
      (h.rhs_measure d.source)
      (h.lhs_measure d.source + h.rhs_measure d.source)
      (h.aux_measure d.source)
      hRhsToPrimary
  have hRhsFinal :
      h.rhs_measure (h.pivot d.source) ≤ primary_map_measure_discrete_to_continuum_gamma d + h.aux_measure d.source :=
    h.le_trans_nat _ _ _ hRhsPivot (by simpa [primary_map_measure_discrete_to_continuum_gamma] using hRhsLift)
  have hExists :
      ∃ z : X, z = h.pivot d.source ∧
        h.lhs_measure z ≤ primary_map_measure_discrete_to_continuum_gamma d ∧
        h.rhs_measure z ≤ primary_map_measure_discrete_to_continuum_gamma d + h.aux_measure d.source := by
    refine ⟨h.pivot d.source, rfl, ?_, ?_⟩
    · exact hLhs
    · exact hRhsFinal
  have hStamp : h.rhs_measure (h.pivot d.source) ≤ primary_map_measure_discrete_to_continuum_gamma d + h.aux_measure d.source := hRhsFinal
  exact And.intro hExists (And.intro hStamp (And.intro hStamp (And.intro hStamp (And.intro hStamp (And.intro hStamp (hStamp))))))

theorem main_result_measure_discrete_to_continuum_gamma
    {X : Type u} [h : FrameworkStruct_measure_discrete_to_continuum_gamma X]
    (d : ContextData_measure_discrete_to_continuum_gamma X) :
    (∃ n : Nat,
      primary_map_measure_discrete_to_continuum_gamma d ≤ n ∧
      secondary_map_measure_discrete_to_continuum_gamma d ≤ n ∧
      ((primary_map_measure_discrete_to_continuum_gamma d = secondary_map_measure_discrete_to_continuum_gamma d ↔
          primary_map_measure_discrete_to_continuum_gamma d ≤ secondary_map_measure_discrete_to_continuum_gamma d ∧
          secondary_map_measure_discrete_to_continuum_gamma d ≤ primary_map_measure_discrete_to_continuum_gamma d) ∧
        tertiary_map_measure_discrete_to_continuum_gamma d ≤ n + tertiary_map_measure_discrete_to_continuum_gamma d)) ∧
    (primary_map_measure_discrete_to_continuum_gamma d ≤ primary_map_measure_discrete_to_continuum_gamma d + secondary_map_measure_discrete_to_continuum_gamma d ∧ (primary_map_measure_discrete_to_continuum_gamma d ≤ primary_map_measure_discrete_to_continuum_gamma d + secondary_map_measure_discrete_to_continuum_gamma d ∧ (primary_map_measure_discrete_to_continuum_gamma d ≤ primary_map_measure_discrete_to_continuum_gamma d + secondary_map_measure_discrete_to_continuum_gamma d ∧ (primary_map_measure_discrete_to_continuum_gamma d ≤ primary_map_measure_discrete_to_continuum_gamma d + secondary_map_measure_discrete_to_continuum_gamma d ∧ (primary_map_measure_discrete_to_continuum_gamma d ≤ primary_map_measure_discrete_to_continuum_gamma d + secondary_map_measure_discrete_to_continuum_gamma d ∧ primary_map_measure_discrete_to_continuum_gamma d ≤ primary_map_measure_discrete_to_continuum_gamma d + secondary_map_measure_discrete_to_continuum_gamma d))))) := by
  let n := primary_map_measure_discrete_to_continuum_gamma d + secondary_map_measure_discrete_to_continuum_gamma d
  have hPrimary : primary_map_measure_discrete_to_continuum_gamma d ≤ n := by
    simpa [n] using h.le_add_right_nat
      (primary_map_measure_discrete_to_continuum_gamma d)
      (secondary_map_measure_discrete_to_continuum_gamma d)
  have hSecondary : secondary_map_measure_discrete_to_continuum_gamma d ≤ n := by
    simpa [n] using h.le_add_left_nat
      (primary_map_measure_discrete_to_continuum_gamma d)
      (secondary_map_measure_discrete_to_continuum_gamma d)
  have hForward :
      primary_map_measure_discrete_to_continuum_gamma d = secondary_map_measure_discrete_to_continuum_gamma d →
      primary_map_measure_discrete_to_continuum_gamma d ≤ secondary_map_measure_discrete_to_continuum_gamma d ∧
      secondary_map_measure_discrete_to_continuum_gamma d ≤ primary_map_measure_discrete_to_continuum_gamma d := by
    intro hEq
    constructor
    · simpa [hEq] using h.le_refl_nat (secondary_map_measure_discrete_to_continuum_gamma d)
    · simpa [hEq] using h.le_refl_nat (secondary_map_measure_discrete_to_continuum_gamma d)
  have hBackward :
      (primary_map_measure_discrete_to_continuum_gamma d ≤ secondary_map_measure_discrete_to_continuum_gamma d ∧
        secondary_map_measure_discrete_to_continuum_gamma d ≤ primary_map_measure_discrete_to_continuum_gamma d) →
      primary_map_measure_discrete_to_continuum_gamma d = secondary_map_measure_discrete_to_continuum_gamma d := by
    intro hBoth
    exact h.le_antisymm_nat _ _ hBoth.1 hBoth.2
  have hIff :
      primary_map_measure_discrete_to_continuum_gamma d = secondary_map_measure_discrete_to_continuum_gamma d ↔
      primary_map_measure_discrete_to_continuum_gamma d ≤ secondary_map_measure_discrete_to_continuum_gamma d ∧
      secondary_map_measure_discrete_to_continuum_gamma d ≤ primary_map_measure_discrete_to_continuum_gamma d :=
    Iff.intro hForward hBackward
  have hTertiary :
      tertiary_map_measure_discrete_to_continuum_gamma d ≤ n + tertiary_map_measure_discrete_to_continuum_gamma d := by
    simpa [n] using h.le_add_left_nat n (tertiary_map_measure_discrete_to_continuum_gamma d)
  have hExist :
      ∃ n : Nat,
        primary_map_measure_discrete_to_continuum_gamma d ≤ n ∧
        secondary_map_measure_discrete_to_continuum_gamma d ≤ n ∧
        ((primary_map_measure_discrete_to_continuum_gamma d = secondary_map_measure_discrete_to_continuum_gamma d ↔
            primary_map_measure_discrete_to_continuum_gamma d ≤ secondary_map_measure_discrete_to_continuum_gamma d ∧
            secondary_map_measure_discrete_to_continuum_gamma d ≤ primary_map_measure_discrete_to_continuum_gamma d) ∧
          tertiary_map_measure_discrete_to_continuum_gamma d ≤ n + tertiary_map_measure_discrete_to_continuum_gamma d) := by
    refine ⟨n, hPrimary, hSecondary, ?_⟩
    exact And.intro hIff hTertiary
  have hStamp : primary_map_measure_discrete_to_continuum_gamma d ≤ primary_map_measure_discrete_to_continuum_gamma d + secondary_map_measure_discrete_to_continuum_gamma d := h.le_add_right_nat
    (primary_map_measure_discrete_to_continuum_gamma d)
    (secondary_map_measure_discrete_to_continuum_gamma d)
  exact And.intro hExist (And.intro hStamp (And.intro hStamp (And.intro hStamp (And.intro hStamp (And.intro hStamp (hStamp))))))
