/-
BENCHMARK_ID: TINY_MATHLIB_BATCH07_ANALYSIS_HEAT_KERNEL_ULTRACONTRACTIVITY_DEEP_LIKE
PAIR_STEM: analysis_heat_kernel_ultracontractivity_deep_like
MATH_DOMAIN: Analysis
SOURCE_MATHLIB: Mathlib/Analysis/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class FrameworkStruct_analysis_heat_kernel_ultracontractivity_deep (E : Type u) where
  mass : E → Nat
  entropy : E → Nat
  dissip : E → Nat
  step : Nat → E → E
  smooth : E → E
  le_trans_nat : ∀ a b c : Nat, a ≤ b → b ≤ c → a ≤ c
  add_le_add_left_nat : ∀ a b c : Nat, a ≤ b → c + a ≤ c + b
  add_le_add_right_nat : ∀ a b c : Nat, a ≤ b → a + c ≤ b + c
  add_le_add_nat : ∀ a b c d : Nat, a ≤ b → c ≤ d → a + c ≤ b + d
  le_add_left_nat : ∀ a b : Nat, b ≤ a + b
  le_add_right_nat : ∀ a b : Nat, a ≤ a + b
  step_zero : ∀ x : E, step 0 x = x
  step_add : ∀ m n : Nat, ∀ x : E, step (m + n) x = step m (step n x)
  mass_decay : ∀ n : Nat, ∀ x : E, mass (step n x) ≤ mass x + n
  entropy_decay : ∀ n : Nat, ∀ x : E, entropy (step n x) ≤ entropy x + dissip x + n
  ultracontractive : ∀ x : E, mass (smooth x) ≤ entropy x + dissip x
  dissip_smooth : ∀ x : E, dissip (smooth x) ≤ dissip x + dissip x
  smoothing_step : ∀ n : Nat, ∀ x : E, mass (smooth (step n x)) ≤ mass (step n x) + dissip (step n x)
  witness_step : ∀ x : E, ∀ n : Nat, ∃ y : E, y = step n x ∧ mass y ≤ mass x + n

structure ContextData_analysis_heat_kernel_ultracontractivity_deep
    (E : Type u) [s : FrameworkStruct_analysis_heat_kernel_ultracontractivity_deep E] where
  seed : E
  time : Nat
  budget : Nat
  budget_time : time ≤ budget
  seed_control : s.entropy seed ≤ s.mass seed + budget

def primary_map_analysis_heat_kernel_ultracontractivity_deep
    {E : Type u} [s : FrameworkStruct_analysis_heat_kernel_ultracontractivity_deep E]
    (d : ContextData_analysis_heat_kernel_ultracontractivity_deep E) : E :=
  s.step d.time d.seed

def secondary_map_analysis_heat_kernel_ultracontractivity_deep
    {E : Type u} [s : FrameworkStruct_analysis_heat_kernel_ultracontractivity_deep E]
    (d : ContextData_analysis_heat_kernel_ultracontractivity_deep E) : Nat :=
  s.mass (primary_map_analysis_heat_kernel_ultracontractivity_deep (E := E) d) + d.budget

def tertiary_map_analysis_heat_kernel_ultracontractivity_deep
    {E : Type u} [s : FrameworkStruct_analysis_heat_kernel_ultracontractivity_deep E]
    (d : ContextData_analysis_heat_kernel_ultracontractivity_deep E) : Prop :=
  s.mass (primary_map_analysis_heat_kernel_ultracontractivity_deep (E := E) d)
    ≤ secondary_map_analysis_heat_kernel_ultracontractivity_deep (E := E) d

theorem stability_step_analysis_heat_kernel_ultracontractivity_deep
    {E : Type u} [s : FrameworkStruct_analysis_heat_kernel_ultracontractivity_deep E]
    (d : ContextData_analysis_heat_kernel_ultracontractivity_deep E) :
    ((fun P : Prop => P) (s.mass (primary_map_analysis_heat_kernel_ultracontractivity_deep (E := E) d) ≤ s.mass d.seed + d.time ∧
    s.entropy (primary_map_analysis_heat_kernel_ultracontractivity_deep (E := E) d) ≤
      s.entropy d.seed + s.dissip d.seed + d.time))


:= by
  have hMassRaw : s.mass (s.step d.time d.seed) ≤ s.mass d.seed + d.time :=
    s.mass_decay d.time d.seed
  have hEntRaw : s.entropy (s.step d.time d.seed) ≤ s.entropy d.seed + s.dissip d.seed + d.time :=
    s.entropy_decay d.time d.seed
  have hMass :
      s.mass (primary_map_analysis_heat_kernel_ultracontractivity_deep (E := E) d) ≤ s.mass d.seed + d.time := by
    simpa [primary_map_analysis_heat_kernel_ultracontractivity_deep] using hMassRaw
  have hEnt :
      s.entropy (primary_map_analysis_heat_kernel_ultracontractivity_deep (E := E) d) ≤
        s.entropy d.seed + s.dissip d.seed + d.time := by
    simpa [primary_map_analysis_heat_kernel_ultracontractivity_deep] using hEntRaw
  exact ⟨hMass, hEnt⟩

theorem factorization_step_analysis_heat_kernel_ultracontractivity_deep
    {E : Type u} [s : FrameworkStruct_analysis_heat_kernel_ultracontractivity_deep E]
    (d : ContextData_analysis_heat_kernel_ultracontractivity_deep E) :
    ((fun P : Prop => P) (s.mass (s.smooth (primary_map_analysis_heat_kernel_ultracontractivity_deep (E := E) d))
      ≤ s.mass (primary_map_analysis_heat_kernel_ultracontractivity_deep (E := E) d)
        + s.dissip (primary_map_analysis_heat_kernel_ultracontractivity_deep (E := E) d) ∧
    s.dissip (s.smooth (primary_map_analysis_heat_kernel_ultracontractivity_deep (E := E) d))
      ≤ s.dissip (primary_map_analysis_heat_kernel_ultracontractivity_deep (E := E) d)
        + s.dissip (primary_map_analysis_heat_kernel_ultracontractivity_deep (E := E) d)))


:= by
  let x := primary_map_analysis_heat_kernel_ultracontractivity_deep (E := E) d
  have hSmoothMass : s.mass (s.smooth x) ≤ s.mass x + s.dissip x :=
    s.smoothing_step d.time d.seed
  have hDissip : s.dissip (s.smooth x) ≤ s.dissip x + s.dissip x :=
    s.dissip_smooth x
  have hx : x = primary_map_analysis_heat_kernel_ultracontractivity_deep (E := E) d := rfl
  constructor
  · simpa [x, hx]
      using hSmoothMass
  · simpa [x, hx]
      using hDissip

theorem comparison_step_analysis_heat_kernel_ultracontractivity_deep
    {E : Type u} [s : FrameworkStruct_analysis_heat_kernel_ultracontractivity_deep E]
    (d : ContextData_analysis_heat_kernel_ultracontractivity_deep E) :
    ((fun P : Prop => P) (tertiary_map_analysis_heat_kernel_ultracontractivity_deep (E := E) d ∧
    s.mass (primary_map_analysis_heat_kernel_ultracontractivity_deep (E := E) d)
      ≤ secondary_map_analysis_heat_kernel_ultracontractivity_deep (E := E) d))


:= by
  have hBase :
      s.mass (primary_map_analysis_heat_kernel_ultracontractivity_deep (E := E) d)
        ≤ s.mass (primary_map_analysis_heat_kernel_ultracontractivity_deep (E := E) d) + d.budget :=
    s.le_add_right_nat _ _
  have hTer : tertiary_map_analysis_heat_kernel_ultracontractivity_deep (E := E) d := by
    simpa [tertiary_map_analysis_heat_kernel_ultracontractivity_deep,
      secondary_map_analysis_heat_kernel_ultracontractivity_deep] using hBase
  constructor
  · exact hTer
  · simpa [secondary_map_analysis_heat_kernel_ultracontractivity_deep] using hBase

theorem transport_step_analysis_heat_kernel_ultracontractivity_deep
    {E : Type u} [s : FrameworkStruct_analysis_heat_kernel_ultracontractivity_deep E]
    (d : ContextData_analysis_heat_kernel_ultracontractivity_deep E) :
    ((fun P : Prop => P) (s.mass (primary_map_analysis_heat_kernel_ultracontractivity_deep (E := E) d) ≤ s.mass d.seed + d.budget ∧
    s.mass (primary_map_analysis_heat_kernel_ultracontractivity_deep (E := E) d)
      ≤ secondary_map_analysis_heat_kernel_ultracontractivity_deep (E := E) d))


:= by
  have hStable := stability_step_analysis_heat_kernel_ultracontractivity_deep (E := E) d
  rcases hStable with ⟨hMassTime, hEntropyTime⟩
  have hBudgetLift : s.mass d.seed + d.time ≤ s.mass d.seed + d.budget :=
    s.add_le_add_left_nat d.time d.budget (s.mass d.seed) d.budget_time
  have hMassBudget : s.mass (primary_map_analysis_heat_kernel_ultracontractivity_deep (E := E) d) ≤ s.mass d.seed + d.budget :=
    s.le_trans_nat _ _ _ hMassTime hBudgetLift
  have hComp := comparison_step_analysis_heat_kernel_ultracontractivity_deep (E := E) d
  rcases hComp with ⟨hTer, hSec⟩
  have _ : s.entropy (primary_map_analysis_heat_kernel_ultracontractivity_deep (E := E) d)
      ≤ s.entropy d.seed + s.dissip d.seed + d.time := hEntropyTime
  have _ : tertiary_map_analysis_heat_kernel_ultracontractivity_deep (E := E) d := hTer
  exact ⟨hMassBudget, hSec⟩

theorem coherence_step_analysis_heat_kernel_ultracontractivity_deep
    {E : Type u} [s : FrameworkStruct_analysis_heat_kernel_ultracontractivity_deep E]
    (d : ContextData_analysis_heat_kernel_ultracontractivity_deep E) :
    ((fun P : Prop => P) (s.entropy (primary_map_analysis_heat_kernel_ultracontractivity_deep (E := E) d)
      ≤ s.mass d.seed + d.budget + s.dissip d.seed + d.budget ∧
    secondary_map_analysis_heat_kernel_ultracontractivity_deep (E := E) d =
      s.mass (primary_map_analysis_heat_kernel_ultracontractivity_deep (E := E) d) + d.budget))


:= by
  have hStable := stability_step_analysis_heat_kernel_ultracontractivity_deep (E := E) d
  rcases hStable with ⟨hMass, hEntropy⟩
  have hSeedLift : s.entropy d.seed + s.dissip d.seed + d.time ≤
      (s.mass d.seed + d.budget) + s.dissip d.seed + d.budget := by
    have h1 : s.entropy d.seed ≤ s.mass d.seed + d.budget := d.seed_control
    have h1' : s.entropy d.seed + s.dissip d.seed ≤ (s.mass d.seed + d.budget) + s.dissip d.seed :=
      s.add_le_add_right_nat (s.entropy d.seed) (s.mass d.seed + d.budget) (s.dissip d.seed) h1
    have h2 : s.entropy d.seed + s.dissip d.seed + d.time ≤
        (s.mass d.seed + d.budget) + s.dissip d.seed + d.time :=
      s.add_le_add_right_nat
        (s.entropy d.seed + s.dissip d.seed)
        ((s.mass d.seed + d.budget) + s.dissip d.seed)
        d.time
        h1'
    have h3 : ((s.mass d.seed + d.budget) + s.dissip d.seed) + d.time ≤
        ((s.mass d.seed + d.budget) + s.dissip d.seed) + d.budget :=
      s.add_le_add_left_nat d.time d.budget ((s.mass d.seed + d.budget) + s.dissip d.seed) d.budget_time
    exact s.le_trans_nat _ _ _ h2 h3
  have hEntropyBudget :
      s.entropy (primary_map_analysis_heat_kernel_ultracontractivity_deep (E := E) d)
        ≤ s.mass d.seed + d.budget + s.dissip d.seed + d.budget :=
    s.le_trans_nat _ _ _ hEntropy hSeedLift
  have hEq :
      secondary_map_analysis_heat_kernel_ultracontractivity_deep (E := E) d =
        s.mass (primary_map_analysis_heat_kernel_ultracontractivity_deep (E := E) d) + d.budget := by
    rfl
  have _ : s.mass (primary_map_analysis_heat_kernel_ultracontractivity_deep (E := E) d) ≤ s.mass d.seed + d.time := hMass
  exact ⟨hEntropyBudget, hEq⟩

theorem iteration_step_analysis_heat_kernel_ultracontractivity_deep
    {E : Type u} [s : FrameworkStruct_analysis_heat_kernel_ultracontractivity_deep E]
    (d : ContextData_analysis_heat_kernel_ultracontractivity_deep E) :
    ((fun P : Prop => P) (∃ y : E,
      y = primary_map_analysis_heat_kernel_ultracontractivity_deep (E := E) d ∧
      s.mass y ≤ s.mass d.seed + d.budget ∧
      s.mass (s.smooth y) ≤ s.mass y + s.dissip y))


:= by
  rcases s.witness_step d.seed d.time with ⟨y, hyEq, hyMassTime⟩
  have hyMassBudget : s.mass y ≤ s.mass d.seed + d.budget :=
    s.le_trans_nat _ _ _ hyMassTime
      (s.add_le_add_left_nat d.time d.budget (s.mass d.seed) d.budget_time)
  have hyIsPrimary : y = primary_map_analysis_heat_kernel_ultracontractivity_deep (E := E) d := by
    simpa [primary_map_analysis_heat_kernel_ultracontractivity_deep] using hyEq
  have hSmooth : s.mass (s.smooth y) ≤ s.mass y + s.dissip y := by
    have hRaw : s.mass (s.smooth (s.step d.time d.seed)) ≤ s.mass (s.step d.time d.seed) + s.dissip (s.step d.time d.seed) :=
      s.smoothing_step d.time d.seed
    simpa [hyEq] using hRaw
  refine ⟨y, hyIsPrimary, hyMassBudget, hSmooth⟩

theorem main_result_analysis_heat_kernel_ultracontractivity_deep
    {E : Type u} [s : FrameworkStruct_analysis_heat_kernel_ultracontractivity_deep E]
    (d : ContextData_analysis_heat_kernel_ultracontractivity_deep E) :
    ((fun P : Prop => P) (∃ y : E,
      y = primary_map_analysis_heat_kernel_ultracontractivity_deep (E := E) d ∧
      s.mass (s.smooth y) ≤ (s.mass d.seed + d.budget) + s.dissip y ∧
      tertiary_map_analysis_heat_kernel_ultracontractivity_deep (E := E) d))


:= by
  rcases iteration_step_analysis_heat_kernel_ultracontractivity_deep (E := E) d with
      ⟨y, hyEq, hyMass, hSmooth⟩
  have hBound : s.mass (s.smooth y) ≤ (s.mass d.seed + d.budget) + s.dissip y := by
    have hLift : s.mass y + s.dissip y ≤ (s.mass d.seed + d.budget) + s.dissip y :=
      s.add_le_add_right_nat (s.mass y) (s.mass d.seed + d.budget) (s.dissip y) hyMass
    exact s.le_trans_nat _ _ _ hSmooth hLift
  have hComp := comparison_step_analysis_heat_kernel_ultracontractivity_deep (E := E) d
  rcases hComp with ⟨hTer, hSec⟩
  have _ : s.mass (primary_map_analysis_heat_kernel_ultracontractivity_deep (E := E) d)
      ≤ secondary_map_analysis_heat_kernel_ultracontractivity_deep (E := E) d := hSec
  refine ⟨y, hyEq, hBound, hTer⟩
