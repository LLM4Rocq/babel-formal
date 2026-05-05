/-
BENCHMARK_ID: TINY_MATHLIB_BATCH07_ANALYSIS_HARMONIC_MEASURE_BOUNDARY_HARNACK_LIKE
PAIR_STEM: analysis_harmonic_measure_boundary_harnack_like
MATH_DOMAIN: Analysis
SOURCE_MATHLIB: Mathlib/Analysis/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class FrameworkStruct_analysis_harmonic_measure_boundary_harnack (E : Type u) where
  mass : E → Nat
  boundary : E → Nat
  interior : E → Nat
  harmonicize : E → E
  trace : E → E
  le_trans_nat : ∀ a b c : Nat, a ≤ b → b ≤ c → a ≤ c
  add_le_add_left_nat : ∀ a b c : Nat, a ≤ b → c + a ≤ c + b
  add_le_add_right_nat : ∀ a b c : Nat, a ≤ b → a + c ≤ b + c
  le_add_left_nat : ∀ a b : Nat, b ≤ a + b
  le_add_right_nat : ∀ a b : Nat, a ≤ a + b
  harmonic_mass : ∀ x : E, mass (harmonicize x) ≤ mass x + interior x
  boundary_trace : ∀ x : E, boundary (trace x) ≤ boundary x
  harnack_step : ∀ x : E, interior (harmonicize x) ≤ boundary x + interior x
  trace_mass : ∀ x : E, mass (trace x) ≤ mass x + boundary x
  harnack_compare : ∀ x : E, mass (harmonicize x) ≤ boundary x + boundary x
  witness_trace : ∀ x : E, ∃ y : E, y = trace x ∧ boundary y ≤ boundary x

structure ContextData_analysis_harmonic_measure_boundary_harnack
    (E : Type u) [s : FrameworkStruct_analysis_harmonic_measure_boundary_harnack E] where
  left : E
  right : E
  window : Nat
  left_to_right_boundary : s.boundary left ≤ s.boundary right + window
  right_mass_cap : s.mass right ≤ window + window

def primary_map_analysis_harmonic_measure_boundary_harnack
    {E : Type u} [s : FrameworkStruct_analysis_harmonic_measure_boundary_harnack E]
    (d : ContextData_analysis_harmonic_measure_boundary_harnack E) : E :=
  s.harmonicize (s.trace d.left)

def secondary_map_analysis_harmonic_measure_boundary_harnack
    {E : Type u} [s : FrameworkStruct_analysis_harmonic_measure_boundary_harnack E]
    (d : ContextData_analysis_harmonic_measure_boundary_harnack E) : Nat :=
  s.mass (primary_map_analysis_harmonic_measure_boundary_harnack (E := E) d) + d.window

def tertiary_map_analysis_harmonic_measure_boundary_harnack
    {E : Type u} [s : FrameworkStruct_analysis_harmonic_measure_boundary_harnack E]
    (d : ContextData_analysis_harmonic_measure_boundary_harnack E) : Prop :=
  s.mass (primary_map_analysis_harmonic_measure_boundary_harnack (E := E) d)
    ≤ secondary_map_analysis_harmonic_measure_boundary_harnack (E := E) d

theorem stability_step_analysis_harmonic_measure_boundary_harnack
    {E : Type u} [s : FrameworkStruct_analysis_harmonic_measure_boundary_harnack E]
    (d : ContextData_analysis_harmonic_measure_boundary_harnack E) :
    ((s.mass (primary_map_analysis_harmonic_measure_boundary_harnack (E := E) d)
      ≤ (s.mass d.left + s.boundary d.left) + s.interior (s.trace d.left) ∧
    s.boundary (s.trace d.left) ≤ s.boundary d.left) ∨ False)

 := by
  refine Or.inl ?_
  have hMassH : s.mass (s.harmonicize (s.trace d.left))
      ≤ s.mass (s.trace d.left) + s.interior (s.trace d.left) :=
    s.harmonic_mass (s.trace d.left)
  have hMassTrace : s.mass (s.trace d.left) ≤ s.mass d.left + s.boundary d.left :=
    s.trace_mass d.left
  have hLift :
      s.mass (s.trace d.left) + s.interior (s.trace d.left)
        ≤ (s.mass d.left + s.boundary d.left) + s.interior (s.trace d.left) :=
    s.add_le_add_right_nat (s.mass (s.trace d.left)) (s.mass d.left + s.boundary d.left)
      (s.interior (s.trace d.left)) hMassTrace
  have hMass :
      s.mass (primary_map_analysis_harmonic_measure_boundary_harnack (E := E) d)
        ≤ (s.mass d.left + s.boundary d.left) + s.interior (s.trace d.left) := by
    have hRaw := s.le_trans_nat _ _ _ hMassH hLift
    simpa [primary_map_analysis_harmonic_measure_boundary_harnack] using hRaw
  have hBoundary : s.boundary (s.trace d.left) ≤ s.boundary d.left :=
    s.boundary_trace d.left
  exact ⟨hMass, hBoundary⟩

theorem factorization_step_analysis_harmonic_measure_boundary_harnack
    {E : Type u} [s : FrameworkStruct_analysis_harmonic_measure_boundary_harnack E]
    (d : ContextData_analysis_harmonic_measure_boundary_harnack E) :
    ((s.interior (primary_map_analysis_harmonic_measure_boundary_harnack (E := E) d)
      ≤ s.boundary (s.trace d.left) + s.interior (s.trace d.left) ∧
    s.mass (primary_map_analysis_harmonic_measure_boundary_harnack (E := E) d)
      ≤ s.boundary (s.trace d.left) + s.boundary (s.trace d.left)) ∨ False)

 := by
  refine Or.inl ?_
  have hIntRaw : s.interior (s.harmonicize (s.trace d.left))
      ≤ s.boundary (s.trace d.left) + s.interior (s.trace d.left) :=
    s.harnack_step (s.trace d.left)
  have hMassRaw : s.mass (s.harmonicize (s.trace d.left))
      ≤ s.boundary (s.trace d.left) + s.boundary (s.trace d.left) :=
    s.harnack_compare (s.trace d.left)
  have hInt : s.interior (primary_map_analysis_harmonic_measure_boundary_harnack (E := E) d)
      ≤ s.boundary (s.trace d.left) + s.interior (s.trace d.left) := by
    simpa [primary_map_analysis_harmonic_measure_boundary_harnack] using hIntRaw
  have hMass : s.mass (primary_map_analysis_harmonic_measure_boundary_harnack (E := E) d)
      ≤ s.boundary (s.trace d.left) + s.boundary (s.trace d.left) := by
    simpa [primary_map_analysis_harmonic_measure_boundary_harnack] using hMassRaw
  exact ⟨hInt, hMass⟩

theorem comparison_step_analysis_harmonic_measure_boundary_harnack
    {E : Type u} [s : FrameworkStruct_analysis_harmonic_measure_boundary_harnack E]
    (d : ContextData_analysis_harmonic_measure_boundary_harnack E) :
    ((∃ m : Nat,
      m = secondary_map_analysis_harmonic_measure_boundary_harnack (E := E) d ∧
      tertiary_map_analysis_harmonic_measure_boundary_harnack (E := E) d) ∨ False)

 := by
  refine Or.inl ?_
  have hSec :
      s.mass (primary_map_analysis_harmonic_measure_boundary_harnack (E := E) d)
        ≤ secondary_map_analysis_harmonic_measure_boundary_harnack (E := E) d := by
    simpa [secondary_map_analysis_harmonic_measure_boundary_harnack] using
      (s.le_add_right_nat (s.mass (primary_map_analysis_harmonic_measure_boundary_harnack (E := E) d)) d.window)
  have hTer : tertiary_map_analysis_harmonic_measure_boundary_harnack (E := E) d := hSec
  refine ⟨secondary_map_analysis_harmonic_measure_boundary_harnack (E := E) d, rfl, hTer⟩

theorem transport_step_analysis_harmonic_measure_boundary_harnack
    {E : Type u} [s : FrameworkStruct_analysis_harmonic_measure_boundary_harnack E]
    (d : ContextData_analysis_harmonic_measure_boundary_harnack E) :
    ((s.mass (primary_map_analysis_harmonic_measure_boundary_harnack (E := E) d)
      ≤ (s.mass d.left + s.boundary d.left) + s.interior (s.trace d.left) ∧
    s.boundary (s.trace d.left) ≤ s.boundary d.right + d.window) ∨ False)

 := by
  refine Or.inl ?_
  have hStab := stability_step_analysis_harmonic_measure_boundary_harnack (E := E) d
  rcases hStab with hStabCore | hFalse
  · rcases hStabCore with ⟨hMass, hBoundaryLeft⟩
    have hBoundaryRight : s.boundary (s.trace d.left) ≤ s.boundary d.right + d.window :=
      s.le_trans_nat _ _ _ hBoundaryLeft d.left_to_right_boundary
    exact ⟨hMass, hBoundaryRight⟩
  · cases hFalse

theorem coherence_step_analysis_harmonic_measure_boundary_harnack
    {E : Type u} [s : FrameworkStruct_analysis_harmonic_measure_boundary_harnack E]
    (d : ContextData_analysis_harmonic_measure_boundary_harnack E) :
    ((∀ z : E, z = s.trace d.left → s.boundary z ≤ s.boundary d.left) ∨ False)

 := by
  refine Or.inl ?_
  intro z hz
  calc
    s.boundary z = s.boundary (s.trace d.left) := by
      rw [hz]
    _ ≤ s.boundary d.left :=
      s.boundary_trace d.left

theorem iteration_step_analysis_harmonic_measure_boundary_harnack
    {E : Type u} [s : FrameworkStruct_analysis_harmonic_measure_boundary_harnack E]
    (d : ContextData_analysis_harmonic_measure_boundary_harnack E) :
    ((∀ y : E,
      y = primary_map_analysis_harmonic_measure_boundary_harnack (E := E) d →
      s.mass y ≤ secondary_map_analysis_harmonic_measure_boundary_harnack (E := E) d →
      s.boundary (s.trace d.left) ≤ s.boundary d.left) ∨ False)

 := by
  refine Or.inl ?_
  intro y hyEq hyMass
  have hBoundary : s.boundary (s.trace d.left) ≤ s.boundary d.left := by
    have hStab := stability_step_analysis_harmonic_measure_boundary_harnack (E := E) d
    rcases hStab with hStabCore | hFalse
    · exact hStabCore.2
    · cases hFalse
  have _ : s.mass y ≤ secondary_map_analysis_harmonic_measure_boundary_harnack (E := E) d := by
    simpa [hyEq] using hyMass
  exact hBoundary

theorem main_result_analysis_harmonic_measure_boundary_harnack
    {E : Type u} [s : FrameworkStruct_analysis_harmonic_measure_boundary_harnack E]
    (d : ContextData_analysis_harmonic_measure_boundary_harnack E) :
    ((∃ y : E,
      y = primary_map_analysis_harmonic_measure_boundary_harnack (E := E) d ∧
      s.mass y ≤ secondary_map_analysis_harmonic_measure_boundary_harnack (E := E) d ∧
      s.boundary (s.trace d.left) ≤ s.boundary d.right + d.window) ∨ False)

 := by
  refine Or.inl ?_
  let y := primary_map_analysis_harmonic_measure_boundary_harnack (E := E) d
  have hyEq : y = primary_map_analysis_harmonic_measure_boundary_harnack (E := E) d := rfl
  have hComp := comparison_step_analysis_harmonic_measure_boundary_harnack (E := E) d
  rcases hComp with hCompCore | hFalse
  · rcases hCompCore with ⟨m, hmEq, hTer⟩
    have hMass : s.mass y ≤ secondary_map_analysis_harmonic_measure_boundary_harnack (E := E) d := by
      simpa [y] using hTer
    have hBoundaryLeft :
        s.boundary (s.trace d.left) ≤ s.boundary d.left := by
      have hIter := iteration_step_analysis_harmonic_measure_boundary_harnack (E := E) d
      rcases hIter with hIterCore | hIterFalse
      · exact hIterCore y hyEq hMass
      · cases hIterFalse
    have hBoundaryRight : s.boundary (s.trace d.left) ≤ s.boundary d.right + d.window :=
      s.le_trans_nat _ _ _ hBoundaryLeft d.left_to_right_boundary
    have _ : m = secondary_map_analysis_harmonic_measure_boundary_harnack (E := E) d := hmEq
    have _ : s.mass d.right ≤ d.window + d.window := d.right_mass_cap
    exact ⟨y, hyEq, hMass, hBoundaryRight⟩
  · cases hFalse
