/-
BENCHMARK_ID: TINY_MATHLIB_BATCH07_MEASURE_DIMENSION_FREE_CONCENTRATION_LIKE
PAIR_STEM: measure_dimension_free_concentration_like
MATH_DOMAIN: Measure Theory
SOURCE_MATHLIB: Mathlib/MeasureTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class FrameworkStruct_measure_dimension_free_concentration (X : Type u) where
  radius : X → Nat
  variance : X → Nat
  deviation : X → Nat
  project : X → X
  average : X → X
  le_trans_nat : ∀ a b c : Nat, a ≤ b → b ≤ c → a ≤ c
  add_le_add_left_nat : ∀ a b c : Nat, a ≤ b → c + a ≤ c + b
  add_le_add_right_nat : ∀ a b c : Nat, a ≤ b → a + c ≤ b + c
  le_add_left_nat : ∀ a b : Nat, b ≤ a + b
  le_add_right_nat : ∀ a b : Nat, a ≤ a + b
  project_radius : ∀ x : X, radius (project x) ≤ radius x
  project_variance : ∀ x : X, variance (project x) ≤ variance x + deviation x
  concentration_step : ∀ x : X, radius (average (project x)) ≤ radius x + variance x
  tail_bound : ∀ x : X, variance (average (project x)) ≤ variance x + deviation x
  average_deviation : ∀ x : X, deviation (average x) ≤ deviation x + deviation x
  witness_project : ∀ x : X, ∃ y : X, y = project x ∧ radius y ≤ radius x

structure ContextData_measure_dimension_free_concentration
    (X : Type u) [s : FrameworkStruct_measure_dimension_free_concentration X] where
  base : X
  ref : X
  budget : Nat
  variance_cap : s.variance base ≤ budget
  deviation_cap : s.deviation base ≤ budget
  radius_bridge : s.radius base ≤ s.radius ref + budget

def primary_map_measure_dimension_free_concentration
    {X : Type u} [s : FrameworkStruct_measure_dimension_free_concentration X]
    (d : ContextData_measure_dimension_free_concentration X) : X :=
  s.average (s.project d.base)

def secondary_map_measure_dimension_free_concentration
    {X : Type u} [s : FrameworkStruct_measure_dimension_free_concentration X]
    (d : ContextData_measure_dimension_free_concentration X) : Nat :=
  s.radius (primary_map_measure_dimension_free_concentration (X := X) d) + d.budget

def tertiary_map_measure_dimension_free_concentration
    {X : Type u} [s : FrameworkStruct_measure_dimension_free_concentration X]
    (d : ContextData_measure_dimension_free_concentration X) : Prop :=
  s.radius (primary_map_measure_dimension_free_concentration (X := X) d)
    ≤ secondary_map_measure_dimension_free_concentration (X := X) d

theorem stability_step_measure_dimension_free_concentration
    {X : Type u} [s : FrameworkStruct_measure_dimension_free_concentration X]
    (d : ContextData_measure_dimension_free_concentration X) :
    ((fun P : Prop => (fun f : Prop → Prop => f P) (fun R : Prop => R)) ((s.radius (primary_map_measure_dimension_free_concentration (X := X) d)
      ≤ s.radius d.base + s.variance d.base ∧
    s.variance (primary_map_measure_dimension_free_concentration (X := X) d)
      ≤ s.variance d.base + s.deviation d.base) ∧
    s.variance (s.project d.base) ≤ s.variance d.base + s.deviation d.base))

 := by
  have hRadiusRaw : s.radius (s.average (s.project d.base)) ≤ s.radius d.base + s.variance d.base :=
    s.concentration_step d.base
  have hVarRaw : s.variance (s.average (s.project d.base)) ≤ s.variance d.base + s.deviation d.base :=
    s.tail_bound d.base
  have hRadius : s.radius (primary_map_measure_dimension_free_concentration (X := X) d)
      ≤ s.radius d.base + s.variance d.base := by
    simpa [primary_map_measure_dimension_free_concentration] using hRadiusRaw
  have hVar : s.variance (primary_map_measure_dimension_free_concentration (X := X) d)
      ≤ s.variance d.base + s.deviation d.base := by
    simpa [primary_map_measure_dimension_free_concentration] using hVarRaw
  have hProjVar : s.variance (s.project d.base) ≤ s.variance d.base + s.deviation d.base :=
    s.project_variance d.base
  exact ⟨⟨hRadius, hVar⟩, hProjVar⟩

theorem factorization_step_measure_dimension_free_concentration
    {X : Type u} [s : FrameworkStruct_measure_dimension_free_concentration X]
    (d : ContextData_measure_dimension_free_concentration X) :
    ((fun P : Prop => (fun f : Prop → Prop => f P) (fun R : Prop => R)) (s.radius (s.project d.base) ≤ s.radius d.base ∧
    s.variance (s.project d.base) ≤ s.variance d.base + s.deviation d.base))

 := by
  have hRad : s.radius (s.project d.base) ≤ s.radius d.base := s.project_radius d.base
  have hVar : s.variance (s.project d.base) ≤ s.variance d.base + s.deviation d.base := s.project_variance d.base
  exact ⟨hRad, hVar⟩

theorem comparison_step_measure_dimension_free_concentration
    {X : Type u} [s : FrameworkStruct_measure_dimension_free_concentration X]
    (d : ContextData_measure_dimension_free_concentration X) :
    ((fun P : Prop => (fun f : Prop → Prop => f P) (fun R : Prop => R)) (secondary_map_measure_dimension_free_concentration (X := X) d =
      s.radius (primary_map_measure_dimension_free_concentration (X := X) d) + d.budget ∧
    tertiary_map_measure_dimension_free_concentration (X := X) d))

 := by
  have hSec :
      s.radius (primary_map_measure_dimension_free_concentration (X := X) d)
        ≤ secondary_map_measure_dimension_free_concentration (X := X) d := by
    simpa [secondary_map_measure_dimension_free_concentration] using
      (s.le_add_right_nat (s.radius (primary_map_measure_dimension_free_concentration (X := X) d)) d.budget)
  exact ⟨rfl, hSec⟩

theorem transport_step_measure_dimension_free_concentration
    {X : Type u} [s : FrameworkStruct_measure_dimension_free_concentration X]
    (d : ContextData_measure_dimension_free_concentration X) :
    ((fun P : Prop => (fun f : Prop → Prop => f P) (fun R : Prop => R)) (s.radius (primary_map_measure_dimension_free_concentration (X := X) d)
      ≤ s.radius d.ref + d.budget + s.variance d.base ∧
    s.deviation (s.average (s.project d.base)) ≤ s.deviation (s.project d.base) + s.deviation (s.project d.base)))

 := by
  have hStab := stability_step_measure_dimension_free_concentration (X := X) d
  rcases hStab with ⟨hPair, hProjVar⟩
  rcases hPair with ⟨hRadBase, hVarBase⟩
  have hBridge : s.radius d.base + s.variance d.base ≤ (s.radius d.ref + d.budget) + s.variance d.base :=
    s.add_le_add_right_nat (s.radius d.base) (s.radius d.ref + d.budget) (s.variance d.base) d.radius_bridge
  have hFirst : s.radius (primary_map_measure_dimension_free_concentration (X := X) d)
      ≤ s.radius d.ref + d.budget + s.variance d.base :=
    s.le_trans_nat _ _ _ hRadBase hBridge
  have hSecond : s.deviation (s.average (s.project d.base))
      ≤ s.deviation (s.project d.base) + s.deviation (s.project d.base) :=
    s.average_deviation (s.project d.base)
  have _ : s.variance (primary_map_measure_dimension_free_concentration (X := X) d)
      ≤ s.variance d.base + s.deviation d.base := hVarBase
  have _ : s.variance (s.project d.base) ≤ s.variance d.base + s.deviation d.base := hProjVar
  exact ⟨hFirst, hSecond⟩

theorem coherence_step_measure_dimension_free_concentration
    {X : Type u} [s : FrameworkStruct_measure_dimension_free_concentration X]
    (d : ContextData_measure_dimension_free_concentration X) :
    ((fun P : Prop => (fun f : Prop → Prop => f P) (fun R : Prop => R)) (tertiary_map_measure_dimension_free_concentration (X := X) d →
    ∃ y : X,
      y = s.project d.base ∧
      s.radius y ≤ s.radius d.base ∧
      s.variance (s.project d.base) ≤ s.variance d.base + s.deviation d.base))

 := by
  intro hTer
  rcases s.witness_project d.base with ⟨y, hyEq, hyRad⟩
  have hVarProj : s.variance (s.project d.base) ≤ s.variance d.base + s.deviation d.base :=
    s.project_variance d.base
  have _ : tertiary_map_measure_dimension_free_concentration (X := X) d := hTer
  exact ⟨y, hyEq, hyRad, hVarProj⟩

theorem iteration_step_measure_dimension_free_concentration
    {X : Type u} [s : FrameworkStruct_measure_dimension_free_concentration X]
    (d : ContextData_measure_dimension_free_concentration X) :
    ((fun P : Prop => (fun f : Prop → Prop => f P) (fun R : Prop => R)) (∃ y : X,
      y = primary_map_measure_dimension_free_concentration (X := X) d ∧
      s.radius y ≤ s.radius d.base + s.variance d.base ∧
      s.variance y ≤ s.variance d.base + s.deviation d.base))

 := by
  let y := primary_map_measure_dimension_free_concentration (X := X) d
  have hyEq : y = primary_map_measure_dimension_free_concentration (X := X) d := rfl
  have hStab := stability_step_measure_dimension_free_concentration (X := X) d
  rcases hStab with ⟨hPair, hProjVar⟩
  rcases hPair with ⟨hRad, hVar⟩
  have _ : s.variance (s.project d.base) ≤ s.variance d.base + s.deviation d.base := hProjVar
  exact ⟨y, hyEq, by simpa [y] using hRad, by simpa [y] using hVar⟩

theorem main_result_measure_dimension_free_concentration
    {X : Type u} [s : FrameworkStruct_measure_dimension_free_concentration X]
    (d : ContextData_measure_dimension_free_concentration X) :
    ((fun P : Prop => (fun f : Prop → Prop => f P) (fun R : Prop => R)) (∃ y : X,
      y = primary_map_measure_dimension_free_concentration (X := X) d ∧
      s.radius y ≤ s.radius d.ref + d.budget + s.variance d.base ∧
      tertiary_map_measure_dimension_free_concentration (X := X) d))

 := by
  rcases iteration_step_measure_dimension_free_concentration (X := X) d with ⟨y, hyEq, hyRad, hyVar⟩
  have hTrans := (transport_step_measure_dimension_free_concentration (X := X) d).1
  have hFinal : s.radius y ≤ s.radius d.ref + d.budget + s.variance d.base := by
    simpa [hyEq] using hTrans
  have hTer := (comparison_step_measure_dimension_free_concentration (X := X) d).2
  have _ : s.variance y ≤ s.variance d.base + s.deviation d.base := hyVar
  exact ⟨y, hyEq, hFinal, hTer⟩
