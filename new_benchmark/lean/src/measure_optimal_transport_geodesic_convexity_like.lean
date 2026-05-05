/-
BENCHMARK_ID: TINY_MATHLIB_BATCH07_MEASURE_OPTIMAL_TRANSPORT_GEODESIC_CONVEXITY_LIKE
PAIR_STEM: measure_optimal_transport_geodesic_convexity_like
MATH_DOMAIN: Measure Theory
SOURCE_MATHLIB: Mathlib/MeasureTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class FrameworkStruct_measure_optimal_transport_geodesic_convexity (M : Type u) where
  mass : M → Nat
  cost : M → Nat
  entropy : M → Nat
  push : M → M
  midpoint : M → M → M
  geodesic : M → M → M
  le_trans_nat : ∀ a b c : Nat, a ≤ b → b ≤ c → a ≤ c
  add_le_add_left_nat : ∀ a b c : Nat, a ≤ b → c + a ≤ c + b
  add_le_add_right_nat : ∀ a b c : Nat, a ≤ b → a + c ≤ b + c
  le_add_left_nat : ∀ a b : Nat, b ≤ a + b
  le_add_right_nat : ∀ a b : Nat, a ≤ a + b
  push_mass : ∀ x : M, mass (push x) ≤ mass x
  push_cost : ∀ x : M, cost (push x) ≤ cost x + entropy x
  midpoint_cost : ∀ x y : M, cost (midpoint x y) ≤ cost x + cost y
  geodesic_cost : ∀ x y : M, cost (geodesic x y) ≤ cost (midpoint x y) + entropy x + entropy y
  entropy_midpoint : ∀ x y : M, entropy (midpoint x y) ≤ entropy x + entropy y
  witness_push : ∀ x : M, ∃ y : M, y = push x ∧ mass y ≤ mass x

structure ContextData_measure_optimal_transport_geodesic_convexity
    (M : Type u) [s : FrameworkStruct_measure_optimal_transport_geodesic_convexity M] where
  left : M
  right : M
  budget : Nat
  entropy_cap_left : s.entropy left ≤ budget
  entropy_cap_right : s.entropy right ≤ budget
  cost_bridge : s.cost left ≤ s.cost right + budget

def primary_map_measure_optimal_transport_geodesic_convexity
    {M : Type u} [s : FrameworkStruct_measure_optimal_transport_geodesic_convexity M]
    (d : ContextData_measure_optimal_transport_geodesic_convexity M) : M :=
  s.geodesic (s.push d.left) d.right

def secondary_map_measure_optimal_transport_geodesic_convexity
    {M : Type u} [s : FrameworkStruct_measure_optimal_transport_geodesic_convexity M]
    (d : ContextData_measure_optimal_transport_geodesic_convexity M) : Nat :=
  s.cost (primary_map_measure_optimal_transport_geodesic_convexity (M := M) d) + d.budget

def tertiary_map_measure_optimal_transport_geodesic_convexity
    {M : Type u} [s : FrameworkStruct_measure_optimal_transport_geodesic_convexity M]
    (d : ContextData_measure_optimal_transport_geodesic_convexity M) : Prop :=
  s.cost (primary_map_measure_optimal_transport_geodesic_convexity (M := M) d)
    ≤ secondary_map_measure_optimal_transport_geodesic_convexity (M := M) d

theorem stability_step_measure_optimal_transport_geodesic_convexity
    {M : Type u} [s : FrameworkStruct_measure_optimal_transport_geodesic_convexity M]
    (d : ContextData_measure_optimal_transport_geodesic_convexity M) :
    ((fun P : Prop => (fun u : Unit => P) ()) (s.cost (primary_map_measure_optimal_transport_geodesic_convexity (M := M) d)
      ≤ s.cost (s.midpoint (s.push d.left) d.right) + s.entropy (s.push d.left) + s.entropy d.right ∧
    s.mass (s.push d.left) ≤ s.mass d.left))

 := by
  have hCostRaw : s.cost (s.geodesic (s.push d.left) d.right)
      ≤ s.cost (s.midpoint (s.push d.left) d.right) + s.entropy (s.push d.left) + s.entropy d.right :=
    s.geodesic_cost (s.push d.left) d.right
  have hCost :
      s.cost (primary_map_measure_optimal_transport_geodesic_convexity (M := M) d)
        ≤ s.cost (s.midpoint (s.push d.left) d.right) + s.entropy (s.push d.left) + s.entropy d.right := by
    simpa [primary_map_measure_optimal_transport_geodesic_convexity] using hCostRaw
  have hMass : s.mass (s.push d.left) ≤ s.mass d.left :=
    s.push_mass d.left
  exact ⟨hCost, hMass⟩

theorem factorization_step_measure_optimal_transport_geodesic_convexity
    {M : Type u} [s : FrameworkStruct_measure_optimal_transport_geodesic_convexity M]
    (d : ContextData_measure_optimal_transport_geodesic_convexity M) :
    ((fun P : Prop => (fun u : Unit => P) ()) (s.cost (s.midpoint (s.push d.left) d.right) ≤ s.cost (s.push d.left) + s.cost d.right ∧
    s.cost (s.push d.left) ≤ s.cost d.left + s.entropy d.left))

 := by
  have hMid : s.cost (s.midpoint (s.push d.left) d.right) ≤ s.cost (s.push d.left) + s.cost d.right :=
    s.midpoint_cost (s.push d.left) d.right
  have hPush : s.cost (s.push d.left) ≤ s.cost d.left + s.entropy d.left :=
    s.push_cost d.left
  exact ⟨hMid, hPush⟩

theorem comparison_step_measure_optimal_transport_geodesic_convexity
    {M : Type u} [s : FrameworkStruct_measure_optimal_transport_geodesic_convexity M]
    (d : ContextData_measure_optimal_transport_geodesic_convexity M) :
    ((fun P : Prop => (fun u : Unit => P) ()) (tertiary_map_measure_optimal_transport_geodesic_convexity (M := M) d ∧
    ∃ c : Nat,
      c = secondary_map_measure_optimal_transport_geodesic_convexity (M := M) d ∧
      s.cost (primary_map_measure_optimal_transport_geodesic_convexity (M := M) d) ≤ c))

 := by
  have hSec :
      s.cost (primary_map_measure_optimal_transport_geodesic_convexity (M := M) d)
        ≤ secondary_map_measure_optimal_transport_geodesic_convexity (M := M) d := by
    simpa [secondary_map_measure_optimal_transport_geodesic_convexity] using
      (s.le_add_right_nat (s.cost (primary_map_measure_optimal_transport_geodesic_convexity (M := M) d)) d.budget)
  refine ⟨hSec, ?_⟩
  refine ⟨secondary_map_measure_optimal_transport_geodesic_convexity (M := M) d, rfl, ?_⟩
  exact hSec

theorem transport_step_measure_optimal_transport_geodesic_convexity
    {M : Type u} [s : FrameworkStruct_measure_optimal_transport_geodesic_convexity M]
    (d : ContextData_measure_optimal_transport_geodesic_convexity M) :
    ((fun P : Prop => (fun u : Unit => P) ()) (s.cost (primary_map_measure_optimal_transport_geodesic_convexity (M := M) d)
      ≤ s.cost (s.midpoint (s.push d.left) d.right) + s.entropy (s.push d.left) + s.entropy d.right ∧
    s.cost (s.push d.left) ≤ s.cost d.left + s.entropy d.left))

 := by
  have hStab := stability_step_measure_optimal_transport_geodesic_convexity (M := M) d
  rcases hStab with ⟨hCost, hMass⟩
  have hFact := factorization_step_measure_optimal_transport_geodesic_convexity (M := M) d
  rcases hFact with ⟨hMid, hPush⟩
  have _ : s.mass (s.push d.left) ≤ s.mass d.left := hMass
  have _ : s.cost (s.midpoint (s.push d.left) d.right) ≤ s.cost (s.push d.left) + s.cost d.right := hMid
  exact ⟨hCost, hPush⟩

theorem coherence_step_measure_optimal_transport_geodesic_convexity
    {M : Type u} [s : FrameworkStruct_measure_optimal_transport_geodesic_convexity M]
    (d : ContextData_measure_optimal_transport_geodesic_convexity M) :
    ((fun P : Prop => (fun u : Unit => P) ()) (∃ y : M,
      y = s.push d.left ∧
      s.mass y ≤ s.mass d.left ∧
      (∀ z : M, z = y → s.mass z ≤ s.mass d.left)))

 := by
  rcases s.witness_push d.left with ⟨y, hyEq, hyMass⟩
  have hAll : ∀ z : M, z = y → s.mass z ≤ s.mass d.left := by
    intro z hz
    simpa [hz] using hyMass
  exact ⟨y, hyEq, hyMass, hAll⟩

theorem iteration_step_measure_optimal_transport_geodesic_convexity
    {M : Type u} [s : FrameworkStruct_measure_optimal_transport_geodesic_convexity M]
    (d : ContextData_measure_optimal_transport_geodesic_convexity M) :
    ((fun P : Prop => (fun u : Unit => P) ()) (∃ y : M,
      y = primary_map_measure_optimal_transport_geodesic_convexity (M := M) d ∧
      s.cost y ≤ secondary_map_measure_optimal_transport_geodesic_convexity (M := M) d ∧
      ∃ w : M, w = s.push d.left ∧ s.mass w ≤ s.mass d.left))

 := by
  let y := primary_map_measure_optimal_transport_geodesic_convexity (M := M) d
  have hyEq : y = primary_map_measure_optimal_transport_geodesic_convexity (M := M) d := rfl
  have hCost : s.cost y ≤ secondary_map_measure_optimal_transport_geodesic_convexity (M := M) d := by
    rcases (comparison_step_measure_optimal_transport_geodesic_convexity (M := M) d).2 with ⟨c, hcEq, hc⟩
    simpa [y, hcEq] using hc
  have hMass := (stability_step_measure_optimal_transport_geodesic_convexity (M := M) d).2
  refine ⟨y, hyEq, hCost, ?_⟩
  exact ⟨s.push d.left, rfl, hMass⟩

theorem main_result_measure_optimal_transport_geodesic_convexity
    {M : Type u} [s : FrameworkStruct_measure_optimal_transport_geodesic_convexity M]
    (d : ContextData_measure_optimal_transport_geodesic_convexity M) :
    ((fun P : Prop => (fun u : Unit => P) ()) (∃ y : M,
      y = primary_map_measure_optimal_transport_geodesic_convexity (M := M) d ∧
      s.cost y ≤ secondary_map_measure_optimal_transport_geodesic_convexity (M := M) d ∧
      s.cost (s.push d.left) ≤ s.cost d.left + s.entropy d.left))

 := by
  rcases iteration_step_measure_optimal_transport_geodesic_convexity (M := M) d with
      ⟨y, hyEq, hCost, hwitness⟩
  rcases hwitness with ⟨w, hwEq, hwMass⟩
  have hPush := (factorization_step_measure_optimal_transport_geodesic_convexity (M := M) d).2
  have _ : s.mass w ≤ s.mass d.left := hwMass
  have _ : w = s.push d.left := hwEq
  have _ : s.entropy d.left ≤ d.budget := d.entropy_cap_left
  have _ : s.entropy d.right ≤ d.budget := d.entropy_cap_right
  have _ : s.cost d.left ≤ s.cost d.right + d.budget := d.cost_bridge
  exact ⟨y, hyEq, hCost, hPush⟩
