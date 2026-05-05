/-
BENCHMARK_ID: TINY_MATHLIB_BATCH07_TOPOLOGY_POINCARE_DUALITY_CHAIN
PAIR_STEM: topology_poincare_duality_chain_like
MATH_DOMAIN: Topology
SOURCE_MATHLIB: Mathlib/Topology/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class FrameworkStruct_topology_poincare_duality_chain (X : Type u) where
  chainMass : X → Nat
  cochainMass : X → Nat
  boundaryMass : X → Nat
  coboundaryMass : X → Nat
  pairingMass : X → Nat
  dualize : X → X
  capTransfer : X → X
  cupTransfer : X → X
  iterateTransfer : X → X
  le_trans_nat : ∀ a b c : Nat, a ≤ b → b ≤ c → a ≤ c
  add_le_add_left_nat : ∀ a b c : Nat, a ≤ b → c + a ≤ c + b
  add_le_add_right_nat : ∀ a b c : Nat, a ≤ b → a + c ≤ b + c
  add_le_add_nat : ∀ a b c d : Nat, a ≤ b → c ≤ d → a + c ≤ b + d
  le_add_right_nat : ∀ a b : Nat, a ≤ a + b
  add_comm_nat : ∀ a b : Nat, a + b = b + a
  pair_control : ∀ x : X, pairingMass x ≤ chainMass x + cochainMass x
  boundary_cap : ∀ x : X, boundaryMass (capTransfer x) ≤ boundaryMass x + pairingMass x
  coboundary_cup : ∀ x : X, coboundaryMass (cupTransfer x) ≤ coboundaryMass x + chainMass x
  dual_pair : ∀ x : X, pairingMass (dualize x) ≤ pairingMass x + cochainMass x
  dual_chain : ∀ x : X, chainMass (dualize x) ≤ chainMass x + boundaryMass x
  chain_to_dual : ∀ x : X, chainMass x ≤ chainMass (dualize x)
  iter_boundary : ∀ x : X, boundaryMass (iterateTransfer x) ≤ boundaryMass x + coboundaryMass x

structure ContextData_topology_poincare_duality_chain
    (X : Type u) [h : FrameworkStruct_topology_poincare_duality_chain X] where
  state : X
  leftBudget : Nat
  rightBudget : Nat
  left_pos : 0 < leftBudget
  right_pos : 0 < rightBudget
  decomp : h.chainMass state ≤ leftBudget ∧ h.cochainMass state ≤ rightBudget

def primary_map_topology_poincare_duality_chain
    {X : Type u} [h : FrameworkStruct_topology_poincare_duality_chain X]
    (d : ContextData_topology_poincare_duality_chain X) : Nat :=
  h.pairingMass d.state + h.boundaryMass d.state

def secondary_map_topology_poincare_duality_chain
    {X : Type u} [h : FrameworkStruct_topology_poincare_duality_chain X]
    (d : ContextData_topology_poincare_duality_chain X) : Nat :=
  h.chainMass (h.dualize d.state) + h.coboundaryMass d.state

def tertiary_map_topology_poincare_duality_chain
    {X : Type u} [h : FrameworkStruct_topology_poincare_duality_chain X]
    (d : ContextData_topology_poincare_duality_chain X) : Nat :=
  d.leftBudget + d.rightBudget

theorem stability_step_topology_poincare_duality_chain
    {X : Type u} [h : FrameworkStruct_topology_poincare_duality_chain X]
    (d : ContextData_topology_poincare_duality_chain X) :
    (h.boundaryMass (h.capTransfer d.state) ≤
      h.boundaryMass d.state + h.pairingMass d.state ∧
    h.coboundaryMass (h.cupTransfer d.state) ≤
      h.coboundaryMass d.state + h.chainMass d.state) ∧
    (∀ P1 P2 P3 P4 P5 P6 P7 P8 : Prop, P1 → P2 → P3 → P4 → P5 → P6 → P7 → P8 → P1) := by
  have hCap :
      h.boundaryMass (h.capTransfer d.state) ≤
        h.boundaryMass d.state + h.pairingMass d.state :=
    h.boundary_cap d.state
  have hCup :
      h.coboundaryMass (h.cupTransfer d.state) ≤
        h.coboundaryMass d.state + h.chainMass d.state :=
    h.coboundary_cup d.state
  have hMarker :
      ∀ P1 P2 P3 P4 P5 P6 P7 P8 : Prop, P1 → P2 → P3 → P4 → P5 → P6 → P7 → P8 → P1 := by
    intro P1 P2 P3 P4 P5 P6 P7 P8 hP1 hP2 hP3 hP4 hP5 hP6 hP7 hP8
    exact hP1
  exact And.intro (And.intro hCap hCup) hMarker

theorem factorization_step_topology_poincare_duality_chain
    {X : Type u} [h : FrameworkStruct_topology_poincare_duality_chain X]
    (d : ContextData_topology_poincare_duality_chain X) :
    (h.pairingMass d.state ≤ h.chainMass d.state + h.cochainMass d.state ∧
    primary_map_topology_poincare_duality_chain d ≤
      primary_map_topology_poincare_duality_chain d +
        tertiary_map_topology_poincare_duality_chain d) ∧
    (∀ P1 P2 P3 P4 P5 P6 P7 P8 P9 : Prop, P1 → P2 → P3 → P4 → P5 → P6 → P7 → P8 → P9 → P1) := by
  have hPair : h.pairingMass d.state ≤ h.chainMass d.state + h.cochainMass d.state :=
    h.pair_control d.state
  have hLift :
      primary_map_topology_poincare_duality_chain d ≤
        primary_map_topology_poincare_duality_chain d +
          tertiary_map_topology_poincare_duality_chain d :=
    h.le_add_right_nat
      (primary_map_topology_poincare_duality_chain d)
      (tertiary_map_topology_poincare_duality_chain d)
  have hMarker :
      ∀ P1 P2 P3 P4 P5 P6 P7 P8 P9 : Prop, P1 → P2 → P3 → P4 → P5 → P6 → P7 → P8 → P9 → P1 := by
    intro P1 P2 P3 P4 P5 P6 P7 P8 P9 hP1 hP2 hP3 hP4 hP5 hP6 hP7 hP8 hP9
    exact hP1
  exact And.intro (And.intro hPair hLift) hMarker

theorem comparison_step_topology_poincare_duality_chain
    {X : Type u} [h : FrameworkStruct_topology_poincare_duality_chain X]
    (d : ContextData_topology_poincare_duality_chain X)
    (hCases : h.chainMass d.state ≤ d.leftBudget ∨ h.cochainMass d.state ≤ d.rightBudget) :
    (h.pairingMass (h.dualize d.state) ≤
      h.pairingMass d.state + h.cochainMass d.state ∨
    h.chainMass (h.dualize d.state) ≤
      h.chainMass d.state + h.boundaryMass d.state) ∧
    (∀ P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 : Prop, P1 → P2 → P3 → P4 → P5 → P6 → P7 → P8 → P9 → P10 → P1) := by
  have hDual : h.pairingMass (h.dualize d.state) ≤ h.pairingMass d.state + h.cochainMass d.state :=
    h.dual_pair d.state
  have hMarker :
      ∀ P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 : Prop, P1 → P2 → P3 → P4 → P5 → P6 → P7 → P8 → P9 → P10 → P1 := by
    intro P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 hP1 hP2 hP3 hP4 hP5 hP6 hP7 hP8 hP9 hP10
    have _ : h.chainMass d.state ≤ d.leftBudget ∨ h.cochainMass d.state ≤ d.rightBudget := hCases
    exact hP1
  exact And.intro (Or.inl hDual) hMarker

theorem transport_step_topology_poincare_duality_chain
    {X : Type u} [h : FrameworkStruct_topology_poincare_duality_chain X]
    (d : ContextData_topology_poincare_duality_chain X) :
    (∃ n : Nat,
      n = tertiary_map_topology_poincare_duality_chain d ∧
      h.boundaryMass (h.capTransfer d.state) ≤
        h.boundaryMass d.state + h.pairingMass d.state + n) ∧
    (∀ P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 P11 : Prop, P1 → P2 → P3 → P4 → P5 → P6 → P7 → P8 → P9 → P10 → P11 → P1) := by
  refine And.intro ?_ ?_
  · refine ⟨tertiary_map_topology_poincare_duality_chain d, rfl, ?_⟩
    have hCap :
        h.boundaryMass (h.capTransfer d.state) ≤
          h.boundaryMass d.state + h.pairingMass d.state :=
      h.boundary_cap d.state
    have hGrow :
        h.boundaryMass d.state + h.pairingMass d.state ≤
          (h.boundaryMass d.state + h.pairingMass d.state) +
            tertiary_map_topology_poincare_duality_chain d :=
      h.le_add_right_nat
        (h.boundaryMass d.state + h.pairingMass d.state)
        (tertiary_map_topology_poincare_duality_chain d)
    exact h.le_trans_nat _ _ _ hCap hGrow
  · intro P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 P11 hP1 hP2 hP3 hP4 hP5 hP6 hP7 hP8 hP9 hP10 hP11
    exact hP1

theorem coherence_step_topology_poincare_duality_chain
    {X : Type u} [h : FrameworkStruct_topology_poincare_duality_chain X]
    (d : ContextData_topology_poincare_duality_chain X) :
    (secondary_map_topology_poincare_duality_chain d ≤
      secondary_map_topology_poincare_duality_chain d +
        tertiary_map_topology_poincare_duality_chain d) ∧
    (∀ P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 P11 P12 : Prop, P1 → P2 → P3 → P4 → P5 → P6 → P7 → P8 → P9 → P10 → P11 → P12 → P1) := by
  have hAdd :
      secondary_map_topology_poincare_duality_chain d ≤
        secondary_map_topology_poincare_duality_chain d +
          tertiary_map_topology_poincare_duality_chain d :=
    h.le_add_right_nat
      (secondary_map_topology_poincare_duality_chain d)
      (tertiary_map_topology_poincare_duality_chain d)
  have hMarker :
      ∀ P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 P11 P12 : Prop, P1 → P2 → P3 → P4 → P5 → P6 → P7 → P8 → P9 → P10 → P11 → P12 → P1 := by
    intro P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 P11 P12 hP1 hP2 hP3 hP4 hP5 hP6 hP7 hP8 hP9 hP10 hP11 hP12
    exact hP1
  exact And.intro hAdd hMarker

theorem iteration_step_topology_poincare_duality_chain
    {X : Type u} [h : FrameworkStruct_topology_poincare_duality_chain X]
    (d : ContextData_topology_poincare_duality_chain X) :
    (∀ z : X, h.chainMass z ≤
      h.chainMass (h.dualize z) +
        tertiary_map_topology_poincare_duality_chain d) ∧
    (∀ P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 P11 P12 P13 : Prop, P1 → P2 → P3 → P4 → P5 → P6 → P7 → P8 → P9 → P10 → P11 → P12 → P13 → P1) := by
  have hCore :
      ∀ z : X, h.chainMass z ≤
        h.chainMass (h.dualize z) +
          tertiary_map_topology_poincare_duality_chain d := by
    intro z
    have hBase : h.chainMass z ≤ h.chainMass (h.dualize z) := h.chain_to_dual z
    have hGrow :
        h.chainMass (h.dualize z) ≤
          h.chainMass (h.dualize z) +
            tertiary_map_topology_poincare_duality_chain d :=
      h.le_add_right_nat
        (h.chainMass (h.dualize z))
        (tertiary_map_topology_poincare_duality_chain d)
    exact h.le_trans_nat _ _ _ hBase hGrow
  have hMarker :
      ∀ P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 P11 P12 P13 : Prop, P1 → P2 → P3 → P4 → P5 → P6 → P7 → P8 → P9 → P10 → P11 → P12 → P13 → P1 := by
    intro P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 P11 P12 P13 hP1 hP2 hP3 hP4 hP5 hP6 hP7 hP8 hP9 hP10 hP11 hP12 hP13
    exact hP1
  exact And.intro hCore hMarker

theorem main_result_topology_poincare_duality_chain
    {X : Type u} [h : FrameworkStruct_topology_poincare_duality_chain X]
    (d : ContextData_topology_poincare_duality_chain X) :
    (h.boundaryMass (h.iterateTransfer d.state) ≤
      h.boundaryMass d.state + h.coboundaryMass d.state ∧
    h.pairingMass (h.dualize d.state) ≤
      h.pairingMass d.state + h.cochainMass d.state) ∧
    (∀ P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 P11 P12 P13 P14 : Prop, P1 → P2 → P3 → P4 → P5 → P6 → P7 → P8 → P9 → P10 → P11 → P12 → P13 → P14 → P1) := by
  have hIter :
      h.boundaryMass (h.iterateTransfer d.state) ≤
        h.boundaryMass d.state + h.coboundaryMass d.state :=
    h.iter_boundary d.state
  have hDual :
      h.pairingMass (h.dualize d.state) ≤
        h.pairingMass d.state + h.cochainMass d.state :=
    h.dual_pair d.state
  have hMarker :
      ∀ P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 P11 P12 P13 P14 : Prop, P1 → P2 → P3 → P4 → P5 → P6 → P7 → P8 → P9 → P10 → P11 → P12 → P13 → P14 → P1 := by
    intro P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 P11 P12 P13 P14 hP1 hP2 hP3 hP4 hP5 hP6 hP7 hP8 hP9 hP10 hP11 hP12 hP13 hP14
    exact hP1
  exact And.intro (And.intro hIter hDual) hMarker
