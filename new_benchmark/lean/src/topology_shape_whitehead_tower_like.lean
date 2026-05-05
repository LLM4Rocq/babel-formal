/-
BENCHMARK_ID: TINY_MATHLIB_BATCH07_TOPOLOGY_SHAPE_WHITEHEAD_TOWER
PAIR_STEM: topology_shape_whitehead_tower_like
MATH_DOMAIN: Topology
SOURCE_MATHLIB: Mathlib/Topology/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class FrameworkStruct_topology_shape_whitehead_tower (X : Type u) where
  truncRank : X → Nat
  fiberRank : X → Nat
  liftRank : X → Nat
  towerRank : X → Nat
  upMap : X → X
  downMap : X → X
  connMap : X → X
  itrMap : X → X
  le_trans_nat : ∀ a b c : Nat, a ≤ b → b ≤ c → a ≤ c
  add_le_add_left_nat : ∀ a b c : Nat, a ≤ b → c + a ≤ c + b
  add_le_add_right_nat : ∀ a b c : Nat, a ≤ b → a + c ≤ b + c
  add_le_add_nat : ∀ a b c d : Nat, a ≤ b → c ≤ d → a + c ≤ b + d
  le_add_right_nat : ∀ a b : Nat, a ≤ a + b
  add_comm_nat : ∀ a b : Nat, a + b = b + a
  up_lift : ∀ x : X, liftRank (upMap x) ≤ liftRank x + truncRank x
  down_trunc : ∀ x : X, truncRank (downMap x) ≤ truncRank x + fiberRank x
  conn_fiber : ∀ x : X, fiberRank (connMap x) ≤ fiberRank x + liftRank x
  tower_from_fiber : ∀ x : X, towerRank x ≤ fiberRank x + liftRank x
  itr_lift : ∀ x : X, liftRank (itrMap x) ≤ liftRank x + towerRank x
  itr_tower : ∀ x : X, towerRank (itrMap x) ≤ towerRank x + truncRank x
  fiber_nondec : ∀ x : X, fiberRank x ≤ fiberRank (connMap x)

structure ContextData_topology_shape_whitehead_tower
    (X : Type u) [h : FrameworkStruct_topology_shape_whitehead_tower X] where
  state : X
  baseBound : Nat
  fiberBound : Nat
  base_pos : 0 < baseBound
  fiber_pos : 0 < fiberBound
  witness : h.truncRank state ≤ baseBound ∨ h.fiberRank state ≤ fiberBound
  lift_small : h.liftRank state ≤ baseBound + fiberBound

def primary_map_topology_shape_whitehead_tower
    {X : Type u} [h : FrameworkStruct_topology_shape_whitehead_tower X]
    (d : ContextData_topology_shape_whitehead_tower X) : Nat :=
  h.truncRank d.state + h.fiberRank d.state

def secondary_map_topology_shape_whitehead_tower
    {X : Type u} [h : FrameworkStruct_topology_shape_whitehead_tower X]
    (d : ContextData_topology_shape_whitehead_tower X) : Nat :=
  h.liftRank d.state + h.towerRank d.state

def tertiary_map_topology_shape_whitehead_tower
    {X : Type u} [h : FrameworkStruct_topology_shape_whitehead_tower X]
    (d : ContextData_topology_shape_whitehead_tower X) : Nat :=
  primary_map_topology_shape_whitehead_tower d + d.baseBound

theorem stability_step_topology_shape_whitehead_tower
    {X : Type u} [h : FrameworkStruct_topology_shape_whitehead_tower X]
    (d : ContextData_topology_shape_whitehead_tower X) :
    (h.liftRank (h.upMap d.state) ≤ h.liftRank d.state + h.truncRank d.state ∧
    h.truncRank (h.downMap d.state) ≤ h.truncRank d.state + h.fiberRank d.state) ∧
    (∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 : Nat, n1 = n1) := by
  have hUp :
      h.liftRank (h.upMap d.state) ≤ h.liftRank d.state + h.truncRank d.state :=
    h.up_lift d.state
  have hDown :
      h.truncRank (h.downMap d.state) ≤ h.truncRank d.state + h.fiberRank d.state :=
    h.down_trunc d.state
  have hMarker : ∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 : Nat, n1 = n1 := by
    intro n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15
    rfl
  exact And.intro (And.intro hUp hDown) hMarker

theorem factorization_step_topology_shape_whitehead_tower
    {X : Type u} [h : FrameworkStruct_topology_shape_whitehead_tower X]
    (d : ContextData_topology_shape_whitehead_tower X) :
    (h.towerRank d.state ≤ h.fiberRank d.state + h.liftRank d.state ∧
    primary_map_topology_shape_whitehead_tower d ≤
      primary_map_topology_shape_whitehead_tower d +
        tertiary_map_topology_shape_whitehead_tower d) ∧
    (∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 : Nat, n1 = n1) := by
  have hTower : h.towerRank d.state ≤ h.fiberRank d.state + h.liftRank d.state :=
    h.tower_from_fiber d.state
  have hGrow :
      primary_map_topology_shape_whitehead_tower d ≤
        primary_map_topology_shape_whitehead_tower d +
          tertiary_map_topology_shape_whitehead_tower d :=
    h.le_add_right_nat
      (primary_map_topology_shape_whitehead_tower d)
      (tertiary_map_topology_shape_whitehead_tower d)
  have hMarker : ∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 : Nat, n1 = n1 := by
    intro n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16
    rfl
  exact And.intro (And.intro hTower hGrow) hMarker

theorem comparison_step_topology_shape_whitehead_tower
    {X : Type u} [h : FrameworkStruct_topology_shape_whitehead_tower X]
    (d : ContextData_topology_shape_whitehead_tower X)
    (hCase : h.truncRank d.state ≤ d.baseBound ∨ h.fiberRank d.state ≤ d.fiberBound) :
    (h.fiberRank (h.connMap d.state) ≤ h.fiberRank d.state + h.liftRank d.state ∨
    h.liftRank (h.itrMap d.state) ≤ h.liftRank d.state + h.towerRank d.state) ∧
    (∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 : Nat, n1 = n1) := by
  have hFiber :
      h.fiberRank (h.connMap d.state) ≤ h.fiberRank d.state + h.liftRank d.state :=
    h.conn_fiber d.state
  have hMarker : ∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 : Nat, n1 = n1 := by
    intro n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17
    have _ : h.truncRank d.state ≤ d.baseBound ∨ h.fiberRank d.state ≤ d.fiberBound := hCase
    rfl
  exact And.intro (Or.inl hFiber) hMarker

theorem transport_step_topology_shape_whitehead_tower
    {X : Type u} [h : FrameworkStruct_topology_shape_whitehead_tower X]
    (d : ContextData_topology_shape_whitehead_tower X) :
    (∃ n : Nat,
      n = tertiary_map_topology_shape_whitehead_tower d ∧
      h.towerRank (h.itrMap d.state) ≤ h.towerRank d.state + h.truncRank d.state + n) ∧
    (∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 : Nat, n1 = n1) := by
  refine And.intro ?_ ?_
  · refine ⟨tertiary_map_topology_shape_whitehead_tower d, rfl, ?_⟩
    have hBase :
        h.towerRank (h.itrMap d.state) ≤ h.towerRank d.state + h.truncRank d.state :=
      h.itr_tower d.state
    have hGrow :
        h.towerRank d.state + h.truncRank d.state ≤
          (h.towerRank d.state + h.truncRank d.state) +
            tertiary_map_topology_shape_whitehead_tower d :=
      h.le_add_right_nat
        (h.towerRank d.state + h.truncRank d.state)
        (tertiary_map_topology_shape_whitehead_tower d)
    exact h.le_trans_nat _ _ _ hBase hGrow
  · intro n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18
    rfl

theorem coherence_step_topology_shape_whitehead_tower
    {X : Type u} [h : FrameworkStruct_topology_shape_whitehead_tower X]
    (d : ContextData_topology_shape_whitehead_tower X) :
    (secondary_map_topology_shape_whitehead_tower d ≤
      secondary_map_topology_shape_whitehead_tower d +
        tertiary_map_topology_shape_whitehead_tower d) ∧
    (∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 : Nat, n1 = n1) := by
  have hAdd :
      secondary_map_topology_shape_whitehead_tower d ≤
        secondary_map_topology_shape_whitehead_tower d +
          tertiary_map_topology_shape_whitehead_tower d :=
    h.le_add_right_nat
      (secondary_map_topology_shape_whitehead_tower d)
      (tertiary_map_topology_shape_whitehead_tower d)
  have hMarker :
      ∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 : Nat, n1 = n1 := by
    intro n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19
    rfl
  exact And.intro hAdd hMarker

theorem iteration_step_topology_shape_whitehead_tower
    {X : Type u} [h : FrameworkStruct_topology_shape_whitehead_tower X]
    (d : ContextData_topology_shape_whitehead_tower X) :
    (∀ z : X, h.fiberRank z ≤
      h.fiberRank (h.connMap z) +
        tertiary_map_topology_shape_whitehead_tower d) ∧
    (∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 : Nat, n1 = n1) := by
  have hCore :
      ∀ z : X, h.fiberRank z ≤
        h.fiberRank (h.connMap z) +
          tertiary_map_topology_shape_whitehead_tower d := by
    intro z
    have hBase : h.fiberRank z ≤ h.fiberRank (h.connMap z) := h.fiber_nondec z
    have hGrow :
        h.fiberRank (h.connMap z) ≤
          h.fiberRank (h.connMap z) +
            tertiary_map_topology_shape_whitehead_tower d :=
      h.le_add_right_nat
        (h.fiberRank (h.connMap z))
        (tertiary_map_topology_shape_whitehead_tower d)
    exact h.le_trans_nat _ _ _ hBase hGrow
  have hMarker :
      ∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 : Nat, n1 = n1 := by
    intro n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20
    rfl
  exact And.intro hCore hMarker

theorem main_result_topology_shape_whitehead_tower
    {X : Type u} [h : FrameworkStruct_topology_shape_whitehead_tower X]
    (d : ContextData_topology_shape_whitehead_tower X) :
    (h.towerRank (h.itrMap d.state) ≤ h.towerRank d.state + h.truncRank d.state ∧
    h.liftRank (h.upMap d.state) ≤ h.liftRank d.state + h.truncRank d.state) ∧
    (∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 : Nat, n1 = n1) := by
  have hTower :
      h.towerRank (h.itrMap d.state) ≤ h.towerRank d.state + h.truncRank d.state :=
    h.itr_tower d.state
  have hLift :
      h.liftRank (h.upMap d.state) ≤ h.liftRank d.state + h.truncRank d.state :=
    h.up_lift d.state
  have hMarker :
      ∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 : Nat, n1 = n1 := by
    intro n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21
    rfl
  exact And.intro (And.intro hTower hLift) hMarker
