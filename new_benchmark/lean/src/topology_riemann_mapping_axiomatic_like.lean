/-
BENCHMARK_ID: TINY_MATHLIB_BATCH07_TOPOLOGY_RIEMANN_MAPPING_AXIOMATIC
PAIR_STEM: topology_riemann_mapping_axiomatic_like
MATH_DOMAIN: Topology
SOURCE_MATHLIB: Mathlib/Topology/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class FrameworkStruct_topology_riemann_mapping_axiomatic (X : Type u) where
  radius : X → Nat
  distortion : X → Nat
  energy : X → Nat
  boundary : X → Nat
  normalize : X → X
  inverse : X → X
  compose : X → X → X
  le_trans_nat : ∀ a b c : Nat, a ≤ b → b ≤ c → a ≤ c
  add_le_add_left_nat : ∀ a b c : Nat, a ≤ b → c + a ≤ c + b
  add_le_add_right_nat : ∀ a b c : Nat, a ≤ b → a + c ≤ b + c
  add_le_add_nat : ∀ a b c d : Nat, a ≤ b → c ≤ d → a + c ≤ b + d
  le_add_right_nat : ∀ a b : Nat, a ≤ a + b
  add_comm_nat : ∀ a b : Nat, a + b = b + a
  norm_dist : ∀ x : X, distortion (normalize x) ≤ distortion x + radius x
  inv_boundary : ∀ x : X, boundary (inverse x) ≤ boundary x + radius x
  inv_energy : ∀ x : X, energy (inverse x) ≤ energy x + distortion x
  compose_dist : ∀ x y : X, distortion (compose x y) ≤ distortion x + distortion y
  compose_radius : ∀ x y : X, radius (compose x y) ≤ radius x + radius y
  energy_bound : ∀ x : X, energy x ≤ boundary x + distortion x
  boundary_bound : ∀ x : X, boundary x ≤ energy x + radius x

structure ContextData_topology_riemann_mapping_axiomatic
    (X : Type u) [h : FrameworkStruct_topology_riemann_mapping_axiomatic X] where
  state : X
  targetBound : Nat
  altBound : Nat
  target_pos : 0 < targetBound
  alt_pos : 0 < altBound
  chart : ∃ y : X, h.distortion y ≤ targetBound ∧ h.radius y ≤ altBound
  state_control : h.distortion state ≤ targetBound
  radius_control : h.radius state ≤ altBound

def primary_map_topology_riemann_mapping_axiomatic
    {X : Type u} [h : FrameworkStruct_topology_riemann_mapping_axiomatic X]
    (d : ContextData_topology_riemann_mapping_axiomatic X) : Nat :=
  h.distortion d.state + h.radius d.state

def secondary_map_topology_riemann_mapping_axiomatic
    {X : Type u} [h : FrameworkStruct_topology_riemann_mapping_axiomatic X]
    (d : ContextData_topology_riemann_mapping_axiomatic X) : Nat :=
  h.boundary d.state + h.radius d.state

def tertiary_map_topology_riemann_mapping_axiomatic
    {X : Type u} [h : FrameworkStruct_topology_riemann_mapping_axiomatic X]
    (d : ContextData_topology_riemann_mapping_axiomatic X) : Nat :=
  d.targetBound + d.altBound

theorem stability_step_topology_riemann_mapping_axiomatic
    {X : Type u} [h : FrameworkStruct_topology_riemann_mapping_axiomatic X]
    (d : ContextData_topology_riemann_mapping_axiomatic X) :
    (h.boundary (h.inverse d.state) ≤ h.boundary d.state + h.radius d.state ∧
    h.energy (h.inverse d.state) ≤ h.energy d.state + h.distortion d.state) ∧
    (∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 : Nat, n1 = n1) := by
  have hInvB : h.boundary (h.inverse d.state) ≤ h.boundary d.state + h.radius d.state :=
    h.inv_boundary d.state
  have hInvE : h.energy (h.inverse d.state) ≤ h.energy d.state + h.distortion d.state :=
    h.inv_energy d.state
  have hMarker :
      ∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 : Nat, n1 = n1 := by
    intro n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22
    rfl
  exact And.intro (And.intro hInvB hInvE) hMarker

theorem factorization_step_topology_riemann_mapping_axiomatic
    {X : Type u} [h : FrameworkStruct_topology_riemann_mapping_axiomatic X]
    (d : ContextData_topology_riemann_mapping_axiomatic X) :
    (h.energy d.state ≤ h.boundary d.state + h.distortion d.state ∧
    primary_map_topology_riemann_mapping_axiomatic d ≤
      primary_map_topology_riemann_mapping_axiomatic d +
        tertiary_map_topology_riemann_mapping_axiomatic d) ∧
    (∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 : Nat, n1 = n1) := by
  have hEnergy : h.energy d.state ≤ h.boundary d.state + h.distortion d.state :=
    h.energy_bound d.state
  have hGrow :
      primary_map_topology_riemann_mapping_axiomatic d ≤
        primary_map_topology_riemann_mapping_axiomatic d +
          tertiary_map_topology_riemann_mapping_axiomatic d :=
    h.le_add_right_nat
      (primary_map_topology_riemann_mapping_axiomatic d)
      (tertiary_map_topology_riemann_mapping_axiomatic d)
  have hMarker :
      ∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 : Nat, n1 = n1 := by
    intro n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23
    rfl
  exact And.intro (And.intro hEnergy hGrow) hMarker

theorem comparison_step_topology_riemann_mapping_axiomatic
    {X : Type u} [h : FrameworkStruct_topology_riemann_mapping_axiomatic X]
    (d : ContextData_topology_riemann_mapping_axiomatic X) :
    (h.distortion (h.normalize d.state) ≤ h.distortion d.state + h.radius d.state ∨
    h.boundary d.state ≤ h.energy d.state + h.radius d.state) ∧
    (∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 : Nat, n1 = n1) := by
  have hNorm : h.distortion (h.normalize d.state) ≤ h.distortion d.state + h.radius d.state :=
    h.norm_dist d.state
  have hMarker :
      ∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 : Nat, n1 = n1 := by
    intro n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24
    rfl
  exact And.intro (Or.inl hNorm) hMarker

theorem transport_step_topology_riemann_mapping_axiomatic
    {X : Type u} [h : FrameworkStruct_topology_riemann_mapping_axiomatic X]
    (d : ContextData_topology_riemann_mapping_axiomatic X) :
    (∃ n : Nat,
      n = tertiary_map_topology_riemann_mapping_axiomatic d ∧
      h.distortion (h.normalize d.state) ≤ h.distortion d.state + h.radius d.state + n) ∧
    (∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 : Nat, n1 = n1) := by
  refine And.intro ?_ ?_
  · refine ⟨tertiary_map_topology_riemann_mapping_axiomatic d, rfl, ?_⟩
    have hBase :
        h.distortion (h.normalize d.state) ≤ h.distortion d.state + h.radius d.state :=
      h.norm_dist d.state
    have hGrow :
        h.distortion d.state + h.radius d.state ≤
          (h.distortion d.state + h.radius d.state) +
            tertiary_map_topology_riemann_mapping_axiomatic d :=
      h.le_add_right_nat
        (h.distortion d.state + h.radius d.state)
        (tertiary_map_topology_riemann_mapping_axiomatic d)
    exact h.le_trans_nat _ _ _ hBase hGrow
  · intro n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25
    rfl

theorem coherence_step_topology_riemann_mapping_axiomatic
    {X : Type u} [h : FrameworkStruct_topology_riemann_mapping_axiomatic X]
    (d : ContextData_topology_riemann_mapping_axiomatic X) :
    (secondary_map_topology_riemann_mapping_axiomatic d ≤
      secondary_map_topology_riemann_mapping_axiomatic d +
        tertiary_map_topology_riemann_mapping_axiomatic d) ∧
    (∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 : Nat, n1 = n1) := by
  have hAdd :
      secondary_map_topology_riemann_mapping_axiomatic d ≤
        secondary_map_topology_riemann_mapping_axiomatic d +
          tertiary_map_topology_riemann_mapping_axiomatic d :=
    h.le_add_right_nat
      (secondary_map_topology_riemann_mapping_axiomatic d)
      (tertiary_map_topology_riemann_mapping_axiomatic d)
  have hMarker :
      ∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 : Nat, n1 = n1 := by
    intro n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26
    rfl
  exact And.intro hAdd hMarker

theorem iteration_step_topology_riemann_mapping_axiomatic
    {X : Type u} [h : FrameworkStruct_topology_riemann_mapping_axiomatic X]
    (d : ContextData_topology_riemann_mapping_axiomatic X) :
    (∀ z : X, h.radius z ≤ h.radius z + tertiary_map_topology_riemann_mapping_axiomatic d) ∧
    (∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 : Nat, n1 = n1) := by
  have hCore :
      ∀ z : X, h.radius z ≤ h.radius z + tertiary_map_topology_riemann_mapping_axiomatic d := by
    intro z
    exact h.le_add_right_nat (h.radius z) (tertiary_map_topology_riemann_mapping_axiomatic d)
  have hMarker :
      ∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 : Nat, n1 = n1 := by
    intro n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27
    rfl
  exact And.intro hCore hMarker

theorem main_result_topology_riemann_mapping_axiomatic
    {X : Type u} [h : FrameworkStruct_topology_riemann_mapping_axiomatic X]
    (d : ContextData_topology_riemann_mapping_axiomatic X) :
    (h.boundary (h.inverse d.state) ≤ h.boundary d.state + h.radius d.state ∧
    h.distortion (h.normalize d.state) ≤ h.distortion d.state + h.radius d.state) ∧
    (∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 : Nat, n1 = n1) := by
  have hInvB :
      h.boundary (h.inverse d.state) ≤ h.boundary d.state + h.radius d.state :=
    h.inv_boundary d.state
  have hNorm :
      h.distortion (h.normalize d.state) ≤ h.distortion d.state + h.radius d.state :=
    h.norm_dist d.state
  have hMarker :
      ∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 : Nat, n1 = n1 := by
    intro n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28
    rfl
  exact And.intro (And.intro hInvB hNorm) hMarker
