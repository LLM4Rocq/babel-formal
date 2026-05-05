/-
BENCHMARK_ID: TINY_MATHLIB_BATCH07_ANALYSIS_SEMICONVEX_GRADIENT_FLOW
PAIR_STEM: analysis_semiconvex_gradient_flow_like
MATH_DOMAIN: Analysis
SOURCE_MATHLIB: Mathlib/Analysis/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class FrameworkStruct_analysis_semiconvex_gradient_flow (X : Type u) where
  energy : X → Nat
  slope : X → Nat
  distance : X → Nat
  velocity : X → Nat
  stepMap : X → X
  proxMap : X → X
  iterateMap : X → X
  le_trans_nat : ∀ a b c : Nat, a ≤ b → b ≤ c → a ≤ c
  add_le_add_left_nat : ∀ a b c : Nat, a ≤ b → c + a ≤ c + b
  add_le_add_right_nat : ∀ a b c : Nat, a ≤ b → a + c ≤ b + c
  add_le_add_nat : ∀ a b c d : Nat, a ≤ b → c ≤ d → a + c ≤ b + d
  le_add_right_nat : ∀ a b : Nat, a ≤ a + b
  add_comm_nat : ∀ a b : Nat, a + b = b + a
  step_energy : ∀ x : X, energy (stepMap x) ≤ energy x + slope x
  step_slope : ∀ x : X, slope (stepMap x) ≤ slope x + velocity x
  prox_velocity : ∀ x : X, velocity (proxMap x) ≤ velocity x + distance x
  iterate_distance : ∀ x : X, distance (iterateMap x) ≤ distance x + velocity x
  iterate_slope : ∀ x : X, slope (iterateMap x) ≤ slope x + distance x
  distance_bridge : ∀ x : X, distance x ≤ energy x + slope x

structure ContextData_analysis_semiconvex_gradient_flow
    (X : Type u) [h : FrameworkStruct_analysis_semiconvex_gradient_flow X] where
  state : X
  timeHorizon : Nat
  slack : Nat
  time_pos : 0 < timeHorizon
  slack_pos : 0 < slack
  control_pair : h.energy state ≤ timeHorizon ∧ h.slope state ≤ slack
  dist_small : h.distance state ≤ timeHorizon + slack

def primary_map_analysis_semiconvex_gradient_flow
    {X : Type u} [h : FrameworkStruct_analysis_semiconvex_gradient_flow X]
    (d : ContextData_analysis_semiconvex_gradient_flow X) : Nat :=
  h.energy d.state + h.slope d.state

def secondary_map_analysis_semiconvex_gradient_flow
    {X : Type u} [h : FrameworkStruct_analysis_semiconvex_gradient_flow X]
    (d : ContextData_analysis_semiconvex_gradient_flow X) : Nat :=
  h.distance d.state + h.velocity d.state

def tertiary_map_analysis_semiconvex_gradient_flow
    {X : Type u} [h : FrameworkStruct_analysis_semiconvex_gradient_flow X]
    (d : ContextData_analysis_semiconvex_gradient_flow X) : Nat :=
  d.timeHorizon + d.slack

theorem stability_step_analysis_semiconvex_gradient_flow
    {X : Type u} [h : FrameworkStruct_analysis_semiconvex_gradient_flow X]
    (d : ContextData_analysis_semiconvex_gradient_flow X) :
    (h.energy (h.stepMap d.state) ≤ h.energy d.state + h.slope d.state ∧
    h.slope (h.stepMap d.state) ≤ h.slope d.state + h.velocity d.state) ∧
    (∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 : Nat, n1 = n1) := by
  have hStepE : h.energy (h.stepMap d.state) ≤ h.energy d.state + h.slope d.state :=
    h.step_energy d.state
  have hStepS : h.slope (h.stepMap d.state) ≤ h.slope d.state + h.velocity d.state :=
    h.step_slope d.state
  have hMarker :
      ∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 : Nat, n1 = n1 := by
    intro n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29
    rfl
  exact And.intro (And.intro hStepE hStepS) hMarker

theorem factorization_step_analysis_semiconvex_gradient_flow
    {X : Type u} [h : FrameworkStruct_analysis_semiconvex_gradient_flow X]
    (d : ContextData_analysis_semiconvex_gradient_flow X) :
    (h.distance d.state ≤ h.energy d.state + h.slope d.state ∧
    primary_map_analysis_semiconvex_gradient_flow d ≤
      primary_map_analysis_semiconvex_gradient_flow d +
        tertiary_map_analysis_semiconvex_gradient_flow d) ∧
    (∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 : Nat, n1 = n1) := by
  have hDist : h.distance d.state ≤ h.energy d.state + h.slope d.state :=
    h.distance_bridge d.state
  have hGrow :
      primary_map_analysis_semiconvex_gradient_flow d ≤
        primary_map_analysis_semiconvex_gradient_flow d +
          tertiary_map_analysis_semiconvex_gradient_flow d :=
    h.le_add_right_nat
      (primary_map_analysis_semiconvex_gradient_flow d)
      (tertiary_map_analysis_semiconvex_gradient_flow d)
  have hMarker :
      ∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 : Nat, n1 = n1 := by
    intro n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30
    rfl
  exact And.intro (And.intro hDist hGrow) hMarker

theorem comparison_step_analysis_semiconvex_gradient_flow
    {X : Type u} [h : FrameworkStruct_analysis_semiconvex_gradient_flow X]
    (d : ContextData_analysis_semiconvex_gradient_flow X) :
    (h.slope (h.iterateMap d.state) ≤ h.slope d.state + h.distance d.state ∨
    h.velocity (h.proxMap d.state) ≤ h.velocity d.state + h.distance d.state) ∧
    (∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 : Nat, n1 = n1) := by
  have hSlope : h.slope (h.iterateMap d.state) ≤ h.slope d.state + h.distance d.state :=
    h.iterate_slope d.state
  have hMarker :
      ∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 : Nat, n1 = n1 := by
    intro n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31
    rfl
  exact And.intro (Or.inl hSlope) hMarker

theorem transport_step_analysis_semiconvex_gradient_flow
    {X : Type u} [h : FrameworkStruct_analysis_semiconvex_gradient_flow X]
    (d : ContextData_analysis_semiconvex_gradient_flow X) :
    (∃ q : Nat,
      q = tertiary_map_analysis_semiconvex_gradient_flow d ∧
      h.distance (h.iterateMap d.state) ≤ h.distance d.state + h.velocity d.state + q) ∧
    (∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 : Nat, n1 = n1) := by
  refine And.intro ?_ ?_
  · refine ⟨tertiary_map_analysis_semiconvex_gradient_flow d, rfl, ?_⟩
    have hBase :
        h.distance (h.iterateMap d.state) ≤ h.distance d.state + h.velocity d.state :=
      h.iterate_distance d.state
    have hGrow :
        h.distance d.state + h.velocity d.state ≤
          (h.distance d.state + h.velocity d.state) +
            tertiary_map_analysis_semiconvex_gradient_flow d :=
      h.le_add_right_nat
        (h.distance d.state + h.velocity d.state)
        (tertiary_map_analysis_semiconvex_gradient_flow d)
    exact h.le_trans_nat _ _ _ hBase hGrow
  · intro n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32
    rfl

theorem coherence_step_analysis_semiconvex_gradient_flow
    {X : Type u} [h : FrameworkStruct_analysis_semiconvex_gradient_flow X]
    (d : ContextData_analysis_semiconvex_gradient_flow X) :
    (secondary_map_analysis_semiconvex_gradient_flow d ≤
      secondary_map_analysis_semiconvex_gradient_flow d +
        tertiary_map_analysis_semiconvex_gradient_flow d) ∧
    (∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 : Nat, n1 = n1) := by
  have hAdd :
      secondary_map_analysis_semiconvex_gradient_flow d ≤
        secondary_map_analysis_semiconvex_gradient_flow d +
          tertiary_map_analysis_semiconvex_gradient_flow d :=
    h.le_add_right_nat
      (secondary_map_analysis_semiconvex_gradient_flow d)
      (tertiary_map_analysis_semiconvex_gradient_flow d)
  have hMarker :
      ∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 : Nat, n1 = n1 := by
    intro n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33
    rfl
  exact And.intro hAdd hMarker

theorem iteration_step_analysis_semiconvex_gradient_flow
    {X : Type u} [h : FrameworkStruct_analysis_semiconvex_gradient_flow X]
    (d : ContextData_analysis_semiconvex_gradient_flow X) :
    (∀ z : X, h.distance z ≤ h.energy z + h.slope z +
      tertiary_map_analysis_semiconvex_gradient_flow d) ∧
    (∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 : Nat, n1 = n1) := by
  have hCore :
      ∀ z : X, h.distance z ≤ h.energy z + h.slope z +
        tertiary_map_analysis_semiconvex_gradient_flow d := by
    intro z
    have hBase : h.distance z ≤ h.energy z + h.slope z := h.distance_bridge z
    have hGrow :
        h.energy z + h.slope z ≤
          (h.energy z + h.slope z) +
            tertiary_map_analysis_semiconvex_gradient_flow d :=
      h.le_add_right_nat
        (h.energy z + h.slope z)
        (tertiary_map_analysis_semiconvex_gradient_flow d)
    exact h.le_trans_nat _ _ _ hBase hGrow
  have hMarker :
      ∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 : Nat, n1 = n1 := by
    intro n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34
    rfl
  exact And.intro hCore hMarker

theorem main_result_analysis_semiconvex_gradient_flow
    {X : Type u} [h : FrameworkStruct_analysis_semiconvex_gradient_flow X]
    (d : ContextData_analysis_semiconvex_gradient_flow X) :
    (h.distance (h.iterateMap d.state) ≤ h.distance d.state + h.velocity d.state ∧
    h.energy (h.stepMap d.state) ≤ h.energy d.state + h.slope d.state) ∧
    (∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 : Nat, n1 = n1) := by
  have hIter :
      h.distance (h.iterateMap d.state) ≤ h.distance d.state + h.velocity d.state :=
    h.iterate_distance d.state
  have hStep :
      h.energy (h.stepMap d.state) ≤ h.energy d.state + h.slope d.state :=
    h.step_energy d.state
  have hMarker :
      ∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 : Nat, n1 = n1 := by
    intro n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35
    rfl
  exact And.intro (And.intro hIter hStep) hMarker
