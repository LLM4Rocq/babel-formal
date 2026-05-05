/-
BENCHMARK_ID: TINY_MATHLIB_BATCH07_TOPOLOGY_HOMOTOPY_COLIMIT_GLUING_ADVANCED
PAIR_STEM: topology_homotopy_colimit_gluing_advanced_like
MATH_DOMAIN: Topology
SOURCE_MATHLIB: Mathlib/Topology/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class FrameworkStruct_topology_homotopy_colimit_gluing_advanced (X : Type u) where
  pointWeight : X → Nat
  patchComplexity : X → Nat
  glueComplexity : X → Nat
  hocolimComplexity : X → Nat
  leftAttach : X → X
  rightAttach : X → X
  glueNode : X → X
  iterateNode : X → X
  le_trans_nat : ∀ a b c : Nat, a ≤ b → b ≤ c → a ≤ c
  add_le_add_left_nat : ∀ a b c : Nat, a ≤ b → c + a ≤ c + b
  add_le_add_right_nat : ∀ a b c : Nat, a ≤ b → a + c ≤ b + c
  add_le_add_nat : ∀ a b c d : Nat, a ≤ b → c ≤ d → a + c ≤ b + d
  le_add_right_nat : ∀ a b : Nat, a ≤ a + b
  add_comm_nat : ∀ a b : Nat, a + b = b + a
  left_attach_weight : ∀ x : X, pointWeight (leftAttach x) ≤ pointWeight x + patchComplexity x
  right_attach_patch : ∀ x : X, patchComplexity (rightAttach x) ≤ patchComplexity x + pointWeight x
  glue_node_bound : ∀ x : X, glueComplexity (glueNode x) ≤ glueComplexity x + patchComplexity x
  hocolim_of_glue : ∀ x : X, hocolimComplexity x ≤ glueComplexity x + pointWeight x
  iterate_glue_bound : ∀ x : X, glueComplexity (iterateNode x) ≤ glueComplexity x + hocolimComplexity x
  iterate_weight_bound : ∀ x : X, pointWeight (iterateNode x) ≤ pointWeight x + patchComplexity x

structure ContextData_topology_homotopy_colimit_gluing_advanced
    (X : Type u) [h : FrameworkStruct_topology_homotopy_colimit_gluing_advanced X] where
  state : X
  budget : Nat
  budget_pos : 0 < budget
  weight_le_budget : h.pointWeight state ≤ budget
  patch_le_budget : h.patchComplexity state ≤ budget

def primary_map_topology_homotopy_colimit_gluing_advanced
    {X : Type u} [h : FrameworkStruct_topology_homotopy_colimit_gluing_advanced X]
    (d : ContextData_topology_homotopy_colimit_gluing_advanced X) : Nat :=
  h.pointWeight d.state + h.patchComplexity d.state

def secondary_map_topology_homotopy_colimit_gluing_advanced
    {X : Type u} [h : FrameworkStruct_topology_homotopy_colimit_gluing_advanced X]
    (d : ContextData_topology_homotopy_colimit_gluing_advanced X) : Nat :=
  h.glueComplexity d.state + h.hocolimComplexity d.state

def tertiary_map_topology_homotopy_colimit_gluing_advanced
    {X : Type u} [h : FrameworkStruct_topology_homotopy_colimit_gluing_advanced X]
    (d : ContextData_topology_homotopy_colimit_gluing_advanced X) : Nat :=
  primary_map_topology_homotopy_colimit_gluing_advanced d + d.budget

theorem stability_step_topology_homotopy_colimit_gluing_advanced
    {X : Type u} [h : FrameworkStruct_topology_homotopy_colimit_gluing_advanced X]
    (d : ContextData_topology_homotopy_colimit_gluing_advanced X) :
    (h.pointWeight (h.leftAttach d.state) ≤
      h.pointWeight d.state + h.patchComplexity d.state ∧
    h.patchComplexity (h.rightAttach d.state) ≤
      h.patchComplexity d.state + h.pointWeight d.state) ∧
    (∀ P1 : Prop, P1 → P1) := by
  have hLeft :
      h.pointWeight (h.leftAttach d.state) ≤
        h.pointWeight d.state + h.patchComplexity d.state :=
    h.left_attach_weight d.state
  have hRight :
      h.patchComplexity (h.rightAttach d.state) ≤
        h.patchComplexity d.state + h.pointWeight d.state :=
    h.right_attach_patch d.state
  have hMarker : ∀ P1 : Prop, P1 → P1 := by
    intro P1 hP1
    exact hP1
  exact And.intro (And.intro hLeft hRight) hMarker

theorem factorization_step_topology_homotopy_colimit_gluing_advanced
    {X : Type u} [h : FrameworkStruct_topology_homotopy_colimit_gluing_advanced X]
    (d : ContextData_topology_homotopy_colimit_gluing_advanced X) :
    (primary_map_topology_homotopy_colimit_gluing_advanced d ≤
      primary_map_topology_homotopy_colimit_gluing_advanced d +
        tertiary_map_topology_homotopy_colimit_gluing_advanced d) ∧
    (∀ P1 P2 : Prop, P1 → P2 → P1) := by
  have hPrim :
      primary_map_topology_homotopy_colimit_gluing_advanced d ≤
        primary_map_topology_homotopy_colimit_gluing_advanced d +
          tertiary_map_topology_homotopy_colimit_gluing_advanced d :=
    h.le_add_right_nat
      (primary_map_topology_homotopy_colimit_gluing_advanced d)
      (tertiary_map_topology_homotopy_colimit_gluing_advanced d)
  have hMarker : ∀ P1 P2 : Prop, P1 → P2 → P1 := by
    intro P1 P2 hP1 hP2
    have _ : P2 := hP2
    exact hP1
  exact And.intro hPrim hMarker

theorem comparison_step_topology_homotopy_colimit_gluing_advanced
    {X : Type u} [h : FrameworkStruct_topology_homotopy_colimit_gluing_advanced X]
    (d : ContextData_topology_homotopy_colimit_gluing_advanced X) :
    (h.glueComplexity (h.iterateNode d.state) ≤
      h.glueComplexity d.state + h.hocolimComplexity d.state ∨
    h.pointWeight (h.iterateNode d.state) ≤
      h.pointWeight d.state + h.patchComplexity d.state) ∧
    (∀ P1 P2 P3 : Prop, P1 → P2 → P3 → P1) := by
  have hGlue :
      h.glueComplexity (h.iterateNode d.state) ≤
        h.glueComplexity d.state + h.hocolimComplexity d.state :=
    h.iterate_glue_bound d.state
  have hMarker : ∀ P1 P2 P3 : Prop, P1 → P2 → P3 → P1 := by
    intro P1 P2 P3 hP1 hP2 hP3
    have _ : P2 := hP2
    have _ : P3 := hP3
    exact hP1
  exact And.intro (Or.inl hGlue) hMarker

theorem transport_step_topology_homotopy_colimit_gluing_advanced
    {X : Type u} [h : FrameworkStruct_topology_homotopy_colimit_gluing_advanced X]
    (d : ContextData_topology_homotopy_colimit_gluing_advanced X)
    (hBudget : h.pointWeight d.state ≤ d.budget ∧ h.patchComplexity d.state ≤ d.budget) :
    (∃ n : Nat,
      n = tertiary_map_topology_homotopy_colimit_gluing_advanced d ∧
      h.hocolimComplexity d.state ≤ h.glueComplexity d.state + h.pointWeight d.state + n) ∧
    (∀ P1 P2 P3 P4 : Prop, P1 → P2 → P3 → P4 → P1) := by
  refine And.intro ?_ ?_
  · refine ⟨tertiary_map_topology_homotopy_colimit_gluing_advanced d, rfl, ?_⟩
    have hBase : h.hocolimComplexity d.state ≤ h.glueComplexity d.state + h.pointWeight d.state :=
      h.hocolim_of_glue d.state
    have hLift :
        h.glueComplexity d.state + h.pointWeight d.state ≤
          (h.glueComplexity d.state + h.pointWeight d.state) +
            tertiary_map_topology_homotopy_colimit_gluing_advanced d :=
      h.le_add_right_nat
        (h.glueComplexity d.state + h.pointWeight d.state)
        (tertiary_map_topology_homotopy_colimit_gluing_advanced d)
    exact h.le_trans_nat _ _ _ hBase hLift
  · intro P1 P2 P3 P4 hP1 hP2 hP3 hP4
    have _ : P2 := hP2
    have _ : P3 := hP3
    have _ : P4 := hP4
    have _ : h.pointWeight d.state ≤ d.budget := hBudget.left
    exact hP1

theorem coherence_step_topology_homotopy_colimit_gluing_advanced
    {X : Type u} [h : FrameworkStruct_topology_homotopy_colimit_gluing_advanced X]
    (d : ContextData_topology_homotopy_colimit_gluing_advanced X) :
    (h.hocolimComplexity d.state ≤
      h.hocolimComplexity d.state +
        tertiary_map_topology_homotopy_colimit_gluing_advanced d) ∧
    (∀ P1 P2 P3 P4 P5 : Prop, P1 → P2 → P3 → P4 → P5 → P1) := by
  have hFirst :
      h.hocolimComplexity d.state ≤
        h.hocolimComplexity d.state +
          tertiary_map_topology_homotopy_colimit_gluing_advanced d :=
    h.le_add_right_nat
      (h.hocolimComplexity d.state)
      (tertiary_map_topology_homotopy_colimit_gluing_advanced d)
  have hMarker : ∀ P1 P2 P3 P4 P5 : Prop, P1 → P2 → P3 → P4 → P5 → P1 := by
    intro P1 P2 P3 P4 P5 hP1 hP2 hP3 hP4 hP5
    have _ : P2 := hP2
    have _ : P3 := hP3
    have _ : P4 := hP4
    have _ : P5 := hP5
    exact hP1
  exact And.intro hFirst hMarker

theorem iteration_step_topology_homotopy_colimit_gluing_advanced
    {X : Type u} [h : FrameworkStruct_topology_homotopy_colimit_gluing_advanced X]
    (d : ContextData_topology_homotopy_colimit_gluing_advanced X) :
    (∀ z : X, h.glueComplexity z ≤
      h.glueComplexity z + tertiary_map_topology_homotopy_colimit_gluing_advanced d) ∧
    (∀ P1 P2 P3 P4 P5 P6 : Prop, P1 → P2 → P3 → P4 → P5 → P6 → P1) := by
  have hCore :
      ∀ z : X, h.glueComplexity z ≤
        h.glueComplexity z + tertiary_map_topology_homotopy_colimit_gluing_advanced d := by
    intro z
    exact h.le_add_right_nat
      (h.glueComplexity z)
      (tertiary_map_topology_homotopy_colimit_gluing_advanced d)
  have hMarker : ∀ P1 P2 P3 P4 P5 P6 : Prop, P1 → P2 → P3 → P4 → P5 → P6 → P1 := by
    intro P1 P2 P3 P4 P5 P6 hP1 hP2 hP3 hP4 hP5 hP6
    have _ : P2 := hP2
    have _ : P3 := hP3
    have _ : P4 := hP4
    have _ : P5 := hP5
    have _ : P6 := hP6
    exact hP1
  exact And.intro hCore hMarker

theorem main_result_topology_homotopy_colimit_gluing_advanced
    {X : Type u} [h : FrameworkStruct_topology_homotopy_colimit_gluing_advanced X]
    (d : ContextData_topology_homotopy_colimit_gluing_advanced X) :
    (h.hocolimComplexity (h.iterateNode d.state) ≤
      h.glueComplexity (h.iterateNode d.state) + h.pointWeight (h.iterateNode d.state) ∧
    h.pointWeight (h.leftAttach d.state) ≤
      h.pointWeight d.state + h.patchComplexity d.state) ∧
    (∀ P1 P2 P3 P4 P5 P6 P7 : Prop, P1 → P2 → P3 → P4 → P5 → P6 → P7 → P1) := by
  have hHocolim :
      h.hocolimComplexity (h.iterateNode d.state) ≤
        h.glueComplexity (h.iterateNode d.state) + h.pointWeight (h.iterateNode d.state) :=
    h.hocolim_of_glue (h.iterateNode d.state)
  have hLeft :
      h.pointWeight (h.leftAttach d.state) ≤
        h.pointWeight d.state + h.patchComplexity d.state :=
    h.left_attach_weight d.state
  have hMarker : ∀ P1 P2 P3 P4 P5 P6 P7 : Prop, P1 → P2 → P3 → P4 → P5 → P6 → P7 → P1 := by
    intro P1 P2 P3 P4 P5 P6 P7 hP1 hP2 hP3 hP4 hP5 hP6 hP7
    have _ : P2 := hP2
    have _ : P3 := hP3
    have _ : P4 := hP4
    have _ : P5 := hP5
    have _ : P6 := hP6
    have _ : P7 := hP7
    exact hP1
  exact And.intro (And.intro hHocolim hLeft) hMarker
