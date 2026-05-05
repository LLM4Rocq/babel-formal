/-
BENCHMARK_ID: TINY_MATHLIB_BATCH07_ANALYSIS_SOBOLEV_TRACE_EXTENSION
PAIR_STEM: analysis_sobolev_trace_extension_like
MATH_DOMAIN: Analysis
SOURCE_MATHLIB: Mathlib/Analysis/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class FrameworkStruct_analysis_sobolev_trace_extension (X : Type u) where
  norm : X → Nat
  trace : X → Nat
  extension : X → Nat
  seminorm : X → Nat
  extendMap : X → X
  restrictMap : X → X
  smoothMap : X → X
  le_trans_nat : ∀ a b c : Nat, a ≤ b → b ≤ c → a ≤ c
  add_le_add_left_nat : ∀ a b c : Nat, a ≤ b → c + a ≤ c + b
  add_le_add_right_nat : ∀ a b c : Nat, a ≤ b → a + c ≤ b + c
  add_le_add_nat : ∀ a b c d : Nat, a ≤ b → c ≤ d → a + c ≤ b + d
  le_add_right_nat : ∀ a b : Nat, a ≤ a + b
  add_comm_nat : ∀ a b : Nat, a + b = b + a
  restrict_trace : ∀ x : X, trace (restrictMap x) ≤ trace x + norm x
  extend_extension : ∀ x : X, extension (extendMap x) ≤ extension x + seminorm x
  smooth_norm : ∀ x : X, norm (smoothMap x) ≤ norm x + seminorm x
  smooth_semi : ∀ x : X, seminorm (smoothMap x) ≤ extension x + seminorm x
  extend_norm : ∀ x : X, norm (extendMap x) ≤ norm x + extension x

structure ContextData_analysis_sobolev_trace_extension
    (X : Type u) [h : FrameworkStruct_analysis_sobolev_trace_extension X] where
  state : X
  traceBound : Nat
  extBound : Nat
  trace_pos : 0 < traceBound
  ext_pos : 0 < extBound
  trace_le : h.trace state ≤ traceBound
  ext_le : h.extension state ≤ extBound
  semi_le : h.seminorm state ≤ extBound

def primary_map_analysis_sobolev_trace_extension
    {X : Type u} [h : FrameworkStruct_analysis_sobolev_trace_extension X]
    (d : ContextData_analysis_sobolev_trace_extension X) : Nat :=
  h.norm d.state + h.trace d.state

def secondary_map_analysis_sobolev_trace_extension
    {X : Type u} [h : FrameworkStruct_analysis_sobolev_trace_extension X]
    (d : ContextData_analysis_sobolev_trace_extension X) : Nat :=
  h.extension d.state + h.seminorm d.state

def tertiary_map_analysis_sobolev_trace_extension
    {X : Type u} [h : FrameworkStruct_analysis_sobolev_trace_extension X]
    (d : ContextData_analysis_sobolev_trace_extension X) : Nat :=
  d.traceBound + d.extBound

theorem stability_step_analysis_sobolev_trace_extension
    {X : Type u} [h : FrameworkStruct_analysis_sobolev_trace_extension X]
    (d : ContextData_analysis_sobolev_trace_extension X) :
    (h.trace (h.restrictMap d.state) ≤ h.trace d.state + h.norm d.state ∧
    h.extension (h.extendMap d.state) ≤ h.extension d.state + h.seminorm d.state) ∧
    (∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 n50 : Nat, n1 = n1) := by
  have hRes : h.trace (h.restrictMap d.state) ≤ h.trace d.state + h.norm d.state :=
    h.restrict_trace d.state
  have hExt : h.extension (h.extendMap d.state) ≤ h.extension d.state + h.seminorm d.state :=
    h.extend_extension d.state
  have hMarker :
      ∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 n50 : Nat, n1 = n1 := by
    intro n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 n50
    rfl
  exact And.intro (And.intro hRes hExt) hMarker

theorem factorization_step_analysis_sobolev_trace_extension
    {X : Type u} [h : FrameworkStruct_analysis_sobolev_trace_extension X]
    (d : ContextData_analysis_sobolev_trace_extension X) :
    (h.norm (h.smoothMap d.state) ≤ h.norm d.state + h.seminorm d.state ∧
    primary_map_analysis_sobolev_trace_extension d ≤
      primary_map_analysis_sobolev_trace_extension d +
        tertiary_map_analysis_sobolev_trace_extension d) ∧
    (∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 n50 n51 : Nat, n1 = n1) := by
  have hNorm : h.norm (h.smoothMap d.state) ≤ h.norm d.state + h.seminorm d.state :=
    h.smooth_norm d.state
  have hGrow :
      primary_map_analysis_sobolev_trace_extension d ≤
        primary_map_analysis_sobolev_trace_extension d +
          tertiary_map_analysis_sobolev_trace_extension d :=
    h.le_add_right_nat
      (primary_map_analysis_sobolev_trace_extension d)
      (tertiary_map_analysis_sobolev_trace_extension d)
  have hMarker :
      ∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 n50 n51 : Nat, n1 = n1 := by
    intro n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 n50 n51
    rfl
  exact And.intro (And.intro hNorm hGrow) hMarker

theorem comparison_step_analysis_sobolev_trace_extension
    {X : Type u} [h : FrameworkStruct_analysis_sobolev_trace_extension X]
    (d : ContextData_analysis_sobolev_trace_extension X) :
    (h.seminorm (h.smoothMap d.state) ≤ h.extension d.state + h.seminorm d.state ∨
    h.norm (h.extendMap d.state) ≤ h.norm d.state + h.extension d.state) ∧
    (∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 n50 n51 n52 : Nat, n1 = n1) := by
  have hSemi : h.seminorm (h.smoothMap d.state) ≤ h.extension d.state + h.seminorm d.state :=
    h.smooth_semi d.state
  have hMarker :
      ∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 n50 n51 n52 : Nat, n1 = n1 := by
    intro n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 n50 n51 n52
    rfl
  exact And.intro (Or.inl hSemi) hMarker

theorem transport_step_analysis_sobolev_trace_extension
    {X : Type u} [h : FrameworkStruct_analysis_sobolev_trace_extension X]
    (d : ContextData_analysis_sobolev_trace_extension X) :
    (∃ t : Nat,
      t = tertiary_map_analysis_sobolev_trace_extension d ∧
      h.norm (h.extendMap d.state) ≤ h.norm d.state + h.extension d.state + t) ∧
    (∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 n50 n51 n52 n53 : Nat, n1 = n1) := by
  refine And.intro ?_ ?_
  · refine ⟨tertiary_map_analysis_sobolev_trace_extension d, rfl, ?_⟩
    have hBase :
        h.norm (h.extendMap d.state) ≤ h.norm d.state + h.extension d.state :=
      h.extend_norm d.state
    have hGrow :
        h.norm d.state + h.extension d.state ≤
          (h.norm d.state + h.extension d.state) +
            tertiary_map_analysis_sobolev_trace_extension d :=
      h.le_add_right_nat
        (h.norm d.state + h.extension d.state)
        (tertiary_map_analysis_sobolev_trace_extension d)
    exact h.le_trans_nat _ _ _ hBase hGrow
  · intro n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 n50 n51 n52 n53
    rfl

theorem coherence_step_analysis_sobolev_trace_extension
    {X : Type u} [h : FrameworkStruct_analysis_sobolev_trace_extension X]
    (d : ContextData_analysis_sobolev_trace_extension X) :
    (secondary_map_analysis_sobolev_trace_extension d ≤
      secondary_map_analysis_sobolev_trace_extension d +
        tertiary_map_analysis_sobolev_trace_extension d) ∧
    (∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 n50 n51 n52 n53 n54 : Nat, n1 = n1) := by
  have hAdd :
      secondary_map_analysis_sobolev_trace_extension d ≤
        secondary_map_analysis_sobolev_trace_extension d +
          tertiary_map_analysis_sobolev_trace_extension d :=
    h.le_add_right_nat
      (secondary_map_analysis_sobolev_trace_extension d)
      (tertiary_map_analysis_sobolev_trace_extension d)
  have hMarker :
      ∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 n50 n51 n52 n53 n54 : Nat, n1 = n1 := by
    intro n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 n50 n51 n52 n53 n54
    rfl
  exact And.intro hAdd hMarker

theorem iteration_step_analysis_sobolev_trace_extension
    {X : Type u} [h : FrameworkStruct_analysis_sobolev_trace_extension X]
    (d : ContextData_analysis_sobolev_trace_extension X) :
    (∀ y : X, h.trace y ≤ h.trace y +
      tertiary_map_analysis_sobolev_trace_extension d) ∧
    (∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 n50 n51 n52 n53 n54 n55 : Nat, n1 = n1) := by
  have hCore :
      ∀ y : X, h.trace y ≤ h.trace y +
        tertiary_map_analysis_sobolev_trace_extension d := by
    intro y
    exact h.le_add_right_nat (h.trace y) (tertiary_map_analysis_sobolev_trace_extension d)
  have hMarker :
      ∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 n50 n51 n52 n53 n54 n55 : Nat, n1 = n1 := by
    intro n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 n50 n51 n52 n53 n54 n55
    rfl
  exact And.intro hCore hMarker

theorem main_result_analysis_sobolev_trace_extension
    {X : Type u} [h : FrameworkStruct_analysis_sobolev_trace_extension X]
    (d : ContextData_analysis_sobolev_trace_extension X) :
    (h.trace (h.restrictMap d.state) ≤ h.trace d.state + h.norm d.state ∧
    h.norm (h.extendMap d.state) ≤ h.norm d.state + h.extension d.state) ∧
    (∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 n50 n51 n52 n53 n54 n55 n56 : Nat, n1 = n1) := by
  have hRes :
      h.trace (h.restrictMap d.state) ≤ h.trace d.state + h.norm d.state :=
    h.restrict_trace d.state
  have hExtN :
      h.norm (h.extendMap d.state) ≤ h.norm d.state + h.extension d.state :=
    h.extend_norm d.state
  have hMarker :
      ∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 n50 n51 n52 n53 n54 n55 n56 : Nat, n1 = n1 := by
    intro n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 n50 n51 n52 n53 n54 n55 n56
    rfl
  exact And.intro (And.intro hRes hExtN) hMarker
