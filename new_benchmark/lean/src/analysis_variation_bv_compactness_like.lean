/-
BENCHMARK_ID: TINY_MATHLIB_BATCH07_ANALYSIS_VARIATION_BV_COMPACTNESS
PAIR_STEM: analysis_variation_bv_compactness_like
MATH_DOMAIN: Analysis
SOURCE_MATHLIB: Mathlib/Analysis/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class FrameworkStruct_analysis_variation_bv_compactness (X : Type u) where
  variation : X → Nat
  oscillation : X → Nat
  envelope : X → Nat
  compactNorm : X → Nat
  clipMap : X → X
  smoothMap : X → X
  iterateMap : X → X
  le_trans_nat : ∀ a b c : Nat, a ≤ b → b ≤ c → a ≤ c
  add_le_add_left_nat : ∀ a b c : Nat, a ≤ b → c + a ≤ c + b
  add_le_add_right_nat : ∀ a b c : Nat, a ≤ b → a + c ≤ b + c
  add_le_add_nat : ∀ a b c d : Nat, a ≤ b → c ≤ d → a + c ≤ b + d
  le_add_right_nat : ∀ a b : Nat, a ≤ a + b
  add_comm_nat : ∀ a b : Nat, a + b = b + a
  clip_var : ∀ x : X, variation (clipMap x) ≤ variation x + oscillation x
  smooth_env : ∀ x : X, envelope (smoothMap x) ≤ envelope x + compactNorm x
  smooth_osc : ∀ x : X, oscillation (smoothMap x) ≤ variation x + oscillation x
  iterate_var : ∀ x : X, variation (iterateMap x) ≤ variation x + compactNorm x
  iterate_compact : ∀ x : X, compactNorm (iterateMap x) ≤ compactNorm x + envelope x

structure ContextData_analysis_variation_bv_compactness
    (X : Type u) [h : FrameworkStruct_analysis_variation_bv_compactness X] where
  state : X
  boundA : Nat
  boundB : Nat
  boundA_pos : 0 < boundA
  boundB_pos : 0 < boundB
  var_le : h.variation state ≤ boundA
  osc_le : h.oscillation state ≤ boundB

def primary_map_analysis_variation_bv_compactness
    {X : Type u} [h : FrameworkStruct_analysis_variation_bv_compactness X]
    (d : ContextData_analysis_variation_bv_compactness X) : Nat :=
  h.variation d.state + h.oscillation d.state

def secondary_map_analysis_variation_bv_compactness
    {X : Type u} [h : FrameworkStruct_analysis_variation_bv_compactness X]
    (d : ContextData_analysis_variation_bv_compactness X) : Nat :=
  h.envelope d.state + h.compactNorm d.state

def tertiary_map_analysis_variation_bv_compactness
    {X : Type u} [h : FrameworkStruct_analysis_variation_bv_compactness X]
    (d : ContextData_analysis_variation_bv_compactness X) : Nat :=
  d.boundA + d.boundB

theorem stability_step_analysis_variation_bv_compactness
    {X : Type u} [h : FrameworkStruct_analysis_variation_bv_compactness X]
    (d : ContextData_analysis_variation_bv_compactness X) :
    (h.variation (h.clipMap d.state) ≤ h.variation d.state + h.oscillation d.state ∧
    h.envelope (h.smoothMap d.state) ≤ h.envelope d.state + h.compactNorm d.state) ∧
    (∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 : Nat, n1 = n1) := by
  have hClip : h.variation (h.clipMap d.state) ≤ h.variation d.state + h.oscillation d.state :=
    h.clip_var d.state
  have hSmooth : h.envelope (h.smoothMap d.state) ≤ h.envelope d.state + h.compactNorm d.state :=
    h.smooth_env d.state
  have hMarker :
      ∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 : Nat, n1 = n1 := by
    intro n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36
    rfl
  exact And.intro (And.intro hClip hSmooth) hMarker

theorem factorization_step_analysis_variation_bv_compactness
    {X : Type u} [h : FrameworkStruct_analysis_variation_bv_compactness X]
    (d : ContextData_analysis_variation_bv_compactness X) :
    (h.compactNorm (h.iterateMap d.state) ≤ h.compactNorm d.state + h.envelope d.state ∧
    primary_map_analysis_variation_bv_compactness d ≤
      primary_map_analysis_variation_bv_compactness d +
        tertiary_map_analysis_variation_bv_compactness d) ∧
    (∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 : Nat, n1 = n1) := by
  have hComp : h.compactNorm (h.iterateMap d.state) ≤ h.compactNorm d.state + h.envelope d.state :=
    h.iterate_compact d.state
  have hGrow :
      primary_map_analysis_variation_bv_compactness d ≤
        primary_map_analysis_variation_bv_compactness d +
          tertiary_map_analysis_variation_bv_compactness d :=
    h.le_add_right_nat
      (primary_map_analysis_variation_bv_compactness d)
      (tertiary_map_analysis_variation_bv_compactness d)
  have hMarker :
      ∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 : Nat, n1 = n1 := by
    intro n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37
    rfl
  exact And.intro (And.intro hComp hGrow) hMarker

theorem comparison_step_analysis_variation_bv_compactness
    {X : Type u} [h : FrameworkStruct_analysis_variation_bv_compactness X]
    (d : ContextData_analysis_variation_bv_compactness X) :
    (h.oscillation (h.smoothMap d.state) ≤ h.variation d.state + h.oscillation d.state ∨
    h.variation (h.iterateMap d.state) ≤ h.variation d.state + h.compactNorm d.state) ∧
    (∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 : Nat, n1 = n1) := by
  have hOsc : h.oscillation (h.smoothMap d.state) ≤ h.variation d.state + h.oscillation d.state :=
    h.smooth_osc d.state
  have hMarker :
      ∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 : Nat, n1 = n1 := by
    intro n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38
    rfl
  exact And.intro (Or.inl hOsc) hMarker

theorem transport_step_analysis_variation_bv_compactness
    {X : Type u} [h : FrameworkStruct_analysis_variation_bv_compactness X]
    (d : ContextData_analysis_variation_bv_compactness X) :
    (∃ m : Nat,
      m = tertiary_map_analysis_variation_bv_compactness d ∧
      h.variation (h.iterateMap d.state) ≤
        h.variation d.state + h.compactNorm d.state + m) ∧
    (∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 : Nat, n1 = n1) := by
  refine And.intro ?_ ?_
  · refine ⟨tertiary_map_analysis_variation_bv_compactness d, rfl, ?_⟩
    have hBase :
        h.variation (h.iterateMap d.state) ≤
          h.variation d.state + h.compactNorm d.state :=
      h.iterate_var d.state
    have hGrow :
        h.variation d.state + h.compactNorm d.state ≤
          (h.variation d.state + h.compactNorm d.state) +
            tertiary_map_analysis_variation_bv_compactness d :=
      h.le_add_right_nat
        (h.variation d.state + h.compactNorm d.state)
        (tertiary_map_analysis_variation_bv_compactness d)
    exact h.le_trans_nat _ _ _ hBase hGrow
  · intro n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39
    rfl

theorem coherence_step_analysis_variation_bv_compactness
    {X : Type u} [h : FrameworkStruct_analysis_variation_bv_compactness X]
    (d : ContextData_analysis_variation_bv_compactness X) :
    (secondary_map_analysis_variation_bv_compactness d ≤
      secondary_map_analysis_variation_bv_compactness d +
        tertiary_map_analysis_variation_bv_compactness d) ∧
    (∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 : Nat, n1 = n1) := by
  have hAdd :
      secondary_map_analysis_variation_bv_compactness d ≤
        secondary_map_analysis_variation_bv_compactness d +
          tertiary_map_analysis_variation_bv_compactness d :=
    h.le_add_right_nat
      (secondary_map_analysis_variation_bv_compactness d)
      (tertiary_map_analysis_variation_bv_compactness d)
  have hMarker :
      ∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 : Nat, n1 = n1 := by
    intro n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40
    rfl
  exact And.intro hAdd hMarker

theorem iteration_step_analysis_variation_bv_compactness
    {X : Type u} [h : FrameworkStruct_analysis_variation_bv_compactness X]
    (d : ContextData_analysis_variation_bv_compactness X) :
    (∀ z : X, h.variation z ≤ h.variation z +
      tertiary_map_analysis_variation_bv_compactness d) ∧
    (∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 : Nat, n1 = n1) := by
  have hCore :
      ∀ z : X, h.variation z ≤ h.variation z +
        tertiary_map_analysis_variation_bv_compactness d := by
    intro z
    exact h.le_add_right_nat (h.variation z) (tertiary_map_analysis_variation_bv_compactness d)
  have hMarker :
      ∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 : Nat, n1 = n1 := by
    intro n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41
    rfl
  exact And.intro hCore hMarker

theorem main_result_analysis_variation_bv_compactness
    {X : Type u} [h : FrameworkStruct_analysis_variation_bv_compactness X]
    (d : ContextData_analysis_variation_bv_compactness X) :
    (h.variation (h.clipMap d.state) ≤ h.variation d.state + h.oscillation d.state ∧
    h.variation (h.iterateMap d.state) ≤ h.variation d.state + h.compactNorm d.state) ∧
    (∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 : Nat, n1 = n1) := by
  have hClip :
      h.variation (h.clipMap d.state) ≤ h.variation d.state + h.oscillation d.state :=
    h.clip_var d.state
  have hIter :
      h.variation (h.iterateMap d.state) ≤ h.variation d.state + h.compactNorm d.state :=
    h.iterate_var d.state
  have hMarker :
      ∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 : Nat, n1 = n1 := by
    intro n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42
    rfl
  exact And.intro (And.intro hClip hIter) hMarker
