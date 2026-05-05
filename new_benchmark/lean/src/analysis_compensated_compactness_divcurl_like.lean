/-
BENCHMARK_ID: TINY_MATHLIB_BATCH07_ANALYSIS_COMPENSATED_COMPACTNESS_DIVCURL
PAIR_STEM: analysis_compensated_compactness_divcurl_like
MATH_DOMAIN: Analysis
SOURCE_MATHLIB: Mathlib/Analysis/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class FrameworkStruct_analysis_compensated_compactness_divcurl (X : Type u) where
  divNorm : X → Nat
  curlNorm : X → Nat
  fluxNorm : X → Nat
  defectNorm : X → Nat
  projMap : X → X
  corrMap : X → X
  pairMap : X → X
  le_trans_nat : ∀ a b c : Nat, a ≤ b → b ≤ c → a ≤ c
  add_le_add_left_nat : ∀ a b c : Nat, a ≤ b → c + a ≤ c + b
  add_le_add_right_nat : ∀ a b c : Nat, a ≤ b → a + c ≤ b + c
  add_le_add_nat : ∀ a b c d : Nat, a ≤ b → c ≤ d → a + c ≤ b + d
  le_add_right_nat : ∀ a b : Nat, a ≤ a + b
  add_comm_nat : ∀ a b : Nat, a + b = b + a
  proj_div : ∀ x : X, divNorm (projMap x) ≤ divNorm x + curlNorm x
  corr_curl : ∀ x : X, curlNorm (corrMap x) ≤ curlNorm x + divNorm x
  pair_flux : ∀ x : X, fluxNorm (pairMap x) ≤ fluxNorm x + defectNorm x
  pair_defect : ∀ x : X, defectNorm (pairMap x) ≤ defectNorm x + fluxNorm x
  flux_bound : ∀ x : X, fluxNorm x ≤ divNorm x + curlNorm x

structure ContextData_analysis_compensated_compactness_divcurl
    (X : Type u) [h : FrameworkStruct_analysis_compensated_compactness_divcurl X] where
  state : X
  divBound : Nat
  curlBound : Nat
  div_pos : 0 < divBound
  curl_pos : 0 < curlBound
  div_le : h.divNorm state ≤ divBound
  curl_le : h.curlNorm state ≤ curlBound

def primary_map_analysis_compensated_compactness_divcurl
    {X : Type u} [h : FrameworkStruct_analysis_compensated_compactness_divcurl X]
    (d : ContextData_analysis_compensated_compactness_divcurl X) : Nat :=
  h.divNorm d.state + h.curlNorm d.state

def secondary_map_analysis_compensated_compactness_divcurl
    {X : Type u} [h : FrameworkStruct_analysis_compensated_compactness_divcurl X]
    (d : ContextData_analysis_compensated_compactness_divcurl X) : Nat :=
  h.fluxNorm d.state + h.defectNorm d.state

def tertiary_map_analysis_compensated_compactness_divcurl
    {X : Type u} [h : FrameworkStruct_analysis_compensated_compactness_divcurl X]
    (d : ContextData_analysis_compensated_compactness_divcurl X) : Nat :=
  d.divBound + d.curlBound

theorem stability_step_analysis_compensated_compactness_divcurl
    {X : Type u} [h : FrameworkStruct_analysis_compensated_compactness_divcurl X]
    (d : ContextData_analysis_compensated_compactness_divcurl X) :
    (h.divNorm (h.projMap d.state) ≤ h.divNorm d.state + h.curlNorm d.state ∧
    h.curlNorm (h.corrMap d.state) ≤ h.curlNorm d.state + h.divNorm d.state) ∧
    (∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 : Nat, n1 = n1) := by
  have hProj : h.divNorm (h.projMap d.state) ≤ h.divNorm d.state + h.curlNorm d.state :=
    h.proj_div d.state
  have hCorr : h.curlNorm (h.corrMap d.state) ≤ h.curlNorm d.state + h.divNorm d.state :=
    h.corr_curl d.state
  have hMarker :
      ∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 : Nat, n1 = n1 := by
    intro n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43
    rfl
  exact And.intro (And.intro hProj hCorr) hMarker

theorem factorization_step_analysis_compensated_compactness_divcurl
    {X : Type u} [h : FrameworkStruct_analysis_compensated_compactness_divcurl X]
    (d : ContextData_analysis_compensated_compactness_divcurl X) :
    (h.fluxNorm d.state ≤ h.divNorm d.state + h.curlNorm d.state ∧
    primary_map_analysis_compensated_compactness_divcurl d ≤
      primary_map_analysis_compensated_compactness_divcurl d +
        tertiary_map_analysis_compensated_compactness_divcurl d) ∧
    (∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 : Nat, n1 = n1) := by
  have hFlux : h.fluxNorm d.state ≤ h.divNorm d.state + h.curlNorm d.state :=
    h.flux_bound d.state
  have hGrow :
      primary_map_analysis_compensated_compactness_divcurl d ≤
        primary_map_analysis_compensated_compactness_divcurl d +
          tertiary_map_analysis_compensated_compactness_divcurl d :=
    h.le_add_right_nat
      (primary_map_analysis_compensated_compactness_divcurl d)
      (tertiary_map_analysis_compensated_compactness_divcurl d)
  have hMarker :
      ∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 : Nat, n1 = n1 := by
    intro n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44
    rfl
  exact And.intro (And.intro hFlux hGrow) hMarker

theorem comparison_step_analysis_compensated_compactness_divcurl
    {X : Type u} [h : FrameworkStruct_analysis_compensated_compactness_divcurl X]
    (d : ContextData_analysis_compensated_compactness_divcurl X) :
    (h.fluxNorm (h.pairMap d.state) ≤ h.fluxNorm d.state + h.defectNorm d.state ∨
    h.defectNorm (h.pairMap d.state) ≤ h.defectNorm d.state + h.fluxNorm d.state) ∧
    (∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 : Nat, n1 = n1) := by
  have hPair : h.fluxNorm (h.pairMap d.state) ≤ h.fluxNorm d.state + h.defectNorm d.state :=
    h.pair_flux d.state
  have hMarker :
      ∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 : Nat, n1 = n1 := by
    intro n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45
    rfl
  exact And.intro (Or.inl hPair) hMarker

theorem transport_step_analysis_compensated_compactness_divcurl
    {X : Type u} [h : FrameworkStruct_analysis_compensated_compactness_divcurl X]
    (d : ContextData_analysis_compensated_compactness_divcurl X) :
    (∃ r : Nat,
      r = tertiary_map_analysis_compensated_compactness_divcurl d ∧
      h.defectNorm (h.pairMap d.state) ≤ h.defectNorm d.state + h.fluxNorm d.state + r) ∧
    (∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 : Nat, n1 = n1) := by
  refine And.intro ?_ ?_
  · refine ⟨tertiary_map_analysis_compensated_compactness_divcurl d, rfl, ?_⟩
    have hBase :
        h.defectNorm (h.pairMap d.state) ≤ h.defectNorm d.state + h.fluxNorm d.state :=
      h.pair_defect d.state
    have hGrow :
        h.defectNorm d.state + h.fluxNorm d.state ≤
          (h.defectNorm d.state + h.fluxNorm d.state) +
            tertiary_map_analysis_compensated_compactness_divcurl d :=
      h.le_add_right_nat
        (h.defectNorm d.state + h.fluxNorm d.state)
        (tertiary_map_analysis_compensated_compactness_divcurl d)
    exact h.le_trans_nat _ _ _ hBase hGrow
  · intro n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46
    rfl

theorem coherence_step_analysis_compensated_compactness_divcurl
    {X : Type u} [h : FrameworkStruct_analysis_compensated_compactness_divcurl X]
    (d : ContextData_analysis_compensated_compactness_divcurl X) :
    (secondary_map_analysis_compensated_compactness_divcurl d ≤
      secondary_map_analysis_compensated_compactness_divcurl d +
        tertiary_map_analysis_compensated_compactness_divcurl d) ∧
    (∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 : Nat, n1 = n1) := by
  have hAdd :
      secondary_map_analysis_compensated_compactness_divcurl d ≤
        secondary_map_analysis_compensated_compactness_divcurl d +
          tertiary_map_analysis_compensated_compactness_divcurl d :=
    h.le_add_right_nat
      (secondary_map_analysis_compensated_compactness_divcurl d)
      (tertiary_map_analysis_compensated_compactness_divcurl d)
  have hMarker :
      ∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 : Nat, n1 = n1 := by
    intro n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47
    rfl
  exact And.intro hAdd hMarker

theorem iteration_step_analysis_compensated_compactness_divcurl
    {X : Type u} [h : FrameworkStruct_analysis_compensated_compactness_divcurl X]
    (d : ContextData_analysis_compensated_compactness_divcurl X) :
    (∀ z : X, h.fluxNorm z ≤ h.divNorm z + h.curlNorm z +
      tertiary_map_analysis_compensated_compactness_divcurl d) ∧
    (∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 : Nat, n1 = n1) := by
  have hCore :
      ∀ z : X, h.fluxNorm z ≤ h.divNorm z + h.curlNorm z +
        tertiary_map_analysis_compensated_compactness_divcurl d := by
    intro z
    have hBase : h.fluxNorm z ≤ h.divNorm z + h.curlNorm z := h.flux_bound z
    have hGrow :
        h.divNorm z + h.curlNorm z ≤
          (h.divNorm z + h.curlNorm z) +
            tertiary_map_analysis_compensated_compactness_divcurl d :=
      h.le_add_right_nat
        (h.divNorm z + h.curlNorm z)
        (tertiary_map_analysis_compensated_compactness_divcurl d)
    exact h.le_trans_nat _ _ _ hBase hGrow
  have hMarker :
      ∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 : Nat, n1 = n1 := by
    intro n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48
    rfl
  exact And.intro hCore hMarker

theorem main_result_analysis_compensated_compactness_divcurl
    {X : Type u} [h : FrameworkStruct_analysis_compensated_compactness_divcurl X]
    (d : ContextData_analysis_compensated_compactness_divcurl X) :
    (h.fluxNorm (h.pairMap d.state) ≤ h.fluxNorm d.state + h.defectNorm d.state ∧
    h.defectNorm (h.pairMap d.state) ≤ h.defectNorm d.state + h.fluxNorm d.state) ∧
    (∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 : Nat, n1 = n1) := by
  have hFlux :
      h.fluxNorm (h.pairMap d.state) ≤ h.fluxNorm d.state + h.defectNorm d.state :=
    h.pair_flux d.state
  have hDef :
      h.defectNorm (h.pairMap d.state) ≤ h.defectNorm d.state + h.fluxNorm d.state :=
    h.pair_defect d.state
  have hMarker :
      ∀ n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 : Nat, n1 = n1 := by
    intro n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49
    rfl
  exact And.intro (And.intro hFlux hDef) hMarker
