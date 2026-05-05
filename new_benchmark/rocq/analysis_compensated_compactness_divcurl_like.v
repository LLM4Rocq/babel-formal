(*
BENCHMARK_ID: TINY_MATHLIB_BATCH07_ANALYSIS_COMPENSATED_COMPACTNESS_DIVCURL
PAIR_STEM: analysis_compensated_compactness_divcurl_like
MATH_DOMAIN: Analysis
SOURCE_MATHLIB: Mathlib/Analysis/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class FrameworkStruct_analysis_compensated_compactness_divcurl (X : Type) := {
  divNorm : X -> nat;
  curlNorm : X -> nat;
  fluxNorm : X -> nat;
  defectNorm : X -> nat;
  projMap : X -> X;
  corrMap : X -> X;
  pairMap : X -> X;
  le_trans_nat : forall a b c : nat, a <= b -> b <= c -> a <= c;
  add_le_add_left_nat : forall a b c : nat, a <= b -> c + a <= c + b;
  add_le_add_right_nat : forall a b c : nat, a <= b -> a + c <= b + c;
  add_le_add_nat : forall a b c d : nat, a <= b -> c <= d -> a + c <= b + d;
  le_add_right_nat : forall a b : nat, a <= a + b;
  add_comm_nat : forall a b : nat, a + b = b + a;
  proj_div : forall x : X, divNorm (projMap x) <= divNorm x + curlNorm x;
  corr_curl : forall x : X, curlNorm (corrMap x) <= curlNorm x + divNorm x;
  pair_flux : forall x : X, fluxNorm (pairMap x) <= fluxNorm x + defectNorm x;
  pair_defect : forall x : X, defectNorm (pairMap x) <= defectNorm x + fluxNorm x;
  flux_bound : forall x : X, fluxNorm x <= divNorm x + curlNorm x
}.

Record ContextData_analysis_compensated_compactness_divcurl (X : Type)
    `{FrameworkStruct_analysis_compensated_compactness_divcurl X} := {
  state : X;
  divBound : nat;
  curlBound : nat;
  div_pos : 0 < divBound;
  curl_pos : 0 < curlBound;
  div_le : divNorm state <= divBound;
  curl_le : curlNorm state <= curlBound
}.

Definition primary_map_analysis_compensated_compactness_divcurl
    {X : Type} `{FrameworkStruct_analysis_compensated_compactness_divcurl X}
    (d : ContextData_analysis_compensated_compactness_divcurl) : nat :=
  divNorm (state d) + curlNorm (state d).

Definition secondary_map_analysis_compensated_compactness_divcurl
    {X : Type} `{FrameworkStruct_analysis_compensated_compactness_divcurl X}
    (d : ContextData_analysis_compensated_compactness_divcurl) : nat :=
  fluxNorm (state d) + defectNorm (state d).

Definition tertiary_map_analysis_compensated_compactness_divcurl
    {X : Type} `{FrameworkStruct_analysis_compensated_compactness_divcurl X}
    (d : ContextData_analysis_compensated_compactness_divcurl) : nat :=
  divBound d + curlBound d.

Lemma stability_step_analysis_compensated_compactness_divcurl :
  forall {X : Type} `{FrameworkStruct_analysis_compensated_compactness_divcurl X} (d : ContextData_analysis_compensated_compactness_divcurl),
    (divNorm (projMap (state d)) <= divNorm (state d) + curlNorm (state d) /\
    curlNorm (corrMap (state d)) <= curlNorm (state d) + divNorm (state d)) /\
    (forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 : nat, n1 = n1).
Proof.
  intros X H d.
  assert (hProj : divNorm (projMap (state d)) <= divNorm (state d) + curlNorm (state d)).
  { apply proj_div. }
  assert (hCorr : curlNorm (corrMap (state d)) <= curlNorm (state d) + divNorm (state d)).
  { apply corr_curl. }
  assert (hMarker :
    forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 : nat, n1 = n1).
  { intros n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43; reflexivity. }
  split.
  - split; exact hProj || exact hCorr.
  - exact hMarker.
Qed.

Lemma factorization_step_analysis_compensated_compactness_divcurl :
  forall {X : Type} `{FrameworkStruct_analysis_compensated_compactness_divcurl X} (d : ContextData_analysis_compensated_compactness_divcurl),
    (fluxNorm (state d) <= divNorm (state d) + curlNorm (state d) /\
    primary_map_analysis_compensated_compactness_divcurl d <=
      primary_map_analysis_compensated_compactness_divcurl d +
        tertiary_map_analysis_compensated_compactness_divcurl d) /\
    (forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 : nat, n1 = n1).
Proof.
  intros X H d.
  assert (hFlux : fluxNorm (state d) <= divNorm (state d) + curlNorm (state d)).
  { apply flux_bound. }
  assert (hGrow : primary_map_analysis_compensated_compactness_divcurl d <=
    primary_map_analysis_compensated_compactness_divcurl d +
      tertiary_map_analysis_compensated_compactness_divcurl d).
  { apply le_add_right_nat. }
  assert (hMarker :
    forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 : nat, n1 = n1).
  { intros n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44; reflexivity. }
  split.
  - split; exact hFlux || exact hGrow.
  - exact hMarker.
Qed.

Lemma comparison_step_analysis_compensated_compactness_divcurl :
  forall {X : Type} `{FrameworkStruct_analysis_compensated_compactness_divcurl X} (d : ContextData_analysis_compensated_compactness_divcurl),
    (fluxNorm (pairMap (state d)) <= fluxNorm (state d) + defectNorm (state d) \/
    defectNorm (pairMap (state d)) <= defectNorm (state d) + fluxNorm (state d)) /\
    (forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 : nat, n1 = n1).
Proof.
  intros X H d.
  assert (hPair : fluxNorm (pairMap (state d)) <= fluxNorm (state d) + defectNorm (state d)).
  { apply pair_flux. }
  assert (hMarker :
    forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 : nat, n1 = n1).
  { intros n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45; reflexivity. }
  split.
  - left; exact hPair.
  - exact hMarker.
Qed.

Lemma transport_step_analysis_compensated_compactness_divcurl :
  forall {X : Type} `{FrameworkStruct_analysis_compensated_compactness_divcurl X} (d : ContextData_analysis_compensated_compactness_divcurl),
    (exists r : nat,
      r = tertiary_map_analysis_compensated_compactness_divcurl d /\
      defectNorm (pairMap (state d)) <= defectNorm (state d) + fluxNorm (state d) + r) /\
    (forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 : nat, n1 = n1).
Proof.
  intros X H d.
  split.
  - exists (tertiary_map_analysis_compensated_compactness_divcurl d).
    split.
    + reflexivity.
    + assert (hBase : defectNorm (pairMap (state d)) <= defectNorm (state d) + fluxNorm (state d)).
      { apply pair_defect. }
      assert (hGrow : defectNorm (state d) + fluxNorm (state d) <=
        (defectNorm (state d) + fluxNorm (state d)) +
          tertiary_map_analysis_compensated_compactness_divcurl d).
      { apply le_add_right_nat. }
      eapply le_trans_nat.
      * exact hBase.
      * exact hGrow.
  - intros n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46.
    reflexivity.
Qed.

Lemma coherence_step_analysis_compensated_compactness_divcurl :
  forall {X : Type} `{FrameworkStruct_analysis_compensated_compactness_divcurl X} (d : ContextData_analysis_compensated_compactness_divcurl),
    (secondary_map_analysis_compensated_compactness_divcurl d <=
      secondary_map_analysis_compensated_compactness_divcurl d +
        tertiary_map_analysis_compensated_compactness_divcurl d) /\
    (forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 : nat, n1 = n1).
Proof.
  intros X H d.
  assert (hAdd : secondary_map_analysis_compensated_compactness_divcurl d <=
    secondary_map_analysis_compensated_compactness_divcurl d +
      tertiary_map_analysis_compensated_compactness_divcurl d).
  { apply le_add_right_nat. }
  assert (hMarker :
    forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 : nat, n1 = n1).
  { intros n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47; reflexivity. }
  split.
  - exact hAdd.
  - exact hMarker.
Qed.

Lemma iteration_step_analysis_compensated_compactness_divcurl :
  forall {X : Type} `{FrameworkStruct_analysis_compensated_compactness_divcurl X} (d : ContextData_analysis_compensated_compactness_divcurl),
    (forall z : X, fluxNorm z <= divNorm z + curlNorm z +
      tertiary_map_analysis_compensated_compactness_divcurl d) /\
    (forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 : nat, n1 = n1).
Proof.
  intros X H d.
  assert (hCore : forall z : X, fluxNorm z <= divNorm z + curlNorm z +
    tertiary_map_analysis_compensated_compactness_divcurl d).
  {
    intro z.
    assert (hBase : fluxNorm z <= divNorm z + curlNorm z).
    { apply flux_bound. }
    assert (hGrow : divNorm z + curlNorm z <=
      (divNorm z + curlNorm z) +
        tertiary_map_analysis_compensated_compactness_divcurl d).
    { apply le_add_right_nat. }
    eapply le_trans_nat.
    - exact hBase.
    - exact hGrow.
  }
  assert (hMarker :
    forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 : nat, n1 = n1).
  { intros n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48; reflexivity. }
  split.
  - exact hCore.
  - exact hMarker.
Qed.

Lemma main_result_analysis_compensated_compactness_divcurl :
  forall {X : Type} `{FrameworkStruct_analysis_compensated_compactness_divcurl X} (d : ContextData_analysis_compensated_compactness_divcurl),
    (fluxNorm (pairMap (state d)) <= fluxNorm (state d) + defectNorm (state d) /\
    defectNorm (pairMap (state d)) <= defectNorm (state d) + fluxNorm (state d)) /\
    (forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 : nat, n1 = n1).
Proof.
  intros X H d.
  assert (hFlux : fluxNorm (pairMap (state d)) <= fluxNorm (state d) + defectNorm (state d)).
  { apply pair_flux. }
  assert (hDef : defectNorm (pairMap (state d)) <= defectNorm (state d) + fluxNorm (state d)).
  { apply pair_defect. }
  assert (hMarker :
    forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 : nat, n1 = n1).
  { intros n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49; reflexivity. }
  split.
  - split; exact hFlux || exact hDef.
  - exact hMarker.
Qed.
