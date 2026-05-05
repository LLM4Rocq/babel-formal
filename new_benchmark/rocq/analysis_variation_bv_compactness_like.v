(*
BENCHMARK_ID: TINY_MATHLIB_BATCH07_ANALYSIS_VARIATION_BV_COMPACTNESS
PAIR_STEM: analysis_variation_bv_compactness_like
MATH_DOMAIN: Analysis
SOURCE_MATHLIB: Mathlib/Analysis/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class FrameworkStruct_analysis_variation_bv_compactness (X : Type) := {
  variation : X -> nat;
  oscillation : X -> nat;
  envelope : X -> nat;
  compactNorm : X -> nat;
  clipMap : X -> X;
  smoothMap : X -> X;
  iterateMap : X -> X;
  le_trans_nat : forall a b c : nat, a <= b -> b <= c -> a <= c;
  add_le_add_left_nat : forall a b c : nat, a <= b -> c + a <= c + b;
  add_le_add_right_nat : forall a b c : nat, a <= b -> a + c <= b + c;
  add_le_add_nat : forall a b c d : nat, a <= b -> c <= d -> a + c <= b + d;
  le_add_right_nat : forall a b : nat, a <= a + b;
  add_comm_nat : forall a b : nat, a + b = b + a;
  clip_var : forall x : X, variation (clipMap x) <= variation x + oscillation x;
  smooth_env : forall x : X, envelope (smoothMap x) <= envelope x + compactNorm x;
  smooth_osc : forall x : X, oscillation (smoothMap x) <= variation x + oscillation x;
  iterate_var : forall x : X, variation (iterateMap x) <= variation x + compactNorm x;
  iterate_compact : forall x : X, compactNorm (iterateMap x) <= compactNorm x + envelope x
}.

Record ContextData_analysis_variation_bv_compactness (X : Type)
    `{FrameworkStruct_analysis_variation_bv_compactness X} := {
  state : X;
  boundA : nat;
  boundB : nat;
  boundA_pos : 0 < boundA;
  boundB_pos : 0 < boundB;
  var_le : variation state <= boundA;
  osc_le : oscillation state <= boundB
}.

Definition primary_map_analysis_variation_bv_compactness
    {X : Type} `{FrameworkStruct_analysis_variation_bv_compactness X}
    (d : ContextData_analysis_variation_bv_compactness) : nat :=
  variation (state d) + oscillation (state d).

Definition secondary_map_analysis_variation_bv_compactness
    {X : Type} `{FrameworkStruct_analysis_variation_bv_compactness X}
    (d : ContextData_analysis_variation_bv_compactness) : nat :=
  envelope (state d) + compactNorm (state d).

Definition tertiary_map_analysis_variation_bv_compactness
    {X : Type} `{FrameworkStruct_analysis_variation_bv_compactness X}
    (d : ContextData_analysis_variation_bv_compactness) : nat :=
  boundA d + boundB d.

Lemma stability_step_analysis_variation_bv_compactness :
  forall {X : Type} `{FrameworkStruct_analysis_variation_bv_compactness X} (d : ContextData_analysis_variation_bv_compactness),
    (variation (clipMap (state d)) <= variation (state d) + oscillation (state d) /\
    envelope (smoothMap (state d)) <= envelope (state d) + compactNorm (state d)) /\
    (forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 : nat, n1 = n1).
Proof.
  intros X H d.
  assert (hClip : variation (clipMap (state d)) <= variation (state d) + oscillation (state d)).
  { apply clip_var. }
  assert (hSmooth : envelope (smoothMap (state d)) <= envelope (state d) + compactNorm (state d)).
  { apply smooth_env. }
  assert (hMarker :
    forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 : nat, n1 = n1).
  { intros n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36; reflexivity. }
  split.
  - split; exact hClip || exact hSmooth.
  - exact hMarker.
Qed.

Lemma factorization_step_analysis_variation_bv_compactness :
  forall {X : Type} `{FrameworkStruct_analysis_variation_bv_compactness X} (d : ContextData_analysis_variation_bv_compactness),
    (compactNorm (iterateMap (state d)) <= compactNorm (state d) + envelope (state d) /\
    primary_map_analysis_variation_bv_compactness d <=
      primary_map_analysis_variation_bv_compactness d +
        tertiary_map_analysis_variation_bv_compactness d) /\
    (forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 : nat, n1 = n1).
Proof.
  intros X H d.
  assert (hComp : compactNorm (iterateMap (state d)) <= compactNorm (state d) + envelope (state d)).
  { apply iterate_compact. }
  assert (hGrow : primary_map_analysis_variation_bv_compactness d <=
    primary_map_analysis_variation_bv_compactness d +
      tertiary_map_analysis_variation_bv_compactness d).
  { apply le_add_right_nat. }
  assert (hMarker :
    forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 : nat, n1 = n1).
  { intros n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37; reflexivity. }
  split.
  - split; exact hComp || exact hGrow.
  - exact hMarker.
Qed.

Lemma comparison_step_analysis_variation_bv_compactness :
  forall {X : Type} `{FrameworkStruct_analysis_variation_bv_compactness X} (d : ContextData_analysis_variation_bv_compactness),
    (oscillation (smoothMap (state d)) <= variation (state d) + oscillation (state d) \/
    variation (iterateMap (state d)) <= variation (state d) + compactNorm (state d)) /\
    (forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 : nat, n1 = n1).
Proof.
  intros X H d.
  assert (hOsc : oscillation (smoothMap (state d)) <= variation (state d) + oscillation (state d)).
  { apply smooth_osc. }
  assert (hMarker :
    forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 : nat, n1 = n1).
  { intros n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38; reflexivity. }
  split.
  - left; exact hOsc.
  - exact hMarker.
Qed.

Lemma transport_step_analysis_variation_bv_compactness :
  forall {X : Type} `{FrameworkStruct_analysis_variation_bv_compactness X} (d : ContextData_analysis_variation_bv_compactness),
    (exists m : nat,
      m = tertiary_map_analysis_variation_bv_compactness d /\
      variation (iterateMap (state d)) <=
        variation (state d) + compactNorm (state d) + m) /\
    (forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 : nat, n1 = n1).
Proof.
  intros X H d.
  split.
  - exists (tertiary_map_analysis_variation_bv_compactness d).
    split.
    + reflexivity.
    + assert (hBase : variation (iterateMap (state d)) <=
        variation (state d) + compactNorm (state d)).
      { apply iterate_var. }
      assert (hGrow :
        variation (state d) + compactNorm (state d) <=
          (variation (state d) + compactNorm (state d)) +
            tertiary_map_analysis_variation_bv_compactness d).
      { apply le_add_right_nat. }
      eapply le_trans_nat.
      * exact hBase.
      * exact hGrow.
  - intros n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39.
    reflexivity.
Qed.

Lemma coherence_step_analysis_variation_bv_compactness :
  forall {X : Type} `{FrameworkStruct_analysis_variation_bv_compactness X} (d : ContextData_analysis_variation_bv_compactness),
    (secondary_map_analysis_variation_bv_compactness d <=
      secondary_map_analysis_variation_bv_compactness d +
        tertiary_map_analysis_variation_bv_compactness d) /\
    (forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 : nat, n1 = n1).
Proof.
  intros X H d.
  assert (hAdd : secondary_map_analysis_variation_bv_compactness d <=
    secondary_map_analysis_variation_bv_compactness d +
      tertiary_map_analysis_variation_bv_compactness d).
  { apply le_add_right_nat. }
  assert (hMarker :
    forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 : nat, n1 = n1).
  { intros n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40; reflexivity. }
  split.
  - exact hAdd.
  - exact hMarker.
Qed.

Lemma iteration_step_analysis_variation_bv_compactness :
  forall {X : Type} `{FrameworkStruct_analysis_variation_bv_compactness X} (d : ContextData_analysis_variation_bv_compactness),
    (forall z : X, variation z <= variation z +
      tertiary_map_analysis_variation_bv_compactness d) /\
    (forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 : nat, n1 = n1).
Proof.
  intros X H d.
  assert (hCore : forall z : X, variation z <= variation z +
    tertiary_map_analysis_variation_bv_compactness d).
  { intro z; apply le_add_right_nat. }
  assert (hMarker :
    forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 : nat, n1 = n1).
  { intros n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41; reflexivity. }
  split.
  - exact hCore.
  - exact hMarker.
Qed.

Lemma main_result_analysis_variation_bv_compactness :
  forall {X : Type} `{FrameworkStruct_analysis_variation_bv_compactness X} (d : ContextData_analysis_variation_bv_compactness),
    (variation (clipMap (state d)) <= variation (state d) + oscillation (state d) /\
    variation (iterateMap (state d)) <= variation (state d) + compactNorm (state d)) /\
    (forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 : nat, n1 = n1).
Proof.
  intros X H d.
  assert (hClip : variation (clipMap (state d)) <= variation (state d) + oscillation (state d)).
  { apply clip_var. }
  assert (hIter : variation (iterateMap (state d)) <= variation (state d) + compactNorm (state d)).
  { apply iterate_var. }
  assert (hMarker :
    forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 : nat, n1 = n1).
  { intros n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42; reflexivity. }
  split.
  - split; exact hClip || exact hIter.
  - exact hMarker.
Qed.
