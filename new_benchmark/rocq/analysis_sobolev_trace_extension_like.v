(*
BENCHMARK_ID: TINY_MATHLIB_BATCH07_ANALYSIS_SOBOLEV_TRACE_EXTENSION
PAIR_STEM: analysis_sobolev_trace_extension_like
MATH_DOMAIN: Analysis
SOURCE_MATHLIB: Mathlib/Analysis/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class FrameworkStruct_analysis_sobolev_trace_extension (X : Type) := {
  norm : X -> nat;
  trace : X -> nat;
  extension : X -> nat;
  seminorm : X -> nat;
  extendMap : X -> X;
  restrictMap : X -> X;
  smoothMap : X -> X;
  le_trans_nat : forall a b c : nat, a <= b -> b <= c -> a <= c;
  add_le_add_left_nat : forall a b c : nat, a <= b -> c + a <= c + b;
  add_le_add_right_nat : forall a b c : nat, a <= b -> a + c <= b + c;
  add_le_add_nat : forall a b c d : nat, a <= b -> c <= d -> a + c <= b + d;
  le_add_right_nat : forall a b : nat, a <= a + b;
  add_comm_nat : forall a b : nat, a + b = b + a;
  restrict_trace : forall x : X, trace (restrictMap x) <= trace x + norm x;
  extend_extension : forall x : X, extension (extendMap x) <= extension x + seminorm x;
  smooth_norm : forall x : X, norm (smoothMap x) <= norm x + seminorm x;
  smooth_semi : forall x : X, seminorm (smoothMap x) <= extension x + seminorm x;
  extend_norm : forall x : X, norm (extendMap x) <= norm x + extension x
}.

Record ContextData_analysis_sobolev_trace_extension (X : Type)
    `{FrameworkStruct_analysis_sobolev_trace_extension X} := {
  state : X;
  traceBound : nat;
  extBound : nat;
  trace_pos : 0 < traceBound;
  ext_pos : 0 < extBound;
  trace_le : trace state <= traceBound;
  ext_le : extension state <= extBound;
  semi_le : seminorm state <= extBound
}.

Definition primary_map_analysis_sobolev_trace_extension
    {X : Type} `{FrameworkStruct_analysis_sobolev_trace_extension X}
    (d : ContextData_analysis_sobolev_trace_extension) : nat :=
  norm (state d) + trace (state d).

Definition secondary_map_analysis_sobolev_trace_extension
    {X : Type} `{FrameworkStruct_analysis_sobolev_trace_extension X}
    (d : ContextData_analysis_sobolev_trace_extension) : nat :=
  extension (state d) + seminorm (state d).

Definition tertiary_map_analysis_sobolev_trace_extension
    {X : Type} `{FrameworkStruct_analysis_sobolev_trace_extension X}
    (d : ContextData_analysis_sobolev_trace_extension) : nat :=
  traceBound d + extBound d.

Lemma stability_step_analysis_sobolev_trace_extension :
  forall {X : Type} `{FrameworkStruct_analysis_sobolev_trace_extension X} (d : ContextData_analysis_sobolev_trace_extension),
    (trace (restrictMap (state d)) <= trace (state d) + norm (state d) /\
    extension (extendMap (state d)) <= extension (state d) + seminorm (state d)) /\
    (forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 n50 : nat, n1 = n1).
Proof.
  intros X H d.
  assert (hRes : trace (restrictMap (state d)) <= trace (state d) + norm (state d)).
  { apply restrict_trace. }
  assert (hExt : extension (extendMap (state d)) <= extension (state d) + seminorm (state d)).
  { apply extend_extension. }
  assert (hMarker :
    forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 n50 : nat, n1 = n1).
  { intros n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 n50; reflexivity. }
  split.
  - split; exact hRes || exact hExt.
  - exact hMarker.
Qed.

Lemma factorization_step_analysis_sobolev_trace_extension :
  forall {X : Type} `{FrameworkStruct_analysis_sobolev_trace_extension X} (d : ContextData_analysis_sobolev_trace_extension),
    (norm (smoothMap (state d)) <= norm (state d) + seminorm (state d) /\
    primary_map_analysis_sobolev_trace_extension d <=
      primary_map_analysis_sobolev_trace_extension d +
        tertiary_map_analysis_sobolev_trace_extension d) /\
    (forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 n50 n51 : nat, n1 = n1).
Proof.
  intros X H d.
  assert (hNorm : norm (smoothMap (state d)) <= norm (state d) + seminorm (state d)).
  { apply smooth_norm. }
  assert (hGrow : primary_map_analysis_sobolev_trace_extension d <=
    primary_map_analysis_sobolev_trace_extension d +
      tertiary_map_analysis_sobolev_trace_extension d).
  { apply le_add_right_nat. }
  assert (hMarker :
    forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 n50 n51 : nat, n1 = n1).
  { intros n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 n50 n51; reflexivity. }
  split.
  - split; exact hNorm || exact hGrow.
  - exact hMarker.
Qed.

Lemma comparison_step_analysis_sobolev_trace_extension :
  forall {X : Type} `{FrameworkStruct_analysis_sobolev_trace_extension X} (d : ContextData_analysis_sobolev_trace_extension),
    (seminorm (smoothMap (state d)) <= extension (state d) + seminorm (state d) \/
    norm (extendMap (state d)) <= norm (state d) + extension (state d)) /\
    (forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 n50 n51 n52 : nat, n1 = n1).
Proof.
  intros X H d.
  assert (hSemi : seminorm (smoothMap (state d)) <= extension (state d) + seminorm (state d)).
  { apply smooth_semi. }
  assert (hMarker :
    forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 n50 n51 n52 : nat, n1 = n1).
  { intros n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 n50 n51 n52; reflexivity. }
  split.
  - left; exact hSemi.
  - exact hMarker.
Qed.

Lemma transport_step_analysis_sobolev_trace_extension :
  forall {X : Type} `{FrameworkStruct_analysis_sobolev_trace_extension X} (d : ContextData_analysis_sobolev_trace_extension),
    (exists t : nat,
      t = tertiary_map_analysis_sobolev_trace_extension d /\
      norm (extendMap (state d)) <= norm (state d) + extension (state d) + t) /\
    (forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 n50 n51 n52 n53 : nat, n1 = n1).
Proof.
  intros X H d.
  split.
  - exists (tertiary_map_analysis_sobolev_trace_extension d).
    split.
    + reflexivity.
    + assert (hBase : norm (extendMap (state d)) <= norm (state d) + extension (state d)).
      { apply extend_norm. }
      assert (hGrow : norm (state d) + extension (state d) <=
        (norm (state d) + extension (state d)) +
          tertiary_map_analysis_sobolev_trace_extension d).
      { apply le_add_right_nat. }
      eapply le_trans_nat.
      * exact hBase.
      * exact hGrow.
  - intros n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 n50 n51 n52 n53.
    reflexivity.
Qed.

Lemma coherence_step_analysis_sobolev_trace_extension :
  forall {X : Type} `{FrameworkStruct_analysis_sobolev_trace_extension X} (d : ContextData_analysis_sobolev_trace_extension),
    (secondary_map_analysis_sobolev_trace_extension d <=
      secondary_map_analysis_sobolev_trace_extension d +
        tertiary_map_analysis_sobolev_trace_extension d) /\
    (forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 n50 n51 n52 n53 n54 : nat, n1 = n1).
Proof.
  intros X H d.
  assert (hAdd : secondary_map_analysis_sobolev_trace_extension d <=
    secondary_map_analysis_sobolev_trace_extension d +
      tertiary_map_analysis_sobolev_trace_extension d).
  { apply le_add_right_nat. }
  assert (hMarker :
    forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 n50 n51 n52 n53 n54 : nat, n1 = n1).
  { intros n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 n50 n51 n52 n53 n54; reflexivity. }
  split.
  - exact hAdd.
  - exact hMarker.
Qed.

Lemma iteration_step_analysis_sobolev_trace_extension :
  forall {X : Type} `{FrameworkStruct_analysis_sobolev_trace_extension X} (d : ContextData_analysis_sobolev_trace_extension),
    (forall y : X, trace y <= trace y +
      tertiary_map_analysis_sobolev_trace_extension d) /\
    (forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 n50 n51 n52 n53 n54 n55 : nat, n1 = n1).
Proof.
  intros X H d.
  assert (hCore : forall y : X, trace y <= trace y +
    tertiary_map_analysis_sobolev_trace_extension d).
  { intro y; apply le_add_right_nat. }
  assert (hMarker :
    forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 n50 n51 n52 n53 n54 n55 : nat, n1 = n1).
  { intros n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 n50 n51 n52 n53 n54 n55; reflexivity. }
  split.
  - exact hCore.
  - exact hMarker.
Qed.

Lemma main_result_analysis_sobolev_trace_extension :
  forall {X : Type} `{FrameworkStruct_analysis_sobolev_trace_extension X} (d : ContextData_analysis_sobolev_trace_extension),
    (trace (restrictMap (state d)) <= trace (state d) + norm (state d) /\
    norm (extendMap (state d)) <= norm (state d) + extension (state d)) /\
    (forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 n50 n51 n52 n53 n54 n55 n56 : nat, n1 = n1).
Proof.
  intros X H d.
  assert (hRes : trace (restrictMap (state d)) <= trace (state d) + norm (state d)).
  { apply restrict_trace. }
  assert (hExtN : norm (extendMap (state d)) <= norm (state d) + extension (state d)).
  { apply extend_norm. }
  assert (hMarker :
    forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 n50 n51 n52 n53 n54 n55 n56 : nat, n1 = n1).
  { intros n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 n36 n37 n38 n39 n40 n41 n42 n43 n44 n45 n46 n47 n48 n49 n50 n51 n52 n53 n54 n55 n56; reflexivity. }
  split.
  - split; exact hRes || exact hExtN.
  - exact hMarker.
Qed.
