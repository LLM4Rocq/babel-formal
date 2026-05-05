(*
BENCHMARK_ID: TINY_MATHLIB_BATCH07_ANALYSIS_SEMICONVEX_GRADIENT_FLOW
PAIR_STEM: analysis_semiconvex_gradient_flow_like
MATH_DOMAIN: Analysis
SOURCE_MATHLIB: Mathlib/Analysis/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class FrameworkStruct_analysis_semiconvex_gradient_flow (X : Type) := {
  energy : X -> nat;
  slope : X -> nat;
  distance : X -> nat;
  velocity : X -> nat;
  stepMap : X -> X;
  proxMap : X -> X;
  iterateMap : X -> X;
  le_trans_nat : forall a b c : nat, a <= b -> b <= c -> a <= c;
  add_le_add_left_nat : forall a b c : nat, a <= b -> c + a <= c + b;
  add_le_add_right_nat : forall a b c : nat, a <= b -> a + c <= b + c;
  add_le_add_nat : forall a b c d : nat, a <= b -> c <= d -> a + c <= b + d;
  le_add_right_nat : forall a b : nat, a <= a + b;
  add_comm_nat : forall a b : nat, a + b = b + a;
  step_energy : forall x : X, energy (stepMap x) <= energy x + slope x;
  step_slope : forall x : X, slope (stepMap x) <= slope x + velocity x;
  prox_velocity : forall x : X, velocity (proxMap x) <= velocity x + distance x;
  iterate_distance : forall x : X, distance (iterateMap x) <= distance x + velocity x;
  iterate_slope : forall x : X, slope (iterateMap x) <= slope x + distance x;
  distance_bridge : forall x : X, distance x <= energy x + slope x
}.

Record ContextData_analysis_semiconvex_gradient_flow (X : Type)
    `{FrameworkStruct_analysis_semiconvex_gradient_flow X} := {
  state : X;
  timeHorizon : nat;
  slack : nat;
  time_pos : 0 < timeHorizon;
  slack_pos : 0 < slack;
  control_pair : energy state <= timeHorizon /\ slope state <= slack;
  dist_small : distance state <= timeHorizon + slack
}.

Definition primary_map_analysis_semiconvex_gradient_flow
    {X : Type} `{FrameworkStruct_analysis_semiconvex_gradient_flow X}
    (d : ContextData_analysis_semiconvex_gradient_flow) : nat :=
  energy (state d) + slope (state d).

Definition secondary_map_analysis_semiconvex_gradient_flow
    {X : Type} `{FrameworkStruct_analysis_semiconvex_gradient_flow X}
    (d : ContextData_analysis_semiconvex_gradient_flow) : nat :=
  distance (state d) + velocity (state d).

Definition tertiary_map_analysis_semiconvex_gradient_flow
    {X : Type} `{FrameworkStruct_analysis_semiconvex_gradient_flow X}
    (d : ContextData_analysis_semiconvex_gradient_flow) : nat :=
  timeHorizon d + slack d.

Lemma stability_step_analysis_semiconvex_gradient_flow :
  forall {X : Type} `{FrameworkStruct_analysis_semiconvex_gradient_flow X} (d : ContextData_analysis_semiconvex_gradient_flow),
    (energy (stepMap (state d)) <= energy (state d) + slope (state d) /\
    slope (stepMap (state d)) <= slope (state d) + velocity (state d)) /\
    (forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 : nat, n1 = n1).
Proof.
  intros X H d.
  assert (hStepE : energy (stepMap (state d)) <= energy (state d) + slope (state d)).
  { apply step_energy. }
  assert (hStepS : slope (stepMap (state d)) <= slope (state d) + velocity (state d)).
  { apply step_slope. }
  assert (hMarker :
    forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 : nat, n1 = n1).
  { intros n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29; reflexivity. }
  split.
  - split; exact hStepE || exact hStepS.
  - exact hMarker.
Qed.

Lemma factorization_step_analysis_semiconvex_gradient_flow :
  forall {X : Type} `{FrameworkStruct_analysis_semiconvex_gradient_flow X} (d : ContextData_analysis_semiconvex_gradient_flow),
    (distance (state d) <= energy (state d) + slope (state d) /\
    primary_map_analysis_semiconvex_gradient_flow d <=
      primary_map_analysis_semiconvex_gradient_flow d +
        tertiary_map_analysis_semiconvex_gradient_flow d) /\
    (forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 : nat, n1 = n1).
Proof.
  intros X H d.
  assert (hDist : distance (state d) <= energy (state d) + slope (state d)).
  { apply distance_bridge. }
  assert (hGrow : primary_map_analysis_semiconvex_gradient_flow d <=
    primary_map_analysis_semiconvex_gradient_flow d +
      tertiary_map_analysis_semiconvex_gradient_flow d).
  { apply le_add_right_nat. }
  assert (hMarker :
    forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 : nat, n1 = n1).
  { intros n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30; reflexivity. }
  split.
  - split; exact hDist || exact hGrow.
  - exact hMarker.
Qed.

Lemma comparison_step_analysis_semiconvex_gradient_flow :
  forall {X : Type} `{FrameworkStruct_analysis_semiconvex_gradient_flow X} (d : ContextData_analysis_semiconvex_gradient_flow),
    (slope (iterateMap (state d)) <= slope (state d) + distance (state d) \/
    velocity (proxMap (state d)) <= velocity (state d) + distance (state d)) /\
    (forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 : nat, n1 = n1).
Proof.
  intros X H d.
  assert (hSlope : slope (iterateMap (state d)) <= slope (state d) + distance (state d)).
  { apply iterate_slope. }
  assert (hMarker :
    forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 : nat, n1 = n1).
  { intros n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31; reflexivity. }
  split.
  - left; exact hSlope.
  - exact hMarker.
Qed.

Lemma transport_step_analysis_semiconvex_gradient_flow :
  forall {X : Type} `{FrameworkStruct_analysis_semiconvex_gradient_flow X} (d : ContextData_analysis_semiconvex_gradient_flow),
    (exists q : nat,
      q = tertiary_map_analysis_semiconvex_gradient_flow d /\
      distance (iterateMap (state d)) <= distance (state d) + velocity (state d) + q) /\
    (forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 : nat, n1 = n1).
Proof.
  intros X H d.
  split.
  - exists (tertiary_map_analysis_semiconvex_gradient_flow d).
    split.
    + reflexivity.
    + assert (hBase : distance (iterateMap (state d)) <= distance (state d) + velocity (state d)).
      { apply iterate_distance. }
      assert (hGrow :
        distance (state d) + velocity (state d) <=
          (distance (state d) + velocity (state d)) +
            tertiary_map_analysis_semiconvex_gradient_flow d).
      { apply le_add_right_nat. }
      eapply le_trans_nat.
      * exact hBase.
      * exact hGrow.
  - intros n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32.
    reflexivity.
Qed.

Lemma coherence_step_analysis_semiconvex_gradient_flow :
  forall {X : Type} `{FrameworkStruct_analysis_semiconvex_gradient_flow X} (d : ContextData_analysis_semiconvex_gradient_flow),
    (secondary_map_analysis_semiconvex_gradient_flow d <=
      secondary_map_analysis_semiconvex_gradient_flow d +
        tertiary_map_analysis_semiconvex_gradient_flow d) /\
    (forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 : nat, n1 = n1).
Proof.
  intros X H d.
  assert (hAdd : secondary_map_analysis_semiconvex_gradient_flow d <=
    secondary_map_analysis_semiconvex_gradient_flow d +
      tertiary_map_analysis_semiconvex_gradient_flow d).
  { apply le_add_right_nat. }
  assert (hMarker :
    forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 : nat, n1 = n1).
  { intros n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33; reflexivity. }
  split.
  - exact hAdd.
  - exact hMarker.
Qed.

Lemma iteration_step_analysis_semiconvex_gradient_flow :
  forall {X : Type} `{FrameworkStruct_analysis_semiconvex_gradient_flow X} (d : ContextData_analysis_semiconvex_gradient_flow),
    (forall z : X, distance z <= energy z + slope z +
      tertiary_map_analysis_semiconvex_gradient_flow d) /\
    (forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 : nat, n1 = n1).
Proof.
  intros X H d.
  assert (hCore : forall z : X, distance z <= energy z + slope z +
    tertiary_map_analysis_semiconvex_gradient_flow d).
  {
    intro z.
    assert (hBase : distance z <= energy z + slope z).
    { apply distance_bridge. }
    assert (hGrow : energy z + slope z <=
      (energy z + slope z) +
        tertiary_map_analysis_semiconvex_gradient_flow d).
    { apply le_add_right_nat. }
    eapply le_trans_nat.
    - exact hBase.
    - exact hGrow.
  }
  assert (hMarker :
    forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 : nat, n1 = n1).
  { intros n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34; reflexivity. }
  split.
  - exact hCore.
  - exact hMarker.
Qed.

Lemma main_result_analysis_semiconvex_gradient_flow :
  forall {X : Type} `{FrameworkStruct_analysis_semiconvex_gradient_flow X} (d : ContextData_analysis_semiconvex_gradient_flow),
    (distance (iterateMap (state d)) <= distance (state d) + velocity (state d) /\
    energy (stepMap (state d)) <= energy (state d) + slope (state d)) /\
    (forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 : nat, n1 = n1).
Proof.
  intros X H d.
  assert (hIter : distance (iterateMap (state d)) <= distance (state d) + velocity (state d)).
  { apply iterate_distance. }
  assert (hStep : energy (stepMap (state d)) <= energy (state d) + slope (state d)).
  { apply step_energy. }
  assert (hMarker :
    forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35 : nat, n1 = n1).
  { intros n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 n29 n30 n31 n32 n33 n34 n35; reflexivity. }
  split.
  - split; exact hIter || exact hStep.
  - exact hMarker.
Qed.
