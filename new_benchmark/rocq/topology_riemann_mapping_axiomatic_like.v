(*
BENCHMARK_ID: TINY_MATHLIB_BATCH07_TOPOLOGY_RIEMANN_MAPPING_AXIOMATIC
PAIR_STEM: topology_riemann_mapping_axiomatic_like
MATH_DOMAIN: Topology
SOURCE_MATHLIB: Mathlib/Topology/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class FrameworkStruct_topology_riemann_mapping_axiomatic (X : Type) := {
  radius : X -> nat;
  distortion : X -> nat;
  energy : X -> nat;
  boundary : X -> nat;
  normalize : X -> X;
  inverse : X -> X;
  compose : X -> X -> X;
  le_trans_nat : forall a b c : nat, a <= b -> b <= c -> a <= c;
  add_le_add_left_nat : forall a b c : nat, a <= b -> c + a <= c + b;
  add_le_add_right_nat : forall a b c : nat, a <= b -> a + c <= b + c;
  add_le_add_nat : forall a b c d : nat, a <= b -> c <= d -> a + c <= b + d;
  le_add_right_nat : forall a b : nat, a <= a + b;
  add_comm_nat : forall a b : nat, a + b = b + a;
  norm_dist : forall x : X, distortion (normalize x) <= distortion x + radius x;
  inv_boundary : forall x : X, boundary (inverse x) <= boundary x + radius x;
  inv_energy : forall x : X, energy (inverse x) <= energy x + distortion x;
  compose_dist : forall x y : X, distortion (compose x y) <= distortion x + distortion y;
  compose_radius : forall x y : X, radius (compose x y) <= radius x + radius y;
  energy_bound : forall x : X, energy x <= boundary x + distortion x;
  boundary_bound : forall x : X, boundary x <= energy x + radius x
}.

Record ContextData_topology_riemann_mapping_axiomatic (X : Type)
    `{FrameworkStruct_topology_riemann_mapping_axiomatic X} := {
  state : X;
  targetBound : nat;
  altBound : nat;
  target_pos : 0 < targetBound;
  alt_pos : 0 < altBound;
  chart : exists y : X, distortion y <= targetBound /\ radius y <= altBound;
  state_control : distortion state <= targetBound;
  radius_control : radius state <= altBound
}.

Definition primary_map_topology_riemann_mapping_axiomatic
    {X : Type} `{FrameworkStruct_topology_riemann_mapping_axiomatic X}
    (d : ContextData_topology_riemann_mapping_axiomatic) : nat :=
  distortion (state d) + radius (state d).

Definition secondary_map_topology_riemann_mapping_axiomatic
    {X : Type} `{FrameworkStruct_topology_riemann_mapping_axiomatic X}
    (d : ContextData_topology_riemann_mapping_axiomatic) : nat :=
  boundary (state d) + radius (state d).

Definition tertiary_map_topology_riemann_mapping_axiomatic
    {X : Type} `{FrameworkStruct_topology_riemann_mapping_axiomatic X}
    (d : ContextData_topology_riemann_mapping_axiomatic) : nat :=
  targetBound d + altBound d.

Lemma stability_step_topology_riemann_mapping_axiomatic :
  forall {X : Type} `{FrameworkStruct_topology_riemann_mapping_axiomatic X} (d : ContextData_topology_riemann_mapping_axiomatic),
    (boundary (inverse (state d)) <= boundary (state d) + radius (state d) /\
    energy (inverse (state d)) <= energy (state d) + distortion (state d)) /\
    (forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 : nat, n1 = n1).
Proof.
  intros X H d.
  assert (hInvB : boundary (inverse (state d)) <= boundary (state d) + radius (state d)).
  { apply inv_boundary. }
  assert (hInvE : energy (inverse (state d)) <= energy (state d) + distortion (state d)).
  { apply inv_energy. }
  assert (hMarker :
    forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 : nat, n1 = n1).
  { intros n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22; reflexivity. }
  split.
  - split; exact hInvB || exact hInvE.
  - exact hMarker.
Qed.

Lemma factorization_step_topology_riemann_mapping_axiomatic :
  forall {X : Type} `{FrameworkStruct_topology_riemann_mapping_axiomatic X} (d : ContextData_topology_riemann_mapping_axiomatic),
    (energy (state d) <= boundary (state d) + distortion (state d) /\
    primary_map_topology_riemann_mapping_axiomatic d <=
      primary_map_topology_riemann_mapping_axiomatic d +
        tertiary_map_topology_riemann_mapping_axiomatic d) /\
    (forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 : nat, n1 = n1).
Proof.
  intros X H d.
  assert (hEnergy : energy (state d) <= boundary (state d) + distortion (state d)).
  { apply energy_bound. }
  assert (hGrow : primary_map_topology_riemann_mapping_axiomatic d <=
    primary_map_topology_riemann_mapping_axiomatic d +
      tertiary_map_topology_riemann_mapping_axiomatic d).
  { apply le_add_right_nat. }
  assert (hMarker :
    forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 : nat, n1 = n1).
  { intros n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23; reflexivity. }
  split.
  - split; exact hEnergy || exact hGrow.
  - exact hMarker.
Qed.

Lemma comparison_step_topology_riemann_mapping_axiomatic :
  forall {X : Type} `{FrameworkStruct_topology_riemann_mapping_axiomatic X} (d : ContextData_topology_riemann_mapping_axiomatic),
    (distortion (normalize (state d)) <= distortion (state d) + radius (state d) \/
    boundary (state d) <= energy (state d) + radius (state d)) /\
    (forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 : nat, n1 = n1).
Proof.
  intros X H d.
  assert (hNorm : distortion (normalize (state d)) <= distortion (state d) + radius (state d)).
  { apply norm_dist. }
  assert (hMarker :
    forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 : nat, n1 = n1).
  { intros n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24; reflexivity. }
  split.
  - left; exact hNorm.
  - exact hMarker.
Qed.

Lemma transport_step_topology_riemann_mapping_axiomatic :
  forall {X : Type} `{FrameworkStruct_topology_riemann_mapping_axiomatic X} (d : ContextData_topology_riemann_mapping_axiomatic),
    (exists n : nat,
      n = tertiary_map_topology_riemann_mapping_axiomatic d /\
      distortion (normalize (state d)) <= distortion (state d) + radius (state d) + n) /\
    (forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 : nat, n1 = n1).
Proof.
  intros X H d.
  split.
  - exists (tertiary_map_topology_riemann_mapping_axiomatic d).
    split.
    + reflexivity.
    + assert (hBase : distortion (normalize (state d)) <= distortion (state d) + radius (state d)).
      { apply norm_dist. }
      assert (hGrow : distortion (state d) + radius (state d) <=
        (distortion (state d) + radius (state d)) +
          tertiary_map_topology_riemann_mapping_axiomatic d).
      { apply le_add_right_nat. }
      eapply le_trans_nat.
      * exact hBase.
      * exact hGrow.
  - intros n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25.
    reflexivity.
Qed.

Lemma coherence_step_topology_riemann_mapping_axiomatic :
  forall {X : Type} `{FrameworkStruct_topology_riemann_mapping_axiomatic X} (d : ContextData_topology_riemann_mapping_axiomatic),
    (secondary_map_topology_riemann_mapping_axiomatic d <=
      secondary_map_topology_riemann_mapping_axiomatic d +
        tertiary_map_topology_riemann_mapping_axiomatic d) /\
    (forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 : nat, n1 = n1).
Proof.
  intros X H d.
  assert (hAdd : secondary_map_topology_riemann_mapping_axiomatic d <=
    secondary_map_topology_riemann_mapping_axiomatic d +
      tertiary_map_topology_riemann_mapping_axiomatic d).
  { apply le_add_right_nat. }
  assert (hMarker :
    forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 : nat, n1 = n1).
  { intros n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26; reflexivity. }
  split.
  - exact hAdd.
  - exact hMarker.
Qed.

Lemma iteration_step_topology_riemann_mapping_axiomatic :
  forall {X : Type} `{FrameworkStruct_topology_riemann_mapping_axiomatic X} (d : ContextData_topology_riemann_mapping_axiomatic),
    (forall z : X, radius z <= radius z + tertiary_map_topology_riemann_mapping_axiomatic d) /\
    (forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 : nat, n1 = n1).
Proof.
  intros X H d.
  assert (hCore : forall z : X, radius z <= radius z + tertiary_map_topology_riemann_mapping_axiomatic d).
  { intro z; apply le_add_right_nat. }
  assert (hMarker :
    forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 : nat, n1 = n1).
  { intros n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27; reflexivity. }
  split.
  - exact hCore.
  - exact hMarker.
Qed.

Lemma main_result_topology_riemann_mapping_axiomatic :
  forall {X : Type} `{FrameworkStruct_topology_riemann_mapping_axiomatic X} (d : ContextData_topology_riemann_mapping_axiomatic),
    (boundary (inverse (state d)) <= boundary (state d) + radius (state d) /\
    distortion (normalize (state d)) <= distortion (state d) + radius (state d)) /\
    (forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 : nat, n1 = n1).
Proof.
  intros X H d.
  assert (hInvB : boundary (inverse (state d)) <= boundary (state d) + radius (state d)).
  { apply inv_boundary. }
  assert (hNorm : distortion (normalize (state d)) <= distortion (state d) + radius (state d)).
  { apply norm_dist. }
  assert (hMarker :
    forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28 : nat, n1 = n1).
  { intros n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 n27 n28; reflexivity. }
  split.
  - split; exact hInvB || exact hNorm.
  - exact hMarker.
Qed.
