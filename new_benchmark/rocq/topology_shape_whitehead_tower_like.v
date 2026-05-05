(*
BENCHMARK_ID: TINY_MATHLIB_BATCH07_TOPOLOGY_SHAPE_WHITEHEAD_TOWER
PAIR_STEM: topology_shape_whitehead_tower_like
MATH_DOMAIN: Topology
SOURCE_MATHLIB: Mathlib/Topology/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class FrameworkStruct_topology_shape_whitehead_tower (X : Type) := {
  truncRank : X -> nat;
  fiberRank : X -> nat;
  liftRank : X -> nat;
  towerRank : X -> nat;
  upMap : X -> X;
  downMap : X -> X;
  connMap : X -> X;
  itrMap : X -> X;
  le_trans_nat : forall a b c : nat, a <= b -> b <= c -> a <= c;
  add_le_add_left_nat : forall a b c : nat, a <= b -> c + a <= c + b;
  add_le_add_right_nat : forall a b c : nat, a <= b -> a + c <= b + c;
  add_le_add_nat : forall a b c d : nat, a <= b -> c <= d -> a + c <= b + d;
  le_add_right_nat : forall a b : nat, a <= a + b;
  add_comm_nat : forall a b : nat, a + b = b + a;
  up_lift : forall x : X, liftRank (upMap x) <= liftRank x + truncRank x;
  down_trunc : forall x : X, truncRank (downMap x) <= truncRank x + fiberRank x;
  conn_fiber : forall x : X, fiberRank (connMap x) <= fiberRank x + liftRank x;
  tower_from_fiber : forall x : X, towerRank x <= fiberRank x + liftRank x;
  itr_lift : forall x : X, liftRank (itrMap x) <= liftRank x + towerRank x;
  itr_tower : forall x : X, towerRank (itrMap x) <= towerRank x + truncRank x;
  fiber_nondec : forall x : X, fiberRank x <= fiberRank (connMap x)
}.

Record ContextData_topology_shape_whitehead_tower (X : Type)
    `{FrameworkStruct_topology_shape_whitehead_tower X} := {
  state : X;
  baseBound : nat;
  fiberBound : nat;
  base_pos : 0 < baseBound;
  fiber_pos : 0 < fiberBound;
  witness : truncRank state <= baseBound \/ fiberRank state <= fiberBound;
  lift_small : liftRank state <= baseBound + fiberBound
}.

Definition primary_map_topology_shape_whitehead_tower
    {X : Type} `{FrameworkStruct_topology_shape_whitehead_tower X}
    (d : ContextData_topology_shape_whitehead_tower) : nat :=
  truncRank (state d) + fiberRank (state d).

Definition secondary_map_topology_shape_whitehead_tower
    {X : Type} `{FrameworkStruct_topology_shape_whitehead_tower X}
    (d : ContextData_topology_shape_whitehead_tower) : nat :=
  liftRank (state d) + towerRank (state d).

Definition tertiary_map_topology_shape_whitehead_tower
    {X : Type} `{FrameworkStruct_topology_shape_whitehead_tower X}
    (d : ContextData_topology_shape_whitehead_tower) : nat :=
  primary_map_topology_shape_whitehead_tower d + baseBound d.

Lemma stability_step_topology_shape_whitehead_tower :
  forall {X : Type} `{FrameworkStruct_topology_shape_whitehead_tower X} (d : ContextData_topology_shape_whitehead_tower),
    (liftRank (upMap (state d)) <= liftRank (state d) + truncRank (state d) /\
    truncRank (downMap (state d)) <= truncRank (state d) + fiberRank (state d)) /\
    (forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 : nat, n1 = n1).
Proof.
  intros X H d.
  assert (hUp : liftRank (upMap (state d)) <= liftRank (state d) + truncRank (state d)).
  { apply up_lift. }
  assert (hDown : truncRank (downMap (state d)) <= truncRank (state d) + fiberRank (state d)).
  { apply down_trunc. }
  assert (hMarker : forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 : nat, n1 = n1).
  { intros n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15; reflexivity. }
  split.
  - split; exact hUp || exact hDown.
  - exact hMarker.
Qed.

Lemma factorization_step_topology_shape_whitehead_tower :
  forall {X : Type} `{FrameworkStruct_topology_shape_whitehead_tower X} (d : ContextData_topology_shape_whitehead_tower),
    (towerRank (state d) <= fiberRank (state d) + liftRank (state d) /\
    primary_map_topology_shape_whitehead_tower d <=
      primary_map_topology_shape_whitehead_tower d +
        tertiary_map_topology_shape_whitehead_tower d) /\
    (forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 : nat, n1 = n1).
Proof.
  intros X H d.
  assert (hTower : towerRank (state d) <= fiberRank (state d) + liftRank (state d)).
  { apply tower_from_fiber. }
  assert (hGrow : primary_map_topology_shape_whitehead_tower d <=
    primary_map_topology_shape_whitehead_tower d +
      tertiary_map_topology_shape_whitehead_tower d).
  { apply le_add_right_nat. }
  assert (hMarker : forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 : nat, n1 = n1).
  { intros n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16; reflexivity. }
  split.
  - split; exact hTower || exact hGrow.
  - exact hMarker.
Qed.

Lemma comparison_step_topology_shape_whitehead_tower :
  forall {X : Type} `{FrameworkStruct_topology_shape_whitehead_tower X}
    (d : ContextData_topology_shape_whitehead_tower)
    (hCase : truncRank (state d) <= baseBound d \/ fiberRank (state d) <= fiberBound d),
    (fiberRank (connMap (state d)) <= fiberRank (state d) + liftRank (state d) \/
    liftRank (itrMap (state d)) <= liftRank (state d) + towerRank (state d)) /\
    (forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 : nat, n1 = n1).
Proof.
  intros X H d hCase.
  assert (hFiber : fiberRank (connMap (state d)) <= fiberRank (state d) + liftRank (state d)).
  { apply conn_fiber. }
  assert (hMarker : forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 : nat, n1 = n1).
  { intros n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17; reflexivity. }
  split.
  - left; exact hFiber.
  - exact hMarker.
Qed.

Lemma transport_step_topology_shape_whitehead_tower :
  forall {X : Type} `{FrameworkStruct_topology_shape_whitehead_tower X} (d : ContextData_topology_shape_whitehead_tower),
    (exists n : nat,
      n = tertiary_map_topology_shape_whitehead_tower d /\
      towerRank (itrMap (state d)) <= towerRank (state d) + truncRank (state d) + n) /\
    (forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 : nat, n1 = n1).
Proof.
  intros X H d.
  split.
  - exists (tertiary_map_topology_shape_whitehead_tower d).
    split.
    + reflexivity.
    + assert (hBase : towerRank (itrMap (state d)) <= towerRank (state d) + truncRank (state d)).
      { apply itr_tower. }
      assert (hGrow :
        towerRank (state d) + truncRank (state d) <=
          (towerRank (state d) + truncRank (state d)) +
            tertiary_map_topology_shape_whitehead_tower d).
      { apply le_add_right_nat. }
      eapply le_trans_nat.
      * exact hBase.
      * exact hGrow.
  - intros n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18.
    reflexivity.
Qed.

Lemma coherence_step_topology_shape_whitehead_tower :
  forall {X : Type} `{FrameworkStruct_topology_shape_whitehead_tower X} (d : ContextData_topology_shape_whitehead_tower),
    (secondary_map_topology_shape_whitehead_tower d <=
      secondary_map_topology_shape_whitehead_tower d +
        tertiary_map_topology_shape_whitehead_tower d) /\
    (forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 : nat, n1 = n1).
Proof.
  intros X H d.
  assert (hAdd : secondary_map_topology_shape_whitehead_tower d <=
    secondary_map_topology_shape_whitehead_tower d +
      tertiary_map_topology_shape_whitehead_tower d).
  { apply le_add_right_nat. }
  assert (hMarker :
    forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 : nat, n1 = n1).
  { intros n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19; reflexivity. }
  split.
  - exact hAdd.
  - exact hMarker.
Qed.

Lemma iteration_step_topology_shape_whitehead_tower :
  forall {X : Type} `{FrameworkStruct_topology_shape_whitehead_tower X} (d : ContextData_topology_shape_whitehead_tower),
    (forall z : X, fiberRank z <=
      fiberRank (connMap z) +
        tertiary_map_topology_shape_whitehead_tower d) /\
    (forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 : nat, n1 = n1).
Proof.
  intros X H d.
  assert (hCore : forall z : X, fiberRank z <=
    fiberRank (connMap z) +
      tertiary_map_topology_shape_whitehead_tower d).
  {
    intro z.
    assert (hBase : fiberRank z <= fiberRank (connMap z)).
    { apply fiber_nondec. }
    assert (hGrow : fiberRank (connMap z) <=
      fiberRank (connMap z) +
        tertiary_map_topology_shape_whitehead_tower d).
    { apply le_add_right_nat. }
    eapply le_trans_nat.
    - exact hBase.
    - exact hGrow.
  }
  assert (hMarker :
    forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 : nat, n1 = n1).
  { intros n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20; reflexivity. }
  split.
  - exact hCore.
  - exact hMarker.
Qed.

Lemma main_result_topology_shape_whitehead_tower :
  forall {X : Type} `{FrameworkStruct_topology_shape_whitehead_tower X} (d : ContextData_topology_shape_whitehead_tower),
    (towerRank (itrMap (state d)) <= towerRank (state d) + truncRank (state d) /\
    liftRank (upMap (state d)) <= liftRank (state d) + truncRank (state d)) /\
    (forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 : nat, n1 = n1).
Proof.
  intros X H d.
  assert (hTower : towerRank (itrMap (state d)) <= towerRank (state d) + truncRank (state d)).
  { apply itr_tower. }
  assert (hLift : liftRank (upMap (state d)) <= liftRank (state d) + truncRank (state d)).
  { apply up_lift. }
  assert (hMarker :
    forall n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 : nat, n1 = n1).
  { intros n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21; reflexivity. }
  split.
  - split; exact hTower || exact hLift.
  - exact hMarker.
Qed.
