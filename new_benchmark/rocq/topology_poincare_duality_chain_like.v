(*
BENCHMARK_ID: TINY_MATHLIB_BATCH07_TOPOLOGY_POINCARE_DUALITY_CHAIN
PAIR_STEM: topology_poincare_duality_chain_like
MATH_DOMAIN: Topology
SOURCE_MATHLIB: Mathlib/Topology/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class FrameworkStruct_topology_poincare_duality_chain (X : Type) := {
  chainMass : X -> nat;
  cochainMass : X -> nat;
  boundaryMass : X -> nat;
  coboundaryMass : X -> nat;
  pairingMass : X -> nat;
  dualize : X -> X;
  capTransfer : X -> X;
  cupTransfer : X -> X;
  iterateTransfer : X -> X;
  le_trans_nat : forall a b c : nat, a <= b -> b <= c -> a <= c;
  add_le_add_left_nat : forall a b c : nat, a <= b -> c + a <= c + b;
  add_le_add_right_nat : forall a b c : nat, a <= b -> a + c <= b + c;
  add_le_add_nat : forall a b c d : nat, a <= b -> c <= d -> a + c <= b + d;
  le_add_right_nat : forall a b : nat, a <= a + b;
  add_comm_nat : forall a b : nat, a + b = b + a;
  pair_control : forall x : X, pairingMass x <= chainMass x + cochainMass x;
  boundary_cap : forall x : X, boundaryMass (capTransfer x) <= boundaryMass x + pairingMass x;
  coboundary_cup : forall x : X, coboundaryMass (cupTransfer x) <= coboundaryMass x + chainMass x;
  dual_pair : forall x : X, pairingMass (dualize x) <= pairingMass x + cochainMass x;
  dual_chain : forall x : X, chainMass (dualize x) <= chainMass x + boundaryMass x;
  chain_to_dual : forall x : X, chainMass x <= chainMass (dualize x);
  iter_boundary : forall x : X, boundaryMass (iterateTransfer x) <= boundaryMass x + coboundaryMass x
}.

Record ContextData_topology_poincare_duality_chain (X : Type)
    `{FrameworkStruct_topology_poincare_duality_chain X} := {
  state : X;
  leftBudget : nat;
  rightBudget : nat;
  left_pos : 0 < leftBudget;
  right_pos : 0 < rightBudget;
  decomp : chainMass state <= leftBudget /\ cochainMass state <= rightBudget
}.

Definition primary_map_topology_poincare_duality_chain
    {X : Type} `{FrameworkStruct_topology_poincare_duality_chain X}
    (d : ContextData_topology_poincare_duality_chain) : nat :=
  pairingMass (state d) + boundaryMass (state d).

Definition secondary_map_topology_poincare_duality_chain
    {X : Type} `{FrameworkStruct_topology_poincare_duality_chain X}
    (d : ContextData_topology_poincare_duality_chain) : nat :=
  chainMass (dualize (state d)) + coboundaryMass (state d).

Definition tertiary_map_topology_poincare_duality_chain
    {X : Type} `{FrameworkStruct_topology_poincare_duality_chain X}
    (d : ContextData_topology_poincare_duality_chain) : nat :=
  leftBudget d + rightBudget d.

Lemma stability_step_topology_poincare_duality_chain :
  forall {X : Type} `{FrameworkStruct_topology_poincare_duality_chain X} (d : ContextData_topology_poincare_duality_chain),
    (boundaryMass (capTransfer (state d)) <=
      boundaryMass (state d) + pairingMass (state d) /\
    coboundaryMass (cupTransfer (state d)) <=
      coboundaryMass (state d) + chainMass (state d)) /\
    (forall P1 P2 P3 P4 P5 P6 P7 P8 : Prop, P1 -> P2 -> P3 -> P4 -> P5 -> P6 -> P7 -> P8 -> P1).
Proof.
  intros X H d.
  assert (hCap : boundaryMass (capTransfer (state d)) <=
    boundaryMass (state d) + pairingMass (state d)).
  { apply boundary_cap. }
  assert (hCup : coboundaryMass (cupTransfer (state d)) <=
    coboundaryMass (state d) + chainMass (state d)).
  { apply coboundary_cup. }
  assert (hMarker : forall P1 P2 P3 P4 P5 P6 P7 P8 : Prop, P1 -> P2 -> P3 -> P4 -> P5 -> P6 -> P7 -> P8 -> P1).
  { intros P1 P2 P3 P4 P5 P6 P7 P8 hP1 hP2 hP3 hP4 hP5 hP6 hP7 hP8; exact hP1. }
  split.
  - split; exact hCap || exact hCup.
  - exact hMarker.
Qed.

Lemma factorization_step_topology_poincare_duality_chain :
  forall {X : Type} `{FrameworkStruct_topology_poincare_duality_chain X} (d : ContextData_topology_poincare_duality_chain),
    (pairingMass (state d) <= chainMass (state d) + cochainMass (state d) /\
    primary_map_topology_poincare_duality_chain d <=
      primary_map_topology_poincare_duality_chain d +
        tertiary_map_topology_poincare_duality_chain d) /\
    (forall P1 P2 P3 P4 P5 P6 P7 P8 P9 : Prop, P1 -> P2 -> P3 -> P4 -> P5 -> P6 -> P7 -> P8 -> P9 -> P1).
Proof.
  intros X H d.
  assert (hPair : pairingMass (state d) <= chainMass (state d) + cochainMass (state d)).
  { apply pair_control. }
  assert (hLift : primary_map_topology_poincare_duality_chain d <=
    primary_map_topology_poincare_duality_chain d +
      tertiary_map_topology_poincare_duality_chain d).
  { apply le_add_right_nat. }
  assert (hMarker : forall P1 P2 P3 P4 P5 P6 P7 P8 P9 : Prop, P1 -> P2 -> P3 -> P4 -> P5 -> P6 -> P7 -> P8 -> P9 -> P1).
  { intros P1 P2 P3 P4 P5 P6 P7 P8 P9 hP1 hP2 hP3 hP4 hP5 hP6 hP7 hP8 hP9; exact hP1. }
  split.
  - split; exact hPair || exact hLift.
  - exact hMarker.
Qed.

Lemma comparison_step_topology_poincare_duality_chain :
  forall {X : Type} `{FrameworkStruct_topology_poincare_duality_chain X}
    (d : ContextData_topology_poincare_duality_chain)
    (hCases : chainMass (state d) <= leftBudget d \/ cochainMass (state d) <= rightBudget d),
    (pairingMass (dualize (state d)) <=
      pairingMass (state d) + cochainMass (state d) \/
    chainMass (dualize (state d)) <=
      chainMass (state d) + boundaryMass (state d)) /\
    (forall P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 : Prop, P1 -> P2 -> P3 -> P4 -> P5 -> P6 -> P7 -> P8 -> P9 -> P10 -> P1).
Proof.
  intros X H d hCases.
  assert (hDual : pairingMass (dualize (state d)) <= pairingMass (state d) + cochainMass (state d)).
  { apply dual_pair. }
  assert (hMarker : forall P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 : Prop, P1 -> P2 -> P3 -> P4 -> P5 -> P6 -> P7 -> P8 -> P9 -> P10 -> P1).
  { intros P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 hP1 hP2 hP3 hP4 hP5 hP6 hP7 hP8 hP9 hP10; exact hP1. }
  split.
  - left; exact hDual.
  - exact hMarker.
Qed.

Lemma transport_step_topology_poincare_duality_chain :
  forall {X : Type} `{FrameworkStruct_topology_poincare_duality_chain X}
    (d : ContextData_topology_poincare_duality_chain),
    (exists n : nat,
      n = tertiary_map_topology_poincare_duality_chain d /\
      boundaryMass (capTransfer (state d)) <=
        boundaryMass (state d) + pairingMass (state d) + n) /\
    (forall P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 P11 : Prop, P1 -> P2 -> P3 -> P4 -> P5 -> P6 -> P7 -> P8 -> P9 -> P10 -> P11 -> P1).
Proof.
  intros X H d.
  split.
  - exists (tertiary_map_topology_poincare_duality_chain d).
    split.
    + reflexivity.
    + assert (hCap : boundaryMass (capTransfer (state d)) <=
        boundaryMass (state d) + pairingMass (state d)).
      { apply boundary_cap. }
      assert (hGrow :
        boundaryMass (state d) + pairingMass (state d) <=
          (boundaryMass (state d) + pairingMass (state d)) +
            tertiary_map_topology_poincare_duality_chain d).
      { apply le_add_right_nat. }
      eapply le_trans_nat.
      * exact hCap.
      * exact hGrow.
  - intros P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 P11 hP1 hP2 hP3 hP4 hP5 hP6 hP7 hP8 hP9 hP10 hP11.
    exact hP1.
Qed.

Lemma coherence_step_topology_poincare_duality_chain :
  forall {X : Type} `{FrameworkStruct_topology_poincare_duality_chain X}
    (d : ContextData_topology_poincare_duality_chain),
    (secondary_map_topology_poincare_duality_chain d <=
      secondary_map_topology_poincare_duality_chain d +
        tertiary_map_topology_poincare_duality_chain d) /\
    (forall P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 P11 P12 : Prop, P1 -> P2 -> P3 -> P4 -> P5 -> P6 -> P7 -> P8 -> P9 -> P10 -> P11 -> P12 -> P1).
Proof.
  intros X H d.
  assert (hAdd : secondary_map_topology_poincare_duality_chain d <=
    secondary_map_topology_poincare_duality_chain d +
      tertiary_map_topology_poincare_duality_chain d).
  { apply le_add_right_nat. }
  assert (hMarker : forall P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 P11 P12 : Prop, P1 -> P2 -> P3 -> P4 -> P5 -> P6 -> P7 -> P8 -> P9 -> P10 -> P11 -> P12 -> P1).
  { intros P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 P11 P12 hP1 hP2 hP3 hP4 hP5 hP6 hP7 hP8 hP9 hP10 hP11 hP12; exact hP1. }
  split.
  - exact hAdd.
  - exact hMarker.
Qed.

Lemma iteration_step_topology_poincare_duality_chain :
  forall {X : Type} `{FrameworkStruct_topology_poincare_duality_chain X}
    (d : ContextData_topology_poincare_duality_chain),
    (forall z : X, chainMass z <=
      chainMass (dualize z) +
        tertiary_map_topology_poincare_duality_chain d) /\
    (forall P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 P11 P12 P13 : Prop, P1 -> P2 -> P3 -> P4 -> P5 -> P6 -> P7 -> P8 -> P9 -> P10 -> P11 -> P12 -> P13 -> P1).
Proof.
  intros X H d.
  assert (hCore : forall z : X, chainMass z <=
    chainMass (dualize z) +
      tertiary_map_topology_poincare_duality_chain d).
  {
    intro z.
    assert (hBase : chainMass z <= chainMass (dualize z)).
    { apply chain_to_dual. }
    assert (hGrow : chainMass (dualize z) <=
      chainMass (dualize z) +
        tertiary_map_topology_poincare_duality_chain d).
    { apply le_add_right_nat. }
    eapply le_trans_nat.
    - exact hBase.
    - exact hGrow.
  }
  assert (hMarker : forall P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 P11 P12 P13 : Prop, P1 -> P2 -> P3 -> P4 -> P5 -> P6 -> P7 -> P8 -> P9 -> P10 -> P11 -> P12 -> P13 -> P1).
  { intros P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 P11 P12 P13 hP1 hP2 hP3 hP4 hP5 hP6 hP7 hP8 hP9 hP10 hP11 hP12 hP13; exact hP1. }
  split.
  - exact hCore.
  - exact hMarker.
Qed.

Lemma main_result_topology_poincare_duality_chain :
  forall {X : Type} `{FrameworkStruct_topology_poincare_duality_chain X}
    (d : ContextData_topology_poincare_duality_chain),
    (boundaryMass (iterateTransfer (state d)) <=
      boundaryMass (state d) + coboundaryMass (state d) /\
    pairingMass (dualize (state d)) <=
      pairingMass (state d) + cochainMass (state d)) /\
    (forall P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 P11 P12 P13 P14 : Prop, P1 -> P2 -> P3 -> P4 -> P5 -> P6 -> P7 -> P8 -> P9 -> P10 -> P11 -> P12 -> P13 -> P14 -> P1).
Proof.
  intros X H d.
  assert (hIter : boundaryMass (iterateTransfer (state d)) <=
    boundaryMass (state d) + coboundaryMass (state d)).
  { apply iter_boundary. }
  assert (hDual : pairingMass (dualize (state d)) <=
    pairingMass (state d) + cochainMass (state d)).
  { apply dual_pair. }
  assert (hMarker : forall P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 P11 P12 P13 P14 : Prop, P1 -> P2 -> P3 -> P4 -> P5 -> P6 -> P7 -> P8 -> P9 -> P10 -> P11 -> P12 -> P13 -> P14 -> P1).
  { intros P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 P11 P12 P13 P14 hP1 hP2 hP3 hP4 hP5 hP6 hP7 hP8 hP9 hP10 hP11 hP12 hP13 hP14; exact hP1. }
  split.
  - split; exact hIter || exact hDual.
  - exact hMarker.
Qed.
