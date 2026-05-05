(*
BENCHMARK_ID: TINY_MATHLIB_BATCH07_TOPOLOGY_HOMOTOPY_COLIMIT_GLUING_ADVANCED
PAIR_STEM: topology_homotopy_colimit_gluing_advanced_like
MATH_DOMAIN: Topology
SOURCE_MATHLIB: Mathlib/Topology/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class FrameworkStruct_topology_homotopy_colimit_gluing_advanced (X : Type) := {
  pointWeight : X -> nat;
  patchComplexity : X -> nat;
  glueComplexity : X -> nat;
  hocolimComplexity : X -> nat;
  leftAttach : X -> X;
  rightAttach : X -> X;
  glueNode : X -> X;
  iterateNode : X -> X;
  le_trans_nat : forall a b c : nat, a <= b -> b <= c -> a <= c;
  add_le_add_left_nat : forall a b c : nat, a <= b -> c + a <= c + b;
  add_le_add_right_nat : forall a b c : nat, a <= b -> a + c <= b + c;
  add_le_add_nat : forall a b c d : nat, a <= b -> c <= d -> a + c <= b + d;
  le_add_right_nat : forall a b : nat, a <= a + b;
  add_comm_nat : forall a b : nat, a + b = b + a;
  left_attach_weight : forall x : X, pointWeight (leftAttach x) <= pointWeight x + patchComplexity x;
  right_attach_patch : forall x : X, patchComplexity (rightAttach x) <= patchComplexity x + pointWeight x;
  glue_node_bound : forall x : X, glueComplexity (glueNode x) <= glueComplexity x + patchComplexity x;
  hocolim_of_glue : forall x : X, hocolimComplexity x <= glueComplexity x + pointWeight x;
  iterate_glue_bound : forall x : X, glueComplexity (iterateNode x) <= glueComplexity x + hocolimComplexity x;
  iterate_weight_bound : forall x : X, pointWeight (iterateNode x) <= pointWeight x + patchComplexity x
}.

Record ContextData_topology_homotopy_colimit_gluing_advanced (X : Type)
    `{FrameworkStruct_topology_homotopy_colimit_gluing_advanced X} := {
  state : X;
  budget : nat;
  budget_pos : 0 < budget;
  weight_le_budget : pointWeight state <= budget;
  patch_le_budget : patchComplexity state <= budget
}.

Definition primary_map_topology_homotopy_colimit_gluing_advanced
    {X : Type} `{FrameworkStruct_topology_homotopy_colimit_gluing_advanced X}
    (d : ContextData_topology_homotopy_colimit_gluing_advanced) : nat :=
  pointWeight (state d) + patchComplexity (state d).

Definition secondary_map_topology_homotopy_colimit_gluing_advanced
    {X : Type} `{FrameworkStruct_topology_homotopy_colimit_gluing_advanced X}
    (d : ContextData_topology_homotopy_colimit_gluing_advanced) : nat :=
  glueComplexity (state d) + hocolimComplexity (state d).

Definition tertiary_map_topology_homotopy_colimit_gluing_advanced
    {X : Type} `{FrameworkStruct_topology_homotopy_colimit_gluing_advanced X}
    (d : ContextData_topology_homotopy_colimit_gluing_advanced) : nat :=
  primary_map_topology_homotopy_colimit_gluing_advanced d + budget d.

Lemma stability_step_topology_homotopy_colimit_gluing_advanced :
  forall {X : Type} `{FrameworkStruct_topology_homotopy_colimit_gluing_advanced X} (d : ContextData_topology_homotopy_colimit_gluing_advanced),
    (pointWeight (leftAttach (state d)) <=
      pointWeight (state d) + patchComplexity (state d) /\
    patchComplexity (rightAttach (state d)) <=
      patchComplexity (state d) + pointWeight (state d)) /\
    (forall P1 : Prop, P1 -> P1).
Proof.
  intros X H d.
  assert (hLeft : pointWeight (leftAttach (state d)) <=
    pointWeight (state d) + patchComplexity (state d)).
  { apply left_attach_weight. }
  assert (hRight : patchComplexity (rightAttach (state d)) <=
    patchComplexity (state d) + pointWeight (state d)).
  { apply right_attach_patch. }
  assert (hMarker : forall P1 : Prop, P1 -> P1).
  { intros P1 hP1; exact hP1. }
  split.
  - split; exact hLeft || exact hRight.
  - exact hMarker.
Qed.

Lemma factorization_step_topology_homotopy_colimit_gluing_advanced :
  forall {X : Type} `{FrameworkStruct_topology_homotopy_colimit_gluing_advanced X} (d : ContextData_topology_homotopy_colimit_gluing_advanced),
    (primary_map_topology_homotopy_colimit_gluing_advanced d <=
      primary_map_topology_homotopy_colimit_gluing_advanced d +
        tertiary_map_topology_homotopy_colimit_gluing_advanced d) /\
    (forall P1 P2 : Prop, P1 -> P2 -> P1).
Proof.
  intros X H d.
  assert (hPrim : primary_map_topology_homotopy_colimit_gluing_advanced d <=
    primary_map_topology_homotopy_colimit_gluing_advanced d +
      tertiary_map_topology_homotopy_colimit_gluing_advanced d).
  { apply le_add_right_nat. }
  assert (hMarker : forall P1 P2 : Prop, P1 -> P2 -> P1).
  { intros P1 P2 hP1 hP2; exact hP1. }
  split.
  - exact hPrim.
  - exact hMarker.
Qed.

Lemma comparison_step_topology_homotopy_colimit_gluing_advanced :
  forall {X : Type} `{FrameworkStruct_topology_homotopy_colimit_gluing_advanced X} (d : ContextData_topology_homotopy_colimit_gluing_advanced),
    (glueComplexity (iterateNode (state d)) <=
      glueComplexity (state d) + hocolimComplexity (state d) \/
    pointWeight (iterateNode (state d)) <=
      pointWeight (state d) + patchComplexity (state d)) /\
    (forall P1 P2 P3 : Prop, P1 -> P2 -> P3 -> P1).
Proof.
  intros X H d.
  assert (hGlue : glueComplexity (iterateNode (state d)) <=
    glueComplexity (state d) + hocolimComplexity (state d)).
  { apply iterate_glue_bound. }
  assert (hMarker : forall P1 P2 P3 : Prop, P1 -> P2 -> P3 -> P1).
  { intros P1 P2 P3 hP1 hP2 hP3; exact hP1. }
  split.
  - left; exact hGlue.
  - exact hMarker.
Qed.

Lemma transport_step_topology_homotopy_colimit_gluing_advanced :
  forall {X : Type} `{FrameworkStruct_topology_homotopy_colimit_gluing_advanced X}
    (d : ContextData_topology_homotopy_colimit_gluing_advanced)
    (hBudget : pointWeight (state d) <= budget d /\ patchComplexity (state d) <= budget d),
    (exists n : nat,
      n = tertiary_map_topology_homotopy_colimit_gluing_advanced d /\
      hocolimComplexity (state d) <= glueComplexity (state d) + pointWeight (state d) + n) /\
    (forall P1 P2 P3 P4 : Prop, P1 -> P2 -> P3 -> P4 -> P1).
Proof.
  intros X H d hBudget.
  split.
  - exists (tertiary_map_topology_homotopy_colimit_gluing_advanced d).
    split.
    + reflexivity.
    + assert (hBase : hocolimComplexity (state d) <=
        glueComplexity (state d) + pointWeight (state d)).
      { apply hocolim_of_glue. }
      assert (hLift :
        glueComplexity (state d) + pointWeight (state d) <=
          (glueComplexity (state d) + pointWeight (state d)) +
            tertiary_map_topology_homotopy_colimit_gluing_advanced d).
      { apply le_add_right_nat. }
      eapply le_trans_nat.
      * exact hBase.
      * exact hLift.
  - intros P1 P2 P3 P4 hP1 hP2 hP3 hP4.
    assert (hkeep : pointWeight (state d) <= budget d).
    { exact (proj1 hBudget). }
    exact hP1.
Qed.

Lemma coherence_step_topology_homotopy_colimit_gluing_advanced :
  forall {X : Type} `{FrameworkStruct_topology_homotopy_colimit_gluing_advanced X}
    (d : ContextData_topology_homotopy_colimit_gluing_advanced),
    (hocolimComplexity (state d) <=
      hocolimComplexity (state d) +
        tertiary_map_topology_homotopy_colimit_gluing_advanced d) /\
    (forall P1 P2 P3 P4 P5 : Prop, P1 -> P2 -> P3 -> P4 -> P5 -> P1).
Proof.
  intros X H d.
  assert (hFirst : hocolimComplexity (state d) <=
    hocolimComplexity (state d) +
      tertiary_map_topology_homotopy_colimit_gluing_advanced d).
  { apply le_add_right_nat. }
  assert (hMarker : forall P1 P2 P3 P4 P5 : Prop, P1 -> P2 -> P3 -> P4 -> P5 -> P1).
  { intros P1 P2 P3 P4 P5 hP1 hP2 hP3 hP4 hP5; exact hP1. }
  split.
  - exact hFirst.
  - exact hMarker.
Qed.

Lemma iteration_step_topology_homotopy_colimit_gluing_advanced :
  forall {X : Type} `{FrameworkStruct_topology_homotopy_colimit_gluing_advanced X}
    (d : ContextData_topology_homotopy_colimit_gluing_advanced),
    (forall z : X, glueComplexity z <=
      glueComplexity z + tertiary_map_topology_homotopy_colimit_gluing_advanced d) /\
    (forall P1 P2 P3 P4 P5 P6 : Prop, P1 -> P2 -> P3 -> P4 -> P5 -> P6 -> P1).
Proof.
  intros X H d.
  assert (hCore : forall z : X, glueComplexity z <=
    glueComplexity z + tertiary_map_topology_homotopy_colimit_gluing_advanced d).
  { intro z; apply le_add_right_nat. }
  assert (hMarker : forall P1 P2 P3 P4 P5 P6 : Prop, P1 -> P2 -> P3 -> P4 -> P5 -> P6 -> P1).
  { intros P1 P2 P3 P4 P5 P6 hP1 hP2 hP3 hP4 hP5 hP6; exact hP1. }
  split.
  - exact hCore.
  - exact hMarker.
Qed.

Lemma main_result_topology_homotopy_colimit_gluing_advanced :
  forall {X : Type} `{FrameworkStruct_topology_homotopy_colimit_gluing_advanced X}
    (d : ContextData_topology_homotopy_colimit_gluing_advanced),
    (hocolimComplexity (iterateNode (state d)) <=
      glueComplexity (iterateNode (state d)) + pointWeight (iterateNode (state d)) /\
    pointWeight (leftAttach (state d)) <=
      pointWeight (state d) + patchComplexity (state d)) /\
    (forall P1 P2 P3 P4 P5 P6 P7 : Prop, P1 -> P2 -> P3 -> P4 -> P5 -> P6 -> P7 -> P1).
Proof.
  intros X H d.
  assert (hHocolim : hocolimComplexity (iterateNode (state d)) <=
    glueComplexity (iterateNode (state d)) + pointWeight (iterateNode (state d))).
  { apply hocolim_of_glue. }
  assert (hLeft : pointWeight (leftAttach (state d)) <=
    pointWeight (state d) + patchComplexity (state d)).
  { apply left_attach_weight. }
  assert (hMarker : forall P1 P2 P3 P4 P5 P6 P7 : Prop, P1 -> P2 -> P3 -> P4 -> P5 -> P6 -> P7 -> P1).
  { intros P1 P2 P3 P4 P5 P6 P7 hP1 hP2 hP3 hP4 hP5 hP6 hP7; exact hP1. }
  split.
  - split; exact hHocolim || exact hLeft.
  - exact hMarker.
Qed.
