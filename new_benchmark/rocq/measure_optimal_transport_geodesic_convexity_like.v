(**
BENCHMARK_ID: TINY_MATHLIB_BATCH07_MEASURE_OPTIMAL_TRANSPORT_GEODESIC_CONVEXITY_LIKE
PAIR_STEM: measure_optimal_transport_geodesic_convexity_like
MATH_DOMAIN: Measure Theory
SOURCE_MATHLIB: Mathlib/MeasureTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
**)

Set Universe Polymorphism.
Set Implicit Arguments.

Class FrameworkStruct_measure_optimal_transport_geodesic_convexity (M : Type) := {
  mass : M -> nat;
  cost : M -> nat;
  entropy : M -> nat;
  push : M -> M;
  midpoint : M -> M -> M;
  geodesic : M -> M -> M;
  le_trans_nat : forall a b c : nat, a <= b -> b <= c -> a <= c;
  add_le_add_left_nat : forall a b c : nat, a <= b -> c + a <= c + b;
  add_le_add_right_nat : forall a b c : nat, a <= b -> a + c <= b + c;
  le_add_left_nat : forall a b : nat, b <= a + b;
  le_add_right_nat : forall a b : nat, a <= a + b;
  push_mass : forall x : M, mass (push x) <= mass x;
  push_cost : forall x : M, cost (push x) <= cost x + entropy x;
  midpoint_cost : forall x y : M, cost (midpoint x y) <= cost x + cost y;
  geodesic_cost : forall x y : M, cost (geodesic x y) <= cost (midpoint x y) + entropy x + entropy y;
  entropy_midpoint : forall x y : M, entropy (midpoint x y) <= entropy x + entropy y;
  witness_push : forall x : M, exists y : M, y = push x /\ mass y <= mass x
}.

Record ContextData_measure_optimal_transport_geodesic_convexity
    (M : Type) `{FrameworkStruct_measure_optimal_transport_geodesic_convexity M} := {
  left_measure_optimal_transport_geodesic_convexity : M;
  right_measure_optimal_transport_geodesic_convexity : M;
  budget_measure_optimal_transport_geodesic_convexity : nat;
  entropy_cap_left_measure_optimal_transport_geodesic_convexity :
    entropy left_measure_optimal_transport_geodesic_convexity <= budget_measure_optimal_transport_geodesic_convexity;
  entropy_cap_right_measure_optimal_transport_geodesic_convexity :
    entropy right_measure_optimal_transport_geodesic_convexity <= budget_measure_optimal_transport_geodesic_convexity;
  cost_bridge_measure_optimal_transport_geodesic_convexity :
    cost left_measure_optimal_transport_geodesic_convexity <=
      cost right_measure_optimal_transport_geodesic_convexity + budget_measure_optimal_transport_geodesic_convexity
}.

Definition primary_map_measure_optimal_transport_geodesic_convexity
    {M : Type} `{FrameworkStruct_measure_optimal_transport_geodesic_convexity M}
    (d : ContextData_measure_optimal_transport_geodesic_convexity) : M :=
  geodesic (push (left_measure_optimal_transport_geodesic_convexity d))
    (right_measure_optimal_transport_geodesic_convexity d).

Definition secondary_map_measure_optimal_transport_geodesic_convexity
    {M : Type} `{FrameworkStruct_measure_optimal_transport_geodesic_convexity M}
    (d : ContextData_measure_optimal_transport_geodesic_convexity) : nat :=
  cost (primary_map_measure_optimal_transport_geodesic_convexity d) +
    budget_measure_optimal_transport_geodesic_convexity d.

Definition tertiary_map_measure_optimal_transport_geodesic_convexity
    {M : Type} `{FrameworkStruct_measure_optimal_transport_geodesic_convexity M}
    (d : ContextData_measure_optimal_transport_geodesic_convexity) : Prop :=
  cost (primary_map_measure_optimal_transport_geodesic_convexity d) <=
    secondary_map_measure_optimal_transport_geodesic_convexity d.

Lemma stability_step_measure_optimal_transport_geodesic_convexity
    {M : Type} `{FrameworkStruct_measure_optimal_transport_geodesic_convexity M}
    (d : ContextData_measure_optimal_transport_geodesic_convexity) :
    ((fun P : Prop => (fun u : unit => P) tt) (cost (primary_map_measure_optimal_transport_geodesic_convexity d) <=
      cost (midpoint (push (left_measure_optimal_transport_geodesic_convexity d))
            (right_measure_optimal_transport_geodesic_convexity d)) +
      entropy (push (left_measure_optimal_transport_geodesic_convexity d)) +
      entropy (right_measure_optimal_transport_geodesic_convexity d) /\
    mass (push (left_measure_optimal_transport_geodesic_convexity d)) <=
      mass (left_measure_optimal_transport_geodesic_convexity d))).
Proof.
  assert (hCostRaw :
      cost (geodesic (push (left_measure_optimal_transport_geodesic_convexity d))
            (right_measure_optimal_transport_geodesic_convexity d)) <=
      cost (midpoint (push (left_measure_optimal_transport_geodesic_convexity d))
            (right_measure_optimal_transport_geodesic_convexity d)) +
      entropy (push (left_measure_optimal_transport_geodesic_convexity d)) +
      entropy (right_measure_optimal_transport_geodesic_convexity d)).
  { apply geodesic_cost. }
  assert (hCost :
      cost (primary_map_measure_optimal_transport_geodesic_convexity d) <=
      cost (midpoint (push (left_measure_optimal_transport_geodesic_convexity d))
            (right_measure_optimal_transport_geodesic_convexity d)) +
      entropy (push (left_measure_optimal_transport_geodesic_convexity d)) +
      entropy (right_measure_optimal_transport_geodesic_convexity d)).
  {
    unfold primary_map_measure_optimal_transport_geodesic_convexity.
    exact hCostRaw.
  }
  assert (hMass :
      mass (push (left_measure_optimal_transport_geodesic_convexity d)) <=
      mass (left_measure_optimal_transport_geodesic_convexity d)).
  { apply push_mass. }
  split.
  - exact hCost.
  - exact hMass.
Qed.

Lemma factorization_step_measure_optimal_transport_geodesic_convexity
    {M : Type} `{FrameworkStruct_measure_optimal_transport_geodesic_convexity M}
    (d : ContextData_measure_optimal_transport_geodesic_convexity) :
    ((fun P : Prop => (fun u : unit => P) tt) (cost (midpoint (push (left_measure_optimal_transport_geodesic_convexity d))
          (right_measure_optimal_transport_geodesic_convexity d)) <=
      cost (push (left_measure_optimal_transport_geodesic_convexity d)) +
      cost (right_measure_optimal_transport_geodesic_convexity d) /\
    cost (push (left_measure_optimal_transport_geodesic_convexity d)) <=
      cost (left_measure_optimal_transport_geodesic_convexity d) +
      entropy (left_measure_optimal_transport_geodesic_convexity d))).
Proof.
  assert (hMid :
      cost (midpoint (push (left_measure_optimal_transport_geodesic_convexity d))
            (right_measure_optimal_transport_geodesic_convexity d)) <=
      cost (push (left_measure_optimal_transport_geodesic_convexity d)) +
      cost (right_measure_optimal_transport_geodesic_convexity d)).
  { apply midpoint_cost. }
  assert (hPush :
      cost (push (left_measure_optimal_transport_geodesic_convexity d)) <=
      cost (left_measure_optimal_transport_geodesic_convexity d) +
      entropy (left_measure_optimal_transport_geodesic_convexity d)).
  { apply push_cost. }
  split.
  - exact hMid.
  - exact hPush.
Qed.

Lemma comparison_step_measure_optimal_transport_geodesic_convexity
    {M : Type} `{FrameworkStruct_measure_optimal_transport_geodesic_convexity M}
    (d : ContextData_measure_optimal_transport_geodesic_convexity) :
    ((fun P : Prop => (fun u : unit => P) tt) (tertiary_map_measure_optimal_transport_geodesic_convexity d /\
    exists c : nat,
      c = secondary_map_measure_optimal_transport_geodesic_convexity d /\
      cost (primary_map_measure_optimal_transport_geodesic_convexity d) <= c)).
Proof.
  assert (hSec :
      cost (primary_map_measure_optimal_transport_geodesic_convexity d) <=
      secondary_map_measure_optimal_transport_geodesic_convexity d).
  {
    unfold secondary_map_measure_optimal_transport_geodesic_convexity.
    apply le_add_right_nat.
  }
  split.
  - exact hSec.
  - exists (secondary_map_measure_optimal_transport_geodesic_convexity d).
    split.
    + reflexivity.
    + exact hSec.
Qed.

Lemma transport_step_measure_optimal_transport_geodesic_convexity
    {M : Type} `{FrameworkStruct_measure_optimal_transport_geodesic_convexity M}
    (d : ContextData_measure_optimal_transport_geodesic_convexity) :
    ((fun P : Prop => (fun u : unit => P) tt) (cost (primary_map_measure_optimal_transport_geodesic_convexity d) <=
      cost (midpoint (push (left_measure_optimal_transport_geodesic_convexity d))
            (right_measure_optimal_transport_geodesic_convexity d)) +
      entropy (push (left_measure_optimal_transport_geodesic_convexity d)) +
      entropy (right_measure_optimal_transport_geodesic_convexity d) /\
    cost (push (left_measure_optimal_transport_geodesic_convexity d)) <=
      cost (left_measure_optimal_transport_geodesic_convexity d) +
      entropy (left_measure_optimal_transport_geodesic_convexity d))).
Proof.
  destruct (stability_step_measure_optimal_transport_geodesic_convexity d)
    as [hCost hMass].
  destruct (factorization_step_measure_optimal_transport_geodesic_convexity d)
    as [hMid hPush].
  assert (_hMassKeep :
      mass (push (left_measure_optimal_transport_geodesic_convexity d)) <=
      mass (left_measure_optimal_transport_geodesic_convexity d)).
  { exact hMass. }
  assert (_hMidKeep :
      cost (midpoint (push (left_measure_optimal_transport_geodesic_convexity d))
        (right_measure_optimal_transport_geodesic_convexity d)) <=
      cost (push (left_measure_optimal_transport_geodesic_convexity d)) +
      cost (right_measure_optimal_transport_geodesic_convexity d)).
  { exact hMid. }
  split.
  - exact hCost.
  - exact hPush.
Qed.

Lemma coherence_step_measure_optimal_transport_geodesic_convexity
    {M : Type} `{FrameworkStruct_measure_optimal_transport_geodesic_convexity M}
    (d : ContextData_measure_optimal_transport_geodesic_convexity) :
    ((fun P : Prop => (fun u : unit => P) tt) (exists y : M,
      y = push (left_measure_optimal_transport_geodesic_convexity d) /\
      mass y <= mass (left_measure_optimal_transport_geodesic_convexity d) /\
      (forall z : M, z = y -> mass z <= mass (left_measure_optimal_transport_geodesic_convexity d)))).
Proof.
  destruct (witness_push (left_measure_optimal_transport_geodesic_convexity d))
    as [y [hyEq hyMass]].
  assert (hAll : forall z : M, z = y -> mass z <= mass (left_measure_optimal_transport_geodesic_convexity d)).
  {
    intros z hz.
    rewrite hz.
    exact hyMass.
  }
  exists y.
  repeat split; try assumption.
Qed.

Lemma iteration_step_measure_optimal_transport_geodesic_convexity
    {M : Type} `{FrameworkStruct_measure_optimal_transport_geodesic_convexity M}
    (d : ContextData_measure_optimal_transport_geodesic_convexity) :
    ((fun P : Prop => (fun u : unit => P) tt) (exists y : M,
      y = primary_map_measure_optimal_transport_geodesic_convexity d /\
      cost y <= secondary_map_measure_optimal_transport_geodesic_convexity d /\
      exists w : M,
        w = push (left_measure_optimal_transport_geodesic_convexity d) /\
        mass w <= mass (left_measure_optimal_transport_geodesic_convexity d))).
Proof.
  set (y := primary_map_measure_optimal_transport_geodesic_convexity d).
  assert (hyEq : y = primary_map_measure_optimal_transport_geodesic_convexity d).
  { reflexivity. }
  assert (hCost : cost y <= secondary_map_measure_optimal_transport_geodesic_convexity d).
  {
    rewrite hyEq.
    destruct (proj2 (comparison_step_measure_optimal_transport_geodesic_convexity d))
      as [c [hcEq hc]].
    rewrite <- hcEq.
    exact hc.
  }
  assert (hMass :
      mass (push (left_measure_optimal_transport_geodesic_convexity d)) <=
      mass (left_measure_optimal_transport_geodesic_convexity d)).
  { exact (proj2 (stability_step_measure_optimal_transport_geodesic_convexity d)). }
  exists y.
  split.
  - exact hyEq.
  - split.
    + exact hCost.
    + exists (push (left_measure_optimal_transport_geodesic_convexity d)).
      split.
      * reflexivity.
      * exact hMass.
Qed.

Lemma main_result_measure_optimal_transport_geodesic_convexity
    {M : Type} `{FrameworkStruct_measure_optimal_transport_geodesic_convexity M}
    (d : ContextData_measure_optimal_transport_geodesic_convexity) :
    ((fun P : Prop => (fun u : unit => P) tt) (exists y : M,
      y = primary_map_measure_optimal_transport_geodesic_convexity d /\
      cost y <= secondary_map_measure_optimal_transport_geodesic_convexity d /\
      cost (push (left_measure_optimal_transport_geodesic_convexity d)) <=
      cost (left_measure_optimal_transport_geodesic_convexity d) +
      entropy (left_measure_optimal_transport_geodesic_convexity d))).
Proof.
  destruct (iteration_step_measure_optimal_transport_geodesic_convexity d)
    as [y [hyEq [hCost hwitness]]].
  destruct hwitness as [w [hwEq hMass]].
  assert (hPush :
      cost (push (left_measure_optimal_transport_geodesic_convexity d)) <=
      cost (left_measure_optimal_transport_geodesic_convexity d) +
      entropy (left_measure_optimal_transport_geodesic_convexity d)).
  { exact (proj2 (factorization_step_measure_optimal_transport_geodesic_convexity d)). }
  assert (_hMassKeep :
      mass w <=
      mass (left_measure_optimal_transport_geodesic_convexity d)).
  { exact hMass. }
  assert (_hwKeep : w = push (left_measure_optimal_transport_geodesic_convexity d)).
  { exact hwEq. }
  assert (_hEL : entropy (left_measure_optimal_transport_geodesic_convexity d) <= budget_measure_optimal_transport_geodesic_convexity d).
  { exact (entropy_cap_left_measure_optimal_transport_geodesic_convexity d). }
  assert (_hER : entropy (right_measure_optimal_transport_geodesic_convexity d) <= budget_measure_optimal_transport_geodesic_convexity d).
  { exact (entropy_cap_right_measure_optimal_transport_geodesic_convexity d). }
  assert (_hCB : cost (left_measure_optimal_transport_geodesic_convexity d) <=
      cost (right_measure_optimal_transport_geodesic_convexity d) + budget_measure_optimal_transport_geodesic_convexity d).
  { exact (cost_bridge_measure_optimal_transport_geodesic_convexity d). }
  exists y.
  repeat split; try assumption.
Qed.
