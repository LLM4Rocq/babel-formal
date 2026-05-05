(**
BENCHMARK_ID: TINY_MATHLIB_BATCH07_MEASURE_DIMENSION_FREE_CONCENTRATION_LIKE
PAIR_STEM: measure_dimension_free_concentration_like
MATH_DOMAIN: Measure Theory
SOURCE_MATHLIB: Mathlib/MeasureTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
**)

Set Universe Polymorphism.
Set Implicit Arguments.

Class FrameworkStruct_measure_dimension_free_concentration (X : Type) := {
  radius : X -> nat;
  variance : X -> nat;
  deviation : X -> nat;
  project : X -> X;
  average : X -> X;
  le_trans_nat : forall a b c : nat, a <= b -> b <= c -> a <= c;
  add_le_add_left_nat : forall a b c : nat, a <= b -> c + a <= c + b;
  add_le_add_right_nat : forall a b c : nat, a <= b -> a + c <= b + c;
  le_add_left_nat : forall a b : nat, b <= a + b;
  le_add_right_nat : forall a b : nat, a <= a + b;
  project_radius : forall x : X, radius (project x) <= radius x;
  project_variance : forall x : X, variance (project x) <= variance x + deviation x;
  concentration_step : forall x : X, radius (average (project x)) <= radius x + variance x;
  tail_bound : forall x : X, variance (average (project x)) <= variance x + deviation x;
  average_deviation : forall x : X, deviation (average x) <= deviation x + deviation x;
  witness_project : forall x : X, exists y : X, y = project x /\ radius y <= radius x
}.

Record ContextData_measure_dimension_free_concentration
    (X : Type) `{FrameworkStruct_measure_dimension_free_concentration X} := {
  base_measure_dimension_free_concentration : X;
  ref_measure_dimension_free_concentration : X;
  budget_measure_dimension_free_concentration : nat;
  variance_cap_measure_dimension_free_concentration :
    variance base_measure_dimension_free_concentration <= budget_measure_dimension_free_concentration;
  deviation_cap_measure_dimension_free_concentration :
    deviation base_measure_dimension_free_concentration <= budget_measure_dimension_free_concentration;
  radius_bridge_measure_dimension_free_concentration :
    radius base_measure_dimension_free_concentration <=
      radius ref_measure_dimension_free_concentration + budget_measure_dimension_free_concentration
}.

Definition primary_map_measure_dimension_free_concentration
    {X : Type} `{FrameworkStruct_measure_dimension_free_concentration X}
    (d : ContextData_measure_dimension_free_concentration) : X :=
  average (project (base_measure_dimension_free_concentration d)).

Definition secondary_map_measure_dimension_free_concentration
    {X : Type} `{FrameworkStruct_measure_dimension_free_concentration X}
    (d : ContextData_measure_dimension_free_concentration) : nat :=
  radius (primary_map_measure_dimension_free_concentration d) +
    budget_measure_dimension_free_concentration d.

Definition tertiary_map_measure_dimension_free_concentration
    {X : Type} `{FrameworkStruct_measure_dimension_free_concentration X}
    (d : ContextData_measure_dimension_free_concentration) : Prop :=
  radius (primary_map_measure_dimension_free_concentration d) <=
    secondary_map_measure_dimension_free_concentration d.

Lemma stability_step_measure_dimension_free_concentration
    {X : Type} `{FrameworkStruct_measure_dimension_free_concentration X}
    (d : ContextData_measure_dimension_free_concentration) :
    ((fun P : Prop => (fun f : Prop -> Prop => f P) (fun R : Prop => R)) ((radius (primary_map_measure_dimension_free_concentration d) <=
      radius (base_measure_dimension_free_concentration d) +
      variance (base_measure_dimension_free_concentration d) /\
    variance (primary_map_measure_dimension_free_concentration d) <=
      variance (base_measure_dimension_free_concentration d) +
      deviation (base_measure_dimension_free_concentration d)) /\
    variance (project (base_measure_dimension_free_concentration d)) <=
      variance (base_measure_dimension_free_concentration d) +
      deviation (base_measure_dimension_free_concentration d))).
Proof.
  assert (hRadiusRaw :
      radius (average (project (base_measure_dimension_free_concentration d))) <=
      radius (base_measure_dimension_free_concentration d) +
      variance (base_measure_dimension_free_concentration d)).
  { apply concentration_step. }
  assert (hVarRaw :
      variance (average (project (base_measure_dimension_free_concentration d))) <=
      variance (base_measure_dimension_free_concentration d) +
      deviation (base_measure_dimension_free_concentration d)).
  { apply tail_bound. }
  assert (hProjVar :
      variance (project (base_measure_dimension_free_concentration d)) <=
      variance (base_measure_dimension_free_concentration d) +
      deviation (base_measure_dimension_free_concentration d)).
  { apply project_variance. }
  split.
  - split.
    + unfold primary_map_measure_dimension_free_concentration.
      exact hRadiusRaw.
    + unfold primary_map_measure_dimension_free_concentration.
      exact hVarRaw.
  - exact hProjVar.
Qed.

Lemma factorization_step_measure_dimension_free_concentration
    {X : Type} `{FrameworkStruct_measure_dimension_free_concentration X}
    (d : ContextData_measure_dimension_free_concentration) :
    ((fun P : Prop => (fun f : Prop -> Prop => f P) (fun R : Prop => R)) (radius (project (base_measure_dimension_free_concentration d)) <=
      radius (base_measure_dimension_free_concentration d) /\
    variance (project (base_measure_dimension_free_concentration d)) <=
      variance (base_measure_dimension_free_concentration d) +
      deviation (base_measure_dimension_free_concentration d))).
Proof.
  assert (hRad : radius (project (base_measure_dimension_free_concentration d)) <= radius (base_measure_dimension_free_concentration d)).
  { apply project_radius. }
  assert (hVar : variance (project (base_measure_dimension_free_concentration d)) <=
      variance (base_measure_dimension_free_concentration d) + deviation (base_measure_dimension_free_concentration d)).
  { apply project_variance. }
  split.
  - exact hRad.
  - exact hVar.
Qed.

Lemma comparison_step_measure_dimension_free_concentration
    {X : Type} `{FrameworkStruct_measure_dimension_free_concentration X}
    (d : ContextData_measure_dimension_free_concentration) :
    ((fun P : Prop => (fun f : Prop -> Prop => f P) (fun R : Prop => R)) (secondary_map_measure_dimension_free_concentration d =
      radius (primary_map_measure_dimension_free_concentration d) +
      budget_measure_dimension_free_concentration d /\
    tertiary_map_measure_dimension_free_concentration d)).
Proof.
  assert (hSec : radius (primary_map_measure_dimension_free_concentration d) <= secondary_map_measure_dimension_free_concentration d).
  {
    unfold secondary_map_measure_dimension_free_concentration.
    apply le_add_right_nat.
  }
  split.
  - reflexivity.
  - exact hSec.
Qed.

Lemma transport_step_measure_dimension_free_concentration
    {X : Type} `{FrameworkStruct_measure_dimension_free_concentration X}
    (d : ContextData_measure_dimension_free_concentration) :
    ((fun P : Prop => (fun f : Prop -> Prop => f P) (fun R : Prop => R)) (radius (primary_map_measure_dimension_free_concentration d) <=
      radius (ref_measure_dimension_free_concentration d) +
      budget_measure_dimension_free_concentration d +
      variance (base_measure_dimension_free_concentration d) /\
    deviation (average (project (base_measure_dimension_free_concentration d))) <=
      deviation (project (base_measure_dimension_free_concentration d)) +
      deviation (project (base_measure_dimension_free_concentration d)))).
Proof.
  destruct (stability_step_measure_dimension_free_concentration d) as [[hRadBase hVarBase] hProjVar].
  assert (hBridge :
      radius (base_measure_dimension_free_concentration d) + variance (base_measure_dimension_free_concentration d) <=
      (radius (ref_measure_dimension_free_concentration d) + budget_measure_dimension_free_concentration d) +
      variance (base_measure_dimension_free_concentration d)).
  {
    apply add_le_add_right_nat.
    exact (radius_bridge_measure_dimension_free_concentration d).
  }
  assert (hFirst :
      radius (primary_map_measure_dimension_free_concentration d) <=
      radius (ref_measure_dimension_free_concentration d) +
      budget_measure_dimension_free_concentration d +
      variance (base_measure_dimension_free_concentration d)).
  {
    apply le_trans_nat with
      (b := radius (base_measure_dimension_free_concentration d) + variance (base_measure_dimension_free_concentration d)).
    - exact hRadBase.
    - exact hBridge.
  }
  assert (hSecond :
      deviation (average (project (base_measure_dimension_free_concentration d))) <=
      deviation (project (base_measure_dimension_free_concentration d)) +
      deviation (project (base_measure_dimension_free_concentration d))).
  { apply average_deviation. }
  assert (_hVarKeep :
      variance (primary_map_measure_dimension_free_concentration d) <=
      variance (base_measure_dimension_free_concentration d) +
      deviation (base_measure_dimension_free_concentration d)).
  { exact hVarBase. }
  assert (_hProjKeep :
      variance (project (base_measure_dimension_free_concentration d)) <=
      variance (base_measure_dimension_free_concentration d) +
      deviation (base_measure_dimension_free_concentration d)).
  { exact hProjVar. }
  split.
  - exact hFirst.
  - exact hSecond.
Qed.

Lemma coherence_step_measure_dimension_free_concentration
    {X : Type} `{FrameworkStruct_measure_dimension_free_concentration X}
    (d : ContextData_measure_dimension_free_concentration) :
    ((fun P : Prop => (fun f : Prop -> Prop => f P) (fun R : Prop => R)) (tertiary_map_measure_dimension_free_concentration d ->
    exists y : X,
      y = project (base_measure_dimension_free_concentration d) /\
      radius y <= radius (base_measure_dimension_free_concentration d) /\
      variance (project (base_measure_dimension_free_concentration d)) <=
        variance (base_measure_dimension_free_concentration d) +
        deviation (base_measure_dimension_free_concentration d))).
Proof.
  intro hTer.
  destruct (witness_project (base_measure_dimension_free_concentration d)) as [y [hyEq hyRad]].
  assert (hProjVar :
      variance (project (base_measure_dimension_free_concentration d)) <=
      variance (base_measure_dimension_free_concentration d) +
      deviation (base_measure_dimension_free_concentration d)).
  { apply project_variance. }
  assert (_hTerKeep : tertiary_map_measure_dimension_free_concentration d).
  { exact hTer. }
  exists y.
  repeat split; try assumption.
Qed.

Lemma iteration_step_measure_dimension_free_concentration
    {X : Type} `{FrameworkStruct_measure_dimension_free_concentration X}
    (d : ContextData_measure_dimension_free_concentration) :
    ((fun P : Prop => (fun f : Prop -> Prop => f P) (fun R : Prop => R)) (exists y : X,
      y = primary_map_measure_dimension_free_concentration d /\
      radius y <= radius (base_measure_dimension_free_concentration d) + variance (base_measure_dimension_free_concentration d) /\
      variance y <= variance (base_measure_dimension_free_concentration d) + deviation (base_measure_dimension_free_concentration d))).
Proof.
  set (y := primary_map_measure_dimension_free_concentration d).
  assert (hyEq : y = primary_map_measure_dimension_free_concentration d).
  { reflexivity. }
  destruct (stability_step_measure_dimension_free_concentration d) as [[hRad hVar] hProjVar].
  assert (_hProjKeep :
      variance (project (base_measure_dimension_free_concentration d)) <=
      variance (base_measure_dimension_free_concentration d) + deviation (base_measure_dimension_free_concentration d)).
  { exact hProjVar. }
  exists y.
  split.
  - exact hyEq.
  - split.
    + rewrite hyEq. exact hRad.
    + rewrite hyEq. exact hVar.
Qed.

Lemma main_result_measure_dimension_free_concentration
    {X : Type} `{FrameworkStruct_measure_dimension_free_concentration X}
    (d : ContextData_measure_dimension_free_concentration) :
    ((fun P : Prop => (fun f : Prop -> Prop => f P) (fun R : Prop => R)) (exists y : X,
      y = primary_map_measure_dimension_free_concentration d /\
      radius y <=
      radius (ref_measure_dimension_free_concentration d) +
      budget_measure_dimension_free_concentration d +
      variance (base_measure_dimension_free_concentration d) /\
      tertiary_map_measure_dimension_free_concentration d)).
Proof.
  destruct (iteration_step_measure_dimension_free_concentration d)
    as [y [hyEq [hyRad hyVar]]].
  assert (hTrans :
      radius (primary_map_measure_dimension_free_concentration d) <=
      radius (ref_measure_dimension_free_concentration d) +
      budget_measure_dimension_free_concentration d +
      variance (base_measure_dimension_free_concentration d)).
  { exact (proj1 (transport_step_measure_dimension_free_concentration d)). }
  assert (hFinal :
      radius y <=
      radius (ref_measure_dimension_free_concentration d) +
      budget_measure_dimension_free_concentration d +
      variance (base_measure_dimension_free_concentration d)).
  {
    rewrite hyEq.
    exact hTrans.
  }
  assert (hTer : tertiary_map_measure_dimension_free_concentration d).
  { exact (proj2 (comparison_step_measure_dimension_free_concentration d)). }
  assert (_hVarKeep :
      variance y <= variance (base_measure_dimension_free_concentration d) + deviation (base_measure_dimension_free_concentration d)).
  { exact hyVar. }
  exists y.
  repeat split; try assumption.
Qed.
