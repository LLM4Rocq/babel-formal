(**
BENCHMARK_ID: TINY_MATHLIB_BATCH07_ANALYSIS_HARMONIC_MEASURE_BOUNDARY_HARNACK_LIKE
PAIR_STEM: analysis_harmonic_measure_boundary_harnack_like
MATH_DOMAIN: Analysis
SOURCE_MATHLIB: Mathlib/Analysis/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
**)

Set Universe Polymorphism.
Set Implicit Arguments.

Class FrameworkStruct_analysis_harmonic_measure_boundary_harnack (E : Type) := {
  mass : E -> nat;
  boundary : E -> nat;
  interior : E -> nat;
  harmonicize : E -> E;
  trace : E -> E;
  le_trans_nat : forall a b c : nat, a <= b -> b <= c -> a <= c;
  add_le_add_left_nat : forall a b c : nat, a <= b -> c + a <= c + b;
  add_le_add_right_nat : forall a b c : nat, a <= b -> a + c <= b + c;
  le_add_left_nat : forall a b : nat, b <= a + b;
  le_add_right_nat : forall a b : nat, a <= a + b;
  harmonic_mass : forall x : E, mass (harmonicize x) <= mass x + interior x;
  boundary_trace : forall x : E, boundary (trace x) <= boundary x;
  harnack_step : forall x : E, interior (harmonicize x) <= boundary x + interior x;
  trace_mass : forall x : E, mass (trace x) <= mass x + boundary x;
  harnack_compare : forall x : E, mass (harmonicize x) <= boundary x + boundary x;
  witness_trace : forall x : E, exists y : E, y = trace x /\ boundary y <= boundary x
}.

Record ContextData_analysis_harmonic_measure_boundary_harnack
    (E : Type) `{FrameworkStruct_analysis_harmonic_measure_boundary_harnack E} := {
  left_analysis_harmonic_measure_boundary_harnack : E;
  right_analysis_harmonic_measure_boundary_harnack : E;
  window_analysis_harmonic_measure_boundary_harnack : nat;
  left_to_right_boundary_analysis_harmonic_measure_boundary_harnack :
    boundary left_analysis_harmonic_measure_boundary_harnack <=
      boundary right_analysis_harmonic_measure_boundary_harnack + window_analysis_harmonic_measure_boundary_harnack;
  right_mass_cap_analysis_harmonic_measure_boundary_harnack :
    mass right_analysis_harmonic_measure_boundary_harnack <=
      window_analysis_harmonic_measure_boundary_harnack + window_analysis_harmonic_measure_boundary_harnack
}.

Definition primary_map_analysis_harmonic_measure_boundary_harnack
    {E : Type} `{FrameworkStruct_analysis_harmonic_measure_boundary_harnack E}
    (d : ContextData_analysis_harmonic_measure_boundary_harnack) : E :=
  harmonicize (trace (left_analysis_harmonic_measure_boundary_harnack d)).

Definition secondary_map_analysis_harmonic_measure_boundary_harnack
    {E : Type} `{FrameworkStruct_analysis_harmonic_measure_boundary_harnack E}
    (d : ContextData_analysis_harmonic_measure_boundary_harnack) : nat :=
  mass (primary_map_analysis_harmonic_measure_boundary_harnack d) +
    window_analysis_harmonic_measure_boundary_harnack d.

Definition tertiary_map_analysis_harmonic_measure_boundary_harnack
    {E : Type} `{FrameworkStruct_analysis_harmonic_measure_boundary_harnack E}
    (d : ContextData_analysis_harmonic_measure_boundary_harnack) : Prop :=
  mass (primary_map_analysis_harmonic_measure_boundary_harnack d) <=
    secondary_map_analysis_harmonic_measure_boundary_harnack d.

Lemma stability_step_analysis_harmonic_measure_boundary_harnack
    {E : Type} `{FrameworkStruct_analysis_harmonic_measure_boundary_harnack E}
    (d : ContextData_analysis_harmonic_measure_boundary_harnack) :
    ((mass (primary_map_analysis_harmonic_measure_boundary_harnack d) <=
      (mass (left_analysis_harmonic_measure_boundary_harnack d) +
       boundary (left_analysis_harmonic_measure_boundary_harnack d)) +
      interior (trace (left_analysis_harmonic_measure_boundary_harnack d)) /\
    boundary (trace (left_analysis_harmonic_measure_boundary_harnack d)) <=
      boundary (left_analysis_harmonic_measure_boundary_harnack d)) \/ False).
Proof.
  left.
  assert (hMassH :
      mass (harmonicize (trace (left_analysis_harmonic_measure_boundary_harnack d))) <=
      mass (trace (left_analysis_harmonic_measure_boundary_harnack d)) +
      interior (trace (left_analysis_harmonic_measure_boundary_harnack d))).
  { apply harmonic_mass. }
  assert (hMassTrace :
      mass (trace (left_analysis_harmonic_measure_boundary_harnack d)) <=
      mass (left_analysis_harmonic_measure_boundary_harnack d) +
      boundary (left_analysis_harmonic_measure_boundary_harnack d)).
  { apply trace_mass. }
  assert (hLift :
      mass (trace (left_analysis_harmonic_measure_boundary_harnack d)) +
      interior (trace (left_analysis_harmonic_measure_boundary_harnack d)) <=
      (mass (left_analysis_harmonic_measure_boundary_harnack d) +
       boundary (left_analysis_harmonic_measure_boundary_harnack d)) +
      interior (trace (left_analysis_harmonic_measure_boundary_harnack d))).
  {
    apply add_le_add_right_nat.
    exact hMassTrace.
  }
  assert (hMass :
      mass (primary_map_analysis_harmonic_measure_boundary_harnack d) <=
      (mass (left_analysis_harmonic_measure_boundary_harnack d) +
       boundary (left_analysis_harmonic_measure_boundary_harnack d)) +
      interior (trace (left_analysis_harmonic_measure_boundary_harnack d))).
  {
    unfold primary_map_analysis_harmonic_measure_boundary_harnack.
    apply le_trans_nat with
      (b := mass (trace (left_analysis_harmonic_measure_boundary_harnack d)) +
            interior (trace (left_analysis_harmonic_measure_boundary_harnack d))).
    - exact hMassH.
    - exact hLift.
  }
  assert (hBoundary :
      boundary (trace (left_analysis_harmonic_measure_boundary_harnack d)) <=
      boundary (left_analysis_harmonic_measure_boundary_harnack d)).
  { apply boundary_trace. }
  split.
  - exact hMass.
  - exact hBoundary.
Qed.

Lemma factorization_step_analysis_harmonic_measure_boundary_harnack
    {E : Type} `{FrameworkStruct_analysis_harmonic_measure_boundary_harnack E}
    (d : ContextData_analysis_harmonic_measure_boundary_harnack) :
    ((interior (primary_map_analysis_harmonic_measure_boundary_harnack d) <=
      boundary (trace (left_analysis_harmonic_measure_boundary_harnack d)) +
      interior (trace (left_analysis_harmonic_measure_boundary_harnack d)) /\
    mass (primary_map_analysis_harmonic_measure_boundary_harnack d) <=
      boundary (trace (left_analysis_harmonic_measure_boundary_harnack d)) +
      boundary (trace (left_analysis_harmonic_measure_boundary_harnack d))) \/ False).
Proof.
  left.
  assert (hIntRaw :
      interior (harmonicize (trace (left_analysis_harmonic_measure_boundary_harnack d))) <=
      boundary (trace (left_analysis_harmonic_measure_boundary_harnack d)) +
      interior (trace (left_analysis_harmonic_measure_boundary_harnack d))).
  { apply harnack_step. }
  assert (hMassRaw :
      mass (harmonicize (trace (left_analysis_harmonic_measure_boundary_harnack d))) <=
      boundary (trace (left_analysis_harmonic_measure_boundary_harnack d)) +
      boundary (trace (left_analysis_harmonic_measure_boundary_harnack d))).
  { apply harnack_compare. }
  split.
  - unfold primary_map_analysis_harmonic_measure_boundary_harnack.
    exact hIntRaw.
  - unfold primary_map_analysis_harmonic_measure_boundary_harnack.
    exact hMassRaw.
Qed.

Lemma comparison_step_analysis_harmonic_measure_boundary_harnack
    {E : Type} `{FrameworkStruct_analysis_harmonic_measure_boundary_harnack E}
    (d : ContextData_analysis_harmonic_measure_boundary_harnack) :
    ((exists m : nat,
      m = secondary_map_analysis_harmonic_measure_boundary_harnack d /\
      tertiary_map_analysis_harmonic_measure_boundary_harnack d) \/ False).
Proof.
  left.
  assert (hSec :
      mass (primary_map_analysis_harmonic_measure_boundary_harnack d) <=
      secondary_map_analysis_harmonic_measure_boundary_harnack d).
  {
    unfold secondary_map_analysis_harmonic_measure_boundary_harnack.
    apply le_add_right_nat.
  }
  exists (secondary_map_analysis_harmonic_measure_boundary_harnack d).
  split.
  - reflexivity.
  - exact hSec.
Qed.

Lemma transport_step_analysis_harmonic_measure_boundary_harnack
    {E : Type} `{FrameworkStruct_analysis_harmonic_measure_boundary_harnack E}
    (d : ContextData_analysis_harmonic_measure_boundary_harnack) :
    ((mass (primary_map_analysis_harmonic_measure_boundary_harnack d) <=
      (mass (left_analysis_harmonic_measure_boundary_harnack d) +
       boundary (left_analysis_harmonic_measure_boundary_harnack d)) +
      interior (trace (left_analysis_harmonic_measure_boundary_harnack d)) /\
    boundary (trace (left_analysis_harmonic_measure_boundary_harnack d)) <=
      boundary (right_analysis_harmonic_measure_boundary_harnack d) +
      window_analysis_harmonic_measure_boundary_harnack d) \/ False).
Proof.
  left.
  destruct (stability_step_analysis_harmonic_measure_boundary_harnack d)
    as [[hMass hBoundaryLeft] | hFalse].
  - assert (hBoundaryRight :
        boundary (trace (left_analysis_harmonic_measure_boundary_harnack d)) <=
        boundary (right_analysis_harmonic_measure_boundary_harnack d) +
        window_analysis_harmonic_measure_boundary_harnack d).
    {
      apply le_trans_nat with
        (b := boundary (left_analysis_harmonic_measure_boundary_harnack d)).
      - exact hBoundaryLeft.
      - exact (left_to_right_boundary_analysis_harmonic_measure_boundary_harnack d).
    }
    split.
    + exact hMass.
    + exact hBoundaryRight.
  - exfalso. exact hFalse.
Qed.

Lemma coherence_step_analysis_harmonic_measure_boundary_harnack
    {E : Type} `{FrameworkStruct_analysis_harmonic_measure_boundary_harnack E}
    (d : ContextData_analysis_harmonic_measure_boundary_harnack) :
    ((forall z : E,
      z = trace (left_analysis_harmonic_measure_boundary_harnack d) ->
      boundary z <= boundary (left_analysis_harmonic_measure_boundary_harnack d)) \/ False).
Proof.
  left.
  intros z hz.
  rewrite hz.
  apply boundary_trace.
Qed.

Lemma iteration_step_analysis_harmonic_measure_boundary_harnack
    {E : Type} `{FrameworkStruct_analysis_harmonic_measure_boundary_harnack E}
    (d : ContextData_analysis_harmonic_measure_boundary_harnack) :
    ((forall y : E,
      y = primary_map_analysis_harmonic_measure_boundary_harnack d ->
      mass y <= secondary_map_analysis_harmonic_measure_boundary_harnack d ->
      boundary (trace (left_analysis_harmonic_measure_boundary_harnack d)) <=
      boundary (left_analysis_harmonic_measure_boundary_harnack d)) \/ False).
Proof.
  left.
  intros y hyEq hyMass.
  assert (hBoundary :
      boundary (trace (left_analysis_harmonic_measure_boundary_harnack d)) <=
      boundary (left_analysis_harmonic_measure_boundary_harnack d)).
  {
    destruct (stability_step_analysis_harmonic_measure_boundary_harnack d)
      as [[_ hBoundaryCore] | hFalse].
    - exact hBoundaryCore.
    - exfalso. exact hFalse.
  }
  assert (_hMassKeep : mass y <= secondary_map_analysis_harmonic_measure_boundary_harnack d).
  { exact hyMass. }
  exact hBoundary.
Qed.

Lemma main_result_analysis_harmonic_measure_boundary_harnack
    {E : Type} `{FrameworkStruct_analysis_harmonic_measure_boundary_harnack E}
    (d : ContextData_analysis_harmonic_measure_boundary_harnack) :
    ((exists y : E,
      y = primary_map_analysis_harmonic_measure_boundary_harnack d /\
      mass y <= secondary_map_analysis_harmonic_measure_boundary_harnack d /\
      boundary (trace (left_analysis_harmonic_measure_boundary_harnack d)) <=
      boundary (right_analysis_harmonic_measure_boundary_harnack d) +
      window_analysis_harmonic_measure_boundary_harnack d) \/ False).
Proof.
  left.
  set (y := primary_map_analysis_harmonic_measure_boundary_harnack d).
  assert (hyEq : y = primary_map_analysis_harmonic_measure_boundary_harnack d).
  { reflexivity. }
  destruct (comparison_step_analysis_harmonic_measure_boundary_harnack d)
    as [[m [hmEq hTer]] | hCompFalse].
  - assert (hMass : mass y <= secondary_map_analysis_harmonic_measure_boundary_harnack d).
    { rewrite hyEq. exact hTer. }
    assert (hBoundaryLeft :
        boundary (trace (left_analysis_harmonic_measure_boundary_harnack d)) <=
        boundary (left_analysis_harmonic_measure_boundary_harnack d)).
    {
      destruct (iteration_step_analysis_harmonic_measure_boundary_harnack d)
        as [hIterCore | hIterFalse].
      - exact (hIterCore y hyEq hMass).
      - exfalso. exact hIterFalse.
    }
    assert (hBoundaryRight :
        boundary (trace (left_analysis_harmonic_measure_boundary_harnack d)) <=
        boundary (right_analysis_harmonic_measure_boundary_harnack d) +
        window_analysis_harmonic_measure_boundary_harnack d).
    {
      apply le_trans_nat with
        (b := boundary (left_analysis_harmonic_measure_boundary_harnack d)).
      - exact hBoundaryLeft.
      - exact (left_to_right_boundary_analysis_harmonic_measure_boundary_harnack d).
    }
    assert (_hRightMassKeep :
        mass (right_analysis_harmonic_measure_boundary_harnack d) <=
        window_analysis_harmonic_measure_boundary_harnack d +
        window_analysis_harmonic_measure_boundary_harnack d).
    { exact (right_mass_cap_analysis_harmonic_measure_boundary_harnack d). }
    assert (_hmKeep : m = secondary_map_analysis_harmonic_measure_boundary_harnack d).
    { exact hmEq. }
    exists y.
    repeat split; try assumption.
  - exfalso. exact hCompFalse.
Qed.
