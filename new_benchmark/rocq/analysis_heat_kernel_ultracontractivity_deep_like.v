(**
BENCHMARK_ID: TINY_MATHLIB_BATCH07_ANALYSIS_HEAT_KERNEL_ULTRACONTRACTIVITY_DEEP_LIKE
PAIR_STEM: analysis_heat_kernel_ultracontractivity_deep_like
MATH_DOMAIN: Analysis
SOURCE_MATHLIB: Mathlib/Analysis/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
**)

Set Universe Polymorphism.
Set Implicit Arguments.

Class FrameworkStruct_analysis_heat_kernel_ultracontractivity_deep (E : Type) := {
  mass : E -> nat;
  entropy : E -> nat;
  dissip : E -> nat;
  step : nat -> E -> E;
  smooth : E -> E;
  le_trans_nat : forall a b c : nat, a <= b -> b <= c -> a <= c;
  add_le_add_left_nat : forall a b c : nat, a <= b -> c + a <= c + b;
  add_le_add_right_nat : forall a b c : nat, a <= b -> a + c <= b + c;
  add_le_add_nat : forall a b c d : nat, a <= b -> c <= d -> a + c <= b + d;
  le_add_left_nat : forall a b : nat, b <= a + b;
  le_add_right_nat : forall a b : nat, a <= a + b;
  step_zero : forall x : E, step 0 x = x;
  step_add : forall m n : nat, forall x : E, step (m + n) x = step m (step n x);
  mass_decay : forall n : nat, forall x : E, mass (step n x) <= mass x + n;
  entropy_decay : forall n : nat, forall x : E, entropy (step n x) <= entropy x + dissip x + n;
  ultracontractive : forall x : E, mass (smooth x) <= entropy x + dissip x;
  dissip_smooth : forall x : E, dissip (smooth x) <= dissip x + dissip x;
  smoothing_step : forall n : nat, forall x : E, mass (smooth (step n x)) <= mass (step n x) + dissip (step n x);
  witness_step : forall x : E, forall n : nat, exists y : E, y = step n x /\ mass y <= mass x + n
}.

Record ContextData_analysis_heat_kernel_ultracontractivity_deep
    (E : Type) `{FrameworkStruct_analysis_heat_kernel_ultracontractivity_deep E} := {
  seed_analysis_heat_kernel_ultracontractivity_deep : E;
  time_analysis_heat_kernel_ultracontractivity_deep : nat;
  budget_analysis_heat_kernel_ultracontractivity_deep : nat;
  budget_time_analysis_heat_kernel_ultracontractivity_deep :
    time_analysis_heat_kernel_ultracontractivity_deep <= budget_analysis_heat_kernel_ultracontractivity_deep;
  seed_control_analysis_heat_kernel_ultracontractivity_deep :
    entropy seed_analysis_heat_kernel_ultracontractivity_deep <=
      mass seed_analysis_heat_kernel_ultracontractivity_deep + budget_analysis_heat_kernel_ultracontractivity_deep
}.

Definition primary_map_analysis_heat_kernel_ultracontractivity_deep
    {E : Type} `{FrameworkStruct_analysis_heat_kernel_ultracontractivity_deep E}
    (d : ContextData_analysis_heat_kernel_ultracontractivity_deep) : E :=
  step (time_analysis_heat_kernel_ultracontractivity_deep d)
    (seed_analysis_heat_kernel_ultracontractivity_deep d).

Definition secondary_map_analysis_heat_kernel_ultracontractivity_deep
    {E : Type} `{FrameworkStruct_analysis_heat_kernel_ultracontractivity_deep E}
    (d : ContextData_analysis_heat_kernel_ultracontractivity_deep) : nat :=
  mass (primary_map_analysis_heat_kernel_ultracontractivity_deep d) +
    budget_analysis_heat_kernel_ultracontractivity_deep d.

Definition tertiary_map_analysis_heat_kernel_ultracontractivity_deep
    {E : Type} `{FrameworkStruct_analysis_heat_kernel_ultracontractivity_deep E}
    (d : ContextData_analysis_heat_kernel_ultracontractivity_deep) : Prop :=
  mass (primary_map_analysis_heat_kernel_ultracontractivity_deep d) <=
    secondary_map_analysis_heat_kernel_ultracontractivity_deep d.

Lemma stability_step_analysis_heat_kernel_ultracontractivity_deep
    {E : Type} `{FrameworkStruct_analysis_heat_kernel_ultracontractivity_deep E}
    (d : ContextData_analysis_heat_kernel_ultracontractivity_deep) :
    ((fun P : Prop => P) (mass (primary_map_analysis_heat_kernel_ultracontractivity_deep d) <=
      mass (seed_analysis_heat_kernel_ultracontractivity_deep d) +
      time_analysis_heat_kernel_ultracontractivity_deep d /\
    entropy (primary_map_analysis_heat_kernel_ultracontractivity_deep d) <=
      entropy (seed_analysis_heat_kernel_ultracontractivity_deep d) +
      dissip (seed_analysis_heat_kernel_ultracontractivity_deep d) +
      time_analysis_heat_kernel_ultracontractivity_deep d)).
Proof.
  assert (hMassRaw :
      mass (step (time_analysis_heat_kernel_ultracontractivity_deep d)
        (seed_analysis_heat_kernel_ultracontractivity_deep d)) <=
      mass (seed_analysis_heat_kernel_ultracontractivity_deep d) +
      time_analysis_heat_kernel_ultracontractivity_deep d).
  { apply mass_decay. }
  assert (hEntRaw :
      entropy (step (time_analysis_heat_kernel_ultracontractivity_deep d)
        (seed_analysis_heat_kernel_ultracontractivity_deep d)) <=
      entropy (seed_analysis_heat_kernel_ultracontractivity_deep d) +
      dissip (seed_analysis_heat_kernel_ultracontractivity_deep d) +
      time_analysis_heat_kernel_ultracontractivity_deep d).
  { apply entropy_decay. }
  split.
  - exact hMassRaw.
  - exact hEntRaw.
Qed.

Lemma factorization_step_analysis_heat_kernel_ultracontractivity_deep
    {E : Type} `{FrameworkStruct_analysis_heat_kernel_ultracontractivity_deep E}
    (d : ContextData_analysis_heat_kernel_ultracontractivity_deep) :
    ((fun P : Prop => P) (mass (smooth (primary_map_analysis_heat_kernel_ultracontractivity_deep d)) <=
      mass (primary_map_analysis_heat_kernel_ultracontractivity_deep d) +
      dissip (primary_map_analysis_heat_kernel_ultracontractivity_deep d) /\
    dissip (smooth (primary_map_analysis_heat_kernel_ultracontractivity_deep d)) <=
      dissip (primary_map_analysis_heat_kernel_ultracontractivity_deep d) +
      dissip (primary_map_analysis_heat_kernel_ultracontractivity_deep d))).
Proof.
  set (x := primary_map_analysis_heat_kernel_ultracontractivity_deep d).
  assert (hSmoothMass : mass (smooth x) <= mass x + dissip x).
  {
    unfold x.
    unfold primary_map_analysis_heat_kernel_ultracontractivity_deep.
    apply smoothing_step.
  }
  assert (hDissip : dissip (smooth x) <= dissip x + dissip x).
  { apply dissip_smooth. }
  split.
  - exact hSmoothMass.
  - exact hDissip.
Qed.

Lemma comparison_step_analysis_heat_kernel_ultracontractivity_deep
    {E : Type} `{FrameworkStruct_analysis_heat_kernel_ultracontractivity_deep E}
    (d : ContextData_analysis_heat_kernel_ultracontractivity_deep) :
    ((fun P : Prop => P) (tertiary_map_analysis_heat_kernel_ultracontractivity_deep d /\
    mass (primary_map_analysis_heat_kernel_ultracontractivity_deep d) <=
      secondary_map_analysis_heat_kernel_ultracontractivity_deep d)).
Proof.
  assert (hBase :
      mass (primary_map_analysis_heat_kernel_ultracontractivity_deep d) <=
        mass (primary_map_analysis_heat_kernel_ultracontractivity_deep d) +
        budget_analysis_heat_kernel_ultracontractivity_deep d).
  { apply le_add_right_nat. }
  assert (hTer : tertiary_map_analysis_heat_kernel_ultracontractivity_deep d).
  {
    unfold tertiary_map_analysis_heat_kernel_ultracontractivity_deep.
    unfold secondary_map_analysis_heat_kernel_ultracontractivity_deep.
    exact hBase.
  }
  split.
  - exact hTer.
  - unfold secondary_map_analysis_heat_kernel_ultracontractivity_deep.
    exact hBase.
Qed.

Lemma transport_step_analysis_heat_kernel_ultracontractivity_deep
    {E : Type} `{FrameworkStruct_analysis_heat_kernel_ultracontractivity_deep E}
    (d : ContextData_analysis_heat_kernel_ultracontractivity_deep) :
    ((fun P : Prop => P) (mass (primary_map_analysis_heat_kernel_ultracontractivity_deep d) <=
      mass (seed_analysis_heat_kernel_ultracontractivity_deep d) +
      budget_analysis_heat_kernel_ultracontractivity_deep d /\
    mass (primary_map_analysis_heat_kernel_ultracontractivity_deep d) <=
      secondary_map_analysis_heat_kernel_ultracontractivity_deep d)).
Proof.
  destruct (stability_step_analysis_heat_kernel_ultracontractivity_deep d)
    as [hMassTime hEntropyTime].
  assert (hBudgetLift :
      mass (seed_analysis_heat_kernel_ultracontractivity_deep d) +
        time_analysis_heat_kernel_ultracontractivity_deep d <=
      mass (seed_analysis_heat_kernel_ultracontractivity_deep d) +
        budget_analysis_heat_kernel_ultracontractivity_deep d).
  {
    apply add_le_add_left_nat.
    exact (budget_time_analysis_heat_kernel_ultracontractivity_deep d).
  }
  assert (hMassBudget :
      mass (primary_map_analysis_heat_kernel_ultracontractivity_deep d) <=
      mass (seed_analysis_heat_kernel_ultracontractivity_deep d) +
      budget_analysis_heat_kernel_ultracontractivity_deep d).
  {
    apply le_trans_nat with
      (b := mass (seed_analysis_heat_kernel_ultracontractivity_deep d) +
            time_analysis_heat_kernel_ultracontractivity_deep d).
    - exact hMassTime.
    - exact hBudgetLift.
  }
  destruct (comparison_step_analysis_heat_kernel_ultracontractivity_deep d)
    as [hTer hSec].
  assert (_hEntropyKeep :
      entropy (primary_map_analysis_heat_kernel_ultracontractivity_deep d) <=
        entropy (seed_analysis_heat_kernel_ultracontractivity_deep d) +
        dissip (seed_analysis_heat_kernel_ultracontractivity_deep d) +
        time_analysis_heat_kernel_ultracontractivity_deep d).
  { exact hEntropyTime. }
  assert (_hTerKeep : tertiary_map_analysis_heat_kernel_ultracontractivity_deep d).
  { exact hTer. }
  split.
  - exact hMassBudget.
  - exact hSec.
Qed.

Lemma coherence_step_analysis_heat_kernel_ultracontractivity_deep
    {E : Type} `{FrameworkStruct_analysis_heat_kernel_ultracontractivity_deep E}
    (d : ContextData_analysis_heat_kernel_ultracontractivity_deep) :
    ((fun P : Prop => P) (entropy (primary_map_analysis_heat_kernel_ultracontractivity_deep d) <=
      mass (seed_analysis_heat_kernel_ultracontractivity_deep d) +
      budget_analysis_heat_kernel_ultracontractivity_deep d +
      dissip (seed_analysis_heat_kernel_ultracontractivity_deep d) +
      budget_analysis_heat_kernel_ultracontractivity_deep d /\
    secondary_map_analysis_heat_kernel_ultracontractivity_deep d =
      mass (primary_map_analysis_heat_kernel_ultracontractivity_deep d) +
      budget_analysis_heat_kernel_ultracontractivity_deep d)).
Proof.
  destruct (stability_step_analysis_heat_kernel_ultracontractivity_deep d)
    as [hMass hEntropy].
  assert (h1 :
      entropy (seed_analysis_heat_kernel_ultracontractivity_deep d) <=
        mass (seed_analysis_heat_kernel_ultracontractivity_deep d) +
        budget_analysis_heat_kernel_ultracontractivity_deep d).
  { exact (seed_control_analysis_heat_kernel_ultracontractivity_deep d). }
  assert (h1' :
      entropy (seed_analysis_heat_kernel_ultracontractivity_deep d) +
        dissip (seed_analysis_heat_kernel_ultracontractivity_deep d) <=
      (mass (seed_analysis_heat_kernel_ultracontractivity_deep d) +
        budget_analysis_heat_kernel_ultracontractivity_deep d) +
      dissip (seed_analysis_heat_kernel_ultracontractivity_deep d)).
  {
    apply add_le_add_right_nat.
    exact h1.
  }
  assert (h2 :
      entropy (seed_analysis_heat_kernel_ultracontractivity_deep d) +
        dissip (seed_analysis_heat_kernel_ultracontractivity_deep d) +
        time_analysis_heat_kernel_ultracontractivity_deep d <=
      ((mass (seed_analysis_heat_kernel_ultracontractivity_deep d) +
        budget_analysis_heat_kernel_ultracontractivity_deep d) +
        dissip (seed_analysis_heat_kernel_ultracontractivity_deep d)) +
      time_analysis_heat_kernel_ultracontractivity_deep d).
  {
    apply add_le_add_right_nat.
    exact h1'.
  }
  assert (h3 :
      ((mass (seed_analysis_heat_kernel_ultracontractivity_deep d) +
        budget_analysis_heat_kernel_ultracontractivity_deep d) +
        dissip (seed_analysis_heat_kernel_ultracontractivity_deep d)) +
      time_analysis_heat_kernel_ultracontractivity_deep d <=
      ((mass (seed_analysis_heat_kernel_ultracontractivity_deep d) +
        budget_analysis_heat_kernel_ultracontractivity_deep d) +
        dissip (seed_analysis_heat_kernel_ultracontractivity_deep d)) +
      budget_analysis_heat_kernel_ultracontractivity_deep d).
  {
    apply add_le_add_left_nat.
    exact (budget_time_analysis_heat_kernel_ultracontractivity_deep d).
  }
  assert (hSeedLift :
      entropy (seed_analysis_heat_kernel_ultracontractivity_deep d) +
        dissip (seed_analysis_heat_kernel_ultracontractivity_deep d) +
        time_analysis_heat_kernel_ultracontractivity_deep d <=
      ((mass (seed_analysis_heat_kernel_ultracontractivity_deep d) +
        budget_analysis_heat_kernel_ultracontractivity_deep d) +
        dissip (seed_analysis_heat_kernel_ultracontractivity_deep d)) +
      budget_analysis_heat_kernel_ultracontractivity_deep d).
  {
    apply le_trans_nat with
      (b := ((mass (seed_analysis_heat_kernel_ultracontractivity_deep d) +
              budget_analysis_heat_kernel_ultracontractivity_deep d) +
              dissip (seed_analysis_heat_kernel_ultracontractivity_deep d)) +
            time_analysis_heat_kernel_ultracontractivity_deep d).
    - exact h2.
    - exact h3.
  }
  assert (hEntropyBudget :
      entropy (primary_map_analysis_heat_kernel_ultracontractivity_deep d) <=
      mass (seed_analysis_heat_kernel_ultracontractivity_deep d) +
      budget_analysis_heat_kernel_ultracontractivity_deep d +
      dissip (seed_analysis_heat_kernel_ultracontractivity_deep d) +
      budget_analysis_heat_kernel_ultracontractivity_deep d).
  {
    apply le_trans_nat with
      (b := entropy (seed_analysis_heat_kernel_ultracontractivity_deep d) +
            dissip (seed_analysis_heat_kernel_ultracontractivity_deep d) +
            time_analysis_heat_kernel_ultracontractivity_deep d).
    - exact hEntropy.
    - exact hSeedLift.
  }
  assert (hEq :
      secondary_map_analysis_heat_kernel_ultracontractivity_deep d =
        mass (primary_map_analysis_heat_kernel_ultracontractivity_deep d) +
        budget_analysis_heat_kernel_ultracontractivity_deep d).
  { reflexivity. }
  assert (_hMassKeep :
      mass (primary_map_analysis_heat_kernel_ultracontractivity_deep d) <=
        mass (seed_analysis_heat_kernel_ultracontractivity_deep d) +
        time_analysis_heat_kernel_ultracontractivity_deep d).
  { exact hMass. }
  split.
  - exact hEntropyBudget.
  - exact hEq.
Qed.

Lemma iteration_step_analysis_heat_kernel_ultracontractivity_deep
    {E : Type} `{FrameworkStruct_analysis_heat_kernel_ultracontractivity_deep E}
    (d : ContextData_analysis_heat_kernel_ultracontractivity_deep) :
    ((fun P : Prop => P) (exists y :
    E,
      y = primary_map_analysis_heat_kernel_ultracontractivity_deep d /\
      mass y <=
        mass (seed_analysis_heat_kernel_ultracontractivity_deep d) +
        budget_analysis_heat_kernel_ultracontractivity_deep d /\
      mass (smooth y) <= mass y + dissip y)).
Proof.
  destruct (witness_step (seed_analysis_heat_kernel_ultracontractivity_deep d)
      (time_analysis_heat_kernel_ultracontractivity_deep d)) as [y [hyEq hyMassTime]].
  assert (hyMassBudget :
      mass y <=
        mass (seed_analysis_heat_kernel_ultracontractivity_deep d) +
        budget_analysis_heat_kernel_ultracontractivity_deep d).
  {
    apply le_trans_nat with
      (b := mass (seed_analysis_heat_kernel_ultracontractivity_deep d) +
            time_analysis_heat_kernel_ultracontractivity_deep d).
    - exact hyMassTime.
    - apply add_le_add_left_nat.
      exact (budget_time_analysis_heat_kernel_ultracontractivity_deep d).
  }
  assert (hyPrimary : y = primary_map_analysis_heat_kernel_ultracontractivity_deep d).
  {
    unfold primary_map_analysis_heat_kernel_ultracontractivity_deep.
    exact hyEq.
  }
  assert (hSmooth : mass (smooth y) <= mass y + dissip y).
  {
    rewrite hyEq.
    apply smoothing_step.
  }
  exists y.
  repeat split; try assumption.
Qed.

Lemma main_result_analysis_heat_kernel_ultracontractivity_deep
    {E : Type} `{FrameworkStruct_analysis_heat_kernel_ultracontractivity_deep E}
    (d : ContextData_analysis_heat_kernel_ultracontractivity_deep) :
    ((fun P : Prop => P) (exists y :
    E,
      y = primary_map_analysis_heat_kernel_ultracontractivity_deep d /\
      mass (smooth y) <=
        (mass (seed_analysis_heat_kernel_ultracontractivity_deep d) +
         budget_analysis_heat_kernel_ultracontractivity_deep d) +
        dissip y /\
      tertiary_map_analysis_heat_kernel_ultracontractivity_deep d)).
Proof.
  destruct (iteration_step_analysis_heat_kernel_ultracontractivity_deep d)
    as [y [hyEq [hyMass hSmooth]]].
  assert (hBound :
      mass (smooth y) <=
      (mass (seed_analysis_heat_kernel_ultracontractivity_deep d) +
       budget_analysis_heat_kernel_ultracontractivity_deep d) +
      dissip y).
  {
    apply le_trans_nat with (b := mass y + dissip y).
    - exact hSmooth.
    - apply add_le_add_right_nat.
      exact hyMass.
  }
  destruct (comparison_step_analysis_heat_kernel_ultracontractivity_deep d)
    as [hTer hSec].
  assert (_hSecKeep :
      mass (primary_map_analysis_heat_kernel_ultracontractivity_deep d) <=
        secondary_map_analysis_heat_kernel_ultracontractivity_deep d).
  { exact hSec. }
  exists y.
  repeat split; try assumption.
Qed.
