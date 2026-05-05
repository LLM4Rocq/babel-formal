(**
BENCHMARK_ID: TINY_MATHLIB_BATCH07_ANALYSIS_CALCULUS_VARIATIONS_RELAXATION_LIKE
PAIR_STEM: analysis_calculus_variations_relaxation_like
MATH_DOMAIN: Analysis
SOURCE_MATHLIB: Mathlib/Analysis/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
**)

Set Universe Polymorphism.
Set Implicit Arguments.

Class FrameworkStruct_analysis_calculus_variations_relaxation (E : Type) := {
  energy : E -> nat;
  relaxed : E -> nat;
  gradient : E -> nat;
  perturb : E -> E;
  envelope : E -> E;
  le_trans_nat : forall a b c : nat, a <= b -> b <= c -> a <= c;
  add_le_add_left_nat : forall a b c : nat, a <= b -> c + a <= c + b;
  add_le_add_right_nat : forall a b c : nat, a <= b -> a + c <= b + c;
  le_add_left_nat : forall a b : nat, b <= a + b;
  le_add_right_nat : forall a b : nat, a <= a + b;
  relaxed_le_energy : forall x : E, relaxed x <= energy x;
  perturb_energy : forall x : E, energy (perturb x) <= energy x + gradient x;
  perturb_relaxed : forall x : E, relaxed (perturb x) <= relaxed x + gradient x;
  envelope_relaxed : forall x : E, relaxed (envelope x) <= relaxed x;
  envelope_of_perturb_energy : forall x : E, energy (envelope (perturb x)) <= energy x + gradient x;
  envelope_of_perturb_relaxed : forall x : E, relaxed (envelope (perturb x)) <= relaxed x + gradient x;
  witness_envelope : forall x : E, exists y : E, y = envelope x /\ relaxed y <= energy x
}.

Record ContextData_analysis_calculus_variations_relaxation
    (E : Type) `{FrameworkStruct_analysis_calculus_variations_relaxation E} := {
  base_analysis_calculus_variations_relaxation : E;
  competitor_analysis_calculus_variations_relaxation : E;
  penalty_analysis_calculus_variations_relaxation : nat;
  gradient_bound_analysis_calculus_variations_relaxation :
    gradient base_analysis_calculus_variations_relaxation <= penalty_analysis_calculus_variations_relaxation;
  base_energy_control_analysis_calculus_variations_relaxation :
    energy base_analysis_calculus_variations_relaxation <=
      energy competitor_analysis_calculus_variations_relaxation +
      penalty_analysis_calculus_variations_relaxation
}.

Definition primary_map_analysis_calculus_variations_relaxation
    {E : Type} `{FrameworkStruct_analysis_calculus_variations_relaxation E}
    (d : ContextData_analysis_calculus_variations_relaxation) : E :=
  envelope (perturb (base_analysis_calculus_variations_relaxation d)).

Definition secondary_map_analysis_calculus_variations_relaxation
    {E : Type} `{FrameworkStruct_analysis_calculus_variations_relaxation E}
    (d : ContextData_analysis_calculus_variations_relaxation) : nat :=
  energy (primary_map_analysis_calculus_variations_relaxation d) +
    penalty_analysis_calculus_variations_relaxation d.

Definition tertiary_map_analysis_calculus_variations_relaxation
    {E : Type} `{FrameworkStruct_analysis_calculus_variations_relaxation E}
    (d : ContextData_analysis_calculus_variations_relaxation) : Prop :=
  relaxed (primary_map_analysis_calculus_variations_relaxation d) <=
    secondary_map_analysis_calculus_variations_relaxation d.

Lemma stability_step_analysis_calculus_variations_relaxation
    {E : Type} `{FrameworkStruct_analysis_calculus_variations_relaxation E}
    (d : ContextData_analysis_calculus_variations_relaxation) :
    (True -> energy (primary_map_analysis_calculus_variations_relaxation d) <=
      energy (base_analysis_calculus_variations_relaxation d) +
      gradient (base_analysis_calculus_variations_relaxation d) /\
    relaxed (primary_map_analysis_calculus_variations_relaxation d) <=
      relaxed (base_analysis_calculus_variations_relaxation d) +
      gradient (base_analysis_calculus_variations_relaxation d)).
Proof.
  intro _hTrue.
  assert (hEnergyRaw :
      energy (envelope (perturb (base_analysis_calculus_variations_relaxation d))) <=
        energy (base_analysis_calculus_variations_relaxation d) +
        gradient (base_analysis_calculus_variations_relaxation d)).
  { apply envelope_of_perturb_energy. }
  assert (hRelaxRaw :
      relaxed (envelope (perturb (base_analysis_calculus_variations_relaxation d))) <=
        relaxed (base_analysis_calculus_variations_relaxation d) +
        gradient (base_analysis_calculus_variations_relaxation d)).
  { apply envelope_of_perturb_relaxed. }
  split.
  - exact hEnergyRaw.
  - exact hRelaxRaw.
Qed.

Lemma factorization_step_analysis_calculus_variations_relaxation
    {E : Type} `{FrameworkStruct_analysis_calculus_variations_relaxation E}
    (d : ContextData_analysis_calculus_variations_relaxation) :
    (True -> relaxed (perturb (base_analysis_calculus_variations_relaxation d)) <=
      energy (base_analysis_calculus_variations_relaxation d) +
      gradient (base_analysis_calculus_variations_relaxation d) /\
    relaxed (envelope (base_analysis_calculus_variations_relaxation d)) <=
      energy (base_analysis_calculus_variations_relaxation d)).
Proof.
  intro _hTrue.
  assert (hRelaxPert :
      relaxed (perturb (base_analysis_calculus_variations_relaxation d)) <=
        energy (perturb (base_analysis_calculus_variations_relaxation d))).
  { apply relaxed_le_energy. }
  assert (hPertEnergy :
      energy (perturb (base_analysis_calculus_variations_relaxation d)) <=
        energy (base_analysis_calculus_variations_relaxation d) +
        gradient (base_analysis_calculus_variations_relaxation d)).
  { apply perturb_energy. }
  assert (hFirst :
      relaxed (perturb (base_analysis_calculus_variations_relaxation d)) <=
        energy (base_analysis_calculus_variations_relaxation d) +
        gradient (base_analysis_calculus_variations_relaxation d)).
  {
    apply le_trans_nat with
      (b := energy (perturb (base_analysis_calculus_variations_relaxation d))).
    - exact hRelaxPert.
    - exact hPertEnergy.
  }
  assert (hEnvRelax :
      relaxed (envelope (base_analysis_calculus_variations_relaxation d)) <=
        relaxed (base_analysis_calculus_variations_relaxation d)).
  { apply envelope_relaxed. }
  assert (hBase :
      relaxed (base_analysis_calculus_variations_relaxation d) <=
        energy (base_analysis_calculus_variations_relaxation d)).
  { apply relaxed_le_energy. }
  assert (hSecond :
      relaxed (envelope (base_analysis_calculus_variations_relaxation d)) <=
        energy (base_analysis_calculus_variations_relaxation d)).
  {
    apply le_trans_nat with
      (b := relaxed (base_analysis_calculus_variations_relaxation d)).
    - exact hEnvRelax.
    - exact hBase.
  }
  split.
  - exact hFirst.
  - exact hSecond.
Qed.

Lemma comparison_step_analysis_calculus_variations_relaxation
    {E : Type} `{FrameworkStruct_analysis_calculus_variations_relaxation E}
    (d : ContextData_analysis_calculus_variations_relaxation) :
    (True -> tertiary_map_analysis_calculus_variations_relaxation d /\
    energy (primary_map_analysis_calculus_variations_relaxation d) <=
      secondary_map_analysis_calculus_variations_relaxation d).
Proof.
  intro _hTrue.
  assert (hRelaxEnergy :
      relaxed (primary_map_analysis_calculus_variations_relaxation d) <=
        energy (primary_map_analysis_calculus_variations_relaxation d)).
  { apply relaxed_le_energy. }
  assert (hEnergySec :
      energy (primary_map_analysis_calculus_variations_relaxation d) <=
        secondary_map_analysis_calculus_variations_relaxation d).
  {
    unfold secondary_map_analysis_calculus_variations_relaxation.
    apply le_add_right_nat.
  }
  assert (hTer : tertiary_map_analysis_calculus_variations_relaxation d).
  {
    unfold tertiary_map_analysis_calculus_variations_relaxation.
    apply le_trans_nat with
      (b := energy (primary_map_analysis_calculus_variations_relaxation d)).
    - exact hRelaxEnergy.
    - exact hEnergySec.
  }
  split.
  - exact hTer.
  - exact hEnergySec.
Qed.

Lemma transport_step_analysis_calculus_variations_relaxation
    {E : Type} `{FrameworkStruct_analysis_calculus_variations_relaxation E}
    (d : ContextData_analysis_calculus_variations_relaxation) :
    (True -> relaxed (primary_map_analysis_calculus_variations_relaxation d) <=
      energy (base_analysis_calculus_variations_relaxation d) +
      penalty_analysis_calculus_variations_relaxation d /\
    energy (primary_map_analysis_calculus_variations_relaxation d) <=
      (energy (competitor_analysis_calculus_variations_relaxation d) +
      penalty_analysis_calculus_variations_relaxation d) +
      gradient (base_analysis_calculus_variations_relaxation d)).
Proof.
  intro _hTrue.
  destruct (stability_step_analysis_calculus_variations_relaxation d I)
    as [hEnergy hRelax].
  assert (hBaseRelax :
      relaxed (base_analysis_calculus_variations_relaxation d) <=
        energy (base_analysis_calculus_variations_relaxation d)).
  { apply relaxed_le_energy. }
  assert (hLift :
      relaxed (base_analysis_calculus_variations_relaxation d) +
        gradient (base_analysis_calculus_variations_relaxation d) <=
      energy (base_analysis_calculus_variations_relaxation d) +
        gradient (base_analysis_calculus_variations_relaxation d)).
  {
    apply add_le_add_right_nat.
    exact hBaseRelax.
  }
  assert (hRelaxBase :
      relaxed (primary_map_analysis_calculus_variations_relaxation d) <=
      energy (base_analysis_calculus_variations_relaxation d) +
      gradient (base_analysis_calculus_variations_relaxation d)).
  {
    apply le_trans_nat with
      (b := relaxed (base_analysis_calculus_variations_relaxation d) +
            gradient (base_analysis_calculus_variations_relaxation d)).
    - exact hRelax.
    - exact hLift.
  }
  assert (hGrad :
      energy (base_analysis_calculus_variations_relaxation d) +
        gradient (base_analysis_calculus_variations_relaxation d) <=
      energy (base_analysis_calculus_variations_relaxation d) +
        penalty_analysis_calculus_variations_relaxation d).
  {
    apply add_le_add_left_nat.
    exact (gradient_bound_analysis_calculus_variations_relaxation d).
  }
  assert (hFirst :
      relaxed (primary_map_analysis_calculus_variations_relaxation d) <=
      energy (base_analysis_calculus_variations_relaxation d) +
      penalty_analysis_calculus_variations_relaxation d).
  {
    apply le_trans_nat with
      (b := energy (base_analysis_calculus_variations_relaxation d) +
            gradient (base_analysis_calculus_variations_relaxation d)).
    - exact hRelaxBase.
    - exact hGrad.
  }
  assert (hSecondLift :
      energy (base_analysis_calculus_variations_relaxation d) +
        gradient (base_analysis_calculus_variations_relaxation d) <=
      (energy (competitor_analysis_calculus_variations_relaxation d) +
      penalty_analysis_calculus_variations_relaxation d) +
      gradient (base_analysis_calculus_variations_relaxation d)).
  {
    apply add_le_add_right_nat.
    exact (base_energy_control_analysis_calculus_variations_relaxation d).
  }
  assert (hSecond :
      energy (primary_map_analysis_calculus_variations_relaxation d) <=
      (energy (competitor_analysis_calculus_variations_relaxation d) +
      penalty_analysis_calculus_variations_relaxation d) +
      gradient (base_analysis_calculus_variations_relaxation d)).
  {
    apply le_trans_nat with
      (b := energy (base_analysis_calculus_variations_relaxation d) +
            gradient (base_analysis_calculus_variations_relaxation d)).
    - exact hEnergy.
    - exact hSecondLift.
  }
  split.
  - exact hFirst.
  - exact hSecond.
Qed.

Lemma coherence_step_analysis_calculus_variations_relaxation
    {E : Type} `{FrameworkStruct_analysis_calculus_variations_relaxation E}
    (d : ContextData_analysis_calculus_variations_relaxation) :
    (True -> secondary_map_analysis_calculus_variations_relaxation d =
      energy (primary_map_analysis_calculus_variations_relaxation d) +
      penalty_analysis_calculus_variations_relaxation d /\
    tertiary_map_analysis_calculus_variations_relaxation d).
Proof.
  intro _hTrue.
  assert (hEq :
      secondary_map_analysis_calculus_variations_relaxation d =
        energy (primary_map_analysis_calculus_variations_relaxation d) +
        penalty_analysis_calculus_variations_relaxation d).
  { reflexivity. }
  destruct (comparison_step_analysis_calculus_variations_relaxation d I)
    as [hTer hSec].
  assert (_hSecKeep :
      energy (primary_map_analysis_calculus_variations_relaxation d) <=
        secondary_map_analysis_calculus_variations_relaxation d).
  { exact hSec. }
  split.
  - exact hEq.
  - exact hTer.
Qed.

Lemma iteration_step_analysis_calculus_variations_relaxation
    {E : Type} `{FrameworkStruct_analysis_calculus_variations_relaxation E}
    (d : ContextData_analysis_calculus_variations_relaxation) :
    (True -> exists y : E,
      y = envelope (base_analysis_calculus_variations_relaxation d) /\
      relaxed y <= energy (base_analysis_calculus_variations_relaxation d) /\
      relaxed (primary_map_analysis_calculus_variations_relaxation d) <=
        energy (base_analysis_calculus_variations_relaxation d) +
        gradient (base_analysis_calculus_variations_relaxation d)).
Proof.
  intro _hTrue.
  destruct (witness_envelope (base_analysis_calculus_variations_relaxation d))
    as [y [hyEq hyRelax]].
  destruct (stability_step_analysis_calculus_variations_relaxation d I)
    as [hEnergy hRelaxPrimary].
  assert (hBaseRelax :
      relaxed (base_analysis_calculus_variations_relaxation d) <=
        energy (base_analysis_calculus_variations_relaxation d)).
  { apply relaxed_le_energy. }
  assert (hLift :
      relaxed (base_analysis_calculus_variations_relaxation d) +
        gradient (base_analysis_calculus_variations_relaxation d) <=
      energy (base_analysis_calculus_variations_relaxation d) +
        gradient (base_analysis_calculus_variations_relaxation d)).
  {
    apply add_le_add_right_nat.
    exact hBaseRelax.
  }
  assert (hFinal :
      relaxed (primary_map_analysis_calculus_variations_relaxation d) <=
      energy (base_analysis_calculus_variations_relaxation d) +
      gradient (base_analysis_calculus_variations_relaxation d)).
  {
    apply le_trans_nat with
      (b := relaxed (base_analysis_calculus_variations_relaxation d) +
            gradient (base_analysis_calculus_variations_relaxation d)).
    - exact hRelaxPrimary.
    - exact hLift.
  }
  assert (_hEnergyKeep :
      energy (primary_map_analysis_calculus_variations_relaxation d) <=
        energy (base_analysis_calculus_variations_relaxation d) +
        gradient (base_analysis_calculus_variations_relaxation d)).
  { exact hEnergy. }
  exists y.
  repeat split; try assumption.
Qed.

Lemma main_result_analysis_calculus_variations_relaxation
    {E : Type} `{FrameworkStruct_analysis_calculus_variations_relaxation E}
    (d : ContextData_analysis_calculus_variations_relaxation) :
    (True -> exists y : E,
      y = primary_map_analysis_calculus_variations_relaxation d /\
      relaxed y <=
        (energy (competitor_analysis_calculus_variations_relaxation d) +
         penalty_analysis_calculus_variations_relaxation d) +
        penalty_analysis_calculus_variations_relaxation d /\
      tertiary_map_analysis_calculus_variations_relaxation d).
Proof.
  intro _hTrue.
  set (y := primary_map_analysis_calculus_variations_relaxation d).
  assert (hyEq : y = primary_map_analysis_calculus_variations_relaxation d).
  { reflexivity. }
  destruct (transport_step_analysis_calculus_variations_relaxation d I)
    as [hRelaxBound hEnergyBound].
  assert (hLift :
      energy (base_analysis_calculus_variations_relaxation d) +
        penalty_analysis_calculus_variations_relaxation d <=
      (energy (competitor_analysis_calculus_variations_relaxation d) +
       penalty_analysis_calculus_variations_relaxation d) +
      penalty_analysis_calculus_variations_relaxation d).
  {
    apply add_le_add_right_nat.
    exact (base_energy_control_analysis_calculus_variations_relaxation d).
  }
  assert (hFinal :
      relaxed y <=
      (energy (competitor_analysis_calculus_variations_relaxation d) +
       penalty_analysis_calculus_variations_relaxation d) +
      penalty_analysis_calculus_variations_relaxation d).
  {
    apply le_trans_nat with
      (b := energy (base_analysis_calculus_variations_relaxation d) +
            penalty_analysis_calculus_variations_relaxation d).
    - rewrite hyEq. exact hRelaxBound.
    - exact hLift.
  }
  destruct (comparison_step_analysis_calculus_variations_relaxation d I)
    as [hTer hSec].
  assert (_hSecKeep :
      energy y <= secondary_map_analysis_calculus_variations_relaxation d).
  { rewrite hyEq. exact hSec. }
  assert (_hEnergyKeep :
      energy y <=
      (energy (competitor_analysis_calculus_variations_relaxation d) +
      penalty_analysis_calculus_variations_relaxation d) +
      gradient (base_analysis_calculus_variations_relaxation d)).
  { rewrite hyEq. exact hEnergyBound. }
  exists y.
  repeat split; try assumption.
Qed.
