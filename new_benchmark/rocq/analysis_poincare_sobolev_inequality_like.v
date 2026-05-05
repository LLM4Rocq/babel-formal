(*
BENCHMARK_ID: TINY_MATHLIB_BATCH06_ANALYSIS_POINCARE_SOBOLEV_INEQUALITY_LIKE
PAIR_STEM: analysis_poincare_sobolev_inequality_like
MATH_DOMAIN: Analysis
SOURCE_MATHLIB: Mathlib/Analysis/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class SobolevStruct_poincare_inequality (E : Type) := {
  mean : E -> nat;
  gradient : E -> nat;
  oscillation : E -> nat;
  energy : E -> nat;
  localPatch : E -> E;
  renorm : E -> E;
  le_trans_nat : forall a b c : nat, a <= b -> b <= c -> a <= c;
  add_le_add_left_nat : forall a b c : nat, a <= b -> c + a <= c + b;
  add_le_add_right_nat : forall a b c : nat, a <= b -> a + c <= b + c;
  add_le_add_nat : forall a b c d : nat, a <= b -> c <= d -> a + c <= b + d;
  le_add_right_nat : forall a b : nat, a <= a + b;
  mean_localPatch_zero : forall x : E, mean (localPatch x) = 0;
  gradient_localPatch_bound : forall x : E, gradient (localPatch x) <= gradient x + mean x;
  oscillation_localPatch_bound : forall x : E, oscillation (localPatch x) <= oscillation x + mean x;
  poincare_step : forall x : E, oscillation x <= gradient x + mean x;
  holder_step : forall x : E, energy (renorm x) <= energy x + gradient x;
  coercive_step : forall x : E, gradient (renorm x) <= gradient x + oscillation x;
  compact_step : forall x : E, oscillation (renorm x) <= energy (renorm x)
}.

Arguments le_trans_nat {E} {_} _ _ _ _ _.
Arguments add_le_add_left_nat {E} {_} _ _ _ _.
Arguments add_le_add_right_nat {E} {_} _ _ _ _.
Arguments add_le_add_nat {E} {_} _ _ _ _ _ _.
Arguments le_add_right_nat {E} {_} _ _.

Record EnergyData_poincare_sobolev_inequality (E : Type)
    `{SobolevStruct_poincare_inequality E} := {
  state : E;
  scale : nat;
  scale_pos : 0 < scale;
  mean_le_scale : mean state <= scale;
  gradient_le_scaled : gradient state <= scale + gradient (localPatch state)
}.

Definition mean_operator_poincare_sobolev_inequality
    {E : Type} `{SobolevStruct_poincare_inequality E}
    (d : EnergyData_poincare_sobolev_inequality) : nat :=
  mean (state d).

Definition gradient_norm_poincare_sobolev_inequality
    {E : Type} `{SobolevStruct_poincare_inequality E}
    (d : EnergyData_poincare_sobolev_inequality) : nat :=
  gradient (state d) + scale d.

Definition oscillation_norm_poincare_sobolev_inequality
    {E : Type} `{SobolevStruct_poincare_inequality E}
    (d : EnergyData_poincare_sobolev_inequality) : nat :=
  oscillation (state d) + mean (state d).

Lemma mean_zero_reduction_poincare_sobolev_inequality
    {E : Type} `{SobolevStruct_poincare_inequality E}
    (d : EnergyData_poincare_sobolev_inequality) :
    mean (localPatch (state d)) = 0 /\
    gradient (localPatch (state d)) <= gradient_norm_poincare_sobolev_inequality d.
Proof.
  assert (hMeanZero : mean (localPatch (state d)) = 0).
  { apply (mean_localPatch_zero (state d)). }
  assert (hGradPatch :
    gradient (localPatch (state d)) <= gradient (state d) + mean (state d)).
  { apply (gradient_localPatch_bound (state d)). }
  assert (hMeanLift :
    gradient (state d) + mean (state d) <= gradient (state d) + scale d).
  {
    apply (add_le_add_left_nat (mean (state d)) (scale d) (gradient (state d))).
    exact (mean_le_scale d).
  }
  assert (hGradScaled : gradient (localPatch (state d)) <= gradient (state d) + scale d).
  { apply (le_trans_nat _ _ _ hGradPatch hMeanLift). }
  split.
  - exact hMeanZero.
  - unfold gradient_norm_poincare_sobolev_inequality.
    exact hGradScaled.
Qed.

Lemma local_patch_bound_poincare_sobolev_inequality
    {E : Type} `{SobolevStruct_poincare_inequality E}
    (d : EnergyData_poincare_sobolev_inequality) :
    oscillation (localPatch (state d)) <= oscillation_norm_poincare_sobolev_inequality d /\
    gradient (localPatch (state d)) <= gradient (state d) + scale d.
Proof.
  assert (hOscPatch :
    oscillation (localPatch (state d)) <= oscillation (state d) + mean (state d)).
  { apply (oscillation_localPatch_bound (state d)). }
  assert (hGradPatch :
    gradient (localPatch (state d)) <= gradient (state d) + mean (state d)).
  { apply (gradient_localPatch_bound (state d)). }
  assert (hMeanLift :
    gradient (state d) + mean (state d) <= gradient (state d) + scale d).
  {
    apply (add_le_add_left_nat (mean (state d)) (scale d) (gradient (state d))).
    exact (mean_le_scale d).
  }
  assert (hGradFinal : gradient (localPatch (state d)) <= gradient (state d) + scale d).
  { apply (le_trans_nat _ _ _ hGradPatch hMeanLift). }
  split.
  - unfold oscillation_norm_poincare_sobolev_inequality.
    exact hOscPatch.
  - exact hGradFinal.
Qed.

Lemma interpolation_step_poincare_sobolev_inequality
    {E : Type} `{SobolevStruct_poincare_inequality E}
    (x : E) :
    oscillation (localPatch x) <= gradient x + mean x + mean x.
Proof.
  assert (hPatch : oscillation (localPatch x) <= oscillation x + mean x).
  { apply (oscillation_localPatch_bound x). }
  assert (hPoincare : oscillation x <= gradient x + mean x).
  { apply (poincare_step x). }
  assert (hLift : oscillation x + mean x <= (gradient x + mean x) + mean x).
  {
    apply (add_le_add_right_nat (oscillation x) (gradient x + mean x) (mean x)).
    exact hPoincare.
  }
  apply (le_trans_nat _ _ _ hPatch hLift).
Qed.

Lemma holder_chain_poincare_sobolev_inequality
    {E : Type} `{SobolevStruct_poincare_inequality E}
    (d : EnergyData_poincare_sobolev_inequality) :
    energy (renorm (state d)) <= energy (state d) + gradient_norm_poincare_sobolev_inequality d.
Proof.
  assert (hHolder : energy (renorm (state d)) <= energy (state d) + gradient (state d)).
  { apply (holder_step (state d)). }
  assert (hGradLift : gradient (state d) <= gradient (state d) + scale d).
  { apply (le_add_right_nat (gradient (state d)) (scale d)). }
  assert (hEnergyLift :
    energy (state d) + gradient (state d) <= energy (state d) + (gradient (state d) + scale d)).
  {
    apply (add_le_add_left_nat (gradient (state d)) (gradient (state d) + scale d) (energy (state d))).
    exact hGradLift.
  }
  assert (hChain :
    energy (renorm (state d)) <= energy (state d) + (gradient (state d) + scale d)).
  { apply (le_trans_nat _ _ _ hHolder hEnergyLift). }
  unfold gradient_norm_poincare_sobolev_inequality.
  exact hChain.
Qed.

Lemma coercive_estimate_poincare_sobolev_inequality
    {E : Type} `{SobolevStruct_poincare_inequality E}
    (d : EnergyData_poincare_sobolev_inequality) :
    gradient (renorm (state d)) <=
      gradient_norm_poincare_sobolev_inequality d +
      oscillation_norm_poincare_sobolev_inequality d /\
    gradient (state d) <= gradient_norm_poincare_sobolev_inequality d.
Proof.
  assert (hCoercive : gradient (renorm (state d)) <= gradient (state d) + oscillation (state d)).
  { apply (coercive_step (state d)). }
  assert (hGradLift : gradient (state d) <= gradient (state d) + scale d).
  { apply (le_add_right_nat (gradient (state d)) (scale d)). }
  assert (hOscLift : oscillation (state d) <= oscillation (state d) + mean (state d)).
  { apply (le_add_right_nat (oscillation (state d)) (mean (state d))). }
  assert (hPairLift :
    gradient (state d) + oscillation (state d) <=
      (gradient (state d) + scale d) + (oscillation (state d) + mean (state d))).
  {
    apply (add_le_add_nat
      (gradient (state d)) (gradient (state d) + scale d)
      (oscillation (state d)) (oscillation (state d) + mean (state d))).
    - exact hGradLift.
    - exact hOscLift.
  }
  assert (hChain :
    gradient (renorm (state d)) <=
      (gradient (state d) + scale d) + (oscillation (state d) + mean (state d))).
  { apply (le_trans_nat _ _ _ hCoercive hPairLift). }
  assert (hGradBase : gradient (state d) <= gradient_norm_poincare_sobolev_inequality d).
  {
    unfold gradient_norm_poincare_sobolev_inequality.
    apply (le_add_right_nat (gradient (state d)) (scale d)).
  }
  assert (hFirst :
    gradient (renorm (state d)) <=
      gradient_norm_poincare_sobolev_inequality d +
      oscillation_norm_poincare_sobolev_inequality d).
  {
    unfold gradient_norm_poincare_sobolev_inequality.
    unfold oscillation_norm_poincare_sobolev_inequality.
    exact hChain.
  }
  split.
  - exact hFirst.
  - exact hGradBase.
Qed.

Lemma compact_embedding_step_poincare_sobolev_inequality
    {E : Type} `{SobolevStruct_poincare_inequality E}
    (d : EnergyData_poincare_sobolev_inequality) :
    oscillation (renorm (state d)) <= energy (state d) + gradient_norm_poincare_sobolev_inequality d.
Proof.
  assert (hCompact : oscillation (renorm (state d)) <= energy (renorm (state d))).
  { apply (compact_step (state d)). }
  assert (hHolder : energy (renorm (state d)) <= energy (state d) + gradient_norm_poincare_sobolev_inequality d).
  { apply (holder_chain_poincare_sobolev_inequality d). }
  apply (le_trans_nat _ _ _ hCompact hHolder).
Qed.

Lemma global_sobolev_bound_poincare_sobolev_inequality
    {E : Type} `{SobolevStruct_poincare_inequality E}
    (d : EnergyData_poincare_sobolev_inequality) :
    oscillation (renorm (state d)) <=
      (energy (state d) + gradient_norm_poincare_sobolev_inequality d) +
        oscillation_norm_poincare_sobolev_inequality d /\
    gradient (localPatch (state d)) <= gradient_norm_poincare_sobolev_inequality d.
Proof.
  assert (hCompact :
    oscillation (renorm (state d)) <= energy (state d) + gradient_norm_poincare_sobolev_inequality d).
  { apply (compact_embedding_step_poincare_sobolev_inequality d). }
  assert (hAddOsc :
    energy (state d) + gradient_norm_poincare_sobolev_inequality d <=
      (energy (state d) + gradient_norm_poincare_sobolev_inequality d) +
        oscillation_norm_poincare_sobolev_inequality d).
  {
    apply (le_add_right_nat
      (energy (state d) + gradient_norm_poincare_sobolev_inequality d)
      (oscillation_norm_poincare_sobolev_inequality d)).
  }
  assert (hFirst :
    oscillation (renorm (state d)) <=
      (energy (state d) + gradient_norm_poincare_sobolev_inequality d) +
        oscillation_norm_poincare_sobolev_inequality d).
  { apply (le_trans_nat _ _ _ hCompact hAddOsc). }
  pose proof (mean_zero_reduction_poincare_sobolev_inequality d) as hReduction.
  split.
  - exact hFirst.
  - exact (proj2 hReduction).
Qed.
