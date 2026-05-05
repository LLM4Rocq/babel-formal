(**
BENCHMARK_ID: TINY_MATHLIB_BATCH07_ANALYSIS_NONARCHIMEDEAN_OPEN_MAPPING_LIKE
PAIR_STEM: analysis_nonarchimedean_open_mapping_like
MATH_DOMAIN: Analysis
SOURCE_MATHLIB: Mathlib/Analysis/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
**)

Set Universe Polymorphism.
Set Implicit Arguments.

Class FrameworkStruct_analysis_nonarchimedean_open_mapping (E : Type) := {
  seminorm : E -> nat;
  add : E -> E -> E;
  zero : E;
  map : E -> E;
  radius : E -> nat;
  le_trans_nat : forall a b c : nat, a <= b -> b <= c -> a <= c;
  add_le_add_left_nat : forall a b c : nat, a <= b -> c + a <= c + b;
  add_le_add_right_nat : forall a b c : nat, a <= b -> a + c <= b + c;
  le_add_left_nat : forall a b : nat, b <= a + b;
  le_add_right_nat : forall a b : nat, a <= a + b;
  map_bound : forall x : E, seminorm (map x) <= seminorm x + radius x;
  map_contract : forall x : E, seminorm (map x) <= seminorm x;
  add_bound : forall x y : E, seminorm (add x y) <= seminorm x + seminorm y;
  zero_norm : seminorm zero = 0;
  add_zero_right : forall x : E, add x zero = x;
  add_zero_left : forall x : E, add zero x = x;
  open_surj : forall x : E, exists y : E, map y = x;
  preimage_bound : forall x y : E, map y = x -> seminorm y <= seminorm x + radius x;
  radius_control : forall x : E, radius (map x) <= radius x + radius x
}.

Record ContextData_analysis_nonarchimedean_open_mapping
    (E : Type) `{FrameworkStruct_analysis_nonarchimedean_open_mapping E} := {
  anchor_analysis_nonarchimedean_open_mapping : E;
  target_analysis_nonarchimedean_open_mapping : E;
  anchor_small_analysis_nonarchimedean_open_mapping :
    seminorm anchor_analysis_nonarchimedean_open_mapping <=
      seminorm target_analysis_nonarchimedean_open_mapping + radius target_analysis_nonarchimedean_open_mapping;
  target_large_analysis_nonarchimedean_open_mapping :
    seminorm target_analysis_nonarchimedean_open_mapping <=
      seminorm target_analysis_nonarchimedean_open_mapping + seminorm target_analysis_nonarchimedean_open_mapping
}.

Definition primary_map_analysis_nonarchimedean_open_mapping
    {E : Type} `{FrameworkStruct_analysis_nonarchimedean_open_mapping E}
    (d : ContextData_analysis_nonarchimedean_open_mapping) : E :=
  add (map (anchor_analysis_nonarchimedean_open_mapping d))
      (target_analysis_nonarchimedean_open_mapping d).

Definition secondary_map_analysis_nonarchimedean_open_mapping
    {E : Type} `{FrameworkStruct_analysis_nonarchimedean_open_mapping E}
    (d : ContextData_analysis_nonarchimedean_open_mapping) : nat :=
  seminorm (primary_map_analysis_nonarchimedean_open_mapping d) +
    radius (target_analysis_nonarchimedean_open_mapping d).

Definition tertiary_map_analysis_nonarchimedean_open_mapping
    {E : Type} `{FrameworkStruct_analysis_nonarchimedean_open_mapping E}
    (d : ContextData_analysis_nonarchimedean_open_mapping) : Prop :=
  radius (target_analysis_nonarchimedean_open_mapping d) <=
    secondary_map_analysis_nonarchimedean_open_mapping d.

Lemma stability_step_analysis_nonarchimedean_open_mapping
    {E : Type} `{FrameworkStruct_analysis_nonarchimedean_open_mapping E}
    (d : ContextData_analysis_nonarchimedean_open_mapping) :
    seminorm (map (anchor_analysis_nonarchimedean_open_mapping d)) <=
      seminorm (anchor_analysis_nonarchimedean_open_mapping d) +
      radius (anchor_analysis_nonarchimedean_open_mapping d) /\
    radius (map (anchor_analysis_nonarchimedean_open_mapping d)) <=
      radius (anchor_analysis_nonarchimedean_open_mapping d) +
      radius (anchor_analysis_nonarchimedean_open_mapping d).
Proof.
  assert (hMap :
      seminorm (map (anchor_analysis_nonarchimedean_open_mapping d)) <=
        seminorm (anchor_analysis_nonarchimedean_open_mapping d) +
        radius (anchor_analysis_nonarchimedean_open_mapping d)).
  { apply map_bound. }
  assert (hRad :
      radius (map (anchor_analysis_nonarchimedean_open_mapping d)) <=
        radius (anchor_analysis_nonarchimedean_open_mapping d) +
        radius (anchor_analysis_nonarchimedean_open_mapping d)).
  { apply radius_control. }
  split.
  - exact hMap.
  - exact hRad.
Qed.

Lemma factorization_step_analysis_nonarchimedean_open_mapping
    {E : Type} `{FrameworkStruct_analysis_nonarchimedean_open_mapping E}
    (d : ContextData_analysis_nonarchimedean_open_mapping)
    (w : E)
    (hw : map w = target_analysis_nonarchimedean_open_mapping d) :
    seminorm w <=
      seminorm (target_analysis_nonarchimedean_open_mapping d) +
      radius (target_analysis_nonarchimedean_open_mapping d) /\
    seminorm (map w) <= seminorm w.
Proof.
  assert (hPre :
      seminorm w <=
        seminorm (target_analysis_nonarchimedean_open_mapping d) +
        radius (target_analysis_nonarchimedean_open_mapping d)).
  { eapply preimage_bound. exact hw. }
  assert (hCon : seminorm (map w) <= seminorm w).
  { apply map_contract. }
  split.
  - exact hPre.
  - exact hCon.
Qed.

Lemma comparison_step_analysis_nonarchimedean_open_mapping
    {E : Type} `{FrameworkStruct_analysis_nonarchimedean_open_mapping E}
    (d : ContextData_analysis_nonarchimedean_open_mapping) :
    seminorm (primary_map_analysis_nonarchimedean_open_mapping d) <=
      seminorm (map (anchor_analysis_nonarchimedean_open_mapping d)) +
      seminorm (target_analysis_nonarchimedean_open_mapping d) /\
    tertiary_map_analysis_nonarchimedean_open_mapping d.
Proof.
  assert (hPrimary :
      seminorm (primary_map_analysis_nonarchimedean_open_mapping d) <=
        seminorm (map (anchor_analysis_nonarchimedean_open_mapping d)) +
        seminorm (target_analysis_nonarchimedean_open_mapping d)).
  {
    unfold primary_map_analysis_nonarchimedean_open_mapping.
    apply add_bound.
  }
  assert (hRadius :
      radius (target_analysis_nonarchimedean_open_mapping d) <=
        seminorm (primary_map_analysis_nonarchimedean_open_mapping d) +
        radius (target_analysis_nonarchimedean_open_mapping d)).
  {
    apply le_add_left_nat.
  }
  assert (hTer : tertiary_map_analysis_nonarchimedean_open_mapping d).
  {
    unfold tertiary_map_analysis_nonarchimedean_open_mapping.
    unfold secondary_map_analysis_nonarchimedean_open_mapping.
    exact hRadius.
  }
  split.
  - exact hPrimary.
  - exact hTer.
Qed.

Lemma transport_step_analysis_nonarchimedean_open_mapping
    {E : Type} `{FrameworkStruct_analysis_nonarchimedean_open_mapping E}
    (d : ContextData_analysis_nonarchimedean_open_mapping) :
    seminorm (primary_map_analysis_nonarchimedean_open_mapping d) <=
      secondary_map_analysis_nonarchimedean_open_mapping d /\
    seminorm (target_analysis_nonarchimedean_open_mapping d) <=
      secondary_map_analysis_nonarchimedean_open_mapping d +
      seminorm (target_analysis_nonarchimedean_open_mapping d) /\
    tertiary_map_analysis_nonarchimedean_open_mapping d.
Proof.
  destruct (comparison_step_analysis_nonarchimedean_open_mapping d) as [hPrimary hTer].
  assert (hFirst :
      seminorm (primary_map_analysis_nonarchimedean_open_mapping d) <=
        secondary_map_analysis_nonarchimedean_open_mapping d).
  {
    unfold secondary_map_analysis_nonarchimedean_open_mapping.
    apply le_add_right_nat.
  }
  assert (hSecond :
      seminorm (target_analysis_nonarchimedean_open_mapping d) <=
        secondary_map_analysis_nonarchimedean_open_mapping d +
        seminorm (target_analysis_nonarchimedean_open_mapping d)).
  {
    apply le_add_left_nat.
  }
  split.
  - exact hFirst.
  - split.
    + exact hSecond.
    + exact hTer.
Qed.

Lemma coherence_step_analysis_nonarchimedean_open_mapping
    {E : Type} `{FrameworkStruct_analysis_nonarchimedean_open_mapping E}
    (d : ContextData_analysis_nonarchimedean_open_mapping) :
    seminorm (target_analysis_nonarchimedean_open_mapping d) <=
      secondary_map_analysis_nonarchimedean_open_mapping d +
      seminorm (target_analysis_nonarchimedean_open_mapping d) /\
    tertiary_map_analysis_nonarchimedean_open_mapping d.
Proof.
  assert (hLeft :
      seminorm (target_analysis_nonarchimedean_open_mapping d) <=
        secondary_map_analysis_nonarchimedean_open_mapping d +
        seminorm (target_analysis_nonarchimedean_open_mapping d)).
  {
    apply le_add_left_nat.
  }
  destruct (comparison_step_analysis_nonarchimedean_open_mapping d) as [_ hTer].
  split.
  - exact hLeft.
  - exact hTer.
Qed.

Lemma iteration_step_analysis_nonarchimedean_open_mapping
    {E : Type} `{FrameworkStruct_analysis_nonarchimedean_open_mapping E}
    (d : ContextData_analysis_nonarchimedean_open_mapping) :
    exists w : E,
      map w = target_analysis_nonarchimedean_open_mapping d /\
      seminorm w <=
        seminorm (target_analysis_nonarchimedean_open_mapping d) +
        radius (target_analysis_nonarchimedean_open_mapping d) /\
      seminorm (map w) <= seminorm w.
Proof.
  destruct (open_surj (target_analysis_nonarchimedean_open_mapping d)) as [w hw].
  destruct (factorization_step_analysis_nonarchimedean_open_mapping d w hw) as [hNorm hCon].
  exists w.
  repeat split; assumption.
Qed.

Lemma main_result_analysis_nonarchimedean_open_mapping
    {E : Type} `{FrameworkStruct_analysis_nonarchimedean_open_mapping E}
    (d : ContextData_analysis_nonarchimedean_open_mapping) :
    exists w : E,
      map w = target_analysis_nonarchimedean_open_mapping d /\
      seminorm w <=
        (seminorm (target_analysis_nonarchimedean_open_mapping d) +
         radius (target_analysis_nonarchimedean_open_mapping d)) +
        radius (target_analysis_nonarchimedean_open_mapping d) /\
      tertiary_map_analysis_nonarchimedean_open_mapping d.
Proof.
  destruct (iteration_step_analysis_nonarchimedean_open_mapping d)
    as [w [hwMap [hwNorm hwCon]]].
  assert (hLift :
      seminorm w <=
        (seminorm (target_analysis_nonarchimedean_open_mapping d) +
         radius (target_analysis_nonarchimedean_open_mapping d)) +
        radius (target_analysis_nonarchimedean_open_mapping d)).
  {
    apply le_trans_nat with
      (b := seminorm (target_analysis_nonarchimedean_open_mapping d) +
            radius (target_analysis_nonarchimedean_open_mapping d)).
    - exact hwNorm.
    - apply le_add_right_nat.
  }
  destruct (coherence_step_analysis_nonarchimedean_open_mapping d)
    as [hTarget hTer].
  assert (_hContract : seminorm (map w) <= seminorm w).
  { exact hwCon. }
  assert (_hTargetUse :
      seminorm (target_analysis_nonarchimedean_open_mapping d) <=
      secondary_map_analysis_nonarchimedean_open_mapping d +
      seminorm (target_analysis_nonarchimedean_open_mapping d)).
  { exact hTarget. }
  exists w.
  repeat split; try assumption.
Qed.
