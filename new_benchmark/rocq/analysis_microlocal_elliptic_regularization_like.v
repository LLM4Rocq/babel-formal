(**
BENCHMARK_ID: TINY_MATHLIB_BATCH07_ANALYSIS_MICROLOCAL_ELLIPTIC_REGULARIZATION_LIKE
PAIR_STEM: analysis_microlocal_elliptic_regularization_like
MATH_DOMAIN: Analysis
SOURCE_MATHLIB: Mathlib/Analysis/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
**)

Set Universe Polymorphism.
Set Implicit Arguments.

Class FrameworkStruct_analysis_microlocal_elliptic_regularization (E : Type) := {
  order : E -> nat;
  defect : E -> nat;
  regularize : E -> E;
  commutator : E -> E;
  parametrix : E -> E;
  le_trans_nat : forall a b c : nat, a <= b -> b <= c -> a <= c;
  add_le_add_left_nat : forall a b c : nat, a <= b -> c + a <= c + b;
  add_le_add_right_nat : forall a b c : nat, a <= b -> a + c <= b + c;
  le_add_left_nat : forall a b : nat, b <= a + b;
  le_add_right_nat : forall a b : nat, a <= a + b;
  order_regularize : forall x : E, order (regularize x) <= order x + defect x;
  defect_regularize : forall x : E, defect (regularize x) <= defect x + defect x;
  order_commutator : forall x : E, order (commutator x) <= order x + defect x;
  parametrix_gain : forall x : E, order (parametrix x) <= order x;
  elliptic_step : forall x : E, order (parametrix (regularize x)) <= order x + defect x;
  witness_param : forall x : E, exists y : E, y = parametrix x /\ order y <= order x
}.

Record ContextData_analysis_microlocal_elliptic_regularization
    (E : Type) `{FrameworkStruct_analysis_microlocal_elliptic_regularization E} := {
  source_analysis_microlocal_elliptic_regularization : E;
  target_analysis_microlocal_elliptic_regularization : E;
  budget_analysis_microlocal_elliptic_regularization : nat;
  source_to_target_analysis_microlocal_elliptic_regularization :
    order source_analysis_microlocal_elliptic_regularization <=
      order target_analysis_microlocal_elliptic_regularization + budget_analysis_microlocal_elliptic_regularization;
  defect_cap_analysis_microlocal_elliptic_regularization :
    defect source_analysis_microlocal_elliptic_regularization <= budget_analysis_microlocal_elliptic_regularization
}.

Definition primary_map_analysis_microlocal_elliptic_regularization
    {E : Type} `{FrameworkStruct_analysis_microlocal_elliptic_regularization E}
    (d : ContextData_analysis_microlocal_elliptic_regularization) : E :=
  parametrix (regularize (source_analysis_microlocal_elliptic_regularization d)).

Definition secondary_map_analysis_microlocal_elliptic_regularization
    {E : Type} `{FrameworkStruct_analysis_microlocal_elliptic_regularization E}
    (d : ContextData_analysis_microlocal_elliptic_regularization) : nat :=
  order (primary_map_analysis_microlocal_elliptic_regularization d) + budget_analysis_microlocal_elliptic_regularization d.

Definition tertiary_map_analysis_microlocal_elliptic_regularization
    {E : Type} `{FrameworkStruct_analysis_microlocal_elliptic_regularization E}
    (d : ContextData_analysis_microlocal_elliptic_regularization) : Prop :=
  order (primary_map_analysis_microlocal_elliptic_regularization d) <=
    secondary_map_analysis_microlocal_elliptic_regularization d.

Lemma stability_step_analysis_microlocal_elliptic_regularization
    {E : Type} `{FrameworkStruct_analysis_microlocal_elliptic_regularization E}
    (d : ContextData_analysis_microlocal_elliptic_regularization) :
    ((fun P : Prop => (fun n : nat => P) O) (order (primary_map_analysis_microlocal_elliptic_regularization d) <=
      order (source_analysis_microlocal_elliptic_regularization d) +
      defect (source_analysis_microlocal_elliptic_regularization d) /\
    defect (regularize (source_analysis_microlocal_elliptic_regularization d)) <=
      defect (source_analysis_microlocal_elliptic_regularization d) +
      defect (source_analysis_microlocal_elliptic_regularization d))).
Proof.
  assert (hOrderRaw :
      order (parametrix (regularize (source_analysis_microlocal_elliptic_regularization d))) <=
      order (source_analysis_microlocal_elliptic_regularization d) +
      defect (source_analysis_microlocal_elliptic_regularization d)).
  { apply elliptic_step. }
  assert (hOrder :
      order (primary_map_analysis_microlocal_elliptic_regularization d) <=
      order (source_analysis_microlocal_elliptic_regularization d) +
      defect (source_analysis_microlocal_elliptic_regularization d)).
  {
    unfold primary_map_analysis_microlocal_elliptic_regularization.
    exact hOrderRaw.
  }
  assert (hDefect :
      defect (regularize (source_analysis_microlocal_elliptic_regularization d)) <=
      defect (source_analysis_microlocal_elliptic_regularization d) +
      defect (source_analysis_microlocal_elliptic_regularization d)).
  { apply defect_regularize. }
  split.
  - exact hOrder.
  - exact hDefect.
Qed.

Lemma factorization_step_analysis_microlocal_elliptic_regularization
    {E : Type} `{FrameworkStruct_analysis_microlocal_elliptic_regularization E}
    (d : ContextData_analysis_microlocal_elliptic_regularization) :
    ((fun P : Prop => (fun n : nat => P) O) (order (commutator (source_analysis_microlocal_elliptic_regularization d)) <=
      order (source_analysis_microlocal_elliptic_regularization d) +
      defect (source_analysis_microlocal_elliptic_regularization d) /\
    (order (parametrix (source_analysis_microlocal_elliptic_regularization d)) <=
      order (source_analysis_microlocal_elliptic_regularization d) ->
      order (parametrix (source_analysis_microlocal_elliptic_regularization d)) <=
      order (source_analysis_microlocal_elliptic_regularization d)))).
Proof.
  assert (hComm :
      order (commutator (source_analysis_microlocal_elliptic_regularization d)) <=
      order (source_analysis_microlocal_elliptic_regularization d) +
      defect (source_analysis_microlocal_elliptic_regularization d)).
  { apply order_commutator. }
  assert (hPar :
      order (parametrix (source_analysis_microlocal_elliptic_regularization d)) <=
      order (source_analysis_microlocal_elliptic_regularization d)).
  { apply parametrix_gain. }
  assert (hId :
      order (parametrix (source_analysis_microlocal_elliptic_regularization d)) <=
      order (source_analysis_microlocal_elliptic_regularization d) ->
      order (parametrix (source_analysis_microlocal_elliptic_regularization d)) <=
      order (source_analysis_microlocal_elliptic_regularization d)).
  {
    intro h.
    exact h.
  }
  assert (_hParKeep :
      order (parametrix (source_analysis_microlocal_elliptic_regularization d)) <=
      order (source_analysis_microlocal_elliptic_regularization d)).
  { exact hPar. }
  split.
  - exact hComm.
  - exact hId.
Qed.

Lemma comparison_step_analysis_microlocal_elliptic_regularization
    {E : Type} `{FrameworkStruct_analysis_microlocal_elliptic_regularization E}
    (d : ContextData_analysis_microlocal_elliptic_regularization) :
    ((fun P : Prop => (fun n : nat => P) O) (tertiary_map_analysis_microlocal_elliptic_regularization d /\
    (tertiary_map_analysis_microlocal_elliptic_regularization d ->
      order (parametrix (source_analysis_microlocal_elliptic_regularization d)) <=
      order (source_analysis_microlocal_elliptic_regularization d)))).
Proof.
  assert (hOrdSec :
      order (primary_map_analysis_microlocal_elliptic_regularization d) <=
      secondary_map_analysis_microlocal_elliptic_regularization d).
  {
    unfold secondary_map_analysis_microlocal_elliptic_regularization.
    apply le_add_right_nat.
  }
  split.
  - exact hOrdSec.
  - intro _h.
    apply parametrix_gain.
Qed.

Lemma transport_step_analysis_microlocal_elliptic_regularization
    {E : Type} `{FrameworkStruct_analysis_microlocal_elliptic_regularization E}
    (d : ContextData_analysis_microlocal_elliptic_regularization) :
    ((fun P : Prop => (fun n : nat => P) O) (order (primary_map_analysis_microlocal_elliptic_regularization d) <=
      order (target_analysis_microlocal_elliptic_regularization d) +
      budget_analysis_microlocal_elliptic_regularization d +
      defect (source_analysis_microlocal_elliptic_regularization d) /\
    defect (regularize (source_analysis_microlocal_elliptic_regularization d)) <=
      defect (source_analysis_microlocal_elliptic_regularization d) +
      defect (source_analysis_microlocal_elliptic_regularization d))).
Proof.
  destruct (stability_step_analysis_microlocal_elliptic_regularization d)
    as [hOrder hDefect].
  assert (hSourceTarget :
      order (source_analysis_microlocal_elliptic_regularization d) +
      defect (source_analysis_microlocal_elliptic_regularization d) <=
      (order (target_analysis_microlocal_elliptic_regularization d) +
      budget_analysis_microlocal_elliptic_regularization d) +
      defect (source_analysis_microlocal_elliptic_regularization d)).
  {
    apply add_le_add_right_nat.
    exact (source_to_target_analysis_microlocal_elliptic_regularization d).
  }
  assert (hFirst :
      order (primary_map_analysis_microlocal_elliptic_regularization d) <=
      order (target_analysis_microlocal_elliptic_regularization d) +
      budget_analysis_microlocal_elliptic_regularization d +
      defect (source_analysis_microlocal_elliptic_regularization d)).
  {
    apply le_trans_nat with
      (b := order (source_analysis_microlocal_elliptic_regularization d) +
            defect (source_analysis_microlocal_elliptic_regularization d)).
    - exact hOrder.
    - exact hSourceTarget.
  }
  assert (hSecond :
      defect (regularize (source_analysis_microlocal_elliptic_regularization d)) <=
      defect (source_analysis_microlocal_elliptic_regularization d) +
      defect (source_analysis_microlocal_elliptic_regularization d)).
  { exact hDefect. }
  split.
  - exact hFirst.
  - exact hSecond.
Qed.

Lemma coherence_step_analysis_microlocal_elliptic_regularization
    {E : Type} `{FrameworkStruct_analysis_microlocal_elliptic_regularization E}
    (d : ContextData_analysis_microlocal_elliptic_regularization) :
    ((fun P : Prop => (fun n : nat => P) O) (forall z : E,
      z = parametrix (source_analysis_microlocal_elliptic_regularization d) ->
      order z <= order (source_analysis_microlocal_elliptic_regularization d))).
Proof.
  intros z hzEq.
  rewrite hzEq.
  apply parametrix_gain.
Qed.

Lemma iteration_step_analysis_microlocal_elliptic_regularization
    {E : Type} `{FrameworkStruct_analysis_microlocal_elliptic_regularization E}
    (d : ContextData_analysis_microlocal_elliptic_regularization) :
    ((fun P : Prop => (fun n : nat => P) O) (exists y : E,
      y = primary_map_analysis_microlocal_elliptic_regularization d /\
      order y <= order (source_analysis_microlocal_elliptic_regularization d) + defect (source_analysis_microlocal_elliptic_regularization d) /\
      tertiary_map_analysis_microlocal_elliptic_regularization d)).
Proof.
  set (y := primary_map_analysis_microlocal_elliptic_regularization d).
  assert (hyEq : y = primary_map_analysis_microlocal_elliptic_regularization d).
  { reflexivity. }
  assert (hOrder :
      order y <= order (source_analysis_microlocal_elliptic_regularization d) + defect (source_analysis_microlocal_elliptic_regularization d)).
  {
    rewrite hyEq.
    exact (proj1 (stability_step_analysis_microlocal_elliptic_regularization d)).
  }
  assert (hTer : tertiary_map_analysis_microlocal_elliptic_regularization d).
  { exact (proj1 (comparison_step_analysis_microlocal_elliptic_regularization d)). }
  exists y.
  repeat split; try assumption.
Qed.

Lemma main_result_analysis_microlocal_elliptic_regularization
    {E : Type} `{FrameworkStruct_analysis_microlocal_elliptic_regularization E}
    (d : ContextData_analysis_microlocal_elliptic_regularization) :
    ((fun P : Prop => (fun n : nat => P) O) (exists y : E,
      y = primary_map_analysis_microlocal_elliptic_regularization d /\
      order y <=
        order (target_analysis_microlocal_elliptic_regularization d) +
        budget_analysis_microlocal_elliptic_regularization d +
        defect (source_analysis_microlocal_elliptic_regularization d) /\
      tertiary_map_analysis_microlocal_elliptic_regularization d)).
Proof.
  destruct (iteration_step_analysis_microlocal_elliptic_regularization d)
    as [y [hyEq [hyOrd hTer]]].
  assert (hTransport :
      order (primary_map_analysis_microlocal_elliptic_regularization d) <=
      order (target_analysis_microlocal_elliptic_regularization d) +
      budget_analysis_microlocal_elliptic_regularization d +
      defect (source_analysis_microlocal_elliptic_regularization d)).
  { exact (proj1 (transport_step_analysis_microlocal_elliptic_regularization d)). }
  assert (hFinal :
      order y <=
      order (target_analysis_microlocal_elliptic_regularization d) +
      budget_analysis_microlocal_elliptic_regularization d +
      defect (source_analysis_microlocal_elliptic_regularization d)).
  {
    rewrite hyEq.
    exact hTransport.
  }
  exists y.
  repeat split; try assumption.
Qed.
