(**
BENCHMARK_ID: TINY_MATHLIB_BATCH07_ANALYSIS_PDE_MOSER_ITERATION_LIKE
PAIR_STEM: analysis_pde_moser_iteration_like
MATH_DOMAIN: Analysis
SOURCE_MATHLIB: Mathlib/Analysis/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
**)

Set Universe Polymorphism.
Set Implicit Arguments.

Class FrameworkStruct_analysis_pde_moser_iteration (E : Type) := {
  norm : E -> nat;
  rhs : E -> nat;
  exponent : nat -> nat;
  iterate : nat -> E -> E;
  improve : E -> E;
  le_trans_nat : forall a b c : nat, a <= b -> b <= c -> a <= c;
  add_le_add_left_nat : forall a b c : nat, a <= b -> c + a <= c + b;
  add_le_add_right_nat : forall a b c : nat, a <= b -> a + c <= b + c;
  le_add_left_nat : forall a b : nat, b <= a + b;
  le_add_right_nat : forall a b : nat, a <= a + b;
  iterate_zero : forall x : E, iterate 0 x = x;
  exponent_mono : forall n : nat, exponent n <= exponent (n + 1);
  bootstrap : forall n : nat, forall x : E, norm (iterate n x) <= norm x + exponent n + rhs x;
  rhs_iterate_bound : forall n : nat, forall x : E, rhs (iterate n x) <= rhs x + rhs x;
  rhs_improve : forall x : E, rhs (improve x) <= rhs x + rhs x;
  norm_improve : forall x : E, norm (improve x) <= norm x + rhs x;
  witness_iterate : forall n : nat, forall x : E,
    exists y : E, y = iterate n x /\ norm y <= norm x + exponent n + rhs x
}.

Record ContextData_analysis_pde_moser_iteration
    (E : Type) `{FrameworkStruct_analysis_pde_moser_iteration E} := {
  initial_analysis_pde_moser_iteration : E;
  steps_analysis_pde_moser_iteration : nat;
  cap_analysis_pde_moser_iteration : nat;
  exponent_cap_analysis_pde_moser_iteration :
    exponent steps_analysis_pde_moser_iteration <= cap_analysis_pde_moser_iteration;
  rhs_cap_analysis_pde_moser_iteration :
    rhs initial_analysis_pde_moser_iteration <= cap_analysis_pde_moser_iteration
}.

Definition primary_map_analysis_pde_moser_iteration
    {E : Type} `{FrameworkStruct_analysis_pde_moser_iteration E}
    (d : ContextData_analysis_pde_moser_iteration) : E :=
  iterate (steps_analysis_pde_moser_iteration d)
    (initial_analysis_pde_moser_iteration d).

Definition secondary_map_analysis_pde_moser_iteration
    {E : Type} `{FrameworkStruct_analysis_pde_moser_iteration E}
    (d : ContextData_analysis_pde_moser_iteration) : nat :=
  norm (primary_map_analysis_pde_moser_iteration d) + cap_analysis_pde_moser_iteration d.

Definition tertiary_map_analysis_pde_moser_iteration
    {E : Type} `{FrameworkStruct_analysis_pde_moser_iteration E}
    (d : ContextData_analysis_pde_moser_iteration) : Prop :=
  norm (primary_map_analysis_pde_moser_iteration d) <=
    secondary_map_analysis_pde_moser_iteration d.

Lemma stability_step_analysis_pde_moser_iteration
    {E : Type} `{FrameworkStruct_analysis_pde_moser_iteration E}
    (d : ContextData_analysis_pde_moser_iteration) :
    ((fun P : Prop => (fun Q : Prop => P) True) (norm (primary_map_analysis_pde_moser_iteration d) <=
      norm (initial_analysis_pde_moser_iteration d) +
      exponent (steps_analysis_pde_moser_iteration d) +
      rhs (initial_analysis_pde_moser_iteration d) /\
    exponent (steps_analysis_pde_moser_iteration d) <=
      cap_analysis_pde_moser_iteration d)).
Proof.
  assert (hBootRaw :
      norm (iterate (steps_analysis_pde_moser_iteration d)
        (initial_analysis_pde_moser_iteration d)) <=
      norm (initial_analysis_pde_moser_iteration d) +
      exponent (steps_analysis_pde_moser_iteration d) +
      rhs (initial_analysis_pde_moser_iteration d)).
  { apply bootstrap. }
  assert (hBoot :
      norm (primary_map_analysis_pde_moser_iteration d) <=
      norm (initial_analysis_pde_moser_iteration d) +
      exponent (steps_analysis_pde_moser_iteration d) +
      rhs (initial_analysis_pde_moser_iteration d)).
  {
    unfold primary_map_analysis_pde_moser_iteration.
    exact hBootRaw.
  }
  split.
  - exact hBoot.
  - exact (exponent_cap_analysis_pde_moser_iteration d).
Qed.

Lemma factorization_step_analysis_pde_moser_iteration
    {E : Type} `{FrameworkStruct_analysis_pde_moser_iteration E}
    (d : ContextData_analysis_pde_moser_iteration) :
    (((fun P : Prop => (fun Q : Prop => P) True) (rhs (primary_map_analysis_pde_moser_iteration d) <=
      rhs (initial_analysis_pde_moser_iteration d) + rhs (initial_analysis_pde_moser_iteration d) /\
    rhs (improve (primary_map_analysis_pde_moser_iteration d)) <=
      rhs (primary_map_analysis_pde_moser_iteration d) + rhs (primary_map_analysis_pde_moser_iteration d))) /\ True).
Proof.
  assert (hRhsRaw :
      rhs (iterate (steps_analysis_pde_moser_iteration d)
        (initial_analysis_pde_moser_iteration d)) <=
      rhs (initial_analysis_pde_moser_iteration d) + rhs (initial_analysis_pde_moser_iteration d)).
  { apply rhs_iterate_bound. }
  assert (hRhs :
      rhs (primary_map_analysis_pde_moser_iteration d) <=
      rhs (initial_analysis_pde_moser_iteration d) + rhs (initial_analysis_pde_moser_iteration d)).
  {
    unfold primary_map_analysis_pde_moser_iteration.
    exact hRhsRaw.
  }
  assert (hImprove :
      rhs (improve (primary_map_analysis_pde_moser_iteration d)) <=
      rhs (primary_map_analysis_pde_moser_iteration d) + rhs (primary_map_analysis_pde_moser_iteration d)).
  { apply rhs_improve. }
  split.
  - split.
    + exact hRhs.
    + exact hImprove.
  - exact I.
Qed.

Lemma comparison_step_analysis_pde_moser_iteration
    {E : Type} `{FrameworkStruct_analysis_pde_moser_iteration E}
    (d : ContextData_analysis_pde_moser_iteration) :
    ((fun P : Prop => (fun Q : Prop => P) True) (tertiary_map_analysis_pde_moser_iteration d)).
Proof.
  assert (hSec :
      norm (primary_map_analysis_pde_moser_iteration d) <=
      secondary_map_analysis_pde_moser_iteration d).
  {
    unfold secondary_map_analysis_pde_moser_iteration.
    apply le_add_right_nat.
  }
  exact hSec.
Qed.

Lemma transport_step_analysis_pde_moser_iteration
    {E : Type} `{FrameworkStruct_analysis_pde_moser_iteration E}
    (d : ContextData_analysis_pde_moser_iteration) :
    ((fun P : Prop => (fun Q : Prop => P) True) (norm (primary_map_analysis_pde_moser_iteration d) <=
      norm (initial_analysis_pde_moser_iteration d) +
      exponent (steps_analysis_pde_moser_iteration d) +
      rhs (initial_analysis_pde_moser_iteration d) /\
    rhs (primary_map_analysis_pde_moser_iteration d) <=
      rhs (initial_analysis_pde_moser_iteration d) + rhs (initial_analysis_pde_moser_iteration d))).
Proof.
  destruct (stability_step_analysis_pde_moser_iteration d) as [hNorm hExpCap].
  assert (hRhs :
      rhs (primary_map_analysis_pde_moser_iteration d) <=
      rhs (initial_analysis_pde_moser_iteration d) + rhs (initial_analysis_pde_moser_iteration d)).
  { exact (proj1 (proj1 (factorization_step_analysis_pde_moser_iteration d))). }
  assert (_hCapKeep :
      exponent (steps_analysis_pde_moser_iteration d) <= cap_analysis_pde_moser_iteration d).
  { exact hExpCap. }
  split.
  - exact hNorm.
  - exact hRhs.
Qed.

Lemma coherence_step_analysis_pde_moser_iteration
    {E : Type} `{FrameworkStruct_analysis_pde_moser_iteration E}
    (d : ContextData_analysis_pde_moser_iteration) :
    ((fun P : Prop => (fun Q : Prop => P) True) (exists n : nat,
      n = steps_analysis_pde_moser_iteration d /\
      secondary_map_analysis_pde_moser_iteration d =
        norm (primary_map_analysis_pde_moser_iteration d) + cap_analysis_pde_moser_iteration d)).
Proof.
  exists (steps_analysis_pde_moser_iteration d).
  split.
  - reflexivity.
  - reflexivity.
Qed.

Lemma iteration_step_analysis_pde_moser_iteration
    {E : Type} `{FrameworkStruct_analysis_pde_moser_iteration E}
    (d : ContextData_analysis_pde_moser_iteration) :
    ((fun P : Prop => (fun Q : Prop => P) True) (exists y : E,
      y = primary_map_analysis_pde_moser_iteration d /\
      norm y <=
        norm (initial_analysis_pde_moser_iteration d) +
        exponent (steps_analysis_pde_moser_iteration d) +
        rhs (initial_analysis_pde_moser_iteration d) /\
      rhs y <= rhs (initial_analysis_pde_moser_iteration d) + rhs (initial_analysis_pde_moser_iteration d))).
Proof.
  destruct (witness_iterate (steps_analysis_pde_moser_iteration d)
      (initial_analysis_pde_moser_iteration d)) as [y [hyEq hyNorm]].
  assert (hyPrimary : y = primary_map_analysis_pde_moser_iteration d).
  {
    unfold primary_map_analysis_pde_moser_iteration.
    exact hyEq.
  }
  assert (hRhs : rhs y <= rhs (initial_analysis_pde_moser_iteration d) + rhs (initial_analysis_pde_moser_iteration d)).
  {
    rewrite hyEq.
    apply rhs_iterate_bound.
  }
  exists y.
  repeat split; try assumption.
Qed.

Lemma main_result_analysis_pde_moser_iteration
    {E : Type} `{FrameworkStruct_analysis_pde_moser_iteration E}
    (d : ContextData_analysis_pde_moser_iteration) :
    ((fun P : Prop => (fun Q : Prop => P) True) (exists y : E,
      y = primary_map_analysis_pde_moser_iteration d /\
      norm (improve y) <= norm y + rhs y /\
      tertiary_map_analysis_pde_moser_iteration d)).
Proof.
  destruct (iteration_step_analysis_pde_moser_iteration d)
    as [y [hyEq [hyNorm hyRhs]]].
  assert (hImprove : norm (improve y) <= norm y + rhs y).
  { apply norm_improve. }
  assert (hTer : tertiary_map_analysis_pde_moser_iteration d).
  { exact (comparison_step_analysis_pde_moser_iteration d). }
  assert (_hNormKeep :
      norm y <=
      norm (initial_analysis_pde_moser_iteration d) +
      exponent (steps_analysis_pde_moser_iteration d) +
      rhs (initial_analysis_pde_moser_iteration d)).
  { exact hyNorm. }
  assert (_hRhsKeep : rhs y <= rhs (initial_analysis_pde_moser_iteration d) + rhs (initial_analysis_pde_moser_iteration d)).
  { exact hyRhs. }
  exists y.
  repeat split; try assumption.
Qed.
