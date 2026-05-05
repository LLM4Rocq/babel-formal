(*
BENCHMARK_ID: TINY_MATHLIB_BATCH06_ANALYSIS_HORMANDER_L2_ESTIMATE_LIKE
PAIR_STEM: analysis_hormander_l2_estimate_like
MATH_DOMAIN: Analysis
SOURCE_MATHLIB: Mathlib/Analysis/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class HormanderStruct_l2_estimate (E : Type) := {
  l2norm : E -> nat;
  rhs : E -> nat;
  curvature : E -> nat;
  error : E -> nat;
  adjoint : E -> E;
  solver : E -> E;
  le_trans_nat : forall a b c : nat, a <= b -> b <= c -> a <= c;
  add_le_add_left_nat : forall a b c : nat, a <= b -> c + a <= c + b;
  add_le_add_right_nat : forall a b c : nat, a <= b -> a + c <= b + c;
  add_le_add_nat : forall a b c d : nat, a <= b -> c <= d -> a + c <= b + d;
  le_add_right_nat : forall a b : nat, a <= a + b;
  le_add_left_nat : forall a b : nat, b <= a + b;
  bochner_axiom :
    forall x : E, l2norm (adjoint x) + curvature x <= l2norm x + rhs x;
  weighted_cauchy_axiom :
    forall x : E, rhs x <= l2norm x + error x;
  coercive_axiom :
    forall x : E, l2norm (solver x) <= rhs x + error x;
  weak_solution_axiom :
    forall x : E, error (solver x) <= error x + curvature x;
  minimizer_axiom :
    forall x : E, l2norm (adjoint (solver x)) <= l2norm (solver x) + curvature x;
  final_axiom :
    forall x : E, l2norm (solver x) + l2norm (adjoint (solver x)) <= rhs x + rhs x + curvature x
}.

Arguments le_trans_nat {E} {_} _ _ _ _ _.
Arguments add_le_add_left_nat {E} {_} _ _ _ _.
Arguments add_le_add_right_nat {E} {_} _ _ _ _.
Arguments add_le_add_nat {E} {_} _ _ _ _ _ _.
Arguments le_add_right_nat {E} {_} _ _.
Arguments le_add_left_nat {E} {_} _ _.

Record WeightData_hormander_l2_estimate (E : Type)
    `{HormanderStruct_l2_estimate E} := {
  state : E;
  weight : nat;
  weight_pos : 0 < weight;
  rhs_le_weight : rhs state <= weight;
  error_le_curved : error state <= curvature state + weight
}.

Definition adjoint_operator_hormander_l2_estimate
    {E : Type} `{HormanderStruct_l2_estimate E}
    (d : WeightData_hormander_l2_estimate) : E :=
  adjoint (state d).

Definition curvature_form_hormander_l2_estimate
    {E : Type} `{HormanderStruct_l2_estimate E}
    (d : WeightData_hormander_l2_estimate) : nat :=
  curvature (state d) + weight d.

Definition solution_operator_hormander_l2_estimate
    {E : Type} `{HormanderStruct_l2_estimate E}
    (d : WeightData_hormander_l2_estimate) : E :=
  solver (state d).

Lemma bochner_identity_step_hormander_l2_estimate
    {E : Type} `{HormanderStruct_l2_estimate E}
    (d : WeightData_hormander_l2_estimate) :
    l2norm (adjoint_operator_hormander_l2_estimate d) + curvature (state d) <=
      l2norm (state d) + weight d /\
    l2norm (adjoint_operator_hormander_l2_estimate d) + curvature (state d) <=
      l2norm (state d) + curvature_form_hormander_l2_estimate d.
Proof.
  assert (hBochner :
    l2norm (adjoint (state d)) + curvature (state d) <= l2norm (state d) + rhs (state d)).
  { apply (bochner_axiom (state d)). }
  assert (hRhsLift : l2norm (state d) + rhs (state d) <= l2norm (state d) + weight d).
  {
    apply (add_le_add_left_nat (rhs (state d)) (weight d) (l2norm (state d))).
    exact (rhs_le_weight d).
  }
  assert (hFirstRaw : l2norm (adjoint (state d)) + curvature (state d) <= l2norm (state d) + weight d).
  { apply (le_trans_nat _ _ _ hBochner hRhsLift). }
  assert (hWeightLift : weight d <= curvature (state d) + weight d).
  { apply (le_add_left_nat (curvature (state d)) (weight d)). }
  assert (hSecondLift :
    l2norm (state d) + weight d <= l2norm (state d) + (curvature (state d) + weight d)).
  {
    apply (add_le_add_left_nat (weight d) (curvature (state d) + weight d) (l2norm (state d))).
    exact hWeightLift.
  }
  assert (hSecondRaw :
    l2norm (adjoint (state d)) + curvature (state d) <=
      l2norm (state d) + (curvature (state d) + weight d)).
  { apply (le_trans_nat _ _ _ hFirstRaw hSecondLift). }
  split.
  - unfold adjoint_operator_hormander_l2_estimate.
    exact hFirstRaw.
  - unfold adjoint_operator_hormander_l2_estimate.
    unfold curvature_form_hormander_l2_estimate.
    exact hSecondRaw.
Qed.

Lemma positivity_curvature_step_hormander_l2_estimate
    {E : Type} `{HormanderStruct_l2_estimate E}
    (d : WeightData_hormander_l2_estimate) :
    rhs (state d) <= curvature_form_hormander_l2_estimate d /\
    error (state d) <= curvature_form_hormander_l2_estimate d + curvature (state d).
Proof.
  assert (hWeightLift : weight d <= curvature (state d) + weight d).
  { apply (le_add_left_nat (curvature (state d)) (weight d)). }
  assert (hRhsCurved : rhs (state d) <= curvature (state d) + weight d).
  { apply (le_trans_nat _ _ _ (rhs_le_weight d) hWeightLift). }
  assert (hErrBase : error (state d) <= curvature (state d) + weight d).
  { exact (error_le_curved d). }
  assert (hErrLift :
    curvature (state d) + weight d <=
      (curvature (state d) + weight d) + curvature (state d)).
  { apply (le_add_right_nat (curvature (state d) + weight d) (curvature (state d))). }
  assert (hErrRaw :
    error (state d) <= (curvature (state d) + weight d) + curvature (state d)).
  { apply (le_trans_nat _ _ _ hErrBase hErrLift). }
  split.
  - unfold curvature_form_hormander_l2_estimate.
    exact hRhsCurved.
  - unfold curvature_form_hormander_l2_estimate.
    exact hErrRaw.
Qed.

Lemma weighted_cauchy_step_hormander_l2_estimate
    {E : Type} `{HormanderStruct_l2_estimate E}
    (d : WeightData_hormander_l2_estimate) :
    rhs (state d) <= l2norm (state d) + error (state d) /\
    rhs (state d) <= l2norm (state d) + (curvature_form_hormander_l2_estimate d + error (state d)).
Proof.
  assert (hBase : rhs (state d) <= l2norm (state d) + error (state d)).
  { apply (weighted_cauchy_axiom (state d)). }
  assert (hErrLift : error (state d) <= curvature_form_hormander_l2_estimate d + error (state d)).
  { apply (le_add_left_nat (curvature_form_hormander_l2_estimate d) (error (state d))). }
  assert (hRightLift :
    l2norm (state d) + error (state d) <=
      l2norm (state d) + (curvature_form_hormander_l2_estimate d + error (state d))).
  {
    apply (add_le_add_left_nat
      (error (state d))
      (curvature_form_hormander_l2_estimate d + error (state d))
      (l2norm (state d))).
    exact hErrLift.
  }
  assert (hSecond : rhs (state d) <= l2norm (state d) + (curvature_form_hormander_l2_estimate d + error (state d))).
  { apply (le_trans_nat _ _ _ hBase hRightLift). }
  split.
  - exact hBase.
  - exact hSecond.
Qed.

Lemma coercivity_chain_hormander_l2_estimate
    {E : Type} `{HormanderStruct_l2_estimate E}
    (d : WeightData_hormander_l2_estimate) :
    l2norm (solution_operator_hormander_l2_estimate d) <= weight d + error (state d) /\
    l2norm (solution_operator_hormander_l2_estimate d) <=
      (weight d + error (state d)) + error (solution_operator_hormander_l2_estimate d).
Proof.
  assert (hCoercive : l2norm (solver (state d)) <= rhs (state d) + error (state d)).
  { apply (coercive_axiom (state d)). }
  assert (hWeightShift : rhs (state d) + error (state d) <= weight d + error (state d)).
  {
    apply (add_le_add_right_nat (rhs (state d)) (weight d) (error (state d))).
    exact (rhs_le_weight d).
  }
  assert (hFirstRaw : l2norm (solver (state d)) <= weight d + error (state d)).
  { apply (le_trans_nat _ _ _ hCoercive hWeightShift). }
  assert (hSecondLift :
    weight d + error (state d) <=
      (weight d + error (state d)) + error (solver (state d))).
  { apply (le_add_right_nat (weight d + error (state d)) (error (solver (state d)))). }
  assert (hSecondRaw :
    l2norm (solver (state d)) <=
      (weight d + error (state d)) + error (solver (state d))).
  { apply (le_trans_nat _ _ _ hFirstRaw hSecondLift). }
  split.
  - unfold solution_operator_hormander_l2_estimate.
    exact hFirstRaw.
  - unfold solution_operator_hormander_l2_estimate.
    exact hSecondRaw.
Qed.

Lemma weak_solution_exists_hormander_l2_estimate
    {E : Type} `{HormanderStruct_l2_estimate E}
    (d : WeightData_hormander_l2_estimate) :
    exists u : E, u = solution_operator_hormander_l2_estimate d /\
      error u <= error (state d) + curvature (state d).
Proof.
  assert (hWeakRaw : error (solver (state d)) <= error (state d) + curvature (state d)).
  { apply (weak_solution_axiom (state d)). }
  exists (solution_operator_hormander_l2_estimate d).
  split.
  - reflexivity.
  - unfold solution_operator_hormander_l2_estimate.
    exact hWeakRaw.
Qed.

Lemma minimizer_characterization_hormander_l2_estimate
    {E : Type} `{HormanderStruct_l2_estimate E}
    (d : WeightData_hormander_l2_estimate) :
    l2norm (adjoint (solution_operator_hormander_l2_estimate d)) <=
      l2norm (solution_operator_hormander_l2_estimate d) + curvature (state d) /\
    l2norm (adjoint (solution_operator_hormander_l2_estimate d)) <=
      l2norm (solution_operator_hormander_l2_estimate d) +
        curvature_form_hormander_l2_estimate d.
Proof.
  assert (hMinRaw :
    l2norm (adjoint (solver (state d))) <= l2norm (solver (state d)) + curvature (state d)).
  { apply (minimizer_axiom (state d)). }
  assert (hCurvLift : curvature (state d) <= curvature (state d) + weight d).
  { apply (le_add_right_nat (curvature (state d)) (weight d)). }
  assert (hSecondLift :
    l2norm (solver (state d)) + curvature (state d) <=
      l2norm (solver (state d)) + (curvature (state d) + weight d)).
  {
    apply (add_le_add_left_nat
      (curvature (state d))
      (curvature (state d) + weight d)
      (l2norm (solver (state d)))).
    exact hCurvLift.
  }
  assert (hSecondRaw :
    l2norm (adjoint (solver (state d))) <=
      l2norm (solver (state d)) + (curvature (state d) + weight d)).
  { apply (le_trans_nat _ _ _ hMinRaw hSecondLift). }
  split.
  - unfold solution_operator_hormander_l2_estimate.
    exact hMinRaw.
  - unfold solution_operator_hormander_l2_estimate.
    unfold curvature_form_hormander_l2_estimate.
    exact hSecondRaw.
Qed.

Lemma l2_estimate_final_hormander_l2_estimate
    {E : Type} `{HormanderStruct_l2_estimate E}
    (d : WeightData_hormander_l2_estimate) :
    exists u : E, u = solution_operator_hormander_l2_estimate d /\
      l2norm u + l2norm (adjoint u) <=
        weight d + weight d + curvature (state d) + error (state d).
Proof.
  assert (hFinalRaw :
    l2norm (solver (state d)) + l2norm (adjoint (solver (state d))) <=
      rhs (state d) + rhs (state d) + curvature (state d)).
  { apply (final_axiom (state d)). }
  assert (hRhsDouble : rhs (state d) + rhs (state d) <= weight d + weight d).
  {
    apply (add_le_add_nat
      (rhs (state d)) (weight d)
      (rhs (state d)) (weight d)).
    - exact (rhs_le_weight d).
    - exact (rhs_le_weight d).
  }
  assert (hCurvLift :
    rhs (state d) + rhs (state d) + curvature (state d) <=
      (weight d + weight d) + curvature (state d)).
  {
    apply (add_le_add_right_nat
      (rhs (state d) + rhs (state d))
      (weight d + weight d)
      (curvature (state d))).
    exact hRhsDouble.
  }
  assert (hMain :
    l2norm (solver (state d)) + l2norm (adjoint (solver (state d))) <=
      (weight d + weight d) + curvature (state d)).
  { apply (le_trans_nat _ _ _ hFinalRaw hCurvLift). }
  assert (hWithError :
    (weight d + weight d) + curvature (state d) <=
      ((weight d + weight d) + curvature (state d)) + error (state d)).
  { apply (le_add_right_nat ((weight d + weight d) + curvature (state d)) (error (state d))). }
  assert (hBound :
    l2norm (solver (state d)) + l2norm (adjoint (solver (state d))) <=
      ((weight d + weight d) + curvature (state d)) + error (state d)).
  { apply (le_trans_nat _ _ _ hMain hWithError). }
  exists (solution_operator_hormander_l2_estimate d).
  split.
  - reflexivity.
  - unfold solution_operator_hormander_l2_estimate.
    exact hBound.
Qed.
