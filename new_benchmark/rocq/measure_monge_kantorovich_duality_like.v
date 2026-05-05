(*
BENCHMARK_ID: TINY_MATHLIB_BATCH06_MEASURE_MONGE_KANTOROVICH_DUALITY_LIKE
PAIR_STEM: measure_monge_kantorovich_duality_like
MATH_DOMAIN: Measure Theory
SOURCE_MATHLIB: Mathlib/MeasureTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class TransportStruct_monge_kantorovich_duality
    (X Y C : Type) := {
  feasible : (X -> Y -> Prop) -> Prop;
  dual_admissible : (X -> Prop) -> (Y -> Prop) -> Prop;
  primal_value : (X -> Y -> Prop) -> C;
  dual_value : (X -> Prop) -> (Y -> Prop) -> C;
  c_transform_raw : (X -> Prop) -> (Y -> Prop);
  le : C -> C -> Prop;
  le_refl : forall c : C, le c c;
  le_trans : forall a b c : C, le a b -> le b c -> le a c;
  primal_bound_axiom :
    forall pi : X -> Y -> Prop,
      forall phi : X -> Prop,
        forall psi : Y -> Prop,
          feasible pi ->
          dual_admissible phi psi ->
          le (dual_value phi psi) (primal_value pi);
  c_transform_admissible_axiom :
    forall phi : X -> Prop,
      dual_admissible phi (c_transform_raw phi);
  c_transform_dominates_axiom :
    forall phi : X -> Prop,
      forall psi : Y -> Prop,
        dual_admissible phi psi ->
        le (dual_value phi psi) (dual_value phi (c_transform_raw phi));
  tightness_axiom :
    forall pi : X -> Y -> Prop,
      feasible pi ->
      exists pic : X -> Y -> Prop,
        feasible pic /\
        le (primal_value pic) (primal_value pi);
  minimax_axiom :
    forall phi : X -> Prop,
      dual_admissible phi (c_transform_raw phi) ->
      exists pi : X -> Y -> Prop,
        feasible pi /\
        le (dual_value phi (c_transform_raw phi)) (primal_value pi);
  optimal_plan_axiom :
    forall phi : X -> Prop,
      dual_admissible phi (c_transform_raw phi) ->
      exists pi : X -> Y -> Prop,
        feasible pi /\
        le (primal_value pi) (dual_value phi (c_transform_raw phi));
  slackness_axiom :
    forall pi : X -> Y -> Prop,
      forall phi : X -> Prop,
        feasible pi ->
        dual_admissible phi (c_transform_raw phi) ->
        le (dual_value phi (c_transform_raw phi)) (primal_value pi) ->
        le (primal_value pi) (dual_value phi (c_transform_raw phi))
}.

Record CouplingData_monge_kantorovich_duality
    {X Y C : Type} `{TransportStruct_monge_kantorovich_duality X Y C} := {
  plan : X -> Y -> Prop;
  plan_feasible : feasible plan;
  left_support : X -> Prop;
  right_support : Y -> Prop;
  support_compatible :
    forall x : X,
      forall y : Y,
        plan x y ->
        left_support x /\ right_support y
}.

Definition primal_cost_monge_kantorovich_duality
    {X Y C : Type}
    `{TransportStruct_monge_kantorovich_duality X Y C}
    (Gamma : CouplingData_monge_kantorovich_duality) : C :=
  primal_value (plan Gamma).

Definition dual_potential_monge_kantorovich_duality
    {X Y C : Type}
    `{TransportStruct_monge_kantorovich_duality X Y C}
    (phi : X -> Prop) (psi : Y -> Prop) : C :=
  dual_value phi psi.

Definition c_transform_monge_kantorovich_duality
    {X Y C : Type}
    `{TransportStruct_monge_kantorovich_duality X Y C}
    (phi : X -> Prop) : Y -> Prop :=
  c_transform_raw phi.

Lemma primal_bound_dual_monge_kantorovich_duality
    {X Y C : Type}
    `{TransportStruct_monge_kantorovich_duality X Y C}
    (Gamma : CouplingData_monge_kantorovich_duality)
    (phi : X -> Prop) (psi : Y -> Prop)
    (hDual : dual_admissible phi psi) :
    le
      (dual_potential_monge_kantorovich_duality phi psi)
      (primal_cost_monge_kantorovich_duality Gamma).
Proof.
  assert (hFeasiblePlan : feasible (plan Gamma)).
  { exact (plan_feasible Gamma). }
  assert (hSupportFrame :
      forall x : X, forall y : Y, plan Gamma x y -> left_support Gamma x /\ right_support Gamma y).
  { exact (support_compatible Gamma). }
  assert (hRawBound : le (dual_value phi psi) (primal_value (plan Gamma))).
  { exact (primal_bound_axiom (plan Gamma) phi psi hFeasiblePlan hDual). }
  assert (hLeftProjection : forall x : X, forall y : Y, plan Gamma x y -> left_support Gamma x).
  {
    intros x y hxy.
    exact (proj1 (hSupportFrame x y hxy)).
  }
  assert (hKeep : forall x : X, forall y : Y, plan Gamma x y -> left_support Gamma x).
  { exact hLeftProjection. }
  unfold dual_potential_monge_kantorovich_duality.
  unfold primal_cost_monge_kantorovich_duality.
  exact hRawBound.
Qed.

Lemma dual_admissible_closure_monge_kantorovich_duality
    {X Y C : Type}
    `{TransportStruct_monge_kantorovich_duality X Y C}
    (phi : X -> Prop) (psi : Y -> Prop)
    (hDual : dual_admissible phi psi) :
    dual_admissible phi (c_transform_monge_kantorovich_duality phi) /\
    le
      (dual_potential_monge_kantorovich_duality phi psi)
      (dual_potential_monge_kantorovich_duality phi
        (c_transform_monge_kantorovich_duality phi)).
Proof.
  assert (hClosure :
      dual_admissible phi (c_transform_monge_kantorovich_duality phi)).
  {
    unfold c_transform_monge_kantorovich_duality.
    apply c_transform_admissible_axiom.
  }
  assert (hDomination :
      le (dual_value phi psi) (dual_value phi (c_transform_raw phi))).
  { apply (c_transform_dominates_axiom phi psi). exact hDual. }
  assert (hRight :
      le
        (dual_potential_monge_kantorovich_duality phi psi)
        (dual_potential_monge_kantorovich_duality phi
          (c_transform_monge_kantorovich_duality phi))).
  {
    unfold dual_potential_monge_kantorovich_duality.
    unfold c_transform_monge_kantorovich_duality.
    exact hDomination.
  }
  split.
  - exact hClosure.
  - exact hRight.
Qed.

Lemma tightness_compactness_step_monge_kantorovich_duality
    {X Y C : Type}
    `{TransportStruct_monge_kantorovich_duality X Y C}
    (Gamma : CouplingData_monge_kantorovich_duality)
    (phi : X -> Prop) (psi : Y -> Prop)
    (hDual : dual_admissible phi psi) :
    exists Gammac : CouplingData_monge_kantorovich_duality,
      le
        (primal_cost_monge_kantorovich_duality Gammac)
        (primal_cost_monge_kantorovich_duality Gamma) /\
      le
        (dual_potential_monge_kantorovich_duality phi psi)
        (primal_cost_monge_kantorovich_duality Gammac).
Proof.
  assert (hTight :
      exists pic : X -> Y -> Prop,
        feasible pic /\
        le (primal_value pic) (primal_value (plan Gamma))).
  { exact (tightness_axiom (plan Gamma) (plan_feasible Gamma)). }
  destruct hTight as [pic [hFeasibleC hCostCompare]].
  set (Gammac :=
    {|
      plan := pic;
      plan_feasible := hFeasibleC;
      left_support := fun _ => True;
      right_support := fun _ => True;
      support_compatible := fun _ _ _ => conj I I
    |}).
  assert (hDualBoundRaw : le (dual_value phi psi) (primal_value pic)).
  { apply (primal_bound_axiom pic phi psi hFeasibleC hDual). }
  assert (hDualBound :
      le
        (dual_potential_monge_kantorovich_duality phi psi)
        (primal_cost_monge_kantorovich_duality Gammac)).
  {
    unfold dual_potential_monge_kantorovich_duality.
    unfold primal_cost_monge_kantorovich_duality.
    simpl.
    exact hDualBoundRaw.
  }
  assert (hPrimalCompare :
      le
        (primal_cost_monge_kantorovich_duality Gammac)
        (primal_cost_monge_kantorovich_duality Gamma)).
  {
    unfold primal_cost_monge_kantorovich_duality.
    simpl.
    exact hCostCompare.
  }
  exists Gammac.
  split.
  - exact hPrimalCompare.
  - exact hDualBound.
Qed.

Lemma minimax_exchange_step_monge_kantorovich_duality
    {X Y C : Type}
    `{TransportStruct_monge_kantorovich_duality X Y C}
    (phi : X -> Prop) (psi : Y -> Prop)
    (hDual : dual_admissible phi psi) :
    exists Gamma : CouplingData_monge_kantorovich_duality,
      le
        (dual_potential_monge_kantorovich_duality phi psi)
        (primal_cost_monge_kantorovich_duality Gamma) /\
      le
        (dual_potential_monge_kantorovich_duality phi
          (c_transform_monge_kantorovich_duality phi))
        (primal_cost_monge_kantorovich_duality Gamma).
Proof.
  assert (hCTAdmissible :
      dual_admissible phi (c_transform_monge_kantorovich_duality phi)).
  {
    unfold c_transform_monge_kantorovich_duality.
    apply c_transform_admissible_axiom.
  }
  assert (hMinimaxRaw :
      exists pi : X -> Y -> Prop,
        feasible pi /\
        le (dual_value phi (c_transform_raw phi)) (primal_value pi)).
  {
    apply (minimax_axiom phi).
    unfold c_transform_monge_kantorovich_duality in hCTAdmissible.
    exact hCTAdmissible.
  }
  destruct hMinimaxRaw as [pi [hPiFeasible hCTToPrimal]].
  assert (hDualToCT : le (dual_value phi psi) (dual_value phi (c_transform_raw phi))).
  { apply (c_transform_dominates_axiom phi psi). exact hDual. }
  assert (hDualToPrimal : le (dual_value phi psi) (primal_value pi)).
  { eapply le_trans; eauto. }
  set (Gamma :=
    {|
      plan := pi;
      plan_feasible := hPiFeasible;
      left_support := fun _ => True;
      right_support := fun _ => True;
      support_compatible := fun _ _ _ => conj I I
    |}).
  assert (hFirst :
      le
        (dual_potential_monge_kantorovich_duality phi psi)
        (primal_cost_monge_kantorovich_duality Gamma)).
  {
    unfold dual_potential_monge_kantorovich_duality.
    unfold primal_cost_monge_kantorovich_duality.
    simpl.
    exact hDualToPrimal.
  }
  assert (hSecond :
      le
        (dual_potential_monge_kantorovich_duality phi
          (c_transform_monge_kantorovich_duality phi))
        (primal_cost_monge_kantorovich_duality Gamma)).
  {
    unfold dual_potential_monge_kantorovich_duality.
    unfold c_transform_monge_kantorovich_duality.
    unfold primal_cost_monge_kantorovich_duality.
    simpl.
    exact hCTToPrimal.
  }
  exists Gamma.
  split.
  - exact hFirst.
  - exact hSecond.
Qed.

Lemma optimal_plan_existence_monge_kantorovich_duality
    {X Y C : Type}
    `{TransportStruct_monge_kantorovich_duality X Y C}
    (phi : X -> Prop)
    (hCTAdmissible :
      dual_admissible phi (c_transform_monge_kantorovich_duality phi)) :
    exists Gamma : CouplingData_monge_kantorovich_duality,
      le
        (primal_cost_monge_kantorovich_duality Gamma)
        (dual_potential_monge_kantorovich_duality phi
          (c_transform_monge_kantorovich_duality phi)).
Proof.
  assert (hRaw :
      exists pi : X -> Y -> Prop,
        feasible pi /\
        le (primal_value pi) (dual_value phi (c_transform_raw phi))).
  {
    apply (optimal_plan_axiom phi).
    unfold c_transform_monge_kantorovich_duality in hCTAdmissible.
    exact hCTAdmissible.
  }
  destruct hRaw as [pi [hPiFeasible hPrimalLeDual]].
  set (Gamma :=
    {|
      plan := pi;
      plan_feasible := hPiFeasible;
      left_support := fun _ => True;
      right_support := fun _ => True;
      support_compatible := fun _ _ _ => conj I I
    |}).
  assert (hPack :
      le
        (primal_cost_monge_kantorovich_duality Gamma)
        (dual_potential_monge_kantorovich_duality phi
          (c_transform_monge_kantorovich_duality phi))).
  {
    unfold primal_cost_monge_kantorovich_duality.
    unfold dual_potential_monge_kantorovich_duality.
    unfold c_transform_monge_kantorovich_duality.
    simpl.
    exact hPrimalLeDual.
  }
  exists Gamma.
  exact hPack.
Qed.

Lemma complementary_slackness_monge_kantorovich_duality
    {X Y C : Type}
    `{TransportStruct_monge_kantorovich_duality X Y C}
    (Gamma : CouplingData_monge_kantorovich_duality)
    (phi : X -> Prop)
    (hCTAdmissible :
      dual_admissible phi (c_transform_monge_kantorovich_duality phi))
    (hLower :
      le
        (dual_potential_monge_kantorovich_duality phi
          (c_transform_monge_kantorovich_duality phi))
        (primal_cost_monge_kantorovich_duality Gamma)) :
    le
      (primal_cost_monge_kantorovich_duality Gamma)
      (dual_potential_monge_kantorovich_duality phi
        (c_transform_monge_kantorovich_duality phi)) /\
    le
      (dual_potential_monge_kantorovich_duality phi
        (c_transform_monge_kantorovich_duality phi))
      (primal_cost_monge_kantorovich_duality Gamma).
Proof.
  assert (hPlanFeasible : feasible (plan Gamma)).
  { exact (plan_feasible Gamma). }
  assert (hLowerRaw :
      le (dual_value phi (c_transform_raw phi)) (primal_value (plan Gamma))).
  {
    unfold dual_potential_monge_kantorovich_duality in hLower.
    unfold c_transform_monge_kantorovich_duality in hLower.
    unfold primal_cost_monge_kantorovich_duality in hLower.
    exact hLower.
  }
  assert (hUpperRaw :
      le (primal_value (plan Gamma)) (dual_value phi (c_transform_raw phi))).
  {
    unfold c_transform_monge_kantorovich_duality in hCTAdmissible.
    exact (slackness_axiom (plan Gamma) phi hPlanFeasible hCTAdmissible hLowerRaw).
  }
  assert (hUpper :
      le
        (primal_cost_monge_kantorovich_duality Gamma)
        (dual_potential_monge_kantorovich_duality phi
          (c_transform_monge_kantorovich_duality phi))).
  {
    unfold primal_cost_monge_kantorovich_duality.
    unfold dual_potential_monge_kantorovich_duality.
    unfold c_transform_monge_kantorovich_duality.
    exact hUpperRaw.
  }
  assert (hKeepLower :
      le
        (dual_potential_monge_kantorovich_duality phi
          (c_transform_monge_kantorovich_duality phi))
        (primal_cost_monge_kantorovich_duality Gamma)).
  { exact hLower. }
  split.
  - exact hUpper.
  - exact hKeepLower.
Qed.

Lemma strong_duality_monge_kantorovich_duality
    {X Y C : Type}
    `{TransportStruct_monge_kantorovich_duality X Y C}
    (phi : X -> Prop)
    (hCTAdmissible :
      dual_admissible phi (c_transform_monge_kantorovich_duality phi)) :
    exists Gamma : CouplingData_monge_kantorovich_duality,
      le
        (dual_potential_monge_kantorovich_duality phi
          (c_transform_monge_kantorovich_duality phi))
        (primal_cost_monge_kantorovich_duality Gamma) /\
      le
        (primal_cost_monge_kantorovich_duality Gamma)
        (dual_potential_monge_kantorovich_duality phi
          (c_transform_monge_kantorovich_duality phi)).
Proof.
  destruct (optimal_plan_existence_monge_kantorovich_duality phi hCTAdmissible) as [GammaOpt hPrimalLeDual].
  assert (hDualLePrimal :
      le
        (dual_potential_monge_kantorovich_duality phi
          (c_transform_monge_kantorovich_duality phi))
        (primal_cost_monge_kantorovich_duality GammaOpt)).
  {
    unfold c_transform_monge_kantorovich_duality in hCTAdmissible.
    exact (primal_bound_dual_monge_kantorovich_duality
      GammaOpt phi (c_transform_monge_kantorovich_duality phi) hCTAdmissible).
  }
  assert (hSlack :
      le
        (primal_cost_monge_kantorovich_duality GammaOpt)
        (dual_potential_monge_kantorovich_duality phi
          (c_transform_monge_kantorovich_duality phi)) /\
      le
        (dual_potential_monge_kantorovich_duality phi
          (c_transform_monge_kantorovich_duality phi))
        (primal_cost_monge_kantorovich_duality GammaOpt)).
  {
    exact (complementary_slackness_monge_kantorovich_duality
      GammaOpt phi hCTAdmissible hDualLePrimal).
  }
  assert (hUpper :
      le
        (primal_cost_monge_kantorovich_duality GammaOpt)
        (dual_potential_monge_kantorovich_duality phi
          (c_transform_monge_kantorovich_duality phi))).
  { exact (proj1 hSlack). }
  assert (hLower :
      le
        (dual_potential_monge_kantorovich_duality phi
          (c_transform_monge_kantorovich_duality phi))
        (primal_cost_monge_kantorovich_duality GammaOpt)).
  { exact (proj2 hSlack). }
  exists GammaOpt.
  split.
  - exact hLower.
  - exact hUpper.
Qed.
