(**
BENCHMARK_ID: TINY_MATHLIB_BATCH06_NUMBER_THEORY_TATE_LOCAL_DUALITY_LIKE
PAIR_STEM: number_theory_tate_local_duality_like
MATH_DOMAIN: Number Theory
SOURCE_MATHLIB: Mathlib/NumberTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
**)

Set Universe Polymorphism.
Set Implicit Arguments.

Class LocalDualityStruct_tate (K : Type) := {
  H1 : Type;
  T : Type;
  zeroH1 : H1;
  addH1 : H1 -> H1 -> H1;
  negH1 : H1 -> H1;
  zeroT : T;
  addT : T -> T -> T;
  negT : T -> T;
  cup : H1 -> H1 -> T;
  character : H1 -> T;
  annihilator : H1 -> Prop;
  addH1_assoc : forall x y z : H1, addH1 (addH1 x y) z = addH1 x (addH1 y z);
  addH1_zero : forall x : H1, addH1 x zeroH1 = x;
  addH1_zero_left : forall x : H1, addH1 zeroH1 x = x;
  addH1_neg : forall x : H1, addH1 x (negH1 x) = zeroH1;
  addT_assoc : forall a b c : T, addT (addT a b) c = addT a (addT b c);
  addT_zero : forall a : T, addT a zeroT = a;
  zeroT_add : forall a : T, addT zeroT a = a;
  addT_neg : forall a : T, addT a (negT a) = zeroT;
  cup_add_left : forall x y z : H1, cup (addH1 x y) z = addT (cup x z) (cup y z);
  cup_add_right : forall x y z : H1, cup x (addH1 y z) = addT (cup x y) (cup x z);
  cup_neg_left : forall x z : H1, cup (negH1 x) z = negT (cup x z);
  cup_zero_left : forall z : H1, cup zeroH1 z = zeroT;
  character_add : forall x y : H1, character (addH1 x y) = addT (character x) (character y);
  character_zero : character zeroH1 = zeroT;
  local_exact : forall x : H1, annihilator x <-> character x = zeroT;
  annihilator_closed_add : forall x y : H1, annihilator x -> annihilator y -> annihilator (addH1 x y);
  annihilator_closed_neg : forall x : H1, annihilator x -> annihilator (negH1 x);
  finite_level : forall x : H1, annihilator x -> exists y : H1, cup x y = zeroT /\ cup y x = zeroT;
  dual_step : forall x : H1, (forall y : H1, cup x y = zeroT) -> annihilator x;
  limit_step :
    forall P : H1 -> Prop,
      (exists x : H1, P x) ->
      (forall x : H1, P x -> annihilator x) ->
      exists x : H1, P x /\ annihilator x;
  nondegenerate :
    forall x : H1,
      annihilator x ->
      (forall y : H1, cup y x = zeroT) ->
      x = zeroH1
}.

Record CohomologyData_tate_local_duality
    (K : Type) `{LocalDualityStruct_tate K} := {
  left_class_tate_local_duality : H1;
  right_class_tate_local_duality : H1;
  left_ann_tate_local_duality : annihilator left_class_tate_local_duality;
  right_ann_tate_local_duality : annihilator right_class_tate_local_duality
}.

Definition cup_pairing_tate_local_duality
    {K : Type} `{LocalDualityStruct_tate K}
    (x y : H1) : T :=
  cup x y.

Definition local_character_tate_local_duality
    {K : Type} `{LocalDualityStruct_tate K}
    (x : H1) : T :=
  character x.

Definition annihilator_subgroup_tate_local_duality
    {K : Type} `{LocalDualityStruct_tate K}
    (x : H1) : Prop :=
  annihilator x.

Lemma cup_product_bilinear_tate_local_duality
    {K : Type} `{LocalDualityStruct_tate K}
    (x y z : H1) :
    cup_pairing_tate_local_duality (addH1 x y) z =
      addT (cup_pairing_tate_local_duality x z) (cup_pairing_tate_local_duality y z) /\
    cup_pairing_tate_local_duality x (addH1 y z) =
      addT (cup_pairing_tate_local_duality x y) (cup_pairing_tate_local_duality x z) /\
    cup_pairing_tate_local_duality (negH1 x) z =
      negT (cup_pairing_tate_local_duality x z).
Proof.
  assert (hLeft :
      cup_pairing_tate_local_duality (addH1 x y) z =
        addT (cup_pairing_tate_local_duality x z) (cup_pairing_tate_local_duality y z)).
  { apply cup_add_left. }
  assert (hRight :
      cup_pairing_tate_local_duality x (addH1 y z) =
        addT (cup_pairing_tate_local_duality x y) (cup_pairing_tate_local_duality x z)).
  { apply cup_add_right. }
  assert (hNeg :
      cup_pairing_tate_local_duality (negH1 x) z =
        negT (cup_pairing_tate_local_duality x z)).
  { apply cup_neg_left. }
  split.
  - exact hLeft.
  - split.
    + exact hRight.
    + exact hNeg.
Qed.

Lemma local_invariant_exactness_tate_local_duality
    {K : Type} `{LocalDualityStruct_tate K}
    (x y : H1)
    (hx : annihilator_subgroup_tate_local_duality x)
    (hy : annihilator_subgroup_tate_local_duality y) :
    local_character_tate_local_duality (addH1 x (negH1 y)) = zeroT /\
    annihilator_subgroup_tate_local_duality (addH1 x (negH1 y)).
Proof.
  assert (hxZero : local_character_tate_local_duality x = zeroT).
  { apply (proj1 (local_exact x)). exact hx. }
  assert (hyNegAnn : annihilator_subgroup_tate_local_duality (negH1 y)).
  { apply annihilator_closed_neg. exact hy. }
  assert (hyNegZero : local_character_tate_local_duality (negH1 y) = zeroT).
  { apply (proj1 (local_exact (negH1 y))). exact hyNegAnn. }
  assert (hAddChar :
      local_character_tate_local_duality (addH1 x (negH1 y)) =
        addT (local_character_tate_local_duality x) (local_character_tate_local_duality (negH1 y))).
  { apply character_add. }
  assert (hCharZero : local_character_tate_local_duality (addH1 x (negH1 y)) = zeroT).
  {
    rewrite hAddChar.
    rewrite hxZero.
    rewrite (zeroT_add (local_character_tate_local_duality (negH1 y))).
    exact hyNegZero.
  }
  assert (hAnn : annihilator_subgroup_tate_local_duality (addH1 x (negH1 y))).
  { apply (proj2 (local_exact (addH1 x (negH1 y)))). exact hCharZero. }
  split.
  - exact hCharZero.
  - exact hAnn.
Qed.

Lemma pontryagin_dual_step_tate_local_duality
    {K : Type} `{LocalDualityStruct_tate K}
    (x : H1)
    (hker : forall y : H1, cup_pairing_tate_local_duality x y = zeroT) :
    annihilator_subgroup_tate_local_duality x /\
    cup_pairing_tate_local_duality (negH1 x) zeroH1 = zeroT.
Proof.
  assert (hAnn : annihilator_subgroup_tate_local_duality x).
  { apply (dual_step x). exact hker. }
  assert (hCupZero : cup_pairing_tate_local_duality x zeroH1 = zeroT).
  { apply hker. }
  assert (hNegCup :
      cup_pairing_tate_local_duality (negH1 x) zeroH1 =
        negT (cup_pairing_tate_local_duality x zeroH1)).
  { apply cup_neg_left. }
  assert (hZeroLeft : addT zeroT (negT zeroT) = negT zeroT).
  { apply zeroT_add. }
  assert (hAddNeg : addT zeroT (negT zeroT) = zeroT).
  { apply addT_neg. }
  assert (hNegZero : negT zeroT = zeroT).
  {
    transitivity (addT zeroT (negT zeroT)).
    - symmetry. exact hZeroLeft.
    - exact hAddNeg.
  }
  assert (hFinal : cup_pairing_tate_local_duality (negH1 x) zeroH1 = zeroT).
  {
    rewrite hNegCup.
    rewrite hCupZero.
    exact hNegZero.
  }
  split.
  - exact hAnn.
  - exact hFinal.
Qed.

Lemma orthogonality_criterion_tate_local_duality
    {K : Type} `{LocalDualityStruct_tate K}
    (d : CohomologyData_tate_local_duality) :
    annihilator_subgroup_tate_local_duality
      (addH1 (left_class_tate_local_duality d) (negH1 (right_class_tate_local_duality d))) /\
    local_character_tate_local_duality
      (addH1 (left_class_tate_local_duality d) (negH1 (right_class_tate_local_duality d))) = zeroT.
Proof.
  destruct (local_invariant_exactness_tate_local_duality
      (left_class_tate_local_duality d)
      (right_class_tate_local_duality d)
      (left_ann_tate_local_duality d)
      (right_ann_tate_local_duality d)) as [hChar hAnn].
  split.
  - exact hAnn.
  - exact hChar.
Qed.

Lemma finite_level_perfectness_tate_local_duality
    {K : Type} `{LocalDualityStruct_tate K}
    (x : H1)
    (hx : annihilator_subgroup_tate_local_duality x) :
    exists y : H1,
      cup_pairing_tate_local_duality x y = zeroT /\
      cup_pairing_tate_local_duality y x = zeroT /\
      (annihilator_subgroup_tate_local_duality y -> local_character_tate_local_duality y = zeroT).
Proof.
  destruct (finite_level x hx) as [y [hxy hyx]].
  assert (hImp :
      annihilator_subgroup_tate_local_duality y ->
      local_character_tate_local_duality y = zeroT).
  {
    intro hy.
    apply (proj1 (local_exact y)).
    exact hy.
  }
  exists y.
  split.
  - exact hxy.
  - split.
    + exact hyx.
    + exact hImp.
Qed.

Lemma passage_to_limit_tate_local_duality
    {K : Type} `{LocalDualityStruct_tate K}
    (P : H1 -> Prop)
    (hnonempty : exists x : H1, P x)
    (hstable : forall x : H1, P x -> annihilator_subgroup_tate_local_duality x) :
    exists x : H1,
      P x /\
      local_character_tate_local_duality x = zeroT /\
      exists y : H1, cup_pairing_tate_local_duality x y = zeroT.
Proof.
  destruct (limit_step P hnonempty hstable) as [x [hxP hxAnn]].
  assert (hxChar : local_character_tate_local_duality x = zeroT).
  { apply (proj1 (local_exact x)). exact hxAnn. }
  destruct (finite_level_perfectness_tate_local_duality x hxAnn) as [y [hxy [hyx hyCharFromAnn]]].
  assert (hyIfAnn :
      annihilator_subgroup_tate_local_duality y ->
      local_character_tate_local_duality y = zeroT).
  { exact hyCharFromAnn. }
  assert (hSymmetricCup : cup_pairing_tate_local_duality y x = zeroT).
  { exact hyx. }
  assert (hAnnAgain : annihilator_subgroup_tate_local_duality x).
  { exact hxAnn. }
  exists x.
  split.
  - exact hxP.
  - split.
    + exact hxChar.
    + exists y.
      exact hxy.
Qed.

Lemma perfect_pairing_final_tate_local_duality
    {K : Type} `{LocalDualityStruct_tate K}
    (P : H1 -> Prop)
    (hnonempty : exists x : H1, P x)
    (hstable : forall x : H1, P x -> annihilator_subgroup_tate_local_duality x)
    (horth : forall x : H1, P x -> forall y : H1, cup_pairing_tate_local_duality y x = zeroT) :
    exists x : H1,
      P x /\
      x = zeroH1 /\
      local_character_tate_local_duality x = zeroT.
Proof.
  destruct (passage_to_limit_tate_local_duality P hnonempty hstable)
    as [x [hxP [hxChar hExistsPartner]]].
  assert (hxAnn : annihilator_subgroup_tate_local_duality x).
  { apply (proj2 (local_exact x)). exact hxChar. }
  assert (hAllRight : forall y : H1, cup_pairing_tate_local_duality y x = zeroT).
  { apply (horth x hxP). }
  assert (hZero : x = zeroH1).
  { apply (nondegenerate x hxAnn). exact hAllRight. }
  destruct hExistsPartner as [yWitness hyWitness].
  assert (hWitnessed : cup_pairing_tate_local_duality x yWitness = zeroT).
  { exact hyWitness. }
  exists x.
  split.
  - exact hxP.
  - split.
    + exact hZero.
    + exact hxChar.
Qed.
