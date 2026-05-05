(*
BENCHMARK_ID: TINY_MATHLIB_BATCH05_NUMBER_THEORY_ARAKELOV_HEIGHT_INEQUALITY_LIKE
PAIR_STEM: number_theory_arakelov_height_inequality_like
MATH_DOMAIN: Number Theory
SOURCE_MATHLIB: Mathlib/NumberTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class NumberStruct_theory_arakelov_height (K : Type) := {
  rep : K -> nat;
  selmer : K -> nat;
  height : K -> nat;
  density : K -> nat;
  nat_le_trans : forall a b c : nat, a <= b -> b <= c -> a <= c;
  nat_add_mono : forall a b c d : nat, a <= b -> c <= d -> a + c <= b + d;
  nat_le_refl : forall a : nat, a <= a;
  modular_lift_axiom :
    forall x : K,
      rep x + rep x <= selmer x + height x + density x;
  control_axiom :
    forall x : K,
      selmer x + height x <= density x + rep x + height x;
  hodge_axiom :
    forall x : K,
      height x + selmer x <= selmer x + density x + selmer x;
  height_axiom :
    forall x : K,
      height x + rep x <= density x + selmer x + rep x;
  chebotarev_axiom :
    forall x : K,
      density x + height x <= rep x + selmer x + height x;
  arithmetic_transfer_axiom :
    forall x : K,
      rep x + height x <= selmer x + density x + height x;
  global_finiteness_axiom :
    forall x : K,
      rep x + selmer x + height x <= density x + density x + rep x + selmer x
}.

Definition GaloisRep_theory_arakelov_height (K : Type) : Type :=
  K -> nat.

Definition SelmerObj_theory_arakelov_height (K : Type) : Type :=
  K -> nat.

Definition HeightObj_theory_arakelov_height (K : Type) : Type :=
  K -> nat.

Definition DensityObj_theory_arakelov_height (K : Type) : Type :=
  K -> nat.

Lemma modular_lift_step_theory_arakelov_height
    {K : Type} `{NumberStruct_theory_arakelov_height K}
    (x : K) :
    exists b : nat, b = selmer x + height x + density x /\ rep x + rep x <= b.
Proof.
  exists (selmer x + height x + density x).
  split.
  - reflexivity.
  - apply modular_lift_axiom.
Qed.

Lemma control_theorem_step_theory_arakelov_height
    {K : Type} `{NumberStruct_theory_arakelov_height K}
    (x : K) :
    selmer x + height x <= density x + rep x + height x /\
    selmer x + height x <= density x + rep x + height x.
Proof.
  assert (hCtrl : selmer x + height x <= density x + rep x + height x).
  { apply control_axiom. }
  split.
  - exact hCtrl.
  - exact hCtrl.
Qed.

Lemma hodge_filtration_step_theory_arakelov_height
    {K : Type} `{NumberStruct_theory_arakelov_height K}
    (x : K) :
    (selmer x + height x <= density x + rep x + height x) ->
    height x + selmer x <= selmer x + density x + selmer x.
Proof.
  intro hcontrol.
  assert (hUse : selmer x + height x <= density x + rep x + height x).
  { exact hcontrol. }
  assert (hHodge : height x + selmer x <= selmer x + density x + selmer x).
  { apply hodge_axiom. }
  exact hHodge.
Qed.

Lemma height_inequality_step_theory_arakelov_height
    {K : Type} `{NumberStruct_theory_arakelov_height K}
    (x : K) :
    exists t : nat, t = density x + selmer x + rep x /\ height x + rep x <= t.
Proof.
  exists (density x + selmer x + rep x).
  split.
  - reflexivity.
  - apply height_axiom.
Qed.

Lemma chebotarev_count_step_theory_arakelov_height
    {K : Type} `{NumberStruct_theory_arakelov_height K}
    (x : K) :
    density x + height x <= rep x + selmer x + height x /\
    rep x + rep x <= selmer x + height x + density x.
Proof.
  assert (hCheb : density x + height x <= rep x + selmer x + height x).
  { apply chebotarev_axiom. }
  destruct (modular_lift_step_theory_arakelov_height (K := K) x) as [b [hbEq hbLe]].
  assert (hLift : rep x + rep x <= selmer x + height x + density x).
  { rewrite <- hbEq. exact hbLe. }
  split.
  - exact hCheb.
  - exact hLift.
Qed.

Lemma arithmetic_transfer_theory_arakelov_height
    {K : Type} `{NumberStruct_theory_arakelov_height K}
    (x : K) :
    exists t : nat, t = selmer x + density x + height x /\ rep x + height x <= t.
Proof.
  assert (hBase : rep x + height x <= selmer x + density x + height x).
  { apply arithmetic_transfer_axiom. }
  assert (hSelf : rep x + height x <= rep x + height x).
  { apply nat_le_refl. }
  assert (hChain : rep x + height x <= selmer x + density x + height x).
  { eapply nat_le_trans; [exact hSelf | exact hBase]. }
  exists (selmer x + density x + height x).
  split.
  - reflexivity.
  - exact hChain.
Qed.

Lemma global_finiteness_theory_arakelov_height
    {K : Type} `{NumberStruct_theory_arakelov_height K}
    (x : K) :
    rep x + selmer x + height x <=
      density x + density x + rep x + selmer x.
Proof.
  assert (hGlobal :
      rep x + selmer x + height x <=
        density x + density x + rep x + selmer x).
  { apply global_finiteness_axiom. }
  destruct (arithmetic_transfer_theory_arakelov_height (K := K) x)
    as [t [ht hle]].
  assert (hKeepLe : rep x + height x <= t).
  { exact hle. }
  assert (hKeepEq : t = selmer x + density x + height x).
  { exact ht. }
  assert (_hTouch : rep x + height x <= selmer x + density x + height x).
  { rewrite <- hKeepEq. exact hKeepLe. }
  exact hGlobal.
Qed.
