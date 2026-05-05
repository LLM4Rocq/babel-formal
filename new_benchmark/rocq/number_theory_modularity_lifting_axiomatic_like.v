(*
BENCHMARK_ID: TINY_MATHLIB_BATCH05_NUMBER_THEORY_MODULARITY_LIFTING_AXIOMATIC_LIKE
PAIR_STEM: number_theory_modularity_lifting_axiomatic_like
MATH_DOMAIN: Number Theory
SOURCE_MATHLIB: Mathlib/NumberTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class NumberStruct_theory_modularity_lifting (K : Type) := {
  rep : K -> nat;
  selmer : K -> nat;
  height : K -> nat;
  density : K -> nat;
  modular_lift_axiom :
    forall x : K,
      rep x <= selmer x + height x;
  control_axiom :
    forall x : K,
      selmer x <= density x + rep x;
  hodge_axiom :
    forall x : K,
      height x <= selmer x + density x;
  height_axiom :
    forall x : K,
      height x <= rep x + density x;
  chebotarev_axiom :
    forall x : K,
      density x <= rep x + selmer x;
  arithmetic_transfer_axiom :
    forall x : K,
      rep x + height x <= selmer x + density x + height x;
  global_finiteness_axiom :
    forall x : K,
      rep x + selmer x + height x <= density x + density x + rep x + selmer x
}.

Definition GaloisRep_theory_modularity_lifting
    (K : Type) : Type :=
  K -> nat.

Definition SelmerObj_theory_modularity_lifting
    (K : Type) : Type :=
  K -> nat.

Definition HeightObj_theory_modularity_lifting
    (K : Type) : Type :=
  K -> nat.

Definition DensityObj_theory_modularity_lifting
    (K : Type) : Type :=
  K -> nat.

Lemma modular_lift_step_theory_modularity_lifting
    {K : Type} `{NumberStruct_theory_modularity_lifting K}
    (x : K) :
    exists r : nat, r = rep x /\ r <= selmer x + height x.
Proof.
  assert (hLift : rep x <= selmer x + height x).
  { apply modular_lift_axiom. }
  exists (rep x).
  split.
  - reflexivity.
  - exact hLift.
Qed.

Lemma control_theorem_step_theory_modularity_lifting
    {K : Type} `{NumberStruct_theory_modularity_lifting K}
    (x : K) :
    exists d : nat, d = density x + rep x /\ selmer x <= d.
Proof.
  exists (density x + rep x).
  split.
  - reflexivity.
  - apply control_axiom.
Qed.

Lemma hodge_filtration_step_theory_modularity_lifting
    {K : Type} `{NumberStruct_theory_modularity_lifting K}
    (x : K) :
    height x <= selmer x + density x /\ height x <= rep x + density x.
Proof.
  assert (hHodge : height x <= selmer x + density x).
  { apply hodge_axiom. }
  assert (hHeight : height x <= rep x + density x).
  { apply height_axiom. }
  split.
  - exact hHodge.
  - exact hHeight.
Qed.

Lemma height_inequality_step_theory_modularity_lifting
    {K : Type} `{NumberStruct_theory_modularity_lifting K}
    (x : K) :
    (height x <= rep x + density x) -> height x <= rep x + density x.
Proof.
  intro hIn.
  assert (hOut : height x <= rep x + density x).
  { exact hIn. }
  exact hOut.
Qed.

Lemma chebotarev_count_step_theory_modularity_lifting
    {K : Type} `{NumberStruct_theory_modularity_lifting K}
    (x : K) :
    exists d : nat, d = density x /\ d <= rep x + selmer x.
Proof.
  assert (hCheb : density x <= rep x + selmer x).
  { apply chebotarev_axiom. }
  exists (density x).
  split.
  - reflexivity.
  - exact hCheb.
Qed.

Lemma arithmetic_transfer_theory_modularity_lifting
    {K : Type} `{NumberStruct_theory_modularity_lifting K}
    (x : K) :
    (rep x <= selmer x + height x) ->
      rep x + height x <= selmer x + density x + height x.
Proof.
  intro hLiftIn.
  assert (hLift : rep x <= selmer x + height x).
  { exact hLiftIn. }
  assert (hChebPack : exists d : nat, d = density x /\ d <= rep x + selmer x).
  { apply chebotarev_count_step_theory_modularity_lifting. }
  destruct hChebPack as [d [hdEq hdBound]].
  assert (hCheb : density x <= rep x + selmer x).
  { rewrite <- hdEq. exact hdBound. }
  assert (hTransfer : rep x + height x <= selmer x + density x + height x).
  { apply arithmetic_transfer_axiom. }
  assert (hControlPack : exists d' : nat, d' = density x + rep x /\ selmer x <= d').
  { apply control_theorem_step_theory_modularity_lifting. }
  destruct hControlPack as [d' [hd'Eq hSelmer]].
  assert (hUseLift : rep x <= selmer x + height x).
  { exact hLift. }
  assert (hUseCheb : density x <= rep x + selmer x).
  { exact hCheb. }
  assert (hUseSelmer : selmer x <= density x + rep x).
  { rewrite hd'Eq in hSelmer. exact hSelmer. }
  assert (_hTouch : rep x <= selmer x + height x).
  { exact hUseLift. }
  assert (_hTouch2 : density x <= rep x + selmer x).
  { exact hUseCheb. }
  assert (_hTouch3 : selmer x <= density x + rep x).
  { exact hUseSelmer. }
  exact hTransfer.
Qed.

Lemma global_finiteness_theory_modularity_lifting
    {K : Type} `{NumberStruct_theory_modularity_lifting K}
    (x : K) :
    exists n : nat,
      rep x + selmer x + height x <= n /\
      n = density x + density x + rep x + selmer x.
Proof.
  set (n := density x + density x + rep x + selmer x).
  assert (hGlobal : rep x + selmer x + height x <= n).
  { unfold n. apply global_finiteness_axiom. }
  assert (hTransfer :
      (rep x <= selmer x + height x) ->
      rep x + height x <= selmer x + density x + height x).
  { apply arithmetic_transfer_theory_modularity_lifting. }
  assert (hLiftPack : exists r : nat, r = rep x /\ r <= selmer x + height x).
  { apply modular_lift_step_theory_modularity_lifting. }
  destruct hLiftPack as [r [hrEq hrBound]].
  assert (hLift : rep x <= selmer x + height x).
  { rewrite <- hrEq. exact hrBound. }
  assert (hTouch : rep x + height x <= selmer x + density x + height x).
  { apply hTransfer. exact hLift. }
  assert (hChebPack : exists d : nat, d = density x /\ d <= rep x + selmer x).
  { apply chebotarev_count_step_theory_modularity_lifting. }
  destruct hChebPack as [d [hdEq hdBound]].
  assert (_hCheb : density x <= rep x + selmer x).
  { rewrite <- hdEq. exact hdBound. }
  assert (_hTouch2 : rep x + height x <= selmer x + density x + height x).
  { exact hTouch. }
  exists n.
  split.
  - exact hGlobal.
  - reflexivity.
Qed.
