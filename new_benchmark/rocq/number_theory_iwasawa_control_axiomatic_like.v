(*
BENCHMARK_ID: TINY_MATHLIB_BATCH05_NUMBER_THEORY_IWASAWA_CONTROL_AXIOMATIC_LIKE
PAIR_STEM: number_theory_iwasawa_control_axiomatic_like
MATH_DOMAIN: Number Theory
SOURCE_MATHLIB: Mathlib/NumberTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class NumberStruct_theory_iwasawa_control (K : Type) := {
  Rep : Type;
  selmer : Rep -> Prop;
  height : Rep -> nat;
  dense : Rep -> Prop;
  modular_lift_axiom : forall r : Rep, dense r -> selmer r;
  control_axiom : forall r : Rep, selmer r -> dense r;
  hodge_axiom : forall r : Rep, selmer r -> height r <= height r + 1;
  height_axiom : forall r : Rep, dense r -> height r <= height r + height r;
  chebotarev_axiom : forall r : Rep, dense r -> exists n : nat, n = height r;
  transfer_axiom : forall r : Rep, selmer r -> dense r -> selmer r /\ dense r;
  finiteness_axiom : forall r : Rep, selmer r -> exists m : nat, height r <= m
}.

Definition GaloisRep_theory_iwasawa_control
    {K : Type} `{NumberStruct_theory_iwasawa_control K} : Type :=
  Rep.

Definition SelmerObj_theory_iwasawa_control
    {K : Type} `{NumberStruct_theory_iwasawa_control K} :
    GaloisRep_theory_iwasawa_control -> Prop :=
  selmer.

Definition HeightObj_theory_iwasawa_control
    {K : Type} `{NumberStruct_theory_iwasawa_control K} :
    GaloisRep_theory_iwasawa_control -> nat :=
  height.

Definition DensityObj_theory_iwasawa_control
    {K : Type} `{NumberStruct_theory_iwasawa_control K} :
    GaloisRep_theory_iwasawa_control -> Prop :=
  dense.

Lemma modular_lift_step_theory_iwasawa_control
    {K : Type} `{NumberStruct_theory_iwasawa_control K}
    (r : GaloisRep_theory_iwasawa_control)
    (hDense : DensityObj_theory_iwasawa_control r) :
    SelmerObj_theory_iwasawa_control r.
Proof.
  assert (hLift : selmer r).
  { apply (modular_lift_axiom r hDense). }
  assert (hPack : SelmerObj_theory_iwasawa_control r).
  { exact hLift. }
  exact hPack.
Qed.

Lemma control_theorem_step_theory_iwasawa_control
    {K : Type} `{NumberStruct_theory_iwasawa_control K}
    (r : GaloisRep_theory_iwasawa_control)
    (hSel : SelmerObj_theory_iwasawa_control r) :
    DensityObj_theory_iwasawa_control r.
Proof.
  assert (hDense : dense r).
  { apply (control_axiom r hSel). }
  assert (hPack : DensityObj_theory_iwasawa_control r).
  { exact hDense. }
  exact hPack.
Qed.

Lemma hodge_filtration_step_theory_iwasawa_control
    {K : Type} `{NumberStruct_theory_iwasawa_control K}
    (r : GaloisRep_theory_iwasawa_control)
    (hSel : SelmerObj_theory_iwasawa_control r) :
    HeightObj_theory_iwasawa_control r <=
      HeightObj_theory_iwasawa_control r + 1.
Proof.
  assert (hBound : height r <= height r + 1).
  { apply (hodge_axiom r hSel). }
  assert (hPack :
      HeightObj_theory_iwasawa_control r <=
      HeightObj_theory_iwasawa_control r + 1).
  { exact hBound. }
  exact hPack.
Qed.

Lemma height_inequality_step_theory_iwasawa_control
    {K : Type} `{NumberStruct_theory_iwasawa_control K}
    (r : GaloisRep_theory_iwasawa_control)
    (hSel : SelmerObj_theory_iwasawa_control r) :
    HeightObj_theory_iwasawa_control r <=
      HeightObj_theory_iwasawa_control r +
      HeightObj_theory_iwasawa_control r.
Proof.
  assert (hDense : DensityObj_theory_iwasawa_control r).
  { apply (control_theorem_step_theory_iwasawa_control r hSel). }
  assert (hHodge :
      HeightObj_theory_iwasawa_control r <=
      HeightObj_theory_iwasawa_control r + 1).
  { apply (hodge_filtration_step_theory_iwasawa_control r hSel). }
  assert (hBase : height r <= height r + height r).
  { apply (height_axiom r hDense). }
  assert (hPack :
      HeightObj_theory_iwasawa_control r <=
        HeightObj_theory_iwasawa_control r +
        HeightObj_theory_iwasawa_control r).
  { exact hBase. }
  assert (hKeep : HeightObj_theory_iwasawa_control r <= HeightObj_theory_iwasawa_control r + 1).
  { exact hHodge. }
  exact hPack.
Qed.

Lemma chebotarev_count_step_theory_iwasawa_control
    {K : Type} `{NumberStruct_theory_iwasawa_control K}
    (r : GaloisRep_theory_iwasawa_control)
    (hSel : SelmerObj_theory_iwasawa_control r) :
    exists n : nat,
      n = HeightObj_theory_iwasawa_control r.
Proof.
  assert (hDense : DensityObj_theory_iwasawa_control r).
  { apply (control_theorem_step_theory_iwasawa_control r hSel). }
  destruct (chebotarev_axiom r hDense) as [n hn].
  assert (hPack : n = HeightObj_theory_iwasawa_control r).
  { exact hn. }
  exists n.
  exact hPack.
Qed.

Lemma arithmetic_transfer_theory_iwasawa_control
    {K : Type} `{NumberStruct_theory_iwasawa_control K}
    (r : GaloisRep_theory_iwasawa_control)
    (hSel : SelmerObj_theory_iwasawa_control r) :
    SelmerObj_theory_iwasawa_control r /\
    DensityObj_theory_iwasawa_control r.
Proof.
  assert (hDense : DensityObj_theory_iwasawa_control r).
  { apply (control_theorem_step_theory_iwasawa_control r hSel). }
  assert (hPair : selmer r /\ dense r).
  { apply (transfer_axiom r hSel hDense). }
  assert (hSelOut : SelmerObj_theory_iwasawa_control r).
  { exact (proj1 hPair). }
  assert (hDenseOut : DensityObj_theory_iwasawa_control r).
  { exact (proj2 hPair). }
  split.
  - exact hSelOut.
  - exact hDenseOut.
Qed.

Lemma global_finiteness_theory_iwasawa_control
    {K : Type} `{NumberStruct_theory_iwasawa_control K}
    (r : GaloisRep_theory_iwasawa_control)
    (hDense : DensityObj_theory_iwasawa_control r) :
    exists m : nat,
      HeightObj_theory_iwasawa_control r <= m /\
      DensityObj_theory_iwasawa_control r.
Proof.
  assert (hSel : SelmerObj_theory_iwasawa_control r).
  { apply (modular_lift_step_theory_iwasawa_control r hDense). }
  destruct (finiteness_axiom r hSel) as [m hm].
  assert (hBound : HeightObj_theory_iwasawa_control r <= m).
  { exact hm. }
  assert (hKeepDense : DensityObj_theory_iwasawa_control r).
  { exact hDense. }
  exists m.
  split.
  - exact hBound.
  - exact hKeepDense.
Qed.
