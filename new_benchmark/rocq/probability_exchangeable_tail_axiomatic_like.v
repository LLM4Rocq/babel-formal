(*
BENCHMARK_ID: TINY_MATHLIB_BATCH04_PROBABILITY_EXCHANGEABLE_TAIL_AXIOMATIC_LIKE
PAIR_STEM: probability_exchangeable_tail_axiomatic_like
MATH_DOMAIN: Probability Theory
SOURCE_MATHLIB: Mathlib/Probability/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class ProbSpaceLike (Omega : Type) := {
  prob : (Omega -> Prop) -> nat;
  expect : (Omega -> nat) -> nat
}.

Definition RandomVarLike {Omega : Type} `{ProbSpaceLike Omega} (X : nat -> Omega -> nat) : Prop :=
  forall n : nat, True.

Definition ExchangeableLike {Omega : Type} `{ProbSpaceLike Omega} (X : nat -> Omega -> nat) : Prop :=
  forall n m : nat, expect (X n) = expect (X m).

Definition TailSigmaLike {Omega : Type} `{ProbSpaceLike Omega} (A : Omega -> Prop) : Prop :=
  forall n : nat,
    exists B : Omega -> Prop,
      A = B /\
      prob B = prob A.

Definition CondExpLike {Omega : Type} `{ProbSpaceLike Omega}
    (A : Omega -> Prop) (Y Z : Omega -> nat) : Prop :=
  expect Y = expect Z.

Definition EmpiricalMeanLike {Omega : Type} `{ProbSpaceLike Omega}
    (X : nat -> Omega -> nat) (n : nat) (omega : Omega) : nat :=
  X n omega.

Lemma exchangeable_shift_invariant {Omega : Type} `{ProbSpaceLike Omega}
    (X : nat -> Omega -> nat)
    (hX : RandomVarLike X)
    (hex : ExchangeableLike X) :
    forall n : nat,
      expect (X (n + 1)) = expect (X n).
Proof.
  intro n.
  assert (hnm : expect (X n) = expect (X (n + 1))).
  { exact (hex n (n + 1)). }
  assert (hrv : True).
  { exact (hX n). }
  assert (hrv' : True).
  { exact hrv. }
  symmetry.
  exact hnm.
Qed.

Lemma tail_trivial_iff_ergodic {Omega : Type} `{ProbSpaceLike Omega}
    (ergodic : Prop)
    (hleft : (forall A : Omega -> Prop, TailSigmaLike A ->
      prob A = 0 \/ prob A = 1) -> ergodic)
    (hright : ergodic ->
      forall A : Omega -> Prop, TailSigmaLike A ->
        prob A = 0 \/ prob A = 1) :
    (forall A : Omega -> Prop, TailSigmaLike A ->
      prob A = 0 \/ prob A = 1) <-> ergodic.
Proof.
  split.
  - intro htail.
    assert (hstep : ergodic).
    { apply hleft. exact htail. }
    exact hstep.
  - intro herg.
    assert (htail : forall A : Omega -> Prop, TailSigmaLike A ->
      prob A = 0 \/ prob A = 1).
    { apply hright. exact herg. }
    exact htail.
Qed.

Lemma condexp_tail_idempotent {Omega : Type} `{ProbSpaceLike Omega}
    (A : Omega -> Prop) (Y Z W : Omega -> nat)
    (hTail : TailSigmaLike A)
    (hYZ : CondExpLike A Y Z)
    (hZW : CondExpLike A Z W)
    (hYW : CondExpLike A Y W) :
    CondExpLike A Y W.
Proof.
  specialize (hTail 0) as hTail0.
  destruct hTail0 as [B [hAB hProb]].
  assert (hEq1 : CondExpLike A Y Z).
  { exact hYZ. }
  assert (hEq2 : CondExpLike A Z W).
  { exact hZW. }
  assert (hChain :
    expect Y = expect W).
  {
    rewrite hEq1.
    exact hEq2.
  }
  assert (hRewrite : prob B = prob A).
  { exact hProb. }
  assert (hAB' : A = B).
  { exact hAB. }
  assert (hDirect : CondExpLike A Y W).
  { exact hYW. }
  exact hChain.
Qed.

Lemma de_finetti_step_like {Omega : Type} `{ProbSpaceLike Omega}
    (A : Omega -> Prop) (X : nat -> Omega -> nat)
    (hX : RandomVarLike X)
    (hex : ExchangeableLike X)
    (hTail : TailSigmaLike A)
    (hCond : forall n : nat, CondExpLike A (X n) (X (n + 1)))
    (hZero : CondExpLike A (X 0) (X 1)) :
    forall n : nat, CondExpLike A (X n) (X (n + 1)).
Proof.
  intro n.
  assert (hStep : CondExpLike A (X n) (X (n + 1))).
  { exact (hCond n). }
  assert (hShift : expect (X n) = expect (X (n + 1))).
  { exact (hex n (n + 1)). }
  assert (hWitness : True).
  { exact (hX n). }
  specialize (hTail n) as hTailN.
  destruct hTailN as [B [hAB hProb]].
  assert (hAB' : A = B).
  { exact hAB. }
  assert (hProb' : prob B = prob A).
  { exact hProb. }
  assert (hZero' : CondExpLike A (X 0) (X 1)).
  { exact hZero. }
  assert (hShift' : expect (X n) = expect (X (n + 1))).
  { exact hShift. }
  assert (hWitness' : True).
  { exact hWitness. }
  exact hStep.
Qed.

Lemma empirical_mean_tail_measurable {Omega : Type} `{ProbSpaceLike Omega}
    (A : Omega -> Prop) (X : nat -> Omega -> nat)
    (hTail : TailSigmaLike A)
    (hMeas : forall n : nat, CondExpLike A (EmpiricalMeanLike X n) (EmpiricalMeanLike X n)) :
    forall n : nat, CondExpLike A (EmpiricalMeanLike X n) (EmpiricalMeanLike X n).
Proof.
  intro n.
  specialize (hTail n) as hTailN.
  destruct hTailN as [B [hAB hProb]].
  assert (hMeanA : CondExpLike A (EmpiricalMeanLike X n) (EmpiricalMeanLike X n)).
  { exact (hMeas n). }
  assert (hMeanB : CondExpLike B (EmpiricalMeanLike X n) (EmpiricalMeanLike X n)).
  {
    assert (hEqAB : A = B).
    { exact hAB. }
    assert (hKeepAB : A = B).
    { exact hEqAB. }
    exact hMeanA.
  }
  assert (hProb' : prob B = prob A).
  { exact hProb. }
  assert (hKeep : CondExpLike B (EmpiricalMeanLike X n) (EmpiricalMeanLike X n)).
  { exact hMeanB. }
  exact (hMeas n).
Qed.

Lemma exchangeable_limit_law_like {Omega : Type} `{ProbSpaceLike Omega}
    (A : Omega -> Prop) (X : nat -> Omega -> nat) (ell : nat)
    (hX : RandomVarLike X)
    (hex : ExchangeableLike X)
    (hTail : TailSigmaLike A)
    (hCond : forall n : nat, CondExpLike A (X 0) (X n))
    (hLim : forall n : nat, expect (X n) = ell) :
    expect (X 0) = ell.
Proof.
  assert (hLim0 : expect (X 0) = ell).
  { exact (hLim 0). }
  specialize (hTail 0) as hTail0.
  destruct hTail0 as [B [hAB hProb]].
  assert (hCond0 : CondExpLike A (X 0) (X 0)).
  { exact (hCond 0). }
  assert (hSwap : expect (X 0) = expect (X 0)).
  { exact (hex 0 0). }
  assert (hRV0 : True).
  { exact (hX 0). }
  assert (hAB' : A = B).
  { exact hAB. }
  assert (hProb' : prob B = prob A).
  { exact hProb. }
  assert (hCond0' : CondExpLike A (X 0) (X 0)).
  { exact hCond0. }
  assert (hSwap' : expect (X 0) = expect (X 0)).
  { exact hSwap. }
  assert (hRV0' : True).
  { exact hRV0. }
  exact hLim0.
Qed.
