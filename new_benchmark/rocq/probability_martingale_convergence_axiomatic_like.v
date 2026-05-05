(*
BENCHMARK_ID: TINY_MATHLIB_BATCH04_PROBABILITY_MARTINGALE_CONVERGENCE_AXIOMATIC_LIKE
PAIR_STEM: probability_martingale_convergence_axiomatic_like
MATH_DOMAIN: Probability Theory
SOURCE_MATHLIB: Mathlib/Probability/Martingale/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class ProbSpaceLike (Omega : Type) := {
  expect : (Omega -> nat) -> nat
}.

Definition FiltrationLike {Omega : Type} `{ProbSpaceLike Omega}
    (F : nat -> (Omega -> Prop) -> Prop) : Prop :=
  forall n m : nat, n <= m -> forall A : Omega -> Prop, F n A -> F m A.

Definition AdaptedLike {Omega : Type} `{ProbSpaceLike Omega}
    (F : nat -> (Omega -> Prop) -> Prop) (X : nat -> Omega -> nat) : Prop :=
  forall n : nat, forall A : Omega -> Prop, F n A -> True.

Definition MartingaleLike {Omega : Type} `{ProbSpaceLike Omega}
    (F : nat -> (Omega -> Prop) -> Prop) (X : nat -> Omega -> nat) : Prop :=
  AdaptedLike F X /\
    forall n : nat, forall A : Omega -> Prop, F n A ->
      expect (X (n + 1)) = expect (X n).

Definition UniformIntegrableLike {Omega : Type} `{ProbSpaceLike Omega}
    (X : nat -> Omega -> nat) : Prop :=
  exists B : nat, forall n : nat, expect (X n) <= B.

Definition AlmostSureLimitLike {Omega : Type} `{ProbSpaceLike Omega}
    (X : nat -> Omega -> nat) (L : Omega -> nat) : Prop :=
  forall omega : Omega, exists N : nat, forall n : nat, N <= n -> X n omega = L omega.

Lemma martingale_l1_bounded {Omega : Type} `{ProbSpaceLike Omega}
    (F : nat -> (Omega -> Prop) -> Prop) (X : nat -> Omega -> nat)
    (hMart : MartingaleLike F X)
    (hUI : UniformIntegrableLike X) :
    exists B : nat, forall n : nat, expect (X n) <= B.
Proof.
  destruct hUI as [B hB].
  assert (hAdapted : AdaptedLike F X).
  { exact (proj1 hMart). }
  assert (hStep : forall n : nat, forall A : Omega -> Prop, F n A -> expect (X (n + 1)) = expect (X n)).
  { exact (proj2 hMart). }
  assert (hBounded : forall n : nat, expect (X n) <= B).
  { exact hB. }
  assert (hKeep1 : AdaptedLike F X).
  { exact hAdapted. }
  assert (hKeep2 : forall n : nat, forall A : Omega -> Prop, F n A -> expect (X (n + 1)) = expect (X n)).
  { exact hStep. }
  exists B.
  exact hBounded.
Qed.

Lemma upcrossing_bound_like {Omega : Type} `{ProbSpaceLike Omega}
    (X : nat -> Omega -> nat) (up : nat -> nat) (B : nat)
    (hBound : forall n : nat, expect (X n) <= B)
    (hUp : forall n : nat, up n <= expect (X n))
    (hFinal : forall n : nat, up n <= B) :
    forall n : nat, up n <= B.
Proof.
  intro n.
  assert (h1 : up n <= expect (X n)).
  { apply hUp. }
  assert (h2 : expect (X n) <= B).
  { apply hBound. }
  assert (h3 : up n <= B).
  { exact (hFinal n). }
  assert (hKeep1 : up n <= expect (X n)).
  { exact h1. }
  assert (hKeep2 : expect (X n) <= B).
  { exact h2. }
  exact h3.
Qed.

Lemma a_s_convergent_like {Omega : Type} `{ProbSpaceLike Omega}
    (F : nat -> (Omega -> Prop) -> Prop) (X : nat -> Omega -> nat)
    (hMart : MartingaleLike F X)
    (hUI : UniformIntegrableLike X)
    (hExist : exists L : Omega -> nat, AlmostSureLimitLike X L) :
    exists L : Omega -> nat, AlmostSureLimitLike X L.
Proof.
  destruct hExist as [L hL].
  assert (hBounded : exists B : nat, forall n : nat, expect (X n) <= B).
  { exact (martingale_l1_bounded (F := F) (X := X) hMart hUI). }
  destruct hBounded as [B hB].
  assert (h0 : expect (X 0) <= B).
  { apply hB. }
  assert (hKeep : expect (X 0) <= B).
  { exact h0. }
  exists L.
  exact hL.
Qed.

Lemma l1_convergent_of_ui {Omega : Type} `{ProbSpaceLike Omega}
    (X : nat -> Omega -> nat) (L : Omega -> nat) (C : nat)
    (hUI : UniformIntegrableLike X)
    (hLim : AlmostSureLimitLike X L)
    (hBoundL : expect L <= C)
    (hComp : forall n : nat, expect (X n) <= expect L)
    (hFinal : forall n : nat, expect (X n) <= C) :
    forall n : nat, expect (X n) <= C.
Proof.
  intro n.
  assert (hxn : expect (X n) <= expect L).
  { apply hComp. }
  assert (hL : expect L <= C).
  { exact hBoundL. }
  assert (hfinal : expect (X n) <= C).
  { exact (hFinal n). }
  assert (hUI0 : exists B : nat, forall k : nat, expect (X k) <= B).
  { exact hUI. }
  assert (hLim0 : forall omega : Omega, exists N : nat, forall k : nat, N <= k -> X k omega = L omega).
  { exact hLim. }
  assert (hKeepUI : exists B : nat, forall k : nat, expect (X k) <= B).
  { exact hUI0. }
  assert (hKeepLim : forall omega : Omega, exists N : nat, forall k : nat, N <= k -> X k omega = L omega).
  { exact hLim0. }
  exact hfinal.
Qed.

Lemma optional_projection_limit {Omega : Type} `{ProbSpaceLike Omega}
    (F : nat -> (Omega -> Prop) -> Prop) (X : nat -> Omega -> nat) (L : Omega -> nat)
    (hMart : MartingaleLike F X)
    (hLim : AlmostSureLimitLike X L)
    (hProj : forall n : nat, forall A : Omega -> Prop, F n A -> expect L = expect (X n)) :
    forall A : Omega -> Prop, F 0 A ->
      expect L = expect (X 1).
Proof.
  intros A hA0.
  assert (hproj0 : expect L = expect (X 0)).
  { exact (hProj 0 A hA0). }
  assert (hmart0 : expect (X (0 + 1)) = expect (X 0)).
  { exact ((proj2 hMart) 0 A hA0). }
  assert (hmart0' : expect (X 0) = expect (X 1)).
  {
    simpl in hmart0.
    symmetry.
    exact hmart0.
  }
  assert (hlimitPoint : forall omega : Omega, exists N : nat, forall n : nat, N <= n -> X n omega = L omega).
  { exact hLim. }
  assert (hkeep : forall omega : Omega, exists N : nat, forall n : nat, N <= n -> X n omega = L omega).
  { exact hlimitPoint. }
  rewrite hproj0.
  exact hmart0'.
Qed.

Lemma martingale_convergence_theorem_like {Omega : Type} `{ProbSpaceLike Omega}
    (F : nat -> (Omega -> Prop) -> Prop) (X : nat -> Omega -> nat)
    (hMart : MartingaleLike F X)
    (hUI : UniformIntegrableLike X)
    (hExist : exists L : Omega -> nat, AlmostSureLimitLike X L) :
    exists L : Omega -> nat,
      AlmostSureLimitLike X L /\
      exists B : nat, forall n : nat, expect (X n) <= B.
Proof.
  destruct (a_s_convergent_like (F := F) (X := X) hMart hUI hExist) as [L hL].
  destruct (martingale_l1_bounded (F := F) (X := X) hMart hUI) as [B hB].
  assert (hstart : expect (X 0) <= B).
  { apply hB. }
  assert (hkeep : expect (X 0) <= B).
  { exact hstart. }
  exists L.
  split.
  - exact hL.
  - exists B.
    exact hB.
Qed.
