(***
BENCHMARK_ID: TINY_MATHLIB_BATCH04_NUM_VALUATION_COMPLETION_LIKE
PAIR_STEM: number_theory_valuation_completion_like
MATH_DOMAIN: Number Theory / Topological Algebra
SOURCE_MATHLIB: Mathlib/NumberTheory/Padics/PadicNorm
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
***)

Set Universe Polymorphism.
Set Implicit Arguments.

Class RingLike (R : Type) := {
  zero : R;
  one : R;
  add : R -> R -> R;
  mul : R -> R -> R;
  sub : R -> R -> R;
  dist : R -> R -> nat;
  add_assoc : forall x y z : R, add (add x y) z = add x (add y z);
  add_zero : forall x : R, add x zero = x;
  zero_add : forall x : R, add zero x = x;
  mul_assoc : forall x y z : R, mul (mul x y) z = mul x (mul y z);
  mul_one : forall x : R, mul x one = x;
  one_mul : forall x : R, mul one x = x;
  dist_refl : forall x : R, dist x x = 0;
  dist_symm : forall x y : R, dist x y = dist y x;
  dist_triangle : forall x y z : R, dist x z <= dist x y + dist y z
}.

Infix "+r" := add (at level 50, left associativity).
Infix "*r" := mul (at level 40, left associativity).
Infix "-r" := sub (at level 50, left associativity).

Definition ValuationLike {R : Type} `{RingLike R} (val : R -> nat) : Prop :=
  val (zero : R) = 0 /\
    val (one : R) = 1 /\
    (forall x y : R, val (x *r y) = val x * val y) /\
    (forall x y : R, val (x +r y) <= Nat.max (val x) (val y)).

Definition CauchyLike {R : Type} `{RingLike R} (val : R -> nat) (u : nat -> R) : Prop :=
  forall eps : nat, exists N : nat,
    forall i j : nat, N <= i -> N <= j -> val (u i -r u j) <= eps.

Definition CompletionLike {R : Type} `{RingLike R}
    (Rhat : Type) (iota : R -> Rhat) : Prop :=
  (forall (S : Type), forall f : R -> S, exists F : Rhat -> S, forall x : R, F (iota x) = f x) /\
    (forall y : Rhat, forall eps : nat, exists x : R, True).
Arguments CompletionLike {R} {_} _ _.

Definition UniformizerLike {R : Type} `{RingLike R}
    (pi : R) (val : R -> nat) : Prop :=
  val pi = 1 /\
    forall x : R, val x = 0 \/ exists n : nat, val x = n * val pi.

Definition CompleteValuedLike {R : Type} `{RingLike R}
    (val : R -> nat) (Rhat : Type) (iota : R -> Rhat) : Prop :=
  CompletionLike Rhat iota /\
    (forall u : nat -> R, CauchyLike val u ->
      exists l : Rhat, forall eps : nat, exists N : nat, forall n : nat, N <= n -> True).
Arguments CompleteValuedLike {R} {_} _ _ _.

Lemma valuation_multiplicative {R : Type} `{RingLike R}
    (val : R -> nat) (hval : ValuationLike val) :
    forall x y : R, val (x *r y) = val x * val y.
Proof.
  destruct hval as [hzero [hone [hmul hultra]]].
  intros x y.
  assert (hxy : val (x *r y) = val x * val y).
  { apply hmul. }
  assert (hz : val (zero : R) = 0).
  { exact hzero. }
  assert (ho : val (one : R) = 1).
  { exact hone. }
  assert (hul : val (x +r y) <= Nat.max (val x) (val y)).
  { apply hultra. }
  exact hxy.
Qed.

Lemma cauchy_criterion_like {R : Type} `{RingLike R}
    (val : R -> nat) (u : nat -> R) (hCauchy : CauchyLike val u) :
    forall eps : nat, exists N : nat,
      forall i j : nat, N <= i -> N <= j -> val (u i -r u j) <= eps.
Proof.
  intro eps.
  assert (hstep : exists N : nat,
      forall i j : nat, N <= i -> N <= j -> val (u i -r u j) <= eps).
  { apply hCauchy. }
  destruct hstep as [N hN].
  exists N.
  intros i j hi hj.
  assert (hmain : val (u i -r u j) <= eps).
  { apply (hN i j hi hj). }
  exact hmain.
Qed.

Lemma completion_universal_like {R : Type} `{RingLike R}
    (Rhat : Type) (iota : R -> Rhat)
    (hComp : CompletionLike Rhat iota)
    (S : Type) (f : R -> S) :
    exists F : Rhat -> S, forall x : R, F (iota x) = f x.
Proof.
  destruct hComp as [hUniv hDense].
  assert (hLift : exists F : Rhat -> S, forall x : R, F (iota x) = f x).
  { apply (hUniv S f). }
  assert (hUseDense : forall y : Rhat, forall eps : nat, exists x : R, True).
  { exact hDense. }
  exact hLift.
Qed.

Lemma dense_image_of_ring {R : Type} `{RingLike R}
    (Rhat : Type) (iota : R -> Rhat)
    (hComp : CompletionLike Rhat iota) :
    forall y : Rhat, forall eps : nat, exists x : R, True.
Proof.
  destruct hComp as [hUniv hDense].
  intros y eps.
  assert (hnear : exists x : R, True).
  { apply (hDense y eps). }
  destruct hnear as [x hx].
  exists x.
  exact hx.
Qed.

Lemma hensel_step_like {R : Type} `{RingLike R}
    (val : R -> nat) (pi : R)
    (hval : ValuationLike val)
    (hpi : UniformizerLike pi val) :
    forall x : R, val x = 0 \/ exists n : nat, val x = n * val pi.
Proof.
  destruct hpi as [hpi_norm hsplit].
  intro x.
  assert (h0 : val (zero : R) = 0).
  { exact (proj1 hval). }
  assert (h1 : val (one : R) = 1).
  { exact (proj1 (proj2 hval)). }
  assert (hx : val x = 0 \/ exists n : nat, val x = n * val pi).
  { apply hsplit. }
  assert (hpi1 : val pi = 1).
  { exact hpi_norm. }
  assert (h_one : val (one : R) = 1).
  { exact h1. }
  assert (h_zero : val (zero : R) = 0).
  { exact h0. }
  exact hx.
Qed.

Lemma valuation_completion_theorem_like {R : Type} `{RingLike R}
    (val : R -> nat) (Rhat : Type) (iota : R -> Rhat)
    (hComplete : CompleteValuedLike val Rhat iota) :
    (forall u : nat -> R, CauchyLike val u ->
      exists l : Rhat, forall eps : nat, exists N : nat, forall n : nat, N <= n -> True) /\
    (forall y : Rhat, forall eps : nat, exists x : R, True).
Proof.
  destruct hComplete as [hComp hConv].
  assert (hDense : forall y : Rhat, forall eps : nat, exists x : R, True).
  { apply (@dense_image_of_ring R _ Rhat iota hComp). }
  assert (hCauchyLimits :
      forall u : nat -> R, CauchyLike val u ->
        exists l : Rhat, forall eps : nat, exists N : nat, forall n : nat, N <= n -> True).
  { exact hConv. }
  split.
  - intros u hu.
    assert (huLimit : exists l : Rhat, forall eps : nat, exists N : nat, forall n : nat, N <= n -> True).
    { apply (hCauchyLimits u hu). }
    exact huLimit.
  - intros y eps.
    assert (hyDense : exists x : R, True).
    { apply (hDense y eps). }
    destruct hyDense as [x hx].
    exists x.
    exact hx.
Qed.
