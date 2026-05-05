(*
BENCHMARK_ID: TINY_MATHLIB_BATCH03_PROB_OPTIONAL_STOPPING_AXIOMATIC
PAIR_STEM: probability_optional_stopping_axiomatic
MATH_DOMAIN: Probability Theory
SOURCE_MATHLIB: Mathlib/Probability/Martingale
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class ProbSpaceLike (Omega : Type) := {
  Expect : (Omega -> nat) -> nat
}.

Definition FiltrationLike (Omega : Type) : Type :=
  nat -> (Omega -> Prop) -> Prop.

Definition AdaptedLike {Omega : Type}
    (F : FiltrationLike Omega) (X : nat -> Omega -> nat) : Prop :=
  forall n m : nat, F n (fun w => X m w = X m w).

Definition MartingaleLike {Omega : Type} `{ProbSpaceLike Omega}
    (F : FiltrationLike Omega) (X : nat -> Omega -> nat) : Prop :=
  AdaptedLike F X /\
    (forall n : nat,
      Expect (X (S n)) = Expect (X n)).

Definition StoppingTimeLike {Omega : Type}
    (F : FiltrationLike Omega) (tau : nat) : Prop :=
  forall n : nat, F n (fun _ : Omega => True).

Definition StoppedValue {Omega : Type}
    (X : nat -> Omega -> nat) (tau : nat) : nat -> Omega -> nat :=
  fun n w => X (Nat.min tau n) w.

Lemma stopped_adapted {Omega : Type}
    (F : FiltrationLike Omega) (X : nat -> Omega -> nat) (tau : nat)
    (hmono : forall m n : nat, m <= n -> forall s : Omega -> Prop, F m s -> F n s)
    (hinter : forall n : nat, forall s t : Omega -> Prop,
      F n s -> F n t -> F n (fun w => s w /\ t w))
    (hsuperset : forall n : nat, forall s t : Omega -> Prop,
      F n s -> (forall w : Omega, s w -> t w) -> F n t)
    (hX : AdaptedLike F X)
    (htau : StoppingTimeLike F tau) :
    AdaptedLike F (StoppedValue X tau).
Proof.
  intro n.
  intro m.
  assert (hlift : F n (fun w => X (Nat.min tau m) w = X (Nat.min tau m) w)).
  { apply (hX n (Nat.min tau m)). }
  assert (hstop : F n (fun _ : Omega => True)).
  { apply htau. }
  assert (hboth : F n (fun w => (X (Nat.min tau m) w = X (Nat.min tau m) w) /\ True)).
  {
    apply (hinter n (fun w => X (Nat.min tau m) w = X (Nat.min tau m) w) (fun _ => True)); assumption.
  }
  assert (hsub : forall w : Omega,
    ((X (Nat.min tau m) w = X (Nat.min tau m) w) /\ True) ->
    StoppedValue X tau m w = StoppedValue X tau m w).
  {
    intros w hw.
    destruct hw as [hEq hT].
    simpl.
    exact hEq.
  }
  apply (hsuperset n (fun w => (X (Nat.min tau m) w = X (Nat.min tau m) w) /\ True)
    (fun w => StoppedValue X tau m w = StoppedValue X tau m w)).
  - exact hboth.
  - exact hsub.
Qed.

Lemma stopped_martingale {Omega : Type} `{ProbSpaceLike Omega}
    (F : FiltrationLike Omega) (X : nat -> Omega -> nat) (tau : nat)
    (hmono : forall m n : nat, m <= n -> forall s : Omega -> Prop, F m s -> F n s)
    (hinter : forall n : nat, forall s t : Omega -> Prop,
      F n s -> F n t -> F n (fun w => s w /\ t w))
    (hsuperset : forall n : nat, forall s t : Omega -> Prop,
      F n s -> (forall w : Omega, s w -> t w) -> F n t)
    (htau : StoppingTimeLike F tau)
    (hM : MartingaleLike F X)
    (hstationary : forall n : nat,
      Expect (X (Nat.min tau (S n))) =
        Expect (X (Nat.min tau n))) :
    MartingaleLike F (StoppedValue X tau).
Proof.
  split.
  - exact (stopped_adapted (F := F) (X := X) (tau := tau) hmono hinter hsuperset (proj1 hM) htau).
  - intro n.
    assert (hstep :
      Expect (X (Nat.min tau (S n))) = Expect (X (Nat.min tau n))).
    { apply hstationary. }
    assert (hleft :
      Expect ((StoppedValue X tau) (S n)) =
        Expect (X (Nat.min tau (S n)))).
    { reflexivity. }
    assert (hright :
      Expect ((StoppedValue X tau) n) =
        Expect (X (Nat.min tau n))).
    { reflexivity. }
    transitivity (Expect (X (Nat.min tau (S n)))).
    + exact hleft.
    + transitivity (Expect (X (Nat.min tau n))).
      * exact hstep.
      * symmetry.
        exact hright.
Qed.

Lemma optional_stopping_submartingale_bound {Omega : Type} `{ProbSpaceLike Omega}
    (X : nat -> Omega -> nat) (tau : nat)
    (hstop_eq : forall n : nat,
      Expect ((StoppedValue X tau) (S n)) =
        Expect ((StoppedValue X tau) n))
    (hbound0n : forall n : nat,
      Expect ((StoppedValue X tau) 0) <= Expect (X n))
    (n : nat) :
    Expect ((StoppedValue X tau) n) <= Expect (X n).
Proof.
  assert (hconst : forall k : nat,
    Expect ((StoppedValue X tau) k) = Expect ((StoppedValue X tau) 0)).
  {
    intro k.
    induction k as [| k ih].
    - reflexivity.
    - transitivity (Expect ((StoppedValue X tau) k)).
      + apply hstop_eq.
      + exact ih.
  }
  rewrite hconst.
  apply hbound0n.
Qed.

Lemma optional_stopping_eq_expectation {Omega : Type} `{ProbSpaceLike Omega}
    (F : FiltrationLike Omega) (X : nat -> Omega -> nat) (tau : nat)
    (hM : MartingaleLike F (StoppedValue X tau)) :
    forall n : nat,
      Expect ((StoppedValue X tau) n) =
        Expect ((StoppedValue X tau) 0).
Proof.
  intro n.
  induction n as [| n ih].
  - reflexivity.
  - transitivity (Expect ((StoppedValue X tau) n)).
    + apply (proj2 hM).
    + exact ih.
Qed.

Lemma optional_stopping_iterated {Omega : Type}
    (X : nat -> Omega -> nat) (tau sigma : nat)
    (hmin_assoc : forall n : nat, Nat.min tau (Nat.min sigma n) = Nat.min (Nat.min tau sigma) n)
    (n : nat) (w : Omega) :
    StoppedValue (StoppedValue X tau) sigma n w = StoppedValue X (Nat.min tau sigma) n w.
Proof.
  assert (hmin : Nat.min tau (Nat.min sigma n) = Nat.min (Nat.min tau sigma) n).
  { apply hmin_assoc. }
  unfold StoppedValue.
  simpl.
  rewrite hmin.
  reflexivity.
Qed.

Lemma uniform_integrable_extension {Omega : Type} `{ProbSpaceLike Omega}
    (X Y : nat -> Omega -> nat) (tau : nat)
    (hlink : forall n : nat,
      Expect (Y n) = Expect ((StoppedValue X tau) n))
    (hstable : forall n : nat,
      Expect ((StoppedValue X tau) n) =
        Expect ((StoppedValue X tau) 0)) :
    forall n : nat,
      Expect (Y n) = Expect ((StoppedValue X tau) 0).
Proof.
  intro n.
  assert (h1 : Expect (Y n) = Expect ((StoppedValue X tau) n)).
  { apply hlink. }
  assert (h2 : Expect ((StoppedValue X tau) n) = Expect ((StoppedValue X tau) 0)).
  { apply hstable. }
  transitivity (Expect ((StoppedValue X tau) n)).
  - exact h1.
  - exact h2.
Qed.
