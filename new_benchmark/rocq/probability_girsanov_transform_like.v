(*
BENCHMARK_ID: TINY_MATHLIB_BATCH05_PROBABILITY_GIRSANOV_TRANSFORM_LIKE
PAIR_STEM: probability_girsanov_transform_like
MATH_DOMAIN: Probability Theory
SOURCE_MATHLIB: Mathlib/Probability/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class ProbStruct_girsanov_transform (Omega : Type) := {
  Expect : (Omega -> nat) -> nat;
  CondExpect : nat -> (Omega -> nat) -> (Omega -> nat);
  cond_tower_axiom :
    forall n m : nat, n <= m -> forall X : Omega -> nat,
      Expect (CondExpect n (CondExpect m X)) = Expect (CondExpect n X);
  cond_idem_axiom :
    forall n : nat, forall X : Omega -> nat,
      CondExpect n (CondExpect n X) = CondExpect n X;
  expect_add_axiom :
    forall X Y : Omega -> nat,
      Expect (fun omega => X omega + Y omega) = Expect X + Expect Y;
  expect_mono_axiom :
    forall X Y : Omega -> nat,
      (forall omega : Omega, X omega <= Y omega) -> Expect X <= Expect Y
}.

Definition Filtration_girsanov_transform (Omega : Type) : Type :=
  nat -> (Omega -> Prop) -> Prop.

Definition MartingaleStep_girsanov_transform {Omega : Type}
    `{ProbStruct_girsanov_transform Omega}
    (X : nat -> Omega -> nat) : Prop :=
  forall n : nat,
    Expect (X (S n)) = Expect (X n).

Definition DriftShift_girsanov_transform {Omega : Type}
    (X theta : nat -> Omega -> nat) : nat -> Omega -> nat :=
  fun n omega => X n omega + theta n omega.

Definition RateFunc_girsanov_transform {Omega : Type}
    `{ProbStruct_girsanov_transform Omega}
    (theta : nat -> Omega -> nat) : nat -> nat :=
  fun n => Expect (theta n).

Lemma adaptivity_rule_girsanov_transform {Omega : Type}
    (F : Filtration_girsanov_transform Omega)
    (X theta : nat -> Omega -> nat)
    (hX : forall n : nat, F n (fun omega => X n omega = X n omega))
    (hTheta : forall n : nat, F n (fun omega => theta n omega = theta n omega))
    (hInter : forall n : nat, forall s t : Omega -> Prop,
      F n s -> F n t -> F n (fun omega => s omega /\ t omega))
    (hSuperset : forall n : nat, forall s t : Omega -> Prop,
      F n s -> (forall omega : Omega, s omega -> t omega) -> F n t) :
    forall n : nat,
      F n (fun omega =>
        DriftShift_girsanov_transform X theta n omega =
          DriftShift_girsanov_transform X theta n omega).
Proof.
  intro n.
  assert (hLocalX : F n (fun omega => X n omega = X n omega)).
  { apply hX. }
  assert (hLocalTheta : F n (fun omega => theta n omega = theta n omega)).
  { apply hTheta. }
  assert (hBoth :
      F n (fun omega =>
        (X n omega = X n omega) /\ (theta n omega = theta n omega))).
  {
    apply (hInter n (fun omega => X n omega = X n omega)
      (fun omega => theta n omega = theta n omega)); assumption.
  }
  assert (hCast :
      forall omega : Omega,
        ((X n omega = X n omega) /\ (theta n omega = theta n omega)) ->
          DriftShift_girsanov_transform X theta n omega =
            DriftShift_girsanov_transform X theta n omega).
  {
    intros omega hPair.
    destruct hPair as [_ _].
    reflexivity.
  }
  apply (hSuperset n
    (fun omega => (X n omega = X n omega) /\ (theta n omega = theta n omega))
    (fun omega =>
      DriftShift_girsanov_transform X theta n omega =
        DriftShift_girsanov_transform X theta n omega)).
  - exact hBoth.
  - exact hCast.
Qed.

Lemma tower_property_girsanov_transform {Omega : Type}
    `{ProbStruct_girsanov_transform Omega}
    (n m : nat) (hnm : n <= m) (X : Omega -> nat) :
    Expect (CondExpect n (CondExpect m X)) =
      Expect (CondExpect n X).
Proof.
  assert (hTower :
      Expect (CondExpect n (CondExpect m X)) =
        Expect (CondExpect n X)).
  { exact (cond_tower_axiom (n := n) (m := m) hnm X). }
  exact hTower.
Qed.

Lemma change_measure_step_girsanov_transform {Omega : Type}
    `{ProbStruct_girsanov_transform Omega}
    (X theta : nat -> Omega -> nat)
    (hMart : MartingaleStep_girsanov_transform X)
    (hZero : forall n : nat,
      Expect (theta n) = 0)
    (n : nat) :
    Expect (DriftShift_girsanov_transform X theta n) =
      Expect (X (S n)).
Proof.
  assert (hAdd :
      Expect (DriftShift_girsanov_transform X theta n) =
        Expect (X n) + Expect (theta n)).
  {
    unfold DriftShift_girsanov_transform.
    apply (expect_add_axiom (X n) (theta n)).
  }
  assert (hZeroN : Expect (theta n) = 0).
  { apply hZero. }
  assert (hStep : Expect (X (S n)) = Expect (X n)).
  { apply hMart. }
  rewrite hAdd.
  rewrite hZeroN.
  rewrite <- hStep.
  rewrite plus_n_O.
  reflexivity.
Qed.

Lemma stopping_control_girsanov_transform {Omega : Type}
    `{ProbStruct_girsanov_transform Omega}
    (X : nat -> Omega -> nat)
    (tau n : nat) (htau : tau <= n)
    (hMonotone :
      forall a b : nat, a <= b ->
        Expect (X a) <= Expect (X b)) :
    Expect (X (Nat.min tau n)) <=
      Expect (X n).
Proof.
  assert (hMinEq : Nat.min tau n = tau).
  { apply min_l. exact htau. }
  assert (hTauBound : Expect (X tau) <= Expect (X n)).
  { apply hMonotone. exact htau. }
  rewrite hMinEq.
  exact hTauBound.
Qed.

Lemma ld_upper_bound_girsanov_transform {Omega : Type}
    `{ProbStruct_girsanov_transform Omega}
    (X theta : nat -> Omega -> nat)
    (hProc : forall n : nat,
      Expect (X n) <=
        Expect (X (S n)))
    (hRate : forall n : nat,
      Expect (theta n) <=
        RateFunc_girsanov_transform theta (S n))
    (hShift : forall n : nat,
      Expect (DriftShift_girsanov_transform X theta n) =
        Expect (X n) + Expect (theta n))
    (n : nat) :
    Expect (DriftShift_girsanov_transform X theta n) <=
      Expect (X (S n)) +
        RateFunc_girsanov_transform theta (S n).
Proof.
  assert (hLeft : Expect (X n) <= Expect (X (S n))).
  { apply hProc. }
  assert (hRight : Expect (theta n) <= RateFunc_girsanov_transform theta (S n)).
  { apply hRate. }
  assert (hAdd :
      Expect (X n) + Expect (theta n) <=
      Expect (X (S n)) + RateFunc_girsanov_transform theta (S n)).
  {
    assert (hTransNat : forall a b c : nat, a <= b -> b <= c -> a <= c).
    {
      intros a b c hab hbc.
      induction hbc.
      - exact hab.
      - apply le_S. apply IHhbc.
    }
    assert (hLiftL : forall w a b : nat, a <= b -> w + a <= w + b).
    {
      intro w.
      induction w as [| w ih].
      - intros a b hab. simpl. exact hab.
      - intros a b hab. simpl. apply le_n_S. apply ih. exact hab.
    }
    assert (hLiftR : forall w a b : nat, a <= b -> a + w <= b + w).
    {
      intros w a b hab.
      induction hab.
      - apply le_n.
      - simpl. apply le_S. exact IHhab.
    }
    assert (hStep1 :
      Expect (X n) + Expect (theta n) <=
      Expect (X (S n)) + Expect (theta n)).
    { apply (hLiftR (Expect (theta n)) (Expect (X n)) (Expect (X (S n)))). exact hLeft. }
    assert (hStep2 :
      Expect (X (S n)) + Expect (theta n) <=
      Expect (X (S n)) + RateFunc_girsanov_transform theta (S n)).
    { apply (hLiftL (Expect (X (S n))) (Expect (theta n)) (RateFunc_girsanov_transform theta (S n))). exact hRight. }
    exact (hTransNat _ _ _ hStep1 hStep2).
  }
  rewrite hShift.
  exact hAdd.
Qed.

Lemma coupling_estimate_girsanov_transform {Omega : Type}
    `{ProbStruct_girsanov_transform Omega}
    (X Y theta phi : nat -> Omega -> nat)
    (hCouple : forall n : nat,
      Expect (X n) <=
        Expect (Y n))
    (hRate : forall n : nat,
      RateFunc_girsanov_transform theta n <= RateFunc_girsanov_transform phi n)
    (hShiftX : forall n : nat,
      Expect (DriftShift_girsanov_transform X theta n) =
        Expect (X n) + RateFunc_girsanov_transform theta n)
    (hShiftY : forall n : nat,
      Expect (DriftShift_girsanov_transform Y phi n) =
        Expect (Y n) + RateFunc_girsanov_transform phi n)
    (n : nat) :
    Expect (DriftShift_girsanov_transform X theta n) <=
      Expect (DriftShift_girsanov_transform Y phi n).
Proof.
  assert (hBase :
      Expect (X n) + RateFunc_girsanov_transform theta n <=
        Expect (Y n) + RateFunc_girsanov_transform phi n).
  {
    assert (hTransNat : forall a b c : nat, a <= b -> b <= c -> a <= c).
    {
      intros a b c hab hbc.
      induction hbc.
      - exact hab.
      - apply le_S. apply IHhbc.
    }
    assert (hLiftL : forall w a b : nat, a <= b -> w + a <= w + b).
    {
      intro w.
      induction w as [| w ih].
      - intros a b hab. simpl. exact hab.
      - intros a b hab. simpl. apply le_n_S. apply ih. exact hab.
    }
    assert (hLiftR : forall w a b : nat, a <= b -> a + w <= b + w).
    {
      intros w a b hab.
      induction hab.
      - apply le_n.
      - simpl. apply le_S. exact IHhab.
    }
    assert (hStep1 :
      Expect (X n) + RateFunc_girsanov_transform theta n <=
      Expect (Y n) + RateFunc_girsanov_transform theta n).
    { apply (hLiftR (RateFunc_girsanov_transform theta n) (Expect (X n)) (Expect (Y n))). apply hCouple. }
    assert (hStep2 :
      Expect (Y n) + RateFunc_girsanov_transform theta n <=
      Expect (Y n) + RateFunc_girsanov_transform phi n).
    { apply (hLiftL (Expect (Y n)) (RateFunc_girsanov_transform theta n) (RateFunc_girsanov_transform phi n)). apply hRate. }
    exact (hTransNat _ _ _ hStep1 hStep2).
  }
  rewrite hShiftX.
  rewrite hShiftY.
  exact hBase.
Qed.

Lemma stochastic_fubini_rule_girsanov_transform {Omega : Type}
    `{ProbStruct_girsanov_transform Omega}
    (A B C : Omega -> nat) :
    Expect (fun omega => A omega + (B omega + C omega)) =
      Expect (fun omega => (A omega + B omega) + C omega).
Proof.
  assert (hLeftSplit :
      Expect (fun omega => A omega + (B omega + C omega)) =
      Expect A + Expect (fun omega => B omega + C omega)).
  { apply (expect_add_axiom A (fun omega => B omega + C omega)). }
  assert (hRightSplit :
      Expect (fun omega => (A omega + B omega) + C omega) =
      Expect (fun omega => A omega + B omega) + Expect C).
  { apply (expect_add_axiom (fun omega => A omega + B omega) C). }
  assert (hMidSplit :
      Expect (fun omega => A omega + B omega) =
      Expect A + Expect B).
  { apply (expect_add_axiom A B). }
  assert (hTailSplit :
      Expect (fun omega => B omega + C omega) =
      Expect B + Expect C).
  { apply (expect_add_axiom B C). }
  assert (hAssocNat : forall a b c : nat, a + (b + c) = (a + b) + c).
  {
    intro a.
    induction a as [| a iha].
    - intros b c. reflexivity.
    - intros b c. simpl. rewrite iha. reflexivity.
  }
  rewrite hLeftSplit.
  rewrite hTailSplit.
  rewrite (hAssocNat (Expect A) (Expect B) (Expect C)).
  rewrite <- hMidSplit.
  symmetry.
  exact hRightSplit.
Qed.
