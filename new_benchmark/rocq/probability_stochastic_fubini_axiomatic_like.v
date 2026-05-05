(*
BENCHMARK_ID: TINY_MATHLIB_BATCH05_PROBABILITY_STOCHASTIC_FUBINI_AXIOMATIC_LIKE
PAIR_STEM: probability_stochastic_fubini_axiomatic_like
MATH_DOMAIN: Probability Theory
SOURCE_MATHLIB: Mathlib/Probability/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class ProbStruct_stochastic_fubini (Omega : Type) := {
  Expect : (Omega -> nat) -> nat;
  CondExp : nat -> (Omega -> nat) -> (Omega -> nat);
  Expect_mono : forall {f g : Omega -> nat}, (forall w : Omega, f w <= g w) -> Expect f <= Expect g;
  Tower_axiom : forall n : nat, forall f : Omega -> nat, Expect (CondExp n f) = Expect f;
  Cond_mono_axiom :
    forall n : nat, forall {f g : Omega -> nat}, (forall w : Omega, f w <= g w) ->
      (forall w : Omega, CondExp n f w <= CondExp n g w);
  Change_measure_axiom :
    forall w f : Omega -> nat, Expect f <= Expect (fun x => f x + w x);
  Stopping_axiom : forall tau : nat, forall f : Omega -> nat, Expect (CondExp tau f) <= Expect f;
  Nat_le_trans_axiom : forall a b c : nat, a <= b -> b <= c -> a <= c;
  Fubini_swap_axiom :
    forall n m : nat, forall f : Omega -> nat,
      Expect (CondExp n (CondExp m f)) = Expect (CondExp m (CondExp n f))
}.

Definition Filtration_stochastic_fubini (Omega : Type) : Type :=
  nat -> (Omega -> Prop) -> Prop.

Definition MartingaleStep_stochastic_fubini {Omega : Type}
    (X : nat -> Omega -> nat) : nat -> Omega -> nat :=
  fun n w => X (S n) w.

Definition DriftShift_stochastic_fubini {Omega : Type}
    (X D : nat -> Omega -> nat) : nat -> Omega -> nat :=
  fun n w => X n w + D n w.

Definition RateFunc_stochastic_fubini (a b : nat -> nat) : nat -> nat :=
  fun n => a n + b n.

Lemma adaptivity_rule_stochastic_fubini
    {Omega : Type} (F : Filtration_stochastic_fubini Omega)
    (X : nat -> Omega -> nat)
    (hAdapt : forall n m : nat, F n (fun w => X m w = X m w)) :
    forall n m : nat,
      F n
        (fun w =>
          MartingaleStep_stochastic_fubini X m w = MartingaleStep_stochastic_fubini X m w).
Proof.
  intros n m.
  assert (hShift : F n (fun w => X (S m) w = X (S m) w)).
  { exact (hAdapt n (S m)). }
  change (F n (fun w => X (S m) w = X (S m) w)).
  exact hShift.
Qed.

Lemma tower_property_stochastic_fubini
    {Omega : Type} `{ProbStruct_stochastic_fubini Omega}
    (n : nat) (f : Omega -> nat) :
    Expect (CondExp n f) = Expect f.
Proof.
  assert (hRaw : Expect (CondExp n f) = Expect f).
  { exact (Tower_axiom n f). }
  assert (hLeft : Expect (CondExp n f) = Expect (fun w => CondExp n f w)).
  { reflexivity. }
  assert (hRight : Expect (fun w => f w) = Expect f).
  { reflexivity. }
  transitivity (Expect (fun w => CondExp n f w)).
  - exact hLeft.
  - transitivity (Expect f).
    + exact hRaw.
    + rewrite <- hRight. reflexivity.
Qed.

Lemma change_measure_step_stochastic_fubini
    {Omega : Type} `{ProbStruct_stochastic_fubini Omega}
    (w f : Omega -> nat) :
    Expect f <= Expect (DriftShift_stochastic_fubini (fun _ : nat => f) (fun _ : nat => w) 0).
Proof.
  assert (hAxiom : Expect f <= Expect (fun x => f x + w x)).
  { exact (Change_measure_axiom w f). }
  change (Expect f <= Expect (fun x => f x + w x)).
  exact hAxiom.
Qed.

Lemma stopping_control_stochastic_fubini
    {Omega : Type} `{ProbStruct_stochastic_fubini Omega}
    (tau : nat) (f : Omega -> nat) :
    Expect (CondExp tau f) <= Expect f.
Proof.
  assert (hStop : Expect (CondExp tau f) <= Expect f).
  { exact (Stopping_axiom tau f). }
  exact hStop.
Qed.

Lemma ld_upper_bound_stochastic_fubini
    {Omega : Type} `{ProbStruct_stochastic_fubini Omega}
    (f : Omega -> nat) (a b : nat -> nat) (n : nat)
    (hBase : Expect f <= RateFunc_stochastic_fubini a b n)
    (hRate : RateFunc_stochastic_fubini a b n <= RateFunc_stochastic_fubini a b (S n)) :
    Expect f <= RateFunc_stochastic_fubini a b (S n).
Proof.
  assert (hStep : Expect f <= RateFunc_stochastic_fubini a b n).
  { exact hBase. }
  exact (Nat_le_trans_axiom
    (a := Expect f)
    (b := RateFunc_stochastic_fubini a b n)
    (c := RateFunc_stochastic_fubini a b (S n))
    hStep hRate).
Qed.

Lemma coupling_estimate_stochastic_fubini
    {Omega : Type} `{ProbStruct_stochastic_fubini Omega}
    (f g w : Omega -> nat)
    (hfg : forall x : Omega, f x <= g x)
    (hgw : forall x : Omega, g x <= g x + w x) :
    Expect f <= Expect (fun x => g x + w x).
Proof.
  assert (hMono1 : Expect f <= Expect g).
  { exact (Expect_mono hfg). }
  assert (hMono2 : Expect g <= Expect (fun x => g x + w x)).
  { exact (Expect_mono hgw). }
  exact (Nat_le_trans_axiom
    (a := Expect f)
    (b := Expect g)
    (c := Expect (fun x => g x + w x))
    hMono1 hMono2).
Qed.

Lemma stochastic_fubini_rule_stochastic_fubini
    {Omega : Type} `{ProbStruct_stochastic_fubini Omega}
    (n m : nat) (f : Omega -> nat) :
    Expect (CondExp n (CondExp m f)) = Expect f.
Proof.
  assert (hSwap : Expect (CondExp n (CondExp m f)) = Expect (CondExp m (CondExp n f))).
  { exact (Fubini_swap_axiom n m f). }
  assert (hTower1 : Expect (CondExp m (CondExp n f)) = Expect (CondExp n f)).
  { exact (Tower_axiom m (CondExp n f)). }
  assert (hTower2 : Expect (CondExp n f) = Expect f).
  { exact (Tower_axiom n f). }
  transitivity (Expect (CondExp m (CondExp n f))).
  - exact hSwap.
  - transitivity (Expect (CondExp n f)).
    + exact hTower1.
    + exact hTower2.
Qed.
