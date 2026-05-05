(*
BENCHMARK_ID: TINY_MATHLIB_BATCH05_PROBABILITY_STOCHASTIC_INTEGRAL_AXIOMATIC_LIKE
PAIR_STEM: probability_stochastic_integral_axiomatic_like
MATH_DOMAIN: Probability Theory
SOURCE_MATHLIB: Mathlib/Probability/StochasticIntegral
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class ProbStruct_stochastic_integral (Omega : Type) := {
  Event : Type;
  filtration : nat -> Event -> Prop;
  integral : (Omega -> nat) -> nat;
  condExp : nat -> (Omega -> nat) -> Omega -> nat;
  driftShift : nat -> (Omega -> nat) -> Omega -> nat;
  stopShift : nat -> (Omega -> nat) -> Omega -> nat;
  rate : nat -> nat;
  filtration_mono_axiom : forall {n m : nat} {A : Event}, n <= m -> filtration n A -> filtration m A;
  tower_axiom : forall {n m : nat} (X : Omega -> nat), n <= m -> condExp n (condExp m X) = condExp n X;
  change_measure_axiom : forall (n : nat) (X : Omega -> nat),
    integral (driftShift n X) <= integral X + rate n;
  stopping_axiom : forall (n : nat) (X : Omega -> nat),
    integral (stopShift n X) <= integral (driftShift n X);
  coupling_axiom : forall (n : nat) (X Y : Omega -> nat),
    integral (driftShift n X) <= rate n ->
    integral (driftShift n Y) <= rate n ->
    integral (fun w => driftShift n X w + driftShift n Y w) <= rate n + rate n;
  fubini_axiom : forall (n : nat) (X : Omega -> nat),
    integral (condExp n (driftShift n X)) = integral (driftShift n X);
  add_right_mono_axiom : forall a b c : nat, a <= b -> a + c <= b + c;
  nat_le_trans_axiom : forall a b c : nat, a <= b -> b <= c -> a <= c
}.

Definition Filtration_stochastic_integral {Omega : Type}
    (P : ProbStruct_stochastic_integral Omega) (n : nat) : @Event Omega P -> Prop :=
  fun A => @filtration Omega P n A.

Definition MartingaleStep_stochastic_integral {Omega : Type}
    (P : ProbStruct_stochastic_integral Omega) (n : nat) (X : Omega -> nat) : Omega -> nat :=
  @condExp Omega P n X.

Definition DriftShift_stochastic_integral {Omega : Type}
    (P : ProbStruct_stochastic_integral Omega) (n : nat) (X : Omega -> nat) : Omega -> nat :=
  @driftShift Omega P n X.

Definition RateFunc_stochastic_integral {Omega : Type}
    (P : ProbStruct_stochastic_integral Omega) (n : nat) : nat :=
  @rate Omega P n.

Lemma adaptivity_rule_stochastic_integral {Omega : Type}
    (P : ProbStruct_stochastic_integral Omega)
    {n m : nat} (hnm : n <= m) {A : @Event Omega P}
    (hA : Filtration_stochastic_integral P n A) :
    Filtration_stochastic_integral P m A.
Proof.
  unfold Filtration_stochastic_integral in *.
  assert (hMono : @filtration Omega P m A).
  { exact (@filtration_mono_axiom Omega P n m A hnm hA). }
  exact hMono.
Qed.

Lemma tower_property_stochastic_integral {Omega : Type}
    (P : ProbStruct_stochastic_integral Omega)
    {n m : nat} (X : Omega -> nat) (hnm : n <= m) :
    MartingaleStep_stochastic_integral P n (MartingaleStep_stochastic_integral P m X)
      = MartingaleStep_stochastic_integral P n X.
Proof.
  unfold MartingaleStep_stochastic_integral.
  exact (@tower_axiom Omega P n m X hnm).
Qed.

Lemma change_measure_step_stochastic_integral {Omega : Type}
    (P : ProbStruct_stochastic_integral Omega)
    (n : nat) (X : Omega -> nat) :
    @integral Omega P (DriftShift_stochastic_integral P n X)
      <= @integral Omega P X + RateFunc_stochastic_integral P n.
Proof.
  unfold DriftShift_stochastic_integral, RateFunc_stochastic_integral.
  exact (@change_measure_axiom Omega P n X).
Qed.

Lemma stopping_control_stochastic_integral {Omega : Type}
    (P : ProbStruct_stochastic_integral Omega)
    (n : nat) (X : Omega -> nat) :
    @integral Omega P (@stopShift Omega P n X) <= @integral Omega P X + RateFunc_stochastic_integral P n.
Proof.
  assert (hStop : @integral Omega P (@stopShift Omega P n X) <= @integral Omega P (@driftShift Omega P n X)).
  { exact (@stopping_axiom Omega P n X). }
  assert (hChange : @integral Omega P (@driftShift Omega P n X)
      <= @integral Omega P X + RateFunc_stochastic_integral P n).
  {
    unfold RateFunc_stochastic_integral.
    exact (@change_measure_axiom Omega P n X).
  }
  exact (@nat_le_trans_axiom Omega P
    (@integral Omega P (@stopShift Omega P n X))
    (@integral Omega P (@driftShift Omega P n X))
    (@integral Omega P X + RateFunc_stochastic_integral P n)
    hStop hChange).
Qed.

Lemma ld_upper_bound_stochastic_integral {Omega : Type}
    (P : ProbStruct_stochastic_integral Omega)
    (n : nat) (X : Omega -> nat)
    (hX : @integral Omega P X <= RateFunc_stochastic_integral P n) :
    @integral Omega P (DriftShift_stochastic_integral P n X)
      <= RateFunc_stochastic_integral P n + RateFunc_stochastic_integral P n.
Proof.
  assert (hChange : @integral Omega P (DriftShift_stochastic_integral P n X)
      <= @integral Omega P X + RateFunc_stochastic_integral P n).
  { exact (change_measure_step_stochastic_integral P n X). }
  assert (hAdd : @integral Omega P X + RateFunc_stochastic_integral P n
      <= RateFunc_stochastic_integral P n + RateFunc_stochastic_integral P n).
  {
    apply (@add_right_mono_axiom Omega P (@integral Omega P X) (RateFunc_stochastic_integral P n)
      (RateFunc_stochastic_integral P n)).
    exact hX.
  }
  exact (@nat_le_trans_axiom Omega P
    (@integral Omega P (DriftShift_stochastic_integral P n X))
    (@integral Omega P X + RateFunc_stochastic_integral P n)
    (RateFunc_stochastic_integral P n + RateFunc_stochastic_integral P n)
    hChange hAdd).
Qed.

Lemma coupling_estimate_stochastic_integral {Omega : Type}
    (P : ProbStruct_stochastic_integral Omega)
    (n : nat) (X Y : Omega -> nat)
    (hX : @integral Omega P (DriftShift_stochastic_integral P n X) <= RateFunc_stochastic_integral P n)
    (hY : @integral Omega P (DriftShift_stochastic_integral P n Y) <= RateFunc_stochastic_integral P n) :
    @integral Omega P (fun w => DriftShift_stochastic_integral P n X w + DriftShift_stochastic_integral P n Y w)
      <= RateFunc_stochastic_integral P n + RateFunc_stochastic_integral P n.
Proof.
  assert (hXraw : @integral Omega P (@driftShift Omega P n X) <= @rate Omega P n).
  {
    unfold DriftShift_stochastic_integral, RateFunc_stochastic_integral in hX.
    exact hX.
  }
  assert (hYraw : @integral Omega P (@driftShift Omega P n Y) <= @rate Omega P n).
  {
    unfold DriftShift_stochastic_integral, RateFunc_stochastic_integral in hY.
    exact hY.
  }
  assert (hCouple : @integral Omega P (fun w => @driftShift Omega P n X w + @driftShift Omega P n Y w)
      <= @rate Omega P n + @rate Omega P n).
  {
    apply (@coupling_axiom Omega P n X Y);
    assumption.
  }
  unfold DriftShift_stochastic_integral, RateFunc_stochastic_integral.
  exact hCouple.
Qed.

Lemma stochastic_fubini_rule_stochastic_integral {Omega : Type}
    (P : ProbStruct_stochastic_integral Omega)
    (n : nat) (X : Omega -> nat) :
    @integral Omega P (MartingaleStep_stochastic_integral P n (DriftShift_stochastic_integral P n X))
      = @integral Omega P (DriftShift_stochastic_integral P n X).
Proof.
  unfold MartingaleStep_stochastic_integral, DriftShift_stochastic_integral.
  exact (@fubini_axiom Omega P n X).
Qed.
