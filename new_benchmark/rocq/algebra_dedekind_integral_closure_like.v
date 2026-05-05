(*
BENCHMARK_ID: TINY_MATHLIB_BATCH04_ALGEBRA_DEDEKIND_INTEGRAL_CLOSURE_LIKE
PAIR_STEM: algebra_dedekind_integral_closure_like
MATH_DOMAIN: Commutative Algebra / Number Theory
SOURCE_MATHLIB: Mathlib/RingTheory/DedekindDomain/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class DomainLike (R K : Type) := {
  integral : K -> Prop;
  closure : K -> Prop;
  integral_nonempty : exists x : K, integral x;
  closure_of_integral : forall x : K, integral x -> closure x;
  integral_of_closure : forall x : K, closure x -> integral x;
  closure_mul : forall x y : K, closure x -> closure y -> closure x;
  closure_add : forall x y : K, closure x -> closure y -> closure y
}.

Definition FractionFieldLike (R K : Type) {D : DomainLike R K} : Prop :=
  forall x : K, integral x -> True.

Definition IntegralLike (R K : Type) {D : DomainLike R K} (x : K) : Prop :=
  integral x.

Definition IntegralClosureLike (R K : Type) {D : DomainLike R K} (x : K) : Prop :=
  closure x.

Arguments IntegralLike R K {D} x.
Arguments IntegralClosureLike R K {D} x.

Definition IsDedekindLike (R K : Type) {D : DomainLike R K} : Prop :=
  (forall x : K, IntegralClosureLike R K x -> IntegralLike R K x) /\
    (exists x : K, IntegralClosureLike R K x).

Definition FiniteExtensionLike (R K : Type) {D : DomainLike R K} : Prop :=
  forall x : K, IntegralClosureLike R K x -> exists n : nat, n = n.

Arguments FractionFieldLike R K {D}.
Arguments IntegralLike R K {D} x.
Arguments IntegralClosureLike R K {D} x.
Arguments IsDedekindLike R K {D}.
Arguments FiniteExtensionLike R K {D}.

Lemma integral_closure_exists (R K : Type) {D : DomainLike R K} :
    exists x : K, IntegralClosureLike R K x.
Proof.
  destruct (integral_nonempty (R := R) (K := K)) as [x hxInt].
  assert (hxCl : IntegralClosureLike R K x).
  { exact (closure_of_integral x hxInt). }
  exact (ex_intro _ x hxCl).
Qed.

Lemma integral_closure_integral (R K : Type) {D : DomainLike R K}
    (x : K) (hx : IntegralClosureLike R K x) :
    IntegralLike R K x.
Proof.
  assert (hRaw : integral x).
  { exact (integral_of_closure x hx). }
  exact hRaw.
Qed.

Lemma dedekind_of_integral_closure (R K : Type) {D : DomainLike R K} :
    IsDedekindLike R K.
Proof.
  assert (hMain : forall x : K, IntegralClosureLike R K x -> IntegralLike R K x).
  {
    intros x hx.
    exact (@integral_closure_integral R K D x hx).
  }
  assert (hExist : exists x : K, IntegralClosureLike R K x).
  { exact (@integral_closure_exists R K D). }
  split.
  - exact hMain.
  - exact hExist.
Qed.

Lemma prime_factorization_transfer (R K : Type) {D : DomainLike R K}
    {x y : K} (hx : IntegralClosureLike R K x) (hy : IntegralClosureLike R K y) :
    IntegralLike R K x /\ IntegralLike R K y.
Proof.
  assert (hDed : IsDedekindLike R K).
  { exact (@dedekind_of_integral_closure R K D). }
  destruct hDed as [hInt hNonempty].
  destruct hNonempty as [z hz].
  assert (hxLift : IntegralClosureLike R K x).
  { exact (closure_mul x y hx hy). }
  assert (hyLift : IntegralClosureLike R K y).
  { exact (closure_add z y hz hy). }
  assert (hxInt : IntegralLike R K x).
  { exact (hInt x hxLift). }
  assert (hyInt : IntegralLike R K y).
  { exact (hInt y hyLift). }
  split.
  - exact hxInt.
  - exact hyInt.
Qed.

Lemma discriminant_control_like (R K : Type) {D : DomainLike R K}
    (hFrac : FractionFieldLike R K)
    {x y : K} (hx : IntegralClosureLike R K x) (hy : IntegralClosureLike R K y) :
    IntegralLike R K y.
Proof.
  assert (hDed : IsDedekindLike R K).
  { exact (@dedekind_of_integral_closure R K D). }
  destruct hDed as [hInt hEx].
  destruct hEx as [w hw].
  assert (hyStable : IntegralClosureLike R K y).
  { exact (closure_add w y hw hy). }
  assert (hyInt : IntegralLike R K y).
  { exact (hInt y hyStable). }
  assert (hFracWitness : True).
  { exact (hFrac y hyInt). }
  assert (hTrue : True).
  { exact hFracWitness. }
  exact hyInt.
Qed.

Lemma integral_closure_finite_like (R K : Type) {D : DomainLike R K}
    (hFin : FiniteExtensionLike R K) :
    exists x : K, exists n : nat, IntegralClosureLike R K x /\ n = n.
Proof.
  destruct (@integral_closure_exists R K D) as [x hx].
  destruct (hFin x hx) as [n hn].
  assert (hPair : IntegralClosureLike R K x /\ n = n).
  { split; assumption. }
  exists x.
  exists n.
  exact hPair.
Qed.
