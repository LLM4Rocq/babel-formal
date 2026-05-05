(***
BENCHMARK_ID: TINY_MATHLIB_BATCH03_NUM_DEDEKIND_NORM_LIKE
PAIR_STEM: number_theory_dedekind_norm_like
MATH_DOMAIN: Algebraic Number Theory
SOURCE_MATHLIB: Mathlib/NumberTheory/NumberField/Basic
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
***)

Set Universe Polymorphism.
Set Implicit Arguments.

Class DomainLike (A : Type) := {
  d_one : A;
  d_mul : A -> A -> A;
  idealMul : (A -> Prop) -> (A -> Prop) -> (A -> Prop)
}.

Infix "*" := d_mul (at level 40, left associativity).

Definition IdealLike (A : Type) : Type :=
  A -> Prop.

Definition PrimeIdealLike {A : Type} `{DomainLike A} (I : IdealLike A) : Prop :=
  forall a b : A, I (a * b) -> I a \/ I b.

Definition IsDedekindLike {A : Type} `{DomainLike A} : Type :=
  { factorRel : IdealLike A -> (IdealLike A -> nat) -> Prop &
    { norm : IdealLike A -> nat &
      (forall I : IdealLike A, exists F : IdealLike A -> nat, factorRel I F) /\
      (forall I : IdealLike A, forall F G : IdealLike A -> nat, factorRel I F -> factorRel I G -> F = G) /\
      (forall I J : IdealLike A, forall F G : IdealLike A -> nat,
        factorRel I F -> factorRel J G ->
        exists H : IdealLike A -> nat,
          factorRel (idealMul I J) H /\ (forall P : IdealLike A, H P = F P + G P)) /\
      (forall I J : IdealLike A, norm (idealMul I J) = Nat.mul (norm I) (norm J)) /\
      (forall P : IdealLike A, PrimeIdealLike P -> norm P > 1) /\
      (forall P : IdealLike A, PrimeIdealLike P -> forall n : nat,
        exists Ipow : IdealLike A, exists F : IdealLike A -> nat,
          factorRel Ipow F /\ F P = n /\ norm Ipow = Nat.pow (norm P) n) /\
      (exists B : nat, B > 0) /\
      (exists h : nat, h > 0)
    }
  }.

Definition IdealFactorizationLike {A : Type} `{DomainLike A}
    (hD : IsDedekindLike (A := A)) (I : IdealLike A) (F : IdealLike A -> nat) : Prop :=
  (projT1 hD) I F.

Definition IdealNormLike {A : Type} `{DomainLike A}
    (hD : IsDedekindLike (A := A)) (I : IdealLike A) : nat :=
  (projT1 (projT2 hD)) I.

Lemma factorization_exists {A : Type} `{DomainLike A}
    (hD : IsDedekindLike (A := A)) (I : IdealLike A) :
    exists F : IdealLike A -> nat, IdealFactorizationLike hD I F.
Proof.
  destruct hD as [factorRel [norm hprops]].
  destruct hprops as [hex [huniq [hmulfac [hnormmul [hprimegt [hprimepow [hfin hclass]]]]]]].
  assert (hI : exists F : IdealLike A -> nat, factorRel I F).
  { apply hex. }
  destruct hI as [F hF].
  exists F.
  exact hF.
Qed.

Lemma factorization_unique {A : Type} `{DomainLike A}
    (hD : IsDedekindLike (A := A)) (I : IdealLike A)
    (F G : IdealLike A -> nat)
    (hF : IdealFactorizationLike hD I F)
    (hG : IdealFactorizationLike hD I G) :
    F = G.
Proof.
  destruct hD as [factorRel [norm hprops]].
  destruct hprops as [hex [huniq [hmulfac [hnormmul [hprimegt [hprimepow [hfin hclass]]]]]]].
  assert (hEq : F = G).
  { apply (huniq I F G); assumption. }
  exact hEq.
Qed.

Lemma norm_mul {A : Type} `{DomainLike A}
    (hD : IsDedekindLike (A := A)) (I J : IdealLike A) :
    IdealNormLike hD (idealMul I J) = Nat.mul (IdealNormLike hD I) (IdealNormLike hD J).
Proof.
  destruct hD as [factorRel [norm hprops]].
  destruct hprops as [hex [huniq [hmulfac [hnormmul [hprimegt [hprimepow [hfin hclass]]]]]]].
  assert (hmul : norm (idealMul I J) = Nat.mul (norm I) (norm J)).
  { apply hnormmul. }
  exact hmul.
Qed.

Lemma norm_prime_power {A : Type} `{DomainLike A}
    (hD : IsDedekindLike (A := A))
    (P : IdealLike A) (hP : PrimeIdealLike P) (n : nat) :
    exists Ipow : IdealLike A, exists F : IdealLike A -> nat,
      IdealFactorizationLike hD Ipow F /\ F P = n /\
      IdealNormLike hD Ipow = Nat.pow (IdealNormLike hD P) n.
Proof.
  destruct hD as [factorRel [norm hprops]].
  destruct hprops as [hex [huniq [hmulfac [hnormmul [hprimegt [hprimepow [hfin hclass]]]]]]].
  assert (hpow :
    exists Ipow : IdealLike A, exists F : IdealLike A -> nat,
      factorRel Ipow F /\ F P = n /\ norm Ipow = Nat.pow (norm P) n).
  { apply (hprimepow P hP n). }
  destruct hpow as [Ipow [F [hFac [hExp hNormPow]]]].
  exists Ipow.
  exists F.
  split.
  - exact hFac.
  - split.
    + exact hExp.
    + exact hNormPow.
Qed.

Lemma finite_ideal_quotient_like {A : Type} `{DomainLike A}
    (hD : IsDedekindLike (A := A)) :
    exists B : nat, B > 0.
Proof.
  destruct hD as [factorRel [norm hprops]].
  destruct hprops as [hex [huniq [hmulfac [hnormmul [hprimegt [hprimepow [hfin hclass]]]]]]].
  destruct hfin as [B hB].
  exists B.
  exact hB.
Qed.

Lemma class_group_finite_like {A : Type} `{DomainLike A}
    (hD : IsDedekindLike (A := A)) :
    exists h : nat, h > 0.
Proof.
  assert (hfiniteQ : exists B : nat, B > 0).
  { apply (finite_ideal_quotient_like hD). }
  destruct hfiniteQ as [B hB].
  destruct hD as [factorRel [norm hprops]].
  destruct hprops as [hex [huniq [hmulfac [hnormmul [hprimegt [hprimepow [hfin hclass]]]]]]].
  destruct hclass as [h hh].
  assert (hh_pos : h > 0).
  { exact hh. }
  assert (hB_pos : B > 0).
  { exact hB. }
  exists h.
  exact hh_pos.
Qed.
