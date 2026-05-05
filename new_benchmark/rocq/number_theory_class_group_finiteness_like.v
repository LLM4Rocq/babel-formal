(***
BENCHMARK_ID: TINY_MATHLIB_BATCH04_NUM_CLASS_GROUP_FINITENESS_LIKE
PAIR_STEM: number_theory_class_group_finiteness_like
MATH_DOMAIN: Algebraic Number Theory
SOURCE_MATHLIB: Mathlib/NumberTheory/ClassGroup
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
***)

Set Universe Polymorphism.
Set Implicit Arguments.

Class DedekindDomainLike (K : Type) := {
  Ideal : Type;
  classOf : Ideal -> nat;
  norm : Ideal -> nat;
  principal : Ideal -> Prop;
  inClass : nat -> Ideal -> Prop;
  reduced : nat -> Ideal -> Prop;
  reduced_witness : forall c : nat, exists I : Ideal, inClass c I /\ reduced c I;
  reduced_norm_bound : exists B : nat, forall c : nat, forall I : Ideal, reduced c I -> norm I <= B;
  class_of_inClass : forall c : nat, forall I : Ideal, inClass c I -> classOf I = c;
  principal_class_zero : forall I : Ideal, principal I -> classOf I = 0;
  class_has_torsion : exists n : nat, n > 0 /\ forall c : nat, n * c = 0 -> exists I : Ideal, inClass c I;
  finite_classes_from_norm : forall B : nat, exists N : nat,
    forall c : nat, (exists I : Ideal, inClass c I /\ norm I <= B) -> c < N
}.

Definition FractionalIdealLike (K : Type) `{DedekindDomainLike K} : Type :=
  Ideal.
Arguments FractionalIdealLike K {_}.

Definition PrincipalLike {K : Type} `{DedekindDomainLike K}
    (I : FractionalIdealLike K) : Prop :=
  principal I.

Definition ClassGroupLike (K : Type) `{DedekindDomainLike K} : Type :=
  nat.
Arguments ClassGroupLike K {_}.

Definition MinkowskiBoundLike {K : Type} `{DedekindDomainLike K}
    (B : nat) : Prop :=
  forall c : ClassGroupLike K, forall I : FractionalIdealLike K,
    reduced c I -> norm I <= B.

Definition ReducedIdealLike {K : Type} `{DedekindDomainLike K}
    (c : ClassGroupLike K) (I : FractionalIdealLike K) : Prop :=
  reduced c I /\ inClass c I.

Lemma reduced_ideal_exists {K : Type} `{DedekindDomainLike K}
    (c : ClassGroupLike K) :
    exists I : FractionalIdealLike K, ReducedIdealLike c I.
Proof.
  destruct (reduced_witness c) as [I [hIn hRed]].
  assert (hPack : ReducedIdealLike c I).
  {
    split.
    - exact hRed.
    - exact hIn.
  }
  exists I.
  exact hPack.
Qed.

Lemma reduced_ideal_finite_set {K : Type} `{DedekindDomainLike K} :
    exists B : nat, MinkowskiBoundLike (K := K) B.
Proof.
  destruct reduced_norm_bound as [B hB].
  exists B.
  intros c I hRed.
  assert (hnorm : norm I <= B).
  { apply (hB c I). exact hRed. }
  exact hnorm.
Qed.

Lemma every_class_has_reduced_rep {K : Type} `{DedekindDomainLike K}
    (c : ClassGroupLike K) :
    exists I : FractionalIdealLike K, ReducedIdealLike c I /\ classOf I = c.
Proof.
  destruct (reduced_ideal_exists (K := K) c) as [I hRedI].
  assert (hIn : inClass c I).
  { exact (proj2 hRedI). }
  assert (hClass : classOf I = c).
  { apply (class_of_inClass c I). exact hIn. }
  exists I.
  split.
  - exact hRedI.
  - exact hClass.
Qed.

Lemma class_group_generated_finitely {K : Type} `{DedekindDomainLike K} :
    exists B : nat,
      forall c : nat,
        exists I : FractionalIdealLike K,
          inClass c I /\ norm I <= B.
Proof.
  destruct (reduced_ideal_finite_set (K := K)) as [B hB].
  exists B.
  intro c.
  destruct (every_class_has_reduced_rep (K := K) c) as [I [hReducedI hClassI]].
  assert (hIn : inClass c I).
  { exact (proj2 hReducedI). }
  assert (hNorm : norm I <= B).
  { apply (hB c I). exact (proj1 hReducedI). }
  assert (hClass : classOf I = c).
  { exact hClassI. }
  exists I.
  split.
  - exact hIn.
  - exact hNorm.
Qed.

Lemma class_group_torsion_like {K : Type} `{DedekindDomainLike K} :
    exists n : nat,
      n > 0 /\
      forall c : nat,
        n * c = 0 -> exists I : FractionalIdealLike K, inClass c I.
Proof.
  destruct class_has_torsion as [n [hnPos hnKill]].
  exists n.
  split.
  - exact hnPos.
  - intros c hc.
    assert (hRep : exists I : FractionalIdealLike K, inClass c I).
    { apply (hnKill c hc). }
    destruct hRep as [I hI].
    exists I.
    exact hI.
Qed.

Lemma class_group_finite_like {K : Type} `{DedekindDomainLike K} :
    exists N : nat, forall c : nat, c < N.
Proof.
  destruct (class_group_generated_finitely (K := K)) as [B hGen].
  destruct (finite_classes_from_norm B) as [N hN].
  exists N.
  intro c.
  destruct (hGen c) as [I [hIn hNorm]].
  assert (hWitness : exists J : FractionalIdealLike K, inClass c J /\ norm J <= B).
  { exists I. split; assumption. }
  assert (hBound : c < N).
  { apply (hN c). exact hWitness. }
  exact hBound.
Qed.
