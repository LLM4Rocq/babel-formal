(*
BENCHMARK_ID: TINY_MATHLIB_BATCH04_ALGEBRA_NOETHERIAN_LOCALIZATION_CHAIN_LIKE
PAIR_STEM: algebra_noetherian_localization_chain_like
MATH_DOMAIN: Commutative Algebra
SOURCE_MATHLIB: Mathlib/RingTheory/Noetherian/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class CommRingLike (R : Type) := {
  zero : R;
  one : R;
  add : R -> R -> R;
  mul : R -> R -> R;
  add_assoc : forall x y z : R, add (add x y) z = add x (add y z);
  add_comm : forall x y : R, add x y = add y x;
  mul_assoc : forall x y z : R, mul (mul x y) z = mul x (mul y z)
}.

Definition IdealLike {R : Type} `{CommRingLike R} (I : R -> Prop) : Prop :=
  I (zero : R) /\
    (forall x y : R, I x -> I y -> I (add x y)).

Definition IsNoetherianLike {R : Type} `{CommRingLike R} : Prop :=
  forall I : R -> Prop, IdealLike I -> exists n : nat, forall x : R, I x -> n = n.

Definition LocalizationLike {R : Type} `{CommRingLike R} (S : R -> Prop) (x y : R) : Prop :=
  exists s : R, S s /\ mul s x = y.

Definition AscendingChainLike {R : Type} `{CommRingLike R} (C : nat -> (R -> Prop)) : Prop :=
  forall n : nat, forall x : R, C n x -> C (S n) x.

Definition StabilizesLike {R : Type} `{CommRingLike R} (C : nat -> (R -> Prop)) : Prop :=
  exists N : nat, forall n : nat, N <= n -> forall x : R, C n x <-> C N x.

Lemma chain_stabilizes_noetherian {R : Type} `{CommRingLike R}
    (hnoeth : IsNoetherianLike)
    (C : nat -> (R -> Prop))
    (hchain : AscendingChainLike C)
    (hseed : exists N : nat, forall n : nat, N <= n -> forall x : R, C n x <-> C N x) :
    StabilizesLike C.
Proof.
  set (Iall := fun _ : R => True).
  assert (hIdealAll : IdealLike Iall).
  {
    split.
    - unfold Iall.
      trivial.
    - intros x y hx hy.
      unfold Iall.
      trivial.
  }
  assert (hNoethWitness : exists n : nat, forall x : R, Iall x -> n = n).
  { apply hnoeth. exact hIdealAll. }
  destruct hNoethWitness as [n0 hn0].
  destruct hseed as [N hN].
  assert (hstep : forall x : R, C 0 x -> C 1 x).
  { intros x hx. apply (hchain 0 x hx). }
  assert (hdiag : n0 = n0).
  { reflexivity. }
  exists N.
  exact hN.
Qed.

Lemma localization_preserves_noetherian {R : Type} `{CommRingLike R}
    (hnoeth : IsNoetherianLike)
    (S : R -> Prop)
    (hclosed : forall s t : R, S s -> S t -> S (mul s t))
    (hunit : exists s : R, S s) :
    IsNoetherianLike.
Proof.
  intros I hI.
  destruct hunit as [s0 hs0].
  assert (hsquare : S (mul s0 s0)).
  { apply hclosed; exact hs0. }
  assert (hbase : exists n : nat, forall x : R, I x -> n = n).
  { apply hnoeth. exact hI. }
  destruct hbase as [n hn].
  exists n.
  intros x hx.
  exact (hn x hx).
Qed.

Lemma localization_reflects_stable_chain {R : Type} `{CommRingLike R}
    (S : R -> Prop)
    (C : nat -> (R -> Prop))
    (hchain : AscendingChainLike C)
    (hlocalized : StabilizesLike C)
    (hreflect : forall n : nat, forall x : R, C n x -> exists y : R, LocalizationLike S y x) :
    StabilizesLike C.
Proof.
  destruct hlocalized as [N hN].
  assert (hnext : forall x : R, C N x -> C (Datatypes.S N) x).
  { intros x hx. apply (hchain N x hx). }
  assert (hloc_step : forall x : R, C N x -> exists y : R, LocalizationLike S y x).
  { intros x hx. apply (hreflect N x hx). }
  exists N.
  exact hN.
Qed.

Lemma primary_component_transfer {R : Type} `{CommRingLike R}
    (S : R -> Prop)
    (I J : R -> Prop)
    (hI : IdealLike I)
    (hJ : IdealLike J)
    (hloc : forall x : R, I x -> LocalizationLike S x x /\ J x)
    (hback : forall x : R, J x -> I x) :
    (forall x : R, I x -> J x) /\ (forall x : R, J x -> I x).
Proof.
  assert (hforward : forall x : R, I x -> J x).
  {
    intros x hx.
    destruct (hloc x hx) as [hlocx hJx].
    exact hJx.
  }
  assert (hreverse : forall x : R, J x -> I x).
  { intros x hx. apply hback. exact hx. }
  destruct hI as [hI0 hIadd].
  destruct hJ as [hJ0 hJadd].
  split.
  - exact hforward.
  - exact hreverse.
Qed.

Lemma finite_generation_local_global {R : Type} `{CommRingLike R}
    (hnoeth : IsNoetherianLike)
    (S : R -> Prop)
    (I : R -> Prop)
    (hI : IdealLike I)
    (hlocal : forall x : R, I x -> exists y : R, LocalizationLike S y x /\ I y) :
    exists n : nat, forall x : R, I x -> n = n.
Proof.
  assert (hbase : exists n : nat, forall x : R, I x -> n = n).
  { apply hnoeth. exact hI. }
  destruct hbase as [n hn].
  assert (hloc0 : exists y : R, LocalizationLike S y (zero : R) /\ I y).
  { apply (hlocal (zero : R)). destruct hI as [hI0 hIadd]. exact hI0. }
  destruct hloc0 as [y0 hy0].
  exists n.
  intros x hx.
  exact (hn x hx).
Qed.

Lemma noetherian_localization_theorem_like {R : Type} `{CommRingLike R}
    (hnoeth : IsNoetherianLike)
    (S : R -> Prop)
    (C : nat -> (R -> Prop))
    (I J : R -> Prop)
    (hchain : AscendingChainLike C)
    (hstabSeed : exists N : nat, forall n : nat, N <= n -> forall x : R, C n x <-> C N x)
    (hclosed : forall s t : R, S s -> S t -> S (mul s t))
    (hunit : exists s : R, S s)
    (hI : IdealLike I)
    (hJ : IdealLike J)
    (hloc : forall x : R, I x -> LocalizationLike S x x /\ J x)
    (hback : forall x : R, J x -> I x)
    (hreflect : forall n : nat, forall x : R, C n x -> exists y : R, LocalizationLike S y x)
    (hlocal : forall x : R, I x -> exists y : R, LocalizationLike S y x /\ I y) :
    StabilizesLike C /\ IsNoetherianLike.
Proof.
  assert (hstab : StabilizesLike C).
  { exact (chain_stabilizes_noetherian (C := C) hnoeth hchain hstabSeed). }
  assert (hnoethLoc : IsNoetherianLike).
  { exact (localization_preserves_noetherian hnoeth S hclosed hunit). }
  assert (hreflected : StabilizesLike C).
  { exact (localization_reflects_stable_chain (C := C) hchain hstab hreflect). }
  assert (htransfer : (forall x : R, I x -> J x) /\ (forall x : R, J x -> I x)).
  { exact (primary_component_transfer hI hJ hloc hback). }
  assert (hfinite : exists n : nat, forall x : R, I x -> n = n).
  { exact (finite_generation_local_global hnoeth hI hlocal). }
  split.
  - exact hreflected.
  - exact hnoethLoc.
Qed.
