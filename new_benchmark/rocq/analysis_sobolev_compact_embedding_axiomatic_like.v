(*
BENCHMARK_ID: TINY_MATHLIB_BATCH04_ANALYSIS_SOBOLEV_COMPACT_EMBEDDING_AXIOMATIC_LIKE
PAIR_STEM: analysis_sobolev_compact_embedding_axiomatic_like
MATH_DOMAIN: Analysis / PDE
SOURCE_MATHLIB: Mathlib/Analysis/NormedSpace/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class NormedSpaceLike (E : Type) := {
  norm : E -> nat
}.

Definition SobolevLike {E : Type} `{NormedSpaceLike E} (u : E) : Prop :=
  exists C : nat, norm u <= C.

Definition WeakConvergenceLike {E : Type} `{NormedSpaceLike E} (seq : nat -> E) (u : E) : Prop :=
  forall eps : nat, exists N : nat, forall n : nat, N <= n -> norm (seq n) <= norm u + eps.

Definition RelCompactLike {E : Type} `{NormedSpaceLike E} (A : (nat -> E) -> Prop) : Prop :=
  forall seq : nat -> E, A seq ->
    exists phi : nat -> nat, (forall n : nat, n <= phi n) /\ exists u : E, WeakConvergenceLike (fun n => seq (phi n)) u.

Definition EmbeddingLike {E : Type} `{NormedSpaceLike E} (T : E -> E) : Prop :=
  forall u : E, SobolevLike u -> SobolevLike (T u).

Definition BoundedSequenceLike {E : Type} `{NormedSpaceLike E} (seq : nat -> E) : Prop :=
  exists C : nat, forall n : nat, norm (seq n) <= C.

Lemma reflexive_subsequence_step {E : Type} `{NormedSpaceLike E}
    (A : (nat -> E) -> Prop)
    (hrel : RelCompactLike A)
    (seq : nat -> E)
    (hA : A seq) :
    exists phi : nat -> nat, exists u : E,
      (forall n : nat, n <= phi n) /\ WeakConvergenceLike (fun n => seq (phi n)) u.
Proof.
  assert (hextract :
    exists phi : nat -> nat,
      (forall n : nat, n <= phi n) /\ exists u : E, WeakConvergenceLike (fun n => seq (phi n)) u).
  { apply hrel. exact hA. }
  destruct hextract as [phi [hphi hrest]].
  destruct hrest as [u hweak].
  assert (hmono : forall n : nat, n <= phi n).
  { intro n. apply hphi. }
  exists phi.
  exists u.
  split.
  - exact hmono.
  - exact hweak.
Qed.

Lemma compactness_on_bounded_sets {E : Type} `{NormedSpaceLike E}
    (T : E -> E)
    (A : (nat -> E) -> Prop)
    (seq : nat -> E)
    (hemb : EmbeddingLike T)
    (hrel : RelCompactLike A)
    (hclosed : forall s : nat -> E, A s -> A (fun n => T (s n)))
    (hA : A seq)
    (hsob : forall n : nat, SobolevLike (seq n)) :
    exists phi : nat -> nat, exists u : E,
      (forall n : nat, n <= phi n) /\ WeakConvergenceLike (fun n => T (seq (phi n))) u.
Proof.
  assert (hAimage : A (fun n => T (seq n))).
  { apply hclosed. exact hA. }
  assert (hsobImage : forall n : nat, SobolevLike (T (seq n))).
  { intro n. apply hemb. apply hsob. }
  assert (hextract :
    exists phi : nat -> nat,
      (forall n : nat, n <= phi n) /\ exists u : E, WeakConvergenceLike (fun n => T (seq (phi n))) u).
  { exact (hrel (fun n => T (seq n)) hAimage). }
  destruct hextract as [phi [hphi hrest]].
  destruct hrest as [u hweak].
  assert (hs0 : SobolevLike (T (seq (phi 0)))).
  { apply hsobImage. }
  exists phi.
  exists u.
  split.
  - exact hphi.
  - exact hweak.
Qed.

Lemma rellich_step_like {E : Type} `{NormedSpaceLike E}
    (A : (nat -> E) -> Prop)
    (seq : nat -> E)
    (hbounded : BoundedSequenceLike seq)
    (hbridge : forall s : nat -> E, BoundedSequenceLike s -> A s)
    (hrel : RelCompactLike A) :
    exists phi : nat -> nat, exists u : E,
      (forall n : nat, n <= phi n) /\ WeakConvergenceLike (fun n => seq (phi n)) u.
Proof.
  assert (hA : A seq).
  { apply hbridge. exact hbounded. }
  assert (hstep :
    exists phi : nat -> nat, exists u : E,
      (forall n : nat, n <= phi n) /\ WeakConvergenceLike (fun n => seq (phi n)) u).
  {
    apply (reflexive_subsequence_step (A := A)).
    - exact hrel.
    - exact hA.
  }
  destruct hstep as [phi [u [hmono hweak]]].
  exists phi.
  exists u.
  split.
  - exact hmono.
  - exact hweak.
Qed.

Lemma compact_embedding_core {E : Type} `{NormedSpaceLike E}
    (T : E -> E)
    (A : (nat -> E) -> Prop)
    (seq : nat -> E)
    (hemb : EmbeddingLike T)
    (hbounded : BoundedSequenceLike seq)
    (hbridge : forall s : nat -> E, BoundedSequenceLike s -> A s)
    (hrel : RelCompactLike A)
    (hclosed : forall s : nat -> E, A s -> A (fun n => T (s n)))
    (hsob : forall n : nat, SobolevLike (seq n)) :
    exists phi : nat -> nat, exists u : E,
      (forall n : nat, n <= phi n) /\ WeakConvergenceLike (fun n => T (seq (phi n))) u.
Proof.
  assert (hA : A seq).
  { apply hbridge. exact hbounded. }
  assert (hcore :
    exists phi : nat -> nat, exists u : E,
      (forall n : nat, n <= phi n) /\ WeakConvergenceLike (fun n => T (seq (phi n))) u).
  {
    exact (compactness_on_bounded_sets (T := T) (A := A) seq hemb hrel hclosed hA hsob).
  }
  destruct hcore as [phi [u [hmono hweak]]].
  assert (hs0 : SobolevLike (T (seq (phi 0)))).
  { apply hemb. apply hsob. }
  exists phi.
  exists u.
  split.
  - exact hmono.
  - exact hweak.
Qed.

Lemma strong_convergence_extraction {E : Type} `{NormedSpaceLike E}
    (seq : nat -> E)
    (u : E)
    (hweak : WeakConvergenceLike seq u)
    (hupgrade : forall eps : nat, exists N : nat, forall n : nat, N <= n -> norm (seq n) <= norm u + eps) :
    exists N : nat, forall n : nat, N <= n -> norm (seq n) <= norm u + 1.
Proof.
  specialize (hweak 1) as hweakOne.
  destruct hweakOne as [Nw hNw].
  specialize (hupgrade 1) as hupOne.
  destruct hupOne as [Nu hNu].
  exists Nu.
  intros n hn.
  assert (hNuBound : norm (seq n) <= norm u + 1).
  { apply hNu. exact hn. }
  assert (hNwBound : norm (seq n) <= norm u + 1).
  { exact hNuBound. }
  exact hNuBound.
Qed.

Lemma sobolev_compact_embedding_like {E : Type} `{NormedSpaceLike E}
    (T : E -> E)
    (A : (nat -> E) -> Prop)
    (seq : nat -> E)
    (hemb : EmbeddingLike T)
    (hbounded : BoundedSequenceLike seq)
    (hbridge : forall s : nat -> E, BoundedSequenceLike s -> A s)
    (hrel : RelCompactLike A)
    (hclosed : forall s : nat -> E, A s -> A (fun n => T (s n)))
    (hsob : forall n : nat, SobolevLike (seq n))
    (hupgrade :
      forall phi : nat -> nat, forall u : E,
        WeakConvergenceLike (fun n => T (seq (phi n))) u ->
          forall eps : nat, exists N : nat, forall n : nat, N <= n ->
            norm (T (seq (phi n))) <= norm u + eps) :
    exists phi : nat -> nat, exists u : E, exists N : nat,
      (forall n : nat, n <= phi n) /\
      (forall n : nat, N <= n -> norm (T (seq (phi n))) <= norm u + 1).
Proof.
  assert (hcompact :
    exists phi : nat -> nat, exists u : E,
      (forall n : nat, n <= phi n) /\ WeakConvergenceLike (fun n => T (seq (phi n))) u).
  { exact (compact_embedding_core (T := T) (A := A) hemb hbounded hbridge hrel hclosed hsob). }
  destruct hcompact as [phi [u [hmono hweak]]].
  assert (hup : forall eps : nat, exists N : nat, forall n : nat, N <= n -> norm (T (seq (phi n))) <= norm u + eps).
  { apply (hupgrade phi u hweak). }
  assert (hstrong :
    exists N : nat, forall n : nat, N <= n -> norm (T (seq (phi n))) <= norm u + 1).
  { apply (strong_convergence_extraction (seq := fun n => T (seq (phi n))) (u := u) hweak hup). }
  destruct hstrong as [N hN].
  exists phi.
  exists u.
  exists N.
  split.
  - exact hmono.
  - exact hN.
Qed.
