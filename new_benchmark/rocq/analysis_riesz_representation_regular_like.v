(*
BENCHMARK_ID: TINY_MATHLIB_BATCH04_ANALYSIS_RIESZ_REPRESENTATION_REGULAR_LIKE
PAIR_STEM: analysis_riesz_representation_regular_like
MATH_DOMAIN: Functional Analysis / Measure Theory
SOURCE_MATHLIB: Mathlib/MeasureTheory/Integral/RieszMarkov
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class LocCompactSpaceLike (X : Type) := {
  empty : X -> Prop;
  univ : X -> Prop;
  compact : (X -> Prop) -> Prop;
  openSet : (X -> Prop) -> Prop;
  integral : ((X -> Prop) -> nat) -> (X -> nat) -> nat
}.

Definition ContinuousCompactSupportLike {X : Type} `{LocCompactSpaceLike X}
    (f : X -> nat) : Prop :=
  exists K : X -> Prop,
    compact K /\
      (forall x : X, f x = 0 \/ K x).

Definition PositiveFunctionalLike {X : Type} `{LocCompactSpaceLike X}
    (L : (X -> nat) -> nat) : Prop :=
  (forall f : X -> nat, 0 <= L f) /\
    (forall f g : X -> nat, L (fun x => f x + g x) = L f + L g).

Definition RadonMeasureLike {X : Type} `{LocCompactSpaceLike X}
    (mu : (X -> Prop) -> nat) : Prop :=
  (forall K : X -> Prop,
    compact K ->
      mu K <= mu univ) /\
    mu empty = 0.

Definition RepresentationLike {X : Type} `{LocCompactSpaceLike X}
    (L : (X -> nat) -> nat) (mu : (X -> Prop) -> nat) : Prop :=
  forall f : X -> nat,
    ContinuousCompactSupportLike f ->
      L f = integral mu f.

Definition RegularLike {X : Type} `{LocCompactSpaceLike X}
    (mu : (X -> Prop) -> nat) : Prop :=
  (forall U : X -> Prop,
    openSet U ->
      exists K : X -> Prop, compact K /\ mu K <= mu U) /\
    (forall K : X -> Prop,
      compact K ->
        exists U : X -> Prop, openSet U /\ mu K <= mu U).

Lemma representation_exists_like {X : Type} `{LocCompactSpaceLike X}
    (L : (X -> nat) -> nat)
    (hpos : PositiveFunctionalLike L)
    (hex : exists mu : (X -> Prop) -> nat,
      RadonMeasureLike mu /\ RepresentationLike L mu /\ RegularLike mu) :
    exists mu : (X -> Prop) -> nat,
      RadonMeasureLike mu /\ RepresentationLike L mu /\ RegularLike mu.
Proof.
  destruct hex as [mu [hmuRadon [hmuRepr hmuReg]]].
  assert (hpos0 : 0 <= L (fun _ : X => 0)).
  { destruct hpos as [hmono hadd]. apply hmono. }
  assert (hcheck : RepresentationLike L mu).
  { exact hmuRepr. }
  exists mu.
  split.
  - exact hmuRadon.
  - split.
    + exact hmuRepr.
    + exact hmuReg.
Qed.

Lemma representation_unique_like {X : Type} `{LocCompactSpaceLike X}
    (L : (X -> nat) -> nat)
    (mu nu : (X -> Prop) -> nat)
    (hmu : RepresentationLike L mu)
    (hnu : RepresentationLike L nu)
    (hall : forall f : X -> nat, ContinuousCompactSupportLike f)
    (hsep : forall mu1 nu1 : (X -> Prop) -> nat,
      (forall f : X -> nat,
        integral mu1 f = integral nu1 f) ->
      mu1 = nu1) :
    mu = nu.
Proof.
  apply (hsep mu nu).
  intro f.
  assert (hcf : ContinuousCompactSupportLike f).
  { apply hall. }
  assert (hmuf : L f = integral mu f).
  { apply hmu. exact hcf. }
  assert (hnuf : L f = integral nu f).
  { apply hnu. exact hcf. }
  rewrite <- hmuf.
  exact hnuf.
Qed.

Lemma positivity_transfer_like {X : Type} `{LocCompactSpaceLike X}
    (L : (X -> nat) -> nat)
    (mu : (X -> Prop) -> nat)
    (hrepr : RepresentationLike L mu)
    (hmono : forall mu1 : (X -> Prop) -> nat, forall f : X -> nat,
      0 <= integral mu1 f)
    (hadd : forall mu1 : (X -> Prop) -> nat, forall f g : X -> nat,
      integral mu1 (fun x => f x + g x) =
        integral mu1 f + integral mu1 g)
    (hall : forall f : X -> nat, ContinuousCompactSupportLike f) :
    PositiveFunctionalLike L.
Proof.
  split.
  - intro f.
    assert (hcf : ContinuousCompactSupportLike f).
    { apply hall. }
    assert (hreprf : L f = integral mu f).
    { apply hrepr. exact hcf. }
    rewrite hreprf.
    apply hmono.
  - intros f g.
    assert (hcf : ContinuousCompactSupportLike f).
    { apply hall. }
    assert (hcg : ContinuousCompactSupportLike g).
    { apply hall. }
    assert (hsum : ContinuousCompactSupportLike (fun x => f x + g x)).
    { apply hall. }
    assert (hrepr_sum : L (fun x => f x + g x) = integral mu (fun x => f x + g x)).
    { apply hrepr. exact hsum. }
    assert (hrepr_f : L f = integral mu f).
    { apply hrepr. exact hcf. }
    assert (hrepr_g : L g = integral mu g).
    { apply hrepr. exact hcg. }
    rewrite hrepr_sum.
    rewrite (hadd mu f g).
    rewrite <- hrepr_f.
    rewrite <- hrepr_g.
    reflexivity.
Qed.

Lemma regularity_inner_like {X : Type} `{LocCompactSpaceLike X}
    (mu : (X -> Prop) -> nat)
    (hreg : RegularLike mu)
    (U : X -> Prop)
    (hU : openSet U) :
    exists K : X -> Prop, compact K /\ mu K <= mu U.
Proof.
  destruct hreg as [hinner houter].
  specialize (hinner U hU).
  destruct hinner as [K [hKc hKle]].
  exists K.
  split.
  - exact hKc.
  - exact hKle.
Qed.

Lemma regularity_outer_like {X : Type} `{LocCompactSpaceLike X}
    (mu : (X -> Prop) -> nat)
    (hreg : RegularLike mu)
    (K : X -> Prop)
    (hK : compact K) :
    exists U : X -> Prop, openSet U /\ mu K <= mu U.
Proof.
  destruct hreg as [hinner houter].
  specialize (houter K hK).
  destruct houter as [U [hUo hKle]].
  exists U.
  split.
  - exact hUo.
  - exact hKle.
Qed.

Lemma riesz_markov_theorem_like {X : Type} `{LocCompactSpaceLike X}
    (L : (X -> nat) -> nat)
    (hpos : PositiveFunctionalLike L)
    (hex : exists mu : (X -> Prop) -> nat,
      RadonMeasureLike mu /\ RepresentationLike L mu /\ RegularLike mu)
    (hall : forall f : X -> nat, ContinuousCompactSupportLike f)
    (hsep : forall mu1 nu1 : (X -> Prop) -> nat,
      (forall f : X -> nat,
        integral mu1 f = integral nu1 f) ->
      mu1 = nu1)
    (hmono : forall mu1 : (X -> Prop) -> nat, forall f : X -> nat,
      0 <= integral mu1 f)
    (hadd : forall mu1 : (X -> Prop) -> nat, forall f g : X -> nat,
      integral mu1 (fun x => f x + g x) =
        integral mu1 f + integral mu1 g) :
    exists mu : (X -> Prop) -> nat,
      RepresentationLike L mu /\
      RegularLike mu /\
      (forall nu : (X -> Prop) -> nat,
        RepresentationLike L nu -> RegularLike nu -> nu = mu) /\
      PositiveFunctionalLike L.
Proof.
  destruct (representation_exists_like (L := L) hpos hex) as [mu [hmuRadon [hmuRepr hmuReg]]].
  assert (hposL : PositiveFunctionalLike L).
  { exact (positivity_transfer_like (L := L) (mu := mu) hmuRepr hmono hadd hall). }
  exists mu.
  split.
  - exact hmuRepr.
  - split.
    + exact hmuReg.
    + split.
      * intros nu hnuRepr hnuReg.
        assert (hkeep : RegularLike nu).
        { exact hnuReg. }
        assert (huniq : mu = nu).
        { exact (representation_unique_like (L := L) (mu := mu) (nu := nu) hmuRepr hnuRepr hall hsep). }
        symmetry.
        exact huniq.
      * exact hposL.
Qed.
