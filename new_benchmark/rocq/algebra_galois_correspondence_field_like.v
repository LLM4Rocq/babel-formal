(*
BENCHMARK_ID: TINY_MATHLIB_BATCH04_ALGEBRA_GALOIS_CORRESPONDENCE_FIELD_LIKE
PAIR_STEM: algebra_galois_correspondence_field_like
MATH_DOMAIN: Field Theory
SOURCE_MATHLIB: Mathlib/FieldTheory/Galois/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class FieldLike (F : Type) := {
  add : F -> F -> F;
  mul : F -> F -> F;
  zero : F;
  one : F
}.

Definition FieldExtensionLike (F : Type) {K : FieldLike F} : Type :=
  F -> Prop.

Definition IntermediateFieldLike (F : Type) {K : FieldLike F} : Type :=
  F -> Prop.

Definition AutomorphismLike (F : Type) {K : FieldLike F} : Type :=
  F -> F.

Arguments FieldExtensionLike F {K}.
Arguments IntermediateFieldLike F {K}.
Arguments AutomorphismLike F {K}.

Definition FixedFieldLike (F : Type) {K : FieldLike F}
    (G : AutomorphismLike F -> Prop) : IntermediateFieldLike F :=
  fun x => forall sigma : AutomorphismLike F, G sigma -> sigma x = x.

Arguments FixedFieldLike F {K} G x.

Definition GaloisLike (F : Type) {K : FieldLike F}
    (E : FieldExtensionLike F) (G : AutomorphismLike F -> Prop) : Prop :=
  (forall x : F, E x -> FixedFieldLike F G x) /\
    (forall x : F, FixedFieldLike F G x -> E x).

Arguments GaloisLike F {K} E G.

Lemma fixedField_mono (F : Type) {K : FieldLike F}
    (G H : AutomorphismLike F -> Prop)
    (hGH : forall sigma : AutomorphismLike F, G sigma -> H sigma) :
    forall x : F, FixedFieldLike F H x -> FixedFieldLike F G x.
Proof.
  intros x hx sigma hsigma.
  assert (hsigmaH : H sigma).
  { exact (hGH sigma hsigma). }
  exact (hx sigma hsigmaH).
Qed.

Lemma closure_group_antitone (F : Type) {K : FieldLike F}
    (E1 E2 : IntermediateFieldLike F)
    (hE : forall x : F, E1 x -> E2 x) :
    forall sigma : AutomorphismLike F,
      (forall x : F, E2 x -> sigma x = x) ->
      (forall x : F, E1 x -> sigma x = x).
Proof.
  intros sigma hFix2 x hx1.
  assert (hx2 : E2 x).
  { exact (hE x hx1). }
  exact (hFix2 x hx2).
Qed.

Lemma gc_left_inverse_like (F : Type) {K : FieldLike F}
    (E : IntermediateFieldLike F) :
    forall x : F, E x ->
      FixedFieldLike F (fun sigma : AutomorphismLike F => forall y : F, E y -> sigma y = y) x.
Proof.
  intros x hx sigma hsigma.
  assert (hAct : sigma x = x).
  { exact (hsigma x hx). }
  exact hAct.
Qed.

Lemma gc_right_inverse_like (F : Type) {K : FieldLike F}
    (G : AutomorphismLike F -> Prop) :
    forall (sigma : AutomorphismLike F) (x : F),
      G sigma -> FixedFieldLike F G x -> sigma x = x.
Proof.
  intros sigma x hsigma hx.
  exact (hx sigma hsigma).
Qed.

Lemma normal_subgroup_quotient_field (F : Type) {K : FieldLike F}
    (G H : AutomorphismLike F -> Prop)
    (hNormal : forall sigma tau : AutomorphismLike F,
      G sigma -> H tau -> G (fun x => tau (sigma x)))
    (hFixH : forall tau : AutomorphismLike F, forall x : F, H tau -> tau x = x)
    {x : F} (hx : FixedFieldLike F G x) :
    forall sigma tau : AutomorphismLike F,
      G sigma -> H tau -> (fun y => tau (sigma y)) x = x.
Proof.
  intros sigma tau hsigma htau.
  assert (hCompInG : G (fun y => tau (sigma y))).
  { exact (hNormal sigma tau hsigma htau). }
  assert (hFixComp : (fun y => tau (sigma y)) x = x).
  { exact (hx (fun y => tau (sigma y)) hCompInG). }
  assert (hFixTau : tau x = x).
  { exact (hFixH tau x htau). }
  assert (hTauAtSigma : tau (sigma x) = sigma x).
  { exact (hFixH tau (sigma x) htau). }
  assert (hSigmaEq : sigma x = x).
  {
    rewrite <- hTauAtSigma.
    exact hFixComp.
  }
  assert (hTarget : tau (sigma x) = x).
  {
    rewrite hTauAtSigma.
    exact hSigmaEq.
  }
  exact hTarget.
Qed.

Lemma galois_correspondence_theorem_like (F : Type) {K : FieldLike F}
    (E : IntermediateFieldLike F)
    (G : AutomorphismLike F -> Prop)
    (hGal : GaloisLike F E G) :
    forall x : F, E x <-> FixedFieldLike F G x.
Proof.
  intro x.
  destruct hGal as [hToFixed hToExt].
  split.
  - intro hx.
    exact (hToFixed x hx).
  - intro hx.
    exact (hToExt x hx).
Qed.
