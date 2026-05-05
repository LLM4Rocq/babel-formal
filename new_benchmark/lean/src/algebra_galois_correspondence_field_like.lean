/-
BENCHMARK_ID: TINY_MATHLIB_BATCH04_ALGEBRA_GALOIS_CORRESPONDENCE_FIELD_LIKE
PAIR_STEM: algebra_galois_correspondence_field_like
MATH_DOMAIN: Field Theory
SOURCE_MATHLIB: Mathlib/FieldTheory/Galois/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class FieldLike (F : Type u) where
  add : F -> F -> F
  mul : F -> F -> F
  zero : F
  one : F

def FieldExtensionLike (F : Type u) [FieldLike F] : Type u :=
  F -> Prop

def IntermediateFieldLike (F : Type u) [FieldLike F] : Type u :=
  F -> Prop

def AutomorphismLike (F : Type u) [FieldLike F] : Type u :=
  F -> F

def FixedFieldLike (F : Type u) [FieldLike F]
    (G : AutomorphismLike F -> Prop) : IntermediateFieldLike F :=
  fun x => forall sigma : AutomorphismLike F, G sigma -> sigma x = x

def GaloisLike (F : Type u) [FieldLike F]
    (E : FieldExtensionLike F) (G : AutomorphismLike F -> Prop) : Prop :=
  (forall x : F, E x -> FixedFieldLike F G x) /\
    (forall x : F, FixedFieldLike F G x -> E x)

theorem fixedField_mono (F : Type u) [FieldLike F]
    (G H : AutomorphismLike F -> Prop)
    (hGH : forall sigma : AutomorphismLike F, G sigma -> H sigma) :
    forall x : F, FixedFieldLike F H x -> FixedFieldLike F G x := by
  intro x hx sigma hsigma
  have hsigmaH : H sigma := hGH sigma hsigma
  exact hx sigma hsigmaH

theorem closure_group_antitone (F : Type u) [FieldLike F]
    (E₁ E₂ : IntermediateFieldLike F)
    (hE : forall x : F, E₁ x -> E₂ x) :
    forall sigma : AutomorphismLike F,
      (forall x : F, E₂ x -> sigma x = x) ->
      (forall x : F, E₁ x -> sigma x = x) := by
  intro sigma hFix₂ x hx₁
  have hx₂ : E₂ x := hE x hx₁
  exact hFix₂ x hx₂

theorem gc_left_inverse_like (F : Type u) [FieldLike F]
    (E : IntermediateFieldLike F) :
    forall x : F, E x ->
      FixedFieldLike F (fun sigma : AutomorphismLike F => forall y : F, E y -> sigma y = y) x := by
  intro x hx sigma hsigma
  have hAct : sigma x = x := hsigma x hx
  exact hAct

theorem gc_right_inverse_like (F : Type u) [FieldLike F]
    (G : AutomorphismLike F -> Prop) :
    forall (sigma : AutomorphismLike F) (x : F),
      G sigma -> FixedFieldLike F G x -> sigma x = x := by
  intro sigma x hsigma hx
  exact hx sigma hsigma

theorem normal_subgroup_quotient_field (F : Type u) [FieldLike F]
    (G H : AutomorphismLike F -> Prop)
    (hNormal : forall sigma tau : AutomorphismLike F,
      G sigma -> H tau -> G (fun x => tau (sigma x)))
    (hFixH : forall tau : AutomorphismLike F, forall x : F, H tau -> tau x = x)
    {x : F} (hx : FixedFieldLike F G x) :
    forall sigma tau : AutomorphismLike F,
      G sigma -> H tau -> (fun y => tau (sigma y)) x = x := by
  intro sigma tau hsigma htau
  have hCompInG : G (fun y => tau (sigma y)) := hNormal sigma tau hsigma htau
  have hFixComp : (fun y => tau (sigma y)) x = x := hx (fun y => tau (sigma y)) hCompInG
  have hFixTau : tau x = x := hFixH tau x htau
  have hSigmaFromComp : tau (sigma x) = x := hFixComp
  have hTauAtSigma : tau (sigma x) = sigma x := hFixH tau (sigma x) htau
  have hSigmaEq : sigma x = x := by
    calc
      sigma x = tau (sigma x) := by
        exact Eq.symm hTauAtSigma
      _ = x := hSigmaFromComp
  have hTarget : tau (sigma x) = x := by
    calc
      tau (sigma x) = sigma x := hTauAtSigma
      _ = x := hSigmaEq
  exact hTarget

theorem galois_correspondence_theorem_like (F : Type u) [FieldLike F]
    (E : IntermediateFieldLike F)
    (G : AutomorphismLike F -> Prop)
    (hGal : GaloisLike F E G) :
    forall x : F, E x <-> FixedFieldLike F G x := by
  intro x
  rcases hGal with ⟨hToFixed, hToExt⟩
  constructor
  · intro hx
    exact hToFixed x hx
  · intro hx
    exact hToExt x hx
