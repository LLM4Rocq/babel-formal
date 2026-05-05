(*
BENCHMARK_ID: TINY_MATHLIB_BATCH05_LOGIC_HOMOTOPY_TYPE_UNIVALENCE_AXIOMATIC_LIKE
PAIR_STEM: logic_homotopy_type_univalence_axiomatic_like
MATH_DOMAIN: Logic
SOURCE_MATHLIB: Mathlib/Logic/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class LogicStruct_homotopy_type_univalence (Ty : Type) := {
  equiv : Ty -> Ty -> Prop;
  transport : (Ty -> Prop) -> Ty -> Ty -> Prop;
  rel : Ty -> Ty -> Prop;
  phase : Ty -> Ty -> Prop;
  game : Ty -> Ty -> Prop;
  univalence_axiom :
    forall A B : Ty,
      equiv A B ->
        transport (fun X : Ty => X = X) A B;
  parametricity_axiom :
    forall A B : Ty,
      rel A B -> equiv A B;
  phase_sound_axiom :
    forall A B : Ty,
      phase A B -> rel A B;
  full_abstraction_axiom :
    forall A B : Ty,
      game A B -> phase A B;
  bisimulation_axiom :
    forall A B : Ty,
      game A B -> game B A;
  normalization_axiom :
    forall A : Ty,
      game A A;
  adequacy_axiom :
    forall A B : Ty,
      equiv A B -> game A B
}.

Definition TypeObj_homotopy_type_univalence
    (Ty : Type) : Type :=
  Ty.

Definition RelObj_homotopy_type_univalence
    (Ty : Type) : Type :=
  Ty -> Ty -> Prop.

Definition PhaseObj_homotopy_type_univalence
    (Ty : Type) : Type :=
  Ty -> Ty -> Prop.

Definition GameObj_homotopy_type_univalence
    (Ty : Type) : Type :=
  Ty -> Ty -> Prop.

Lemma univalence_transport_homotopy_type_univalence
    {Ty : Type} `{LogicStruct_homotopy_type_univalence Ty}
    (A B : Ty)
    (hEqv : equiv A B) :
    transport (fun X : Ty => X = X) A B /\ (equiv A B -> game A B).
Proof.
  assert (hTr : transport (fun X : Ty => X = X) A B).
  { apply (univalence_axiom A B). exact hEqv. }
  split.
  - exact hTr.
  - intro hEqvAgain.
    exact (adequacy_axiom A B hEqvAgain).
Qed.

Lemma parametricity_free_theorem_homotopy_type_univalence
    {Ty : Type} `{LogicStruct_homotopy_type_univalence Ty}
    (A B : Ty)
    (hRel : rel A B) :
    exists p : phase A B, transport (fun X : Ty => X = X) A B /\ rel A B.
Proof.
  assert (hEqv : equiv A B).
  { apply (parametricity_axiom A B). exact hRel. }
  assert (hGame : game A B).
  { apply (adequacy_axiom A B). exact hEqv. }
  assert (hPhase : phase A B).
  { apply (full_abstraction_axiom A B). exact hGame. }
  assert (hTr : transport (fun X : Ty => X = X) A B).
  { apply (univalence_axiom A B). exact hEqv. }
  exists hPhase.
  split.
  - exact hTr.
  - exact hRel.
Qed.

Lemma phase_soundness_homotopy_type_univalence
    {Ty : Type} `{LogicStruct_homotopy_type_univalence Ty}
    (A B : Ty)
    (hPhase : phase A B) :
    (rel A B -> equiv A B) /\ rel A B.
Proof.
  assert (hRel : rel A B).
  { apply (phase_sound_axiom A B). exact hPhase. }
  split.
  - intro hRelInput.
    exact (parametricity_axiom A B hRelInput).
  - exact hRel.
Qed.

Lemma full_abstraction_step_homotopy_type_univalence
    {Ty : Type} `{LogicStruct_homotopy_type_univalence Ty}
    (A B : Ty)
    (hGame : game A B) :
    exists e : equiv A B, phase A B /\ rel A B.
Proof.
  assert (hPhase : phase A B).
  { apply (full_abstraction_axiom A B). exact hGame. }
  assert (hRel : rel A B).
  { apply (phase_sound_axiom A B). exact hPhase. }
  assert (hEqv : equiv A B).
  { apply (parametricity_axiom A B). exact hRel. }
  exists hEqv.
  split.
  - exact hPhase.
  - exact hRel.
Qed.

Lemma bisimulation_up_to_step_homotopy_type_univalence
    {Ty : Type} `{LogicStruct_homotopy_type_univalence Ty}
    (A B : Ty)
    (hGame : game A B) :
    game B A \/ (phase B A /\ rel B A).
Proof.
  assert (hBack : game B A).
  { apply (bisimulation_axiom A B). exact hGame. }
  assert (hBackPhase : phase B A).
  { apply (full_abstraction_axiom B A). exact hBack. }
  assert (hBackRel : rel B A).
  { apply (phase_sound_axiom B A). exact hBackPhase. }
  right.
  split.
  - exact hBackPhase.
  - exact hBackRel.
Qed.

Lemma normalization_bridge_homotopy_type_univalence
    {Ty : Type} `{LogicStruct_homotopy_type_univalence Ty}
    (A : Ty) :
    exists p : phase A A, transport (fun X : Ty => X = X) A A /\ game A A.
Proof.
  assert (hGame : game A A).
  { apply normalization_axiom. }
  assert (hPhase : phase A A).
  { apply (full_abstraction_axiom A A). exact hGame. }
  assert (hRel : rel A A).
  { apply (phase_sound_axiom A A). exact hPhase. }
  assert (hEqv : equiv A A).
  { apply (parametricity_axiom A A). exact hRel. }
  assert (hTr : transport (fun X : Ty => X = X) A A).
  { apply (univalence_axiom A A). exact hEqv. }
  exists hPhase.
  split.
  - exact hTr.
  - exact hGame.
Qed.

Lemma adequacy_closure_homotopy_type_univalence
    {Ty : Type} `{LogicStruct_homotopy_type_univalence Ty}
    (A B : Ty)
    (hEqv : equiv A B) :
    (game A B /\ game B A) /\ (rel A B -> equiv B A).
Proof.
  assert (hForward : game A B).
  { apply (adequacy_axiom A B). exact hEqv. }
  assert (hBackward : game B A).
  { apply (bisimulation_axiom A B). exact hForward. }
  assert (hPhaseAB : phase A B).
  { apply (full_abstraction_axiom A B). exact hForward. }
  assert (hRelAB : rel A B).
  { apply (phase_sound_axiom A B). exact hPhaseAB. }
  split.
  - split.
    + exact hForward.
    + exact hBackward.
  - intro hRelIn.
    assert (hGameFromRel : game A B).
    {
      apply (adequacy_axiom A B).
      apply (parametricity_axiom A B).
      exact hRelIn.
    }
    assert (hGameBack : game B A).
    { apply (bisimulation_axiom A B). exact hGameFromRel. }
    assert (hPhaseBack : phase B A).
    { apply (full_abstraction_axiom B A). exact hGameBack. }
    assert (hRelBack : rel B A).
    { apply (phase_sound_axiom B A). exact hPhaseBack. }
    assert (hKeep : rel A B).
    { exact hRelAB. }
    apply (parametricity_axiom B A).
    exact hRelBack.
Qed.
