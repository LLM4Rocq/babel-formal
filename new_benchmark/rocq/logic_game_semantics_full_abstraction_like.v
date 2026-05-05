(*
BENCHMARK_ID: TINY_MATHLIB_BATCH05_LOGIC_GAME_SEMANTICS_FULL_ABSTRACTION_LIKE
PAIR_STEM: logic_game_semantics_full_abstraction_like
MATH_DOMAIN: Logic
SOURCE_MATHLIB: Mathlib/Logic/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class LogicStruct_game_semantics_full (Ty : Type) := {
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
  phase_complete_axiom :
    forall A B : Ty,
      rel A B -> phase A B;
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

Definition TypeObj_game_semantics_full (Ty : Type) : Type :=
  Ty.

Definition RelObj_game_semantics_full (Ty : Type) : Type :=
  Ty -> Ty -> Prop.

Definition PhaseObj_game_semantics_full (Ty : Type) : Type :=
  Ty -> Ty -> Prop.

Definition GameObj_game_semantics_full (Ty : Type) : Type :=
  Ty -> Ty -> Prop.

Lemma univalence_transport_game_semantics_full
    {Ty : Type} `{LogicStruct_game_semantics_full Ty}
    (A B : Ty)
    (hEqv : equiv A B) :
    transport (fun X : Ty => X = X) A B.
Proof.
  assert (hGame : game A B).
  { apply (adequacy_axiom A B). exact hEqv. }
  assert (hPhase : phase A B).
  { apply (full_abstraction_axiom A B). exact hGame. }
  assert (hRel : rel A B).
  { apply (phase_sound_axiom A B). exact hPhase. }
  assert (hEqvAgain : equiv A B).
  { apply (parametricity_axiom A B). exact hRel. }
  exact (univalence_axiom A B hEqvAgain).
Qed.

Lemma parametricity_free_theorem_game_semantics_full
    {Ty : Type} `{LogicStruct_game_semantics_full Ty}
    (A B : Ty)
    (hRel : rel A B) :
    exists q : phase A B, game A B \/ rel A B.
Proof.
  assert (hPhase : phase A B).
  { apply (phase_complete_axiom A B). exact hRel. }
  assert (hEqv : equiv A B).
  { apply (parametricity_axiom A B). exact hRel. }
  assert (hGame : game A B).
  { apply (adequacy_axiom A B). exact hEqv. }
  exists hPhase.
  left.
  exact hGame.
Qed.

Lemma phase_soundness_game_semantics_full
    {Ty : Type} `{LogicStruct_game_semantics_full Ty}
    (A B : Ty)
    (hPhase : phase A B) :
    rel A B /\ (phase A B <-> phase A B).
Proof.
  assert (hRel : rel A B).
  { apply (phase_sound_axiom A B). exact hPhase. }
  split.
  - exact hRel.
  - split; intro hp; exact hp.
Qed.

Lemma full_abstraction_step_game_semantics_full
    {Ty : Type} `{LogicStruct_game_semantics_full Ty}
    (A B : Ty)
    (hGame : game A B) :
    transport (fun X : Ty => X = X) A B /\ equiv A B /\ phase A B.
Proof.
  assert (hPhase : phase A B).
  { apply (full_abstraction_axiom A B). exact hGame. }
  assert (hRel : rel A B).
  { apply (phase_sound_axiom A B). exact hPhase. }
  assert (hEqv : equiv A B).
  { apply (parametricity_axiom A B). exact hRel. }
  assert (hTr : transport (fun X : Ty => X = X) A B).
  { apply (univalence_axiom A B). exact hEqv. }
  split.
  - exact hTr.
  - split.
    + exact hEqv.
    + exact hPhase.
Qed.

Lemma bisimulation_up_to_step_game_semantics_full
    {Ty : Type} `{LogicStruct_game_semantics_full Ty}
    (A B : Ty)
    (hGame : game A B) :
    game B A /\ (phase B A -> rel B A).
Proof.
  assert (hBack : game B A).
  { apply (bisimulation_axiom A B). exact hGame. }
  split.
  - exact hBack.
  - intro hPhaseBack.
    exact (phase_sound_axiom B A hPhaseBack).
Qed.

Lemma normalization_bridge_game_semantics_full
    {Ty : Type} `{LogicStruct_game_semantics_full Ty}
    (A : Ty) :
    phase A A /\ (rel A A /\ transport (fun X : Ty => X = X) A A).
Proof.
  assert (hGameAA : game A A).
  { apply normalization_axiom. }
  assert (hPhaseAA : phase A A).
  { apply (full_abstraction_axiom A A). exact hGameAA. }
  assert (hRelAA : rel A A).
  { apply (phase_sound_axiom A A). exact hPhaseAA. }
  assert (hEqvAA : equiv A A).
  { apply (parametricity_axiom A A). exact hRelAA. }
  assert (hTrAA : transport (fun X : Ty => X = X) A A).
  { apply (univalence_axiom A A). exact hEqvAA. }
  split.
  - exact hPhaseAA.
  - split.
    + exact hRelAA.
    + exact hTrAA.
Qed.

Lemma adequacy_closure_game_semantics_full
    {Ty : Type} `{LogicStruct_game_semantics_full Ty}
    (A B : Ty)
    (hEqv : equiv A B) :
    game A B /\ (game B A \/ transport (fun X : Ty => X = X) A B).
Proof.
  assert (hForward : game A B).
  { apply (adequacy_axiom A B). exact hEqv. }
  assert (hBackward : game B A).
  { apply (bisimulation_axiom A B). exact hForward. }
  assert (hTr : transport (fun X : Ty => X = X) A B).
  { apply (univalence_axiom A B). exact hEqv. }
  assert (hKeep : transport (fun X : Ty => X = X) A B).
  { exact hTr. }
  split.
  - exact hForward.
  - left.
    exact hBackward.
Qed.
