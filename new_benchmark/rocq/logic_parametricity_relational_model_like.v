(*
BENCHMARK_ID: TINY_MATHLIB_BATCH05_LOGIC_PARAMETRICITY_RELATIONAL_MODEL_LIKE
PAIR_STEM: logic_parametricity_relational_model_like
MATH_DOMAIN: Logic
SOURCE_MATHLIB: Mathlib/Logic/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class LogicStruct_parametricity_relational_model (Ty : Type) := {
  rel : Ty -> Ty -> Prop;
  phase : Ty -> Ty -> Prop;
  game : Ty -> Ty -> Prop;
  transport : Ty -> Ty -> Prop;
  rel_refl : forall A : Ty, rel A A;
  rel_trans : forall A B C : Ty, rel A B -> rel B C -> rel A C;
  univalence_axiom : forall A B : Ty, rel A B -> transport A B;
  free_axiom : forall A B : Ty, transport A B -> game A B;
  phase_sound_axiom : forall A B : Ty, phase A B -> rel A B;
  full_abstraction_axiom : forall A B : Ty, game A B -> phase A B;
  bisimulation_axiom : forall A B : Ty, game A B -> game B A;
  normalization_axiom : forall A : Ty, game A A;
  adequacy_axiom : forall A B : Ty, transport A B -> phase A B
}.

Definition TypeObj_parametricity_relational_model
    (Ty : Type) : Type :=
  Ty.

Definition RelObj_parametricity_relational_model
    (Ty : Type) : Type :=
  Ty -> Ty -> Prop.

Definition PhaseObj_parametricity_relational_model
    (Ty : Type) : Type :=
  Ty -> Ty -> Prop.

Definition GameObj_parametricity_relational_model
    (Ty : Type) : Type :=
  Ty -> Ty -> Prop.

Lemma univalence_transport_parametricity_relational_model
    {Ty : Type} `{LogicStruct_parametricity_relational_model Ty}
    (A B : TypeObj_parametricity_relational_model Ty)
    (hRel : rel A B) :
    transport A B /\ (phase A B -> game A B).
Proof.
  assert (hTr : transport A B).
  { apply (univalence_axiom A B hRel). }
  split.
  - exact hTr.
  - intro hPhaseIn.
    assert (hRelIn : rel A B).
    { apply (phase_sound_axiom A B hPhaseIn). }
    assert (hTrIn : transport A B).
    { apply (univalence_axiom A B hRelIn). }
    exact (free_axiom A B hTrIn).
Qed.

Lemma parametricity_free_theorem_parametricity_relational_model
    {Ty : Type} `{LogicStruct_parametricity_relational_model Ty}
    (A B : TypeObj_parametricity_relational_model Ty)
    (hPhase : phase A B) :
    game A B \/ (transport A B /\ rel A B).
Proof.
  assert (hRel : rel A B).
  { apply (phase_sound_axiom A B hPhase). }
  assert (hTr : transport A B).
  { apply (univalence_axiom A B hRel). }
  assert (hGame : game A B).
  { apply (free_axiom A B hTr). }
  left.
  exact hGame.
Qed.

Lemma phase_soundness_parametricity_relational_model
    {Ty : Type} `{LogicStruct_parametricity_relational_model Ty}
    (A B : TypeObj_parametricity_relational_model Ty)
    (hGame : game A B) :
    (phase A B /\ rel A B) \/ game B A.
Proof.
  assert (hPhase : phase A B).
  { apply (full_abstraction_axiom A B hGame). }
  assert (hRel : rel A B).
  { apply (phase_sound_axiom A B hPhase). }
  left.
  split.
  - exact hPhase.
  - exact hRel.
Qed.

Lemma full_abstraction_step_parametricity_relational_model
    {Ty : Type} `{LogicStruct_parametricity_relational_model Ty}
    (A B : TypeObj_parametricity_relational_model Ty)
    (hTr : transport A B) :
    game A B /\ (phase A B \/ game B A).
Proof.
  assert (hGameAB : game A B).
  { apply (free_axiom A B hTr). }
  assert (hGameBA : game B A).
  { apply (bisimulation_axiom A B hGameAB). }
  assert (hPhaseAB : phase A B).
  { apply (adequacy_axiom A B hTr). }
  assert (hKeep : phase A B).
  { exact hPhaseAB. }
  split.
  - exact hGameAB.
  - right.
    exact hGameBA.
Qed.

Lemma bisimulation_up_to_step_parametricity_relational_model
    {Ty : Type} `{LogicStruct_parametricity_relational_model Ty}
    (A B : TypeObj_parametricity_relational_model Ty)
    (hTr : transport A B) :
    game B A /\ transport B B.
Proof.
  assert (hGameAB : game A B).
  { apply (free_axiom A B hTr). }
  assert (hGameBA : game B A).
  { apply (bisimulation_axiom A B hGameAB). }
  assert (hRelBB : rel B B).
  { apply (rel_refl B). }
  assert (hTrBB : transport B B).
  { apply (univalence_axiom B B hRelBB). }
  split.
  - exact hGameBA.
  - exact hTrBB.
Qed.

Lemma normalization_bridge_parametricity_relational_model
    {Ty : Type} `{LogicStruct_parametricity_relational_model Ty}
    (A : TypeObj_parametricity_relational_model Ty) :
    exists Z : Ty, Z = A /\ transport A Z /\ rel Z Z.
Proof.
  assert (hRelAA : rel A A).
  { apply (rel_refl A). }
  assert (hTrAA : transport A A).
  { apply (univalence_axiom A A hRelAA). }
  exists A.
  split.
  - reflexivity.
  - split.
    + exact hTrAA.
    + exact hRelAA.
Qed.

Lemma adequacy_closure_parametricity_relational_model
    {Ty : Type} `{LogicStruct_parametricity_relational_model Ty}
    (A B : TypeObj_parametricity_relational_model Ty)
    (hTr : transport A B) :
    phase A B /\ (rel A B /\ (transport B B \/ game B B)).
Proof.
  assert (hPhaseAB : phase A B).
  { apply (adequacy_axiom A B hTr). }
  assert (hRelAB : rel A B).
  { apply (phase_sound_axiom A B hPhaseAB). }
  assert (hRelBB : rel B B).
  { apply (rel_refl B). }
  assert (hTrBB : transport B B).
  { apply (univalence_axiom B B hRelBB). }
  assert (hGameBB : game B B).
  { apply (normalization_axiom B). }
  assert (hKeep : game B B).
  { exact hGameBB. }
  split.
  - exact hPhaseAB.
  - split.
    + exact hRelAB.
    + left.
      exact hTrBB.
Qed.
