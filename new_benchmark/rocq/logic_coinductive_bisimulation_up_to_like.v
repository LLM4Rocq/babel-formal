(*
BENCHMARK_ID: TINY_MATHLIB_BATCH05_LOGIC_COINDUCTIVE_BISIMULATION_UP_TO_LIKE
PAIR_STEM: logic_coinductive_bisimulation_up_to_like
MATH_DOMAIN: Logic
SOURCE_MATHLIB: Mathlib/Logic/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class LogicStruct_coinductive_bisimulation_up (Ty : Type) := {
  rel : Ty -> Ty -> Prop;
  phase : Ty -> nat;
  game : Ty -> Ty -> Prop;
  bisim : Ty -> Ty -> Prop;
  upTo : (Ty -> Ty -> Prop) -> Ty -> Ty -> Prop;
  univalence_axiom :
    forall A B : Ty,
      A = B ->
      rel A B;
  parametricity_axiom :
    forall R : Ty -> Ty -> Prop,
      (forall x y : Ty, R x y -> rel x y) ->
      forall x y : Ty, upTo R x y -> rel x y;
  phase_sound_axiom :
    forall A B : Ty,
      game A B ->
      phase A = phase B;
  full_abstraction_axiom :
    forall A B : Ty,
      rel A B ->
      game A B;
  upTo_sound_axiom :
    forall R : Ty -> Ty -> Prop,
      forall A B : Ty,
        upTo R A B ->
        R A B \/ bisim A B;
  bisim_to_rel_axiom :
    forall A B : Ty,
      bisim A B ->
      rel A B;
  normalization_axiom :
    forall A : Ty,
      exists B : Ty,
        game A B /\ upTo (fun x y => x = y) B B;
  adequacy_axiom :
    forall A B : Ty,
      rel A B ->
      game A B ->
      bisim A B
}.

Definition TypeObj_coinductive_bisimulation_up
    (Ty : Type) : Type :=
  Ty.

Definition RelObj_coinductive_bisimulation_up
    (Ty : Type) : Type :=
  Ty -> Ty -> Prop.

Definition PhaseObj_coinductive_bisimulation_up
    (Ty : Type) : Type :=
  Ty -> nat.

Definition GameObj_coinductive_bisimulation_up
    (Ty : Type) : Type :=
  Ty -> Ty -> Prop.

Lemma univalence_transport_coinductive_bisimulation_up
    {Ty : Type} `{LogicStruct_coinductive_bisimulation_up Ty}
    (A B : Ty)
    (hEq : A = B) :
    rel A B.
Proof.
  exact (univalence_axiom hEq).
Qed.

Lemma parametricity_free_theorem_coinductive_bisimulation_up
    {Ty : Type} `{LogicStruct_coinductive_bisimulation_up Ty}
    (R : RelObj_coinductive_bisimulation_up Ty)
    (hR : forall x y : Ty, R x y -> rel x y)
    (A B : Ty)
    (hUp : upTo R A B) :
    rel A B /\ (R A B \/ bisim A B).
Proof.
  assert (hParam : forall x y : Ty, upTo R x y -> rel x y).
  { exact (parametricity_axiom R hR). }
  assert (hRelAB : rel A B).
  { apply (hParam A B hUp). }
  assert (hSound : R A B \/ bisim A B).
  { apply (upTo_sound_axiom R A B hUp). }
  split.
  - exact hRelAB.
  - exact hSound.
Qed.

Lemma phase_soundness_coinductive_bisimulation_up
    {Ty : Type} `{LogicStruct_coinductive_bisimulation_up Ty}
    (A B : Ty)
    (hGame : game A B) :
    phase A = phase B.
Proof.
  exact (phase_sound_axiom A B hGame).
Qed.

Lemma full_abstraction_step_coinductive_bisimulation_up
    {Ty : Type} `{LogicStruct_coinductive_bisimulation_up Ty}
    (A B : Ty)
    (hRel : rel A B) :
    game A B /\ phase A = phase B.
Proof.
  assert (hGame : game A B).
  { apply (full_abstraction_axiom A B hRel). }
  assert (hPhase : phase A = phase B).
  { apply (phase_sound_axiom A B hGame). }
  split.
  - exact hGame.
  - exact hPhase.
Qed.

Lemma bisimulation_up_to_step_coinductive_bisimulation_up
    {Ty : Type} `{LogicStruct_coinductive_bisimulation_up Ty}
    (A B : Ty)
    (hUp : upTo bisim A B) :
    bisim A B /\ rel A B.
Proof.
  assert (hSound : bisim A B \/ bisim A B).
  { apply (upTo_sound_axiom bisim A B hUp). }
  assert (hBis : bisim A B).
  {
    destruct hSound as [hLeft | hRight].
    - exact hLeft.
    - exact hRight.
  }
  assert (hRel : rel A B).
  { apply (bisim_to_rel_axiom A B hBis). }
  split.
  - exact hBis.
  - exact hRel.
Qed.

Lemma normalization_bridge_coinductive_bisimulation_up
    {Ty : Type} `{LogicStruct_coinductive_bisimulation_up Ty}
    (A : Ty) :
    exists B : Ty, game A B /\ rel B B /\ phase A = phase B.
Proof.
  destruct (normalization_axiom A) as [B [hGame hDiagUp]].
  assert (hEqRel : forall x y : Ty, x = y -> rel x y).
  {
    intros x y hxy.
    exact (univalence_axiom hxy).
  }
  assert (hDiag : rel B B /\ ((B = B) \/ bisim B B)).
  {
    apply (parametricity_free_theorem_coinductive_bisimulation_up (fun x y : Ty => x = y) hEqRel B B hDiagUp).
  }
  assert (hRelBB : rel B B).
  { exact (proj1 hDiag). }
  assert (hPhase : phase A = phase B).
  { apply (phase_sound_axiom A B hGame). }
  exists B.
  split.
  - exact hGame.
  - split.
    + exact hRelBB.
    + exact hPhase.
Qed.

Lemma adequacy_closure_coinductive_bisimulation_up
    {Ty : Type} `{LogicStruct_coinductive_bisimulation_up Ty}
    (A B : Ty)
    (hRel : rel A B) :
    bisim A B /\ rel A B /\ phase A = phase B.
Proof.
  assert (hPack : game A B /\ phase A = phase B).
  { apply (full_abstraction_step_coinductive_bisimulation_up A B hRel). }
  assert (hGame : game A B).
  { exact (proj1 hPack). }
  assert (hPhase : phase A = phase B).
  { exact (proj2 hPack). }
  assert (hBis : bisim A B).
  { apply (adequacy_axiom A B hRel hGame). }
  split.
  - exact hBis.
  - split.
    + exact hRel.
    + exact hPhase.
Qed.
