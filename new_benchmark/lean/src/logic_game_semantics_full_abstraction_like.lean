/-
BENCHMARK_ID: TINY_MATHLIB_BATCH05_LOGIC_GAME_SEMANTICS_FULL_ABSTRACTION_LIKE
PAIR_STEM: logic_game_semantics_full_abstraction_like
MATH_DOMAIN: Logic
SOURCE_MATHLIB: Mathlib/Logic/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class LogicStruct_game_semantics_full (Ty : Type u) where
  equiv : Ty → Ty → Prop
  transport : (Ty → Prop) → Ty → Ty → Prop
  rel : Ty → Ty → Prop
  phase : Ty → Ty → Prop
  game : Ty → Ty → Prop
  univalence_axiom :
    ∀ A B : Ty,
      equiv A B →
        transport (fun X : Ty => X = X) A B
  parametricity_axiom :
    ∀ A B : Ty,
      rel A B → equiv A B
  phase_sound_axiom :
    ∀ A B : Ty,
      phase A B → rel A B
  phase_complete_axiom :
    ∀ A B : Ty,
      rel A B → phase A B
  full_abstraction_axiom :
    ∀ A B : Ty,
      game A B → phase A B
  bisimulation_axiom :
    ∀ A B : Ty,
      game A B → game B A
  normalization_axiom :
    ∀ A : Ty,
      game A A
  adequacy_axiom :
    ∀ A B : Ty,
      equiv A B → game A B

def TypeObj_game_semantics_full (Ty : Type u) : Type u :=
  Ty

def RelObj_game_semantics_full (Ty : Type u) : Type u :=
  Ty → Ty → Prop

def PhaseObj_game_semantics_full (Ty : Type u) : Type u :=
  Ty → Ty → Prop

def GameObj_game_semantics_full (Ty : Type u) : Type u :=
  Ty → Ty → Prop

theorem univalence_transport_game_semantics_full
    {Ty : Type u} [h : LogicStruct_game_semantics_full Ty]
    (A B : Ty)
    (hEqv : h.equiv A B) :
    h.transport (fun X : Ty => X = X) A B := by
  have hGame : h.game A B := h.adequacy_axiom A B hEqv
  have hPhase : h.phase A B := h.full_abstraction_axiom A B hGame
  have hRel : h.rel A B := h.phase_sound_axiom A B hPhase
  have hEqvAgain : h.equiv A B := h.parametricity_axiom A B hRel
  exact h.univalence_axiom A B hEqvAgain

theorem parametricity_free_theorem_game_semantics_full
    {Ty : Type u} [h : LogicStruct_game_semantics_full Ty]
    (A B : Ty)
    (hRel : h.rel A B) :
    ∃ q : h.phase A B, h.game A B ∨ h.rel A B := by
  have hPhase : h.phase A B := h.phase_complete_axiom A B hRel
  have hEqv : h.equiv A B := h.parametricity_axiom A B hRel
  have hGame : h.game A B := h.adequacy_axiom A B hEqv
  exact ⟨hPhase, Or.inl hGame⟩

theorem phase_soundness_game_semantics_full
    {Ty : Type u} [h : LogicStruct_game_semantics_full Ty]
    (A B : Ty)
    (hPhase : h.phase A B) :
    h.rel A B ∧ (h.phase A B ↔ h.phase A B) := by
  have hRel : h.rel A B := h.phase_sound_axiom A B hPhase
  refine And.intro hRel ?_
  constructor <;> intro hp <;> exact hp

theorem full_abstraction_step_game_semantics_full
    {Ty : Type u} [h : LogicStruct_game_semantics_full Ty]
    (A B : Ty)
    (hGame : h.game A B) :
    h.transport (fun X : Ty => X = X) A B ∧ h.equiv A B ∧ h.phase A B := by
  have hPhase : h.phase A B := h.full_abstraction_axiom A B hGame
  have hRel : h.rel A B := h.phase_sound_axiom A B hPhase
  have hEqv : h.equiv A B := h.parametricity_axiom A B hRel
  have hTr : h.transport (fun X : Ty => X = X) A B := h.univalence_axiom A B hEqv
  exact And.intro hTr (And.intro hEqv hPhase)

theorem bisimulation_up_to_step_game_semantics_full
    {Ty : Type u} [h : LogicStruct_game_semantics_full Ty]
    (A B : Ty)
    (hGame : h.game A B) :
    h.game B A ∧ (h.phase B A → h.rel B A) := by
  have hBack : h.game B A := h.bisimulation_axiom A B hGame
  refine And.intro hBack ?_
  intro hPhaseBack
  exact h.phase_sound_axiom B A hPhaseBack

theorem normalization_bridge_game_semantics_full
    {Ty : Type u} [h : LogicStruct_game_semantics_full Ty]
    (A : Ty) :
    h.phase A A ∧ (h.rel A A ∧ h.transport (fun X : Ty => X = X) A A) := by
  have hGameAA : h.game A A := h.normalization_axiom A
  have hPhaseAA : h.phase A A := h.full_abstraction_axiom A A hGameAA
  have hRelAA : h.rel A A := h.phase_sound_axiom A A hPhaseAA
  have hEqvAA : h.equiv A A := h.parametricity_axiom A A hRelAA
  have hTrAA : h.transport (fun X : Ty => X = X) A A := h.univalence_axiom A A hEqvAA
  exact And.intro hPhaseAA (And.intro hRelAA hTrAA)

theorem adequacy_closure_game_semantics_full
    {Ty : Type u} [h : LogicStruct_game_semantics_full Ty]
    (A B : Ty)
    (hEqv : h.equiv A B) :
    h.game A B ∧ (h.game B A ∨ h.transport (fun X : Ty => X = X) A B) := by
  have hForward : h.game A B := h.adequacy_axiom A B hEqv
  have hBackward : h.game B A := h.bisimulation_axiom A B hForward
  have hTr : h.transport (fun X : Ty => X = X) A B := h.univalence_axiom A B hEqv
  have _ : h.transport (fun X : Ty => X = X) A B := hTr
  exact And.intro hForward (Or.inl hBackward)
