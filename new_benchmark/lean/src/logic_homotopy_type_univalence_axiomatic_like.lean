/-
BENCHMARK_ID: TINY_MATHLIB_BATCH05_LOGIC_HOMOTOPY_TYPE_UNIVALENCE_AXIOMATIC_LIKE
PAIR_STEM: logic_homotopy_type_univalence_axiomatic_like
MATH_DOMAIN: Logic
SOURCE_MATHLIB: Mathlib/Logic/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class LogicStruct_homotopy_type_univalence (Ty : Type u) where
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

def TypeObj_homotopy_type_univalence
    (Ty : Type u) : Type u :=
  Ty

def RelObj_homotopy_type_univalence
    (Ty : Type u) : Type u :=
  Ty → Ty → Prop

def PhaseObj_homotopy_type_univalence
    (Ty : Type u) : Type u :=
  Ty → Ty → Prop

def GameObj_homotopy_type_univalence
    (Ty : Type u) : Type u :=
  Ty → Ty → Prop

theorem univalence_transport_homotopy_type_univalence
    {Ty : Type u} [h : LogicStruct_homotopy_type_univalence Ty]
    (A B : Ty)
    (hEqv : h.equiv A B) :
    h.transport (fun X : Ty => X = X) A B ∧ (h.equiv A B → h.game A B) := by
  have hTr : h.transport (fun X : Ty => X = X) A B := h.univalence_axiom A B hEqv
  refine And.intro hTr ?_
  intro hEqvAgain
  exact h.adequacy_axiom A B hEqvAgain

theorem parametricity_free_theorem_homotopy_type_univalence
    {Ty : Type u} [h : LogicStruct_homotopy_type_univalence Ty]
    (A B : Ty)
    (hRel : h.rel A B) :
    ∃ p : h.phase A B, h.transport (fun X : Ty => X = X) A B ∧ h.rel A B := by
  have hEqv : h.equiv A B := h.parametricity_axiom A B hRel
  have hGame : h.game A B := h.adequacy_axiom A B hEqv
  have hPhase : h.phase A B := h.full_abstraction_axiom A B hGame
  have hTr : h.transport (fun X : Ty => X = X) A B := h.univalence_axiom A B hEqv
  exact ⟨hPhase, And.intro hTr hRel⟩

theorem phase_soundness_homotopy_type_univalence
    {Ty : Type u} [h : LogicStruct_homotopy_type_univalence Ty]
    (A B : Ty)
    (hPhase : h.phase A B) :
    (h.rel A B → h.equiv A B) ∧ h.rel A B := by
  have hRel : h.rel A B := h.phase_sound_axiom A B hPhase
  constructor
  · intro hRelInput
    exact h.parametricity_axiom A B hRelInput
  · exact hRel

theorem full_abstraction_step_homotopy_type_univalence
    {Ty : Type u} [h : LogicStruct_homotopy_type_univalence Ty]
    (A B : Ty)
    (hGame : h.game A B) :
    ∃ e : h.equiv A B, h.phase A B ∧ h.rel A B := by
  have hPhase : h.phase A B := h.full_abstraction_axiom A B hGame
  have hRel : h.rel A B := h.phase_sound_axiom A B hPhase
  have hEqv : h.equiv A B := h.parametricity_axiom A B hRel
  exact ⟨hEqv, And.intro hPhase hRel⟩

theorem bisimulation_up_to_step_homotopy_type_univalence
    {Ty : Type u} [h : LogicStruct_homotopy_type_univalence Ty]
    (A B : Ty)
    (hGame : h.game A B) :
    h.game B A ∨ (h.phase B A ∧ h.rel B A) := by
  have hBack : h.game B A := h.bisimulation_axiom A B hGame
  have hBackPhase : h.phase B A := h.full_abstraction_axiom B A hBack
  have hBackRel : h.rel B A := h.phase_sound_axiom B A hBackPhase
  exact Or.inr (And.intro hBackPhase hBackRel)

theorem normalization_bridge_homotopy_type_univalence
    {Ty : Type u} [h : LogicStruct_homotopy_type_univalence Ty]
    (A : Ty) :
    ∃ p : h.phase A A, h.transport (fun X : Ty => X = X) A A ∧ h.game A A := by
  have hGame : h.game A A := h.normalization_axiom A
  have hPhase : h.phase A A := h.full_abstraction_axiom A A hGame
  have hRel : h.rel A A := h.phase_sound_axiom A A hPhase
  have hEqv : h.equiv A A := h.parametricity_axiom A A hRel
  have hTr : h.transport (fun X : Ty => X = X) A A := h.univalence_axiom A A hEqv
  exact ⟨hPhase, And.intro hTr hGame⟩

theorem adequacy_closure_homotopy_type_univalence
    {Ty : Type u} [h : LogicStruct_homotopy_type_univalence Ty]
    (A B : Ty)
    (hEqv : h.equiv A B) :
    (h.game A B ∧ h.game B A) ∧ (h.rel A B → h.equiv B A) := by
  have hForward : h.game A B := h.adequacy_axiom A B hEqv
  have hBackward : h.game B A := h.bisimulation_axiom A B hForward
  have hPhaseAB : h.phase A B := h.full_abstraction_axiom A B hForward
  have hRelAB : h.rel A B := h.phase_sound_axiom A B hPhaseAB
  refine And.intro (And.intro hForward hBackward) ?_
  intro hRelIn
  have hGameFromRel : h.game A B := h.adequacy_axiom A B (h.parametricity_axiom A B hRelIn)
  have hGameBack : h.game B A := h.bisimulation_axiom A B hGameFromRel
  have hPhaseBack : h.phase B A := h.full_abstraction_axiom B A hGameBack
  have hRelBack : h.rel B A := h.phase_sound_axiom B A hPhaseBack
  have _ : h.rel A B := hRelAB
  exact h.parametricity_axiom B A hRelBack
