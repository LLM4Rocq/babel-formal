/-
BENCHMARK_ID: TINY_MATHLIB_BATCH05_LOGIC_PARAMETRICITY_RELATIONAL_MODEL_LIKE
PAIR_STEM: logic_parametricity_relational_model_like
MATH_DOMAIN: Logic
SOURCE_MATHLIB: Mathlib/Logic/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class LogicStruct_parametricity_relational_model (Ty : Type u) where
  rel : Ty → Ty → Prop
  phase : Ty → Ty → Prop
  game : Ty → Ty → Prop
  transport : Ty → Ty → Prop
  rel_refl : ∀ A : Ty, rel A A
  rel_trans : ∀ A B C : Ty, rel A B → rel B C → rel A C
  univalence_axiom : ∀ A B : Ty, rel A B → transport A B
  free_axiom : ∀ A B : Ty, transport A B → game A B
  phase_sound_axiom : ∀ A B : Ty, phase A B → rel A B
  full_abstraction_axiom : ∀ A B : Ty, game A B → phase A B
  bisimulation_axiom : ∀ A B : Ty, game A B → game B A
  normalization_axiom : ∀ A : Ty, game A A
  adequacy_axiom : ∀ A B : Ty, transport A B → phase A B

def TypeObj_parametricity_relational_model
    (Ty : Type u) : Type u :=
  Ty

def RelObj_parametricity_relational_model
    (Ty : Type u) : Type u :=
  Ty → Ty → Prop

def PhaseObj_parametricity_relational_model
    (Ty : Type u) : Type u :=
  Ty → Ty → Prop

def GameObj_parametricity_relational_model
    (Ty : Type u) : Type u :=
  Ty → Ty → Prop

theorem univalence_transport_parametricity_relational_model
    {Ty : Type u} [h : LogicStruct_parametricity_relational_model Ty]
    (A B : TypeObj_parametricity_relational_model Ty)
    (hRel : h.rel A B) :
    h.transport A B ∧ (h.phase A B → h.game A B) := by
  have hTr : h.transport A B := h.univalence_axiom A B hRel
  refine And.intro hTr ?_
  intro hPhaseIn
  have hRelIn : h.rel A B := h.phase_sound_axiom A B hPhaseIn
  have hTrIn : h.transport A B := h.univalence_axiom A B hRelIn
  exact h.free_axiom A B hTrIn

theorem parametricity_free_theorem_parametricity_relational_model
    {Ty : Type u} [h : LogicStruct_parametricity_relational_model Ty]
    (A B : TypeObj_parametricity_relational_model Ty)
    (hPhase : h.phase A B) :
    h.game A B ∨ (h.transport A B ∧ h.rel A B) := by
  have hRel : h.rel A B := h.phase_sound_axiom A B hPhase
  have hTr : h.transport A B := h.univalence_axiom A B hRel
  have hGame : h.game A B := h.free_axiom A B hTr
  exact Or.inl hGame

theorem phase_soundness_parametricity_relational_model
    {Ty : Type u} [h : LogicStruct_parametricity_relational_model Ty]
    (A B : TypeObj_parametricity_relational_model Ty)
    (hGame : h.game A B) :
    (h.phase A B ∧ h.rel A B) ∨ h.game B A := by
  have hPhase : h.phase A B := h.full_abstraction_axiom A B hGame
  have hRel : h.rel A B := h.phase_sound_axiom A B hPhase
  exact Or.inl (And.intro hPhase hRel)

theorem full_abstraction_step_parametricity_relational_model
    {Ty : Type u} [h : LogicStruct_parametricity_relational_model Ty]
    (A B : TypeObj_parametricity_relational_model Ty)
    (hTr : h.transport A B) :
    h.game A B ∧ (h.phase A B ∨ h.game B A) := by
  have hGameAB : h.game A B := h.free_axiom A B hTr
  have hGameBA : h.game B A := h.bisimulation_axiom A B hGameAB
  have hPhaseAB : h.phase A B := h.adequacy_axiom A B hTr
  have _ : h.phase A B := hPhaseAB
  exact And.intro hGameAB (Or.inr hGameBA)

theorem bisimulation_up_to_step_parametricity_relational_model
    {Ty : Type u} [h : LogicStruct_parametricity_relational_model Ty]
    (A B : TypeObj_parametricity_relational_model Ty)
    (hTr : h.transport A B) :
    h.game B A ∧ h.transport B B := by
  have hGameAB : h.game A B := h.free_axiom A B hTr
  have hGameBA : h.game B A := h.bisimulation_axiom A B hGameAB
  have hRelBB : h.rel B B := h.rel_refl B
  have hTrBB : h.transport B B := h.univalence_axiom B B hRelBB
  exact And.intro hGameBA hTrBB

theorem normalization_bridge_parametricity_relational_model
    {Ty : Type u} [h : LogicStruct_parametricity_relational_model Ty]
    (A : TypeObj_parametricity_relational_model Ty) :
    ∃ Z : Ty, Z = A ∧ h.transport A Z ∧ h.rel Z Z := by
  have hRelAA : h.rel A A := h.rel_refl A
  have hTrAA : h.transport A A := h.univalence_axiom A A hRelAA
  exact ⟨A, rfl, hTrAA, hRelAA⟩

theorem adequacy_closure_parametricity_relational_model
    {Ty : Type u} [h : LogicStruct_parametricity_relational_model Ty]
    (A B : TypeObj_parametricity_relational_model Ty)
    (hTr : h.transport A B) :
    h.phase A B ∧ (h.rel A B ∧ (h.transport B B ∨ h.game B B)) := by
  have hPhaseAB : h.phase A B := h.adequacy_axiom A B hTr
  have hRelAB : h.rel A B := h.phase_sound_axiom A B hPhaseAB
  have hRelBB : h.rel B B := h.rel_refl B
  have hTrBB : h.transport B B := h.univalence_axiom B B hRelBB
  have hGameBB : h.game B B := h.normalization_axiom B
  have _ : h.game B B := hGameBB
  exact And.intro hPhaseAB (And.intro hRelAB (Or.inl hTrBB))
