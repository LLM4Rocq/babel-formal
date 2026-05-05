/-
BENCHMARK_ID: TINY_MATHLIB_BATCH05_LOGIC_COINDUCTIVE_BISIMULATION_UP_TO_LIKE
PAIR_STEM: logic_coinductive_bisimulation_up_to_like
MATH_DOMAIN: Logic
SOURCE_MATHLIB: Mathlib/Logic/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class LogicStruct_coinductive_bisimulation_up (Ty : Type u) where
  rel : Ty → Ty → Prop
  phase : Ty → Nat
  game : Ty → Ty → Prop
  bisim : Ty → Ty → Prop
  upTo : (Ty → Ty → Prop) → Ty → Ty → Prop
  univalence_axiom :
    ∀ A B : Ty,
      A = B →
      rel A B
  parametricity_axiom :
    ∀ R : Ty → Ty → Prop,
      (∀ x y : Ty, R x y → rel x y) →
      ∀ x y : Ty, upTo R x y → rel x y
  phase_sound_axiom :
    ∀ A B : Ty,
      game A B →
      phase A = phase B
  full_abstraction_axiom :
    ∀ A B : Ty,
      rel A B →
      game A B
  upTo_sound_axiom :
    ∀ R : Ty → Ty → Prop,
      ∀ A B : Ty,
        upTo R A B →
        R A B ∨ bisim A B
  bisim_to_rel_axiom :
    ∀ A B : Ty,
      bisim A B →
      rel A B
  normalization_axiom :
    ∀ A : Ty,
      ∃ B : Ty,
        game A B ∧ upTo (fun x y => x = y) B B
  adequacy_axiom :
    ∀ A B : Ty,
      rel A B →
      game A B →
      bisim A B

def TypeObj_coinductive_bisimulation_up
    (Ty : Type u) : Type u :=
  Ty

def RelObj_coinductive_bisimulation_up
    (Ty : Type u) : Type u :=
  Ty → Ty → Prop

def PhaseObj_coinductive_bisimulation_up
    (Ty : Type u) : Type u :=
  Ty → Nat

def GameObj_coinductive_bisimulation_up
    (Ty : Type u) : Type u :=
  Ty → Ty → Prop

theorem univalence_transport_coinductive_bisimulation_up
    {Ty : Type u} [h : LogicStruct_coinductive_bisimulation_up Ty]
    (A B : Ty)
    (hEq : A = B) :
    h.rel A B := by
  exact h.univalence_axiom A B hEq

theorem parametricity_free_theorem_coinductive_bisimulation_up
    {Ty : Type u} [h : LogicStruct_coinductive_bisimulation_up Ty]
    (R : RelObj_coinductive_bisimulation_up Ty)
    (hR : ∀ x y : Ty, R x y → h.rel x y)
    (A B : Ty)
    (hUp : h.upTo R A B) :
    h.rel A B ∧ (R A B ∨ h.bisim A B) := by
  have hParam : ∀ x y : Ty, h.upTo R x y → h.rel x y := h.parametricity_axiom R hR
  have hRelAB : h.rel A B := hParam A B hUp
  have hSound : R A B ∨ h.bisim A B := h.upTo_sound_axiom R A B hUp
  exact And.intro hRelAB hSound

theorem phase_soundness_coinductive_bisimulation_up
    {Ty : Type u} [h : LogicStruct_coinductive_bisimulation_up Ty]
    (A B : Ty)
    (hGame : h.game A B) :
    h.phase A = h.phase B := by
  exact h.phase_sound_axiom A B hGame

theorem full_abstraction_step_coinductive_bisimulation_up
    {Ty : Type u} [h : LogicStruct_coinductive_bisimulation_up Ty]
    (A B : Ty)
    (hRel : h.rel A B) :
    h.game A B ∧ h.phase A = h.phase B := by
  have hGame : h.game A B := h.full_abstraction_axiom A B hRel
  have hPhase : h.phase A = h.phase B := h.phase_sound_axiom A B hGame
  exact And.intro hGame hPhase

theorem bisimulation_up_to_step_coinductive_bisimulation_up
    {Ty : Type u} [h : LogicStruct_coinductive_bisimulation_up Ty]
    (A B : Ty)
    (hUp : h.upTo h.bisim A B) :
    h.bisim A B ∧ h.rel A B := by
  have hSound : h.bisim A B ∨ h.bisim A B := h.upTo_sound_axiom h.bisim A B hUp
  have hBis : h.bisim A B := by
    cases hSound with
    | inl hLeft =>
        exact hLeft
    | inr hRight =>
        exact hRight
  have hRel : h.rel A B := h.bisim_to_rel_axiom A B hBis
  exact And.intro hBis hRel

theorem normalization_bridge_coinductive_bisimulation_up
    {Ty : Type u} [h : LogicStruct_coinductive_bisimulation_up Ty]
    (A : Ty) :
    ∃ B : Ty, h.game A B ∧ h.rel B B ∧ h.phase A = h.phase B := by
  rcases h.normalization_axiom A with ⟨B, hGame, hDiagUp⟩
  have hEqRel : ∀ x y : Ty, x = y → h.rel x y := by
    intro x y hxy
    exact h.univalence_axiom x y hxy
  have hDiag :
      h.rel B B ∧ ((B = B) ∨ h.bisim B B) :=
    parametricity_free_theorem_coinductive_bisimulation_up
      (R := fun x y : Ty => x = y) hEqRel B B hDiagUp
  have hRelBB : h.rel B B := hDiag.left
  have hPhase : h.phase A = h.phase B := h.phase_sound_axiom A B hGame
  exact ⟨B, hGame, hRelBB, hPhase⟩

theorem adequacy_closure_coinductive_bisimulation_up
    {Ty : Type u} [h : LogicStruct_coinductive_bisimulation_up Ty]
    (A B : Ty)
    (hRel : h.rel A B) :
    h.bisim A B ∧ h.rel A B ∧ h.phase A = h.phase B := by
  have hPack : h.game A B ∧ h.phase A = h.phase B :=
    full_abstraction_step_coinductive_bisimulation_up A B hRel
  have hGame : h.game A B := hPack.left
  have hPhase : h.phase A = h.phase B := hPack.right
  have hBis : h.bisim A B := h.adequacy_axiom A B hRel hGame
  exact And.intro hBis (And.intro hRel hPhase)
