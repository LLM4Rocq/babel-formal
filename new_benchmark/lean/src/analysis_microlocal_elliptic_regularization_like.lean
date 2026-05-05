/-
BENCHMARK_ID: TINY_MATHLIB_BATCH07_ANALYSIS_MICROLOCAL_ELLIPTIC_REGULARIZATION_LIKE
PAIR_STEM: analysis_microlocal_elliptic_regularization_like
MATH_DOMAIN: Analysis
SOURCE_MATHLIB: Mathlib/Analysis/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class FrameworkStruct_analysis_microlocal_elliptic_regularization (E : Type u) where
  order : E → Nat
  defect : E → Nat
  regularize : E → E
  commutator : E → E
  parametrix : E → E
  le_trans_nat : ∀ a b c : Nat, a ≤ b → b ≤ c → a ≤ c
  add_le_add_left_nat : ∀ a b c : Nat, a ≤ b → c + a ≤ c + b
  add_le_add_right_nat : ∀ a b c : Nat, a ≤ b → a + c ≤ b + c
  le_add_left_nat : ∀ a b : Nat, b ≤ a + b
  le_add_right_nat : ∀ a b : Nat, a ≤ a + b
  order_regularize : ∀ x : E, order (regularize x) ≤ order x + defect x
  defect_regularize : ∀ x : E, defect (regularize x) ≤ defect x + defect x
  order_commutator : ∀ x : E, order (commutator x) ≤ order x + defect x
  parametrix_gain : ∀ x : E, order (parametrix x) ≤ order x
  elliptic_step : ∀ x : E, order (parametrix (regularize x)) ≤ order x + defect x
  witness_param : ∀ x : E, ∃ y : E, y = parametrix x ∧ order y ≤ order x

structure ContextData_analysis_microlocal_elliptic_regularization
    (E : Type u) [s : FrameworkStruct_analysis_microlocal_elliptic_regularization E] where
  source : E
  target : E
  budget : Nat
  source_to_target : s.order source ≤ s.order target + budget
  defect_cap : s.defect source ≤ budget

def primary_map_analysis_microlocal_elliptic_regularization
    {E : Type u} [s : FrameworkStruct_analysis_microlocal_elliptic_regularization E]
    (d : ContextData_analysis_microlocal_elliptic_regularization E) : E :=
  s.parametrix (s.regularize d.source)

def secondary_map_analysis_microlocal_elliptic_regularization
    {E : Type u} [s : FrameworkStruct_analysis_microlocal_elliptic_regularization E]
    (d : ContextData_analysis_microlocal_elliptic_regularization E) : Nat :=
  s.order (primary_map_analysis_microlocal_elliptic_regularization (E := E) d) + d.budget

def tertiary_map_analysis_microlocal_elliptic_regularization
    {E : Type u} [s : FrameworkStruct_analysis_microlocal_elliptic_regularization E]
    (d : ContextData_analysis_microlocal_elliptic_regularization E) : Prop :=
  s.order (primary_map_analysis_microlocal_elliptic_regularization (E := E) d)
    ≤ secondary_map_analysis_microlocal_elliptic_regularization (E := E) d

theorem stability_step_analysis_microlocal_elliptic_regularization
    {E : Type u} [s : FrameworkStruct_analysis_microlocal_elliptic_regularization E]
    (d : ContextData_analysis_microlocal_elliptic_regularization E) :
    ((fun P : Prop => (fun n : Nat => P) 0) (s.order (primary_map_analysis_microlocal_elliptic_regularization (E := E) d)
      ≤ s.order d.source + s.defect d.source ∧
    s.defect (s.regularize d.source) ≤ s.defect d.source + s.defect d.source))

 := by
  have hOrderRaw : s.order (s.parametrix (s.regularize d.source)) ≤ s.order d.source + s.defect d.source :=
    s.elliptic_step d.source
  have hOrder :
      s.order (primary_map_analysis_microlocal_elliptic_regularization (E := E) d)
        ≤ s.order d.source + s.defect d.source := by
    simpa [primary_map_analysis_microlocal_elliptic_regularization] using hOrderRaw
  have hDefect : s.defect (s.regularize d.source) ≤ s.defect d.source + s.defect d.source :=
    s.defect_regularize d.source
  exact ⟨hOrder, hDefect⟩

theorem factorization_step_analysis_microlocal_elliptic_regularization
    {E : Type u} [s : FrameworkStruct_analysis_microlocal_elliptic_regularization E]
    (d : ContextData_analysis_microlocal_elliptic_regularization E) :
    ((fun P : Prop => (fun n : Nat => P) 0) (s.order (s.commutator d.source) ≤ s.order d.source + s.defect d.source ∧
    (s.order (s.parametrix d.source) ≤ s.order d.source →
      s.order (s.parametrix d.source) ≤ s.order d.source)))

 := by
  have hComm : s.order (s.commutator d.source) ≤ s.order d.source + s.defect d.source :=
    s.order_commutator d.source
  have hPar : s.order (s.parametrix d.source) ≤ s.order d.source := s.parametrix_gain d.source
  have hId :
      s.order (s.parametrix d.source) ≤ s.order d.source →
      s.order (s.parametrix d.source) ≤ s.order d.source := by
    intro h
    exact h
  have _ : s.order (s.parametrix d.source) ≤ s.order d.source := hPar
  exact ⟨hComm, hId⟩

theorem comparison_step_analysis_microlocal_elliptic_regularization
    {E : Type u} [s : FrameworkStruct_analysis_microlocal_elliptic_regularization E]
    (d : ContextData_analysis_microlocal_elliptic_regularization E) :
    ((fun P : Prop => (fun n : Nat => P) 0) (tertiary_map_analysis_microlocal_elliptic_regularization (E := E) d ∧
    (tertiary_map_analysis_microlocal_elliptic_regularization (E := E) d →
      s.order (s.parametrix d.source) ≤ s.order d.source)))

 := by
  have hOrdSec :
      s.order (primary_map_analysis_microlocal_elliptic_regularization (E := E) d)
        ≤ secondary_map_analysis_microlocal_elliptic_regularization (E := E) d := by
    simpa [secondary_map_analysis_microlocal_elliptic_regularization] using
      (s.le_add_right_nat (s.order (primary_map_analysis_microlocal_elliptic_regularization (E := E) d)) d.budget)
  have hTer : tertiary_map_analysis_microlocal_elliptic_regularization (E := E) d := hOrdSec
  have hImp :
      tertiary_map_analysis_microlocal_elliptic_regularization (E := E) d →
      s.order (s.parametrix d.source) ≤ s.order d.source := by
    intro _h
    exact s.parametrix_gain d.source
  exact ⟨hTer, hImp⟩

theorem transport_step_analysis_microlocal_elliptic_regularization
    {E : Type u} [s : FrameworkStruct_analysis_microlocal_elliptic_regularization E]
    (d : ContextData_analysis_microlocal_elliptic_regularization E) :
    ((fun P : Prop => (fun n : Nat => P) 0) (s.order (primary_map_analysis_microlocal_elliptic_regularization (E := E) d)
      ≤ s.order d.target + d.budget + s.defect d.source ∧
    s.defect (s.regularize d.source) ≤ s.defect d.source + s.defect d.source))

 := by
  have hStab := stability_step_analysis_microlocal_elliptic_regularization (E := E) d
  rcases hStab with ⟨hOrder, hDefect⟩
  have hSourceTarget : s.order d.source + s.defect d.source ≤ (s.order d.target + d.budget) + s.defect d.source :=
    s.add_le_add_right_nat (s.order d.source) (s.order d.target + d.budget) (s.defect d.source) d.source_to_target
  have hFirst :
      s.order (primary_map_analysis_microlocal_elliptic_regularization (E := E) d)
        ≤ s.order d.target + d.budget + s.defect d.source :=
    s.le_trans_nat _ _ _ hOrder hSourceTarget
  have hSecond : s.defect (s.regularize d.source) ≤ s.defect d.source + s.defect d.source := hDefect
  exact ⟨hFirst, hSecond⟩

theorem coherence_step_analysis_microlocal_elliptic_regularization
    {E : Type u} [s : FrameworkStruct_analysis_microlocal_elliptic_regularization E]
    (d : ContextData_analysis_microlocal_elliptic_regularization E) :
    ((fun P : Prop => (fun n : Nat => P) 0) (∀ z : E, z = s.parametrix d.source → s.order z ≤ s.order d.source))

 := by
  intro z hzEq
  calc
    s.order z = s.order (s.parametrix d.source) := by
      rw [hzEq]
    _ ≤ s.order d.source :=
      s.parametrix_gain d.source

theorem iteration_step_analysis_microlocal_elliptic_regularization
    {E : Type u} [s : FrameworkStruct_analysis_microlocal_elliptic_regularization E]
    (d : ContextData_analysis_microlocal_elliptic_regularization E) :
    ((fun P : Prop => (fun n : Nat => P) 0) (∃ y : E,
      y = primary_map_analysis_microlocal_elliptic_regularization (E := E) d ∧
      s.order y ≤ s.order d.source + s.defect d.source ∧
      tertiary_map_analysis_microlocal_elliptic_regularization (E := E) d))

 := by
  let y := primary_map_analysis_microlocal_elliptic_regularization (E := E) d
  have hyEq : y = primary_map_analysis_microlocal_elliptic_regularization (E := E) d := rfl
  have hOrder := (stability_step_analysis_microlocal_elliptic_regularization (E := E) d).1
  have hTer := (comparison_step_analysis_microlocal_elliptic_regularization (E := E) d).1
  exact ⟨y, hyEq, by simpa [y] using hOrder, hTer⟩

theorem main_result_analysis_microlocal_elliptic_regularization
    {E : Type u} [s : FrameworkStruct_analysis_microlocal_elliptic_regularization E]
    (d : ContextData_analysis_microlocal_elliptic_regularization E) :
    ((fun P : Prop => (fun n : Nat => P) 0) (∃ y : E,
      y = primary_map_analysis_microlocal_elliptic_regularization (E := E) d ∧
      s.order y ≤ s.order d.target + d.budget + s.defect d.source ∧
      tertiary_map_analysis_microlocal_elliptic_regularization (E := E) d))

 := by
  rcases iteration_step_analysis_microlocal_elliptic_regularization (E := E) d with ⟨y, hyEq, hyOrd, hTer⟩
  have hTransport := (transport_step_analysis_microlocal_elliptic_regularization (E := E) d).1
  have hFinal : s.order y ≤ s.order d.target + d.budget + s.defect d.source := by
    simpa [hyEq] using hTransport
  exact ⟨y, hyEq, hFinal, hTer⟩
