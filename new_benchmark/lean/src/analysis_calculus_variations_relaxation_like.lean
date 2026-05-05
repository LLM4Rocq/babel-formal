/-
BENCHMARK_ID: TINY_MATHLIB_BATCH07_ANALYSIS_CALCULUS_VARIATIONS_RELAXATION_LIKE
PAIR_STEM: analysis_calculus_variations_relaxation_like
MATH_DOMAIN: Analysis
SOURCE_MATHLIB: Mathlib/Analysis/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class FrameworkStruct_analysis_calculus_variations_relaxation (E : Type u) where
  energy : E → Nat
  relaxed : E → Nat
  gradient : E → Nat
  perturb : E → E
  envelope : E → E
  le_trans_nat : ∀ a b c : Nat, a ≤ b → b ≤ c → a ≤ c
  add_le_add_left_nat : ∀ a b c : Nat, a ≤ b → c + a ≤ c + b
  add_le_add_right_nat : ∀ a b c : Nat, a ≤ b → a + c ≤ b + c
  le_add_left_nat : ∀ a b : Nat, b ≤ a + b
  le_add_right_nat : ∀ a b : Nat, a ≤ a + b
  relaxed_le_energy : ∀ x : E, relaxed x ≤ energy x
  perturb_energy : ∀ x : E, energy (perturb x) ≤ energy x + gradient x
  perturb_relaxed : ∀ x : E, relaxed (perturb x) ≤ relaxed x + gradient x
  envelope_relaxed : ∀ x : E, relaxed (envelope x) ≤ relaxed x
  envelope_of_perturb_energy : ∀ x : E, energy (envelope (perturb x)) ≤ energy x + gradient x
  envelope_of_perturb_relaxed : ∀ x : E, relaxed (envelope (perturb x)) ≤ relaxed x + gradient x
  witness_envelope : ∀ x : E, ∃ y : E, y = envelope x ∧ relaxed y ≤ energy x

structure ContextData_analysis_calculus_variations_relaxation
    (E : Type u) [s : FrameworkStruct_analysis_calculus_variations_relaxation E] where
  base : E
  competitor : E
  penalty : Nat
  gradient_bound : s.gradient base ≤ penalty
  base_energy_control : s.energy base ≤ s.energy competitor + penalty


def primary_map_analysis_calculus_variations_relaxation
    {E : Type u} [s : FrameworkStruct_analysis_calculus_variations_relaxation E]
    (d : ContextData_analysis_calculus_variations_relaxation E) : E :=
  s.envelope (s.perturb d.base)

def secondary_map_analysis_calculus_variations_relaxation
    {E : Type u} [s : FrameworkStruct_analysis_calculus_variations_relaxation E]
    (d : ContextData_analysis_calculus_variations_relaxation E) : Nat :=
  s.energy (primary_map_analysis_calculus_variations_relaxation (E := E) d) + d.penalty

def tertiary_map_analysis_calculus_variations_relaxation
    {E : Type u} [s : FrameworkStruct_analysis_calculus_variations_relaxation E]
    (d : ContextData_analysis_calculus_variations_relaxation E) : Prop :=
  s.relaxed (primary_map_analysis_calculus_variations_relaxation (E := E) d)
    ≤ secondary_map_analysis_calculus_variations_relaxation (E := E) d

theorem stability_step_analysis_calculus_variations_relaxation
    {E : Type u} [s : FrameworkStruct_analysis_calculus_variations_relaxation E]
    (d : ContextData_analysis_calculus_variations_relaxation E) :
    (True → (s.energy (primary_map_analysis_calculus_variations_relaxation (E := E) d)
      ≤ s.energy d.base + s.gradient d.base ∧
    s.relaxed (primary_map_analysis_calculus_variations_relaxation (E := E) d)
      ≤ s.relaxed d.base + s.gradient d.base))
 := by
  intro _hTrue
  have hEnergyRaw : s.energy (s.envelope (s.perturb d.base)) ≤ s.energy d.base + s.gradient d.base :=
    s.envelope_of_perturb_energy d.base
  have hRelaxRaw : s.relaxed (s.envelope (s.perturb d.base)) ≤ s.relaxed d.base + s.gradient d.base :=
    s.envelope_of_perturb_relaxed d.base
  have hEnergy :
      s.energy (primary_map_analysis_calculus_variations_relaxation (E := E) d)
        ≤ s.energy d.base + s.gradient d.base := by
    simpa [primary_map_analysis_calculus_variations_relaxation] using hEnergyRaw
  have hRelax :
      s.relaxed (primary_map_analysis_calculus_variations_relaxation (E := E) d)
        ≤ s.relaxed d.base + s.gradient d.base := by
    simpa [primary_map_analysis_calculus_variations_relaxation] using hRelaxRaw
  exact ⟨hEnergy, hRelax⟩

theorem factorization_step_analysis_calculus_variations_relaxation
    {E : Type u} [s : FrameworkStruct_analysis_calculus_variations_relaxation E]
    (d : ContextData_analysis_calculus_variations_relaxation E) :
    (True → (s.relaxed (s.perturb d.base) ≤ s.energy d.base + s.gradient d.base ∧
    s.relaxed (s.envelope d.base) ≤ s.energy d.base))
 := by
  intro _hTrue
  have hRelaxPert : s.relaxed (s.perturb d.base) ≤ s.energy (s.perturb d.base) :=
    s.relaxed_le_energy (s.perturb d.base)
  have hPertEnergy : s.energy (s.perturb d.base) ≤ s.energy d.base + s.gradient d.base :=
    s.perturb_energy d.base
  have hFirst : s.relaxed (s.perturb d.base) ≤ s.energy d.base + s.gradient d.base :=
    s.le_trans_nat _ _ _ hRelaxPert hPertEnergy
  have hEnvRelax : s.relaxed (s.envelope d.base) ≤ s.relaxed d.base :=
    s.envelope_relaxed d.base
  have hBase : s.relaxed d.base ≤ s.energy d.base :=
    s.relaxed_le_energy d.base
  have hSecond : s.relaxed (s.envelope d.base) ≤ s.energy d.base :=
    s.le_trans_nat _ _ _ hEnvRelax hBase
  exact ⟨hFirst, hSecond⟩

theorem comparison_step_analysis_calculus_variations_relaxation
    {E : Type u} [s : FrameworkStruct_analysis_calculus_variations_relaxation E]
    (d : ContextData_analysis_calculus_variations_relaxation E) :
    (True → (tertiary_map_analysis_calculus_variations_relaxation (E := E) d ∧
    s.energy (primary_map_analysis_calculus_variations_relaxation (E := E) d)
      ≤ secondary_map_analysis_calculus_variations_relaxation (E := E) d))
 := by
  intro _hTrue
  have hRelaxEnergy :
      s.relaxed (primary_map_analysis_calculus_variations_relaxation (E := E) d)
        ≤ s.energy (primary_map_analysis_calculus_variations_relaxation (E := E) d) :=
    s.relaxed_le_energy _
  have hEnergySec :
      s.energy (primary_map_analysis_calculus_variations_relaxation (E := E) d)
        ≤ secondary_map_analysis_calculus_variations_relaxation (E := E) d := by
    simpa [secondary_map_analysis_calculus_variations_relaxation] using
      (s.le_add_right_nat (s.energy (primary_map_analysis_calculus_variations_relaxation (E := E) d)) d.penalty)
  have hTer : tertiary_map_analysis_calculus_variations_relaxation (E := E) d :=
    s.le_trans_nat _ _ _ hRelaxEnergy hEnergySec
  exact ⟨hTer, hEnergySec⟩

theorem transport_step_analysis_calculus_variations_relaxation
    {E : Type u} [s : FrameworkStruct_analysis_calculus_variations_relaxation E]
    (d : ContextData_analysis_calculus_variations_relaxation E) :
    (True → (s.relaxed (primary_map_analysis_calculus_variations_relaxation (E := E) d)
      ≤ s.energy d.base + d.penalty ∧
    s.energy (primary_map_analysis_calculus_variations_relaxation (E := E) d)
      ≤ (s.energy d.competitor + d.penalty) + s.gradient d.base))
 := by
  intro _hTrue
  have hStab := (stability_step_analysis_calculus_variations_relaxation (E := E) d) trivial
  rcases hStab with ⟨hEnergy, hRelax⟩
  have hBaseRelax : s.relaxed d.base ≤ s.energy d.base :=
    s.relaxed_le_energy d.base
  have hLift : s.relaxed d.base + s.gradient d.base ≤ s.energy d.base + s.gradient d.base :=
    s.add_le_add_right_nat (s.relaxed d.base) (s.energy d.base) (s.gradient d.base) hBaseRelax
  have hRelaxBase :
      s.relaxed (primary_map_analysis_calculus_variations_relaxation (E := E) d)
        ≤ s.energy d.base + s.gradient d.base :=
    s.le_trans_nat _ _ _ hRelax hLift
  have hGrad : s.energy d.base + s.gradient d.base ≤ s.energy d.base + d.penalty :=
    s.add_le_add_left_nat (s.gradient d.base) d.penalty (s.energy d.base) d.gradient_bound
  have hFirst :
      s.relaxed (primary_map_analysis_calculus_variations_relaxation (E := E) d)
        ≤ s.energy d.base + d.penalty :=
    s.le_trans_nat _ _ _ hRelaxBase hGrad
  have hSecondLift :
      s.energy d.base + s.gradient d.base ≤ (s.energy d.competitor + d.penalty) + s.gradient d.base :=
    s.add_le_add_right_nat (s.energy d.base) (s.energy d.competitor + d.penalty) (s.gradient d.base)
      d.base_energy_control
  have hSecond :
      s.energy (primary_map_analysis_calculus_variations_relaxation (E := E) d)
        ≤ (s.energy d.competitor + d.penalty) + s.gradient d.base :=
    s.le_trans_nat _ _ _ hEnergy hSecondLift
  exact ⟨hFirst, hSecond⟩

theorem coherence_step_analysis_calculus_variations_relaxation
    {E : Type u} [s : FrameworkStruct_analysis_calculus_variations_relaxation E]
    (d : ContextData_analysis_calculus_variations_relaxation E) :
    (True → (secondary_map_analysis_calculus_variations_relaxation (E := E) d =
      s.energy (primary_map_analysis_calculus_variations_relaxation (E := E) d) + d.penalty ∧
    tertiary_map_analysis_calculus_variations_relaxation (E := E) d))
 := by
  intro _hTrue
  have hEq :
      secondary_map_analysis_calculus_variations_relaxation (E := E) d =
        s.energy (primary_map_analysis_calculus_variations_relaxation (E := E) d) + d.penalty := rfl
  have hComp := (comparison_step_analysis_calculus_variations_relaxation (E := E) d) trivial
  rcases hComp with ⟨hTer, hSec⟩
  have _ : s.energy (primary_map_analysis_calculus_variations_relaxation (E := E) d)
      ≤ secondary_map_analysis_calculus_variations_relaxation (E := E) d := hSec
  exact ⟨hEq, hTer⟩

theorem iteration_step_analysis_calculus_variations_relaxation
    {E : Type u} [s : FrameworkStruct_analysis_calculus_variations_relaxation E]
    (d : ContextData_analysis_calculus_variations_relaxation E) :
    (True → (∃ y : E,
      y = s.envelope d.base ∧
      s.relaxed y ≤ s.energy d.base ∧
      s.relaxed (primary_map_analysis_calculus_variations_relaxation (E := E) d)
        ≤ s.energy d.base + s.gradient d.base))
 := by
  intro _hTrue
  rcases s.witness_envelope d.base with ⟨y, hyEq, hyRelax⟩
  have hStab := (stability_step_analysis_calculus_variations_relaxation (E := E) d) trivial
  rcases hStab with ⟨hEnergy, hRelaxPrimary⟩
  have hBaseRelax : s.relaxed d.base ≤ s.energy d.base :=
    s.relaxed_le_energy d.base
  have hLift : s.relaxed d.base + s.gradient d.base ≤ s.energy d.base + s.gradient d.base :=
    s.add_le_add_right_nat (s.relaxed d.base) (s.energy d.base) (s.gradient d.base) hBaseRelax
  have hFinal :
      s.relaxed (primary_map_analysis_calculus_variations_relaxation (E := E) d)
        ≤ s.energy d.base + s.gradient d.base :=
    s.le_trans_nat _ _ _ hRelaxPrimary hLift
  have _ : s.energy (primary_map_analysis_calculus_variations_relaxation (E := E) d)
      ≤ s.energy d.base + s.gradient d.base := hEnergy
  refine ⟨y, hyEq, hyRelax, hFinal⟩

theorem main_result_analysis_calculus_variations_relaxation
    {E : Type u} [s : FrameworkStruct_analysis_calculus_variations_relaxation E]
    (d : ContextData_analysis_calculus_variations_relaxation E) :
    (True → (∃ y : E,
      y = primary_map_analysis_calculus_variations_relaxation (E := E) d ∧
      s.relaxed y ≤ (s.energy d.competitor + d.penalty) + d.penalty ∧
      tertiary_map_analysis_calculus_variations_relaxation (E := E) d))
 := by
  intro _hTrue
  let y := primary_map_analysis_calculus_variations_relaxation (E := E) d
  have hyEq : y = primary_map_analysis_calculus_variations_relaxation (E := E) d := rfl
  have hTrans := (transport_step_analysis_calculus_variations_relaxation (E := E) d) trivial
  rcases hTrans with ⟨hRelaxBound, hEnergyBound⟩
  have hLift : s.energy d.base + d.penalty ≤ (s.energy d.competitor + d.penalty) + d.penalty :=
    s.add_le_add_right_nat (s.energy d.base) (s.energy d.competitor + d.penalty) d.penalty d.base_energy_control
  have hFinal : s.relaxed y ≤ (s.energy d.competitor + d.penalty) + d.penalty :=
    s.le_trans_nat _ _ _ (by simpa [y] using hRelaxBound) hLift
  have hComp := (comparison_step_analysis_calculus_variations_relaxation (E := E) d) trivial
  rcases hComp with ⟨hTer, hSec⟩
  have _ : s.energy y ≤ secondary_map_analysis_calculus_variations_relaxation (E := E) d := by
    simpa [y] using hSec
  have _ : s.energy y ≤ (s.energy d.competitor + d.penalty) + s.gradient d.base := by
    simpa [y] using hEnergyBound
  exact ⟨y, hyEq, hFinal, hTer⟩
