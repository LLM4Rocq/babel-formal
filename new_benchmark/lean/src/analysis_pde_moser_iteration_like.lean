/-
BENCHMARK_ID: TINY_MATHLIB_BATCH07_ANALYSIS_PDE_MOSER_ITERATION_LIKE
PAIR_STEM: analysis_pde_moser_iteration_like
MATH_DOMAIN: Analysis
SOURCE_MATHLIB: Mathlib/Analysis/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class FrameworkStruct_analysis_pde_moser_iteration (E : Type u) where
  norm : E → Nat
  rhs : E → Nat
  exponent : Nat → Nat
  iterate : Nat → E → E
  improve : E → E
  le_trans_nat : ∀ a b c : Nat, a ≤ b → b ≤ c → a ≤ c
  add_le_add_left_nat : ∀ a b c : Nat, a ≤ b → c + a ≤ c + b
  add_le_add_right_nat : ∀ a b c : Nat, a ≤ b → a + c ≤ b + c
  le_add_left_nat : ∀ a b : Nat, b ≤ a + b
  le_add_right_nat : ∀ a b : Nat, a ≤ a + b
  iterate_zero : ∀ x : E, iterate 0 x = x
  exponent_mono : ∀ n : Nat, exponent n ≤ exponent (n + 1)
  bootstrap : ∀ n : Nat, ∀ x : E, norm (iterate n x) ≤ norm x + exponent n + rhs x
  rhs_iterate_bound : ∀ n : Nat, ∀ x : E, rhs (iterate n x) ≤ rhs x + rhs x
  rhs_improve : ∀ x : E, rhs (improve x) ≤ rhs x + rhs x
  norm_improve : ∀ x : E, norm (improve x) ≤ norm x + rhs x
  witness_iterate : ∀ n : Nat, ∀ x : E, ∃ y : E, y = iterate n x ∧ norm y ≤ norm x + exponent n + rhs x

structure ContextData_analysis_pde_moser_iteration
    (E : Type u) [s : FrameworkStruct_analysis_pde_moser_iteration E] where
  initial : E
  steps : Nat
  cap : Nat
  exponent_cap : s.exponent steps ≤ cap
  rhs_cap : s.rhs initial ≤ cap

def primary_map_analysis_pde_moser_iteration
    {E : Type u} [s : FrameworkStruct_analysis_pde_moser_iteration E]
    (d : ContextData_analysis_pde_moser_iteration E) : E :=
  s.iterate d.steps d.initial

def secondary_map_analysis_pde_moser_iteration
    {E : Type u} [s : FrameworkStruct_analysis_pde_moser_iteration E]
    (d : ContextData_analysis_pde_moser_iteration E) : Nat :=
  s.norm (primary_map_analysis_pde_moser_iteration (E := E) d) + d.cap

def tertiary_map_analysis_pde_moser_iteration
    {E : Type u} [s : FrameworkStruct_analysis_pde_moser_iteration E]
    (d : ContextData_analysis_pde_moser_iteration E) : Prop :=
  s.norm (primary_map_analysis_pde_moser_iteration (E := E) d)
    ≤ secondary_map_analysis_pde_moser_iteration (E := E) d

theorem stability_step_analysis_pde_moser_iteration
    {E : Type u} [s : FrameworkStruct_analysis_pde_moser_iteration E]
    (d : ContextData_analysis_pde_moser_iteration E) :
    ((fun P : Prop => (fun Q : Prop => P) True) (s.norm (primary_map_analysis_pde_moser_iteration (E := E) d)
      ≤ s.norm d.initial + s.exponent d.steps + s.rhs d.initial ∧
    s.exponent d.steps ≤ d.cap))

 := by
  have hBootRaw :
      s.norm (s.iterate d.steps d.initial) ≤ s.norm d.initial + s.exponent d.steps + s.rhs d.initial :=
    s.bootstrap d.steps d.initial
  have hBoot :
      s.norm (primary_map_analysis_pde_moser_iteration (E := E) d)
        ≤ s.norm d.initial + s.exponent d.steps + s.rhs d.initial := by
    simpa [primary_map_analysis_pde_moser_iteration] using hBootRaw
  exact ⟨hBoot, d.exponent_cap⟩

theorem factorization_step_analysis_pde_moser_iteration
    {E : Type u} [s : FrameworkStruct_analysis_pde_moser_iteration E]
    (d : ContextData_analysis_pde_moser_iteration E) :
    (((fun P : Prop => (fun Q : Prop => P) True) (s.rhs (primary_map_analysis_pde_moser_iteration (E := E) d) ≤ s.rhs d.initial + s.rhs d.initial ∧
    s.rhs (s.improve (primary_map_analysis_pde_moser_iteration (E := E) d))
      ≤ s.rhs (primary_map_analysis_pde_moser_iteration (E := E) d)
        + s.rhs (primary_map_analysis_pde_moser_iteration (E := E) d))) ∧ True)

 := by
  have hRhsRaw : s.rhs (s.iterate d.steps d.initial) ≤ s.rhs d.initial + s.rhs d.initial :=
    s.rhs_iterate_bound d.steps d.initial
  have hRhs : s.rhs (primary_map_analysis_pde_moser_iteration (E := E) d)
      ≤ s.rhs d.initial + s.rhs d.initial := by
    simpa [primary_map_analysis_pde_moser_iteration] using hRhsRaw
  have hImprove :
      s.rhs (s.improve (primary_map_analysis_pde_moser_iteration (E := E) d))
        ≤ s.rhs (primary_map_analysis_pde_moser_iteration (E := E) d)
          + s.rhs (primary_map_analysis_pde_moser_iteration (E := E) d) :=
    s.rhs_improve (primary_map_analysis_pde_moser_iteration (E := E) d)
  exact ⟨⟨hRhs, hImprove⟩, trivial⟩

theorem comparison_step_analysis_pde_moser_iteration
    {E : Type u} [s : FrameworkStruct_analysis_pde_moser_iteration E]
    (d : ContextData_analysis_pde_moser_iteration E) :
    ((fun P : Prop => (fun Q : Prop => P) True) (tertiary_map_analysis_pde_moser_iteration (E := E) d))

 := by
  have hSec :
      s.norm (primary_map_analysis_pde_moser_iteration (E := E) d)
        ≤ secondary_map_analysis_pde_moser_iteration (E := E) d := by
    simpa [secondary_map_analysis_pde_moser_iteration] using
      (s.le_add_right_nat (s.norm (primary_map_analysis_pde_moser_iteration (E := E) d)) d.cap)
  exact hSec

theorem transport_step_analysis_pde_moser_iteration
    {E : Type u} [s : FrameworkStruct_analysis_pde_moser_iteration E]
    (d : ContextData_analysis_pde_moser_iteration E) :
    ((fun P : Prop => (fun Q : Prop => P) True) (s.norm (primary_map_analysis_pde_moser_iteration (E := E) d)
      ≤ s.norm d.initial + s.exponent d.steps + s.rhs d.initial ∧
    s.rhs (primary_map_analysis_pde_moser_iteration (E := E) d) ≤ s.rhs d.initial + s.rhs d.initial))

 := by
  have hStable := stability_step_analysis_pde_moser_iteration (E := E) d
  rcases hStable with ⟨hNorm, hExpCap⟩
  have hRhs := ((factorization_step_analysis_pde_moser_iteration (E := E) d).1).1
  have _ : s.exponent d.steps ≤ d.cap := hExpCap
  exact ⟨hNorm, hRhs⟩

theorem coherence_step_analysis_pde_moser_iteration
    {E : Type u} [s : FrameworkStruct_analysis_pde_moser_iteration E]
    (d : ContextData_analysis_pde_moser_iteration E) :
    ((fun P : Prop => (fun Q : Prop => P) True) (∃ n : Nat,
      n = d.steps ∧
      secondary_map_analysis_pde_moser_iteration (E := E) d =
        s.norm (primary_map_analysis_pde_moser_iteration (E := E) d) + d.cap))

 := by
  refine ⟨d.steps, rfl, ?_⟩
  rfl

theorem iteration_step_analysis_pde_moser_iteration
    {E : Type u} [s : FrameworkStruct_analysis_pde_moser_iteration E]
    (d : ContextData_analysis_pde_moser_iteration E) :
    ((fun P : Prop => (fun Q : Prop => P) True) (∃ y : E,
      y = primary_map_analysis_pde_moser_iteration (E := E) d ∧
      s.norm y ≤ s.norm d.initial + s.exponent d.steps + s.rhs d.initial ∧
      s.rhs y ≤ s.rhs d.initial + s.rhs d.initial))

 := by
  rcases s.witness_iterate d.steps d.initial with ⟨y, hyEq, hyNorm⟩
  have hyPrimary : y = primary_map_analysis_pde_moser_iteration (E := E) d := by
    simpa [primary_map_analysis_pde_moser_iteration] using hyEq
  have hRhs : s.rhs y ≤ s.rhs d.initial + s.rhs d.initial := by
    have hRaw : s.rhs (s.iterate d.steps d.initial) ≤ s.rhs d.initial + s.rhs d.initial :=
      s.rhs_iterate_bound d.steps d.initial
    simpa [hyEq] using hRaw
  refine ⟨y, hyPrimary, hyNorm, hRhs⟩

theorem main_result_analysis_pde_moser_iteration
    {E : Type u} [s : FrameworkStruct_analysis_pde_moser_iteration E]
    (d : ContextData_analysis_pde_moser_iteration E) :
    ((fun P : Prop => (fun Q : Prop => P) True) (∃ y : E,
      y = primary_map_analysis_pde_moser_iteration (E := E) d ∧
      s.norm (s.improve y) ≤ s.norm y + s.rhs y ∧
      tertiary_map_analysis_pde_moser_iteration (E := E) d))

 := by
  rcases iteration_step_analysis_pde_moser_iteration (E := E) d with ⟨y, hyEq, hyNorm, hyRhs⟩
  have hImprove : s.norm (s.improve y) ≤ s.norm y + s.rhs y :=
    s.norm_improve y
  have hTer := comparison_step_analysis_pde_moser_iteration (E := E) d
  have _ : s.norm y ≤ s.norm d.initial + s.exponent d.steps + s.rhs d.initial := hyNorm
  have _ : s.rhs y ≤ s.rhs d.initial + s.rhs d.initial := hyRhs
  exact ⟨y, hyEq, hImprove, hTer⟩
