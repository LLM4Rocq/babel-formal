/-
BENCHMARK_ID: TINY_MATHLIB_BATCH07_PROBABILITY_RANDOM_MATRIX_LOCAL_LAW_LIKE
PAIR_STEM: probability_random_matrix_local_law_like
MATH_DOMAIN: Probability
SOURCE_MATHLIB: Mathlib/Probability/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

class FrameworkStruct_probability_random_matrix_local_law where
  energy : Nat -> Nat
  drift : Nat -> Nat
  step : Nat -> Nat
  monotone_energy : forall n m : Nat, n = m -> energy n = energy m
  drift_step : forall n : Nat, drift (step n) = drift n
  energy_drift : forall n : Nat, energy n = drift n
  step_idem : forall n : Nat, step (step n) = step n

structure ContextData_probability_random_matrix_local_law
    [FrameworkStruct_probability_random_matrix_local_law] where
  n : Nat
  m : Nat
  hnm : n = m

def primary_map_probability_random_matrix_local_law
    [FrameworkStruct_probability_random_matrix_local_law]
    (ctx : ContextData_probability_random_matrix_local_law) : Nat :=
  FrameworkStruct_probability_random_matrix_local_law.energy
    (FrameworkStruct_probability_random_matrix_local_law.step ctx.n)

def secondary_map_probability_random_matrix_local_law
    [FrameworkStruct_probability_random_matrix_local_law]
    (ctx : ContextData_probability_random_matrix_local_law) : Nat :=
  FrameworkStruct_probability_random_matrix_local_law.drift
    (FrameworkStruct_probability_random_matrix_local_law.step ctx.m)

def tertiary_map_probability_random_matrix_local_law
    [FrameworkStruct_probability_random_matrix_local_law]
    (ctx : ContextData_probability_random_matrix_local_law) : Nat :=
  FrameworkStruct_probability_random_matrix_local_law.energy
    (FrameworkStruct_probability_random_matrix_local_law.step
      (FrameworkStruct_probability_random_matrix_local_law.step ctx.n))

theorem stability_step_probability_random_matrix_local_law
    [FrameworkStruct_probability_random_matrix_local_law]
    (ctx : ContextData_probability_random_matrix_local_law)
    (hneq : primary_map_probability_random_matrix_local_law ctx ≠
      secondary_map_probability_random_matrix_local_law ctx) :
    False := by
  have hstepEq :
      FrameworkStruct_probability_random_matrix_local_law.step ctx.n =
      FrameworkStruct_probability_random_matrix_local_law.step ctx.m := by
    rw [ctx.hnm]
  have hEnergy :
      FrameworkStruct_probability_random_matrix_local_law.energy
          (FrameworkStruct_probability_random_matrix_local_law.step ctx.n) =
      FrameworkStruct_probability_random_matrix_local_law.energy
          (FrameworkStruct_probability_random_matrix_local_law.step ctx.m) :=
    FrameworkStruct_probability_random_matrix_local_law.monotone_energy _ _ hstepEq
  have hBridge :
      FrameworkStruct_probability_random_matrix_local_law.energy
          (FrameworkStruct_probability_random_matrix_local_law.step ctx.m) =
      FrameworkStruct_probability_random_matrix_local_law.drift
          (FrameworkStruct_probability_random_matrix_local_law.step ctx.m) :=
    FrameworkStruct_probability_random_matrix_local_law.energy_drift _
  have hEq :
      primary_map_probability_random_matrix_local_law ctx =
      secondary_map_probability_random_matrix_local_law ctx := by
    unfold primary_map_probability_random_matrix_local_law
    unfold secondary_map_probability_random_matrix_local_law
    calc
      FrameworkStruct_probability_random_matrix_local_law.energy
          (FrameworkStruct_probability_random_matrix_local_law.step ctx.n)
          = FrameworkStruct_probability_random_matrix_local_law.energy
              (FrameworkStruct_probability_random_matrix_local_law.step ctx.m) := hEnergy
      _ = FrameworkStruct_probability_random_matrix_local_law.drift
            (FrameworkStruct_probability_random_matrix_local_law.step ctx.m) := hBridge
  exact hneq hEq

theorem factorization_step_probability_random_matrix_local_law
    [FrameworkStruct_probability_random_matrix_local_law]
    (ctx : ContextData_probability_random_matrix_local_law) :
    tertiary_map_probability_random_matrix_local_law ctx =
      primary_map_probability_random_matrix_local_law ctx := by
  unfold tertiary_map_probability_random_matrix_local_law
  unfold primary_map_probability_random_matrix_local_law
  rw [FrameworkStruct_probability_random_matrix_local_law.step_idem]

theorem comparison_step_probability_random_matrix_local_law
    [FrameworkStruct_probability_random_matrix_local_law]
    (ctx : ContextData_probability_random_matrix_local_law) :
    primary_map_probability_random_matrix_local_law ctx =
      secondary_map_probability_random_matrix_local_law ctx /\
      exists k : Nat,
        FrameworkStruct_probability_random_matrix_local_law.energy k =
        primary_map_probability_random_matrix_local_law ctx := by
  refine And.intro ?hEq ?hEx
  · by_cases h :
      primary_map_probability_random_matrix_local_law ctx =
      secondary_map_probability_random_matrix_local_law ctx
    · exact h
    · exact False.elim (stability_step_probability_random_matrix_local_law ctx h)
  · refine Exists.intro
      (FrameworkStruct_probability_random_matrix_local_law.step ctx.n) ?_
    unfold primary_map_probability_random_matrix_local_law
    rfl

theorem transport_step_probability_random_matrix_local_law
    [FrameworkStruct_probability_random_matrix_local_law]
    (ctx : ContextData_probability_random_matrix_local_law)
    (hT : forall k : Nat,
      FrameworkStruct_probability_random_matrix_local_law.energy k =
        primary_map_probability_random_matrix_local_law ctx ->
      FrameworkStruct_probability_random_matrix_local_law.drift k =
        secondary_map_probability_random_matrix_local_law ctx) :
    exists k : Nat,
      FrameworkStruct_probability_random_matrix_local_law.drift k =
      secondary_map_probability_random_matrix_local_law ctx := by
  refine Exists.intro
    (FrameworkStruct_probability_random_matrix_local_law.step ctx.n) ?_
  have hk :
      FrameworkStruct_probability_random_matrix_local_law.energy
        (FrameworkStruct_probability_random_matrix_local_law.step ctx.n) =
      primary_map_probability_random_matrix_local_law ctx := by
    unfold primary_map_probability_random_matrix_local_law
    rfl
  exact hT _ hk

theorem coherence_step_probability_random_matrix_local_law
    [FrameworkStruct_probability_random_matrix_local_law]
    (ctx : ContextData_probability_random_matrix_local_law)
    (hcollapse : (forall k : Nat,
      FrameworkStruct_probability_random_matrix_local_law.drift k ≠
      secondary_map_probability_random_matrix_local_law ctx) -> False) :
    exists k : Nat,
      FrameworkStruct_probability_random_matrix_local_law.drift k =
      secondary_map_probability_random_matrix_local_law ctx := by
  refine Exists.intro
    (FrameworkStruct_probability_random_matrix_local_law.step ctx.m) ?_
  unfold secondary_map_probability_random_matrix_local_law
  rfl

theorem iteration_step_probability_random_matrix_local_law
    [FrameworkStruct_probability_random_matrix_local_law]
    (ctx : ContextData_probability_random_matrix_local_law) :
    exists k : Nat,
      FrameworkStruct_probability_random_matrix_local_law.step k = k /\
      FrameworkStruct_probability_random_matrix_local_law.energy k =
        primary_map_probability_random_matrix_local_law ctx := by
  refine Exists.intro
    (FrameworkStruct_probability_random_matrix_local_law.step ctx.n) ?_
  refine And.intro ?hfix ?hval
  · exact FrameworkStruct_probability_random_matrix_local_law.step_idem _
  · unfold primary_map_probability_random_matrix_local_law
    rfl

theorem main_result_probability_random_matrix_local_law
    [FrameworkStruct_probability_random_matrix_local_law]
    (ctx : ContextData_probability_random_matrix_local_law) :
    primary_map_probability_random_matrix_local_law ctx =
      secondary_map_probability_random_matrix_local_law ctx /\
    exists k : Nat,
      FrameworkStruct_probability_random_matrix_local_law.energy k =
        primary_map_probability_random_matrix_local_law ctx /\
      tertiary_map_probability_random_matrix_local_law ctx =
        FrameworkStruct_probability_random_matrix_local_law.energy
          (FrameworkStruct_probability_random_matrix_local_law.step k) := by
  have hEq : primary_map_probability_random_matrix_local_law ctx =
      secondary_map_probability_random_matrix_local_law ctx :=
    (comparison_step_probability_random_matrix_local_law ctx).1
  refine And.intro hEq ?_
  refine Exists.intro
    (FrameworkStruct_probability_random_matrix_local_law.step ctx.n) ?_
  refine And.intro ?hE ?hT
  · unfold primary_map_probability_random_matrix_local_law
    rfl
  · unfold tertiary_map_probability_random_matrix_local_law
    rw [FrameworkStruct_probability_random_matrix_local_law.step_idem]
