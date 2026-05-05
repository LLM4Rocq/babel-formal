/-
BENCHMARK_ID: TINY_MATHLIB_BATCH07_PROBABILITY_INTERACTING_PARTICLE_HYDRODYNAMIC_LIMIT_LIKE
PAIR_STEM: probability_interacting_particle_hydrodynamic_limit_like
MATH_DOMAIN: Probability
SOURCE_MATHLIB: Mathlib/Probability/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

class FrameworkStruct_probability_interacting_particle_hydrodynamic_limit where
  flux : Nat -> Nat
  density : Nat -> Nat
  push : Nat -> Nat
  push_zero : push 0 = 0
  flux_push : forall n : Nat, flux (push n) = flux n
  density_flux : forall n : Nat, density n = flux n
  push_succ : forall n : Nat, push (n + 1) = push n + 1

structure ContextData_probability_interacting_particle_hydrodynamic_limit
    [FrameworkStruct_probability_interacting_particle_hydrodynamic_limit] where
  i : Nat
  j : Nat
  hij : i = FrameworkStruct_probability_interacting_particle_hydrodynamic_limit.push j

def primary_map_probability_interacting_particle_hydrodynamic_limit
    [FrameworkStruct_probability_interacting_particle_hydrodynamic_limit]
    (ctx : ContextData_probability_interacting_particle_hydrodynamic_limit) : Nat :=
  FrameworkStruct_probability_interacting_particle_hydrodynamic_limit.flux ctx.i

def secondary_map_probability_interacting_particle_hydrodynamic_limit
    [FrameworkStruct_probability_interacting_particle_hydrodynamic_limit]
    (ctx : ContextData_probability_interacting_particle_hydrodynamic_limit) : Nat :=
  FrameworkStruct_probability_interacting_particle_hydrodynamic_limit.density
    (FrameworkStruct_probability_interacting_particle_hydrodynamic_limit.push ctx.j)

def tertiary_map_probability_interacting_particle_hydrodynamic_limit
    [FrameworkStruct_probability_interacting_particle_hydrodynamic_limit]
    (ctx : ContextData_probability_interacting_particle_hydrodynamic_limit) : Nat :=
  FrameworkStruct_probability_interacting_particle_hydrodynamic_limit.flux
    (FrameworkStruct_probability_interacting_particle_hydrodynamic_limit.push
      (FrameworkStruct_probability_interacting_particle_hydrodynamic_limit.push ctx.j))

theorem stability_step_probability_interacting_particle_hydrodynamic_limit
    [FrameworkStruct_probability_interacting_particle_hydrodynamic_limit]
    (ctx : ContextData_probability_interacting_particle_hydrodynamic_limit)
    (hneq : primary_map_probability_interacting_particle_hydrodynamic_limit ctx ≠
      secondary_map_probability_interacting_particle_hydrodynamic_limit ctx) :
    False /\
    primary_map_probability_interacting_particle_hydrodynamic_limit ctx =
      secondary_map_probability_interacting_particle_hydrodynamic_limit ctx := by
  have hprim :
      primary_map_probability_interacting_particle_hydrodynamic_limit ctx =
      FrameworkStruct_probability_interacting_particle_hydrodynamic_limit.flux
        (FrameworkStruct_probability_interacting_particle_hydrodynamic_limit.push ctx.j) := by
    unfold primary_map_probability_interacting_particle_hydrodynamic_limit
    rw [ctx.hij]
  have hsec :
      secondary_map_probability_interacting_particle_hydrodynamic_limit ctx =
      FrameworkStruct_probability_interacting_particle_hydrodynamic_limit.flux
        (FrameworkStruct_probability_interacting_particle_hydrodynamic_limit.push ctx.j) := by
    unfold secondary_map_probability_interacting_particle_hydrodynamic_limit
    rw [FrameworkStruct_probability_interacting_particle_hydrodynamic_limit.density_flux]
  have hEq :
      primary_map_probability_interacting_particle_hydrodynamic_limit ctx =
      secondary_map_probability_interacting_particle_hydrodynamic_limit ctx := by
    calc
      primary_map_probability_interacting_particle_hydrodynamic_limit ctx
          = FrameworkStruct_probability_interacting_particle_hydrodynamic_limit.flux
              (FrameworkStruct_probability_interacting_particle_hydrodynamic_limit.push ctx.j) := hprim
      _ = secondary_map_probability_interacting_particle_hydrodynamic_limit ctx := by
            symm
            exact hsec
  exact And.intro (hneq hEq) hEq

theorem factorization_step_probability_interacting_particle_hydrodynamic_limit
    [FrameworkStruct_probability_interacting_particle_hydrodynamic_limit]
    (ctx : ContextData_probability_interacting_particle_hydrodynamic_limit) :
    tertiary_map_probability_interacting_particle_hydrodynamic_limit ctx =
      secondary_map_probability_interacting_particle_hydrodynamic_limit ctx := by
  unfold tertiary_map_probability_interacting_particle_hydrodynamic_limit
  unfold secondary_map_probability_interacting_particle_hydrodynamic_limit
  rw [FrameworkStruct_probability_interacting_particle_hydrodynamic_limit.flux_push]
  rw [FrameworkStruct_probability_interacting_particle_hydrodynamic_limit.density_flux]

theorem comparison_step_probability_interacting_particle_hydrodynamic_limit
    [FrameworkStruct_probability_interacting_particle_hydrodynamic_limit]
    (ctx : ContextData_probability_interacting_particle_hydrodynamic_limit) :
    exists k : Nat,
      FrameworkStruct_probability_interacting_particle_hydrodynamic_limit.flux k =
        secondary_map_probability_interacting_particle_hydrodynamic_limit ctx /\
      primary_map_probability_interacting_particle_hydrodynamic_limit ctx =
        secondary_map_probability_interacting_particle_hydrodynamic_limit ctx := by
  have hEq :
      primary_map_probability_interacting_particle_hydrodynamic_limit ctx =
      secondary_map_probability_interacting_particle_hydrodynamic_limit ctx := by
    by_cases h :
      primary_map_probability_interacting_particle_hydrodynamic_limit ctx =
      secondary_map_probability_interacting_particle_hydrodynamic_limit ctx
    · exact h
    · exact False.elim (stability_step_probability_interacting_particle_hydrodynamic_limit ctx h).1
  refine Exists.intro
    (FrameworkStruct_probability_interacting_particle_hydrodynamic_limit.push ctx.j) ?_
  refine And.intro ?hflux hEq
  unfold secondary_map_probability_interacting_particle_hydrodynamic_limit
  rw [FrameworkStruct_probability_interacting_particle_hydrodynamic_limit.density_flux]

theorem transport_step_probability_interacting_particle_hydrodynamic_limit
    [FrameworkStruct_probability_interacting_particle_hydrodynamic_limit]
    (ctx : ContextData_probability_interacting_particle_hydrodynamic_limit)
    (htrans : forall k : Nat,
      FrameworkStruct_probability_interacting_particle_hydrodynamic_limit.flux k =
        secondary_map_probability_interacting_particle_hydrodynamic_limit ctx ->
      FrameworkStruct_probability_interacting_particle_hydrodynamic_limit.flux
        (FrameworkStruct_probability_interacting_particle_hydrodynamic_limit.push k) =
        secondary_map_probability_interacting_particle_hydrodynamic_limit ctx) :
    FrameworkStruct_probability_interacting_particle_hydrodynamic_limit.flux
      (FrameworkStruct_probability_interacting_particle_hydrodynamic_limit.push
        (FrameworkStruct_probability_interacting_particle_hydrodynamic_limit.push ctx.j)) =
      secondary_map_probability_interacting_particle_hydrodynamic_limit ctx := by
  have hk :
      FrameworkStruct_probability_interacting_particle_hydrodynamic_limit.flux
        (FrameworkStruct_probability_interacting_particle_hydrodynamic_limit.push ctx.j) =
      secondary_map_probability_interacting_particle_hydrodynamic_limit ctx := by
    unfold secondary_map_probability_interacting_particle_hydrodynamic_limit
    rw [FrameworkStruct_probability_interacting_particle_hydrodynamic_limit.density_flux]
  exact htrans _ hk

theorem coherence_step_probability_interacting_particle_hydrodynamic_limit
    [FrameworkStruct_probability_interacting_particle_hydrodynamic_limit]
    (ctx : ContextData_probability_interacting_particle_hydrodynamic_limit)
    (hno : (forall k : Nat,
      FrameworkStruct_probability_interacting_particle_hydrodynamic_limit.flux k ≠
        secondary_map_probability_interacting_particle_hydrodynamic_limit ctx) -> False) :
    exists k : Nat,
      FrameworkStruct_probability_interacting_particle_hydrodynamic_limit.flux k =
      secondary_map_probability_interacting_particle_hydrodynamic_limit ctx /\ True := by
  refine Exists.intro
    (FrameworkStruct_probability_interacting_particle_hydrodynamic_limit.push ctx.j) ?_
  refine And.intro ?hflux ?htrue
  unfold secondary_map_probability_interacting_particle_hydrodynamic_limit
  rw [FrameworkStruct_probability_interacting_particle_hydrodynamic_limit.density_flux]
  · exact True.intro

theorem iteration_step_probability_interacting_particle_hydrodynamic_limit
    [FrameworkStruct_probability_interacting_particle_hydrodynamic_limit]
    (ctx : ContextData_probability_interacting_particle_hydrodynamic_limit) :
    exists k : Nat,
      FrameworkStruct_probability_interacting_particle_hydrodynamic_limit.push k = 0 /\
      FrameworkStruct_probability_interacting_particle_hydrodynamic_limit.flux
        (FrameworkStruct_probability_interacting_particle_hydrodynamic_limit.push k) =
      FrameworkStruct_probability_interacting_particle_hydrodynamic_limit.flux 0 := by
  refine Exists.intro 0 ?_
  refine And.intro ?hpush ?hflux
  · exact FrameworkStruct_probability_interacting_particle_hydrodynamic_limit.push_zero
  · rw [FrameworkStruct_probability_interacting_particle_hydrodynamic_limit.push_zero]

theorem main_result_probability_interacting_particle_hydrodynamic_limit
    [FrameworkStruct_probability_interacting_particle_hydrodynamic_limit]
    (ctx : ContextData_probability_interacting_particle_hydrodynamic_limit) :
    exists k : Nat,
      (FrameworkStruct_probability_interacting_particle_hydrodynamic_limit.flux k =
        secondary_map_probability_interacting_particle_hydrodynamic_limit ctx /\
      tertiary_map_probability_interacting_particle_hydrodynamic_limit ctx =
        FrameworkStruct_probability_interacting_particle_hydrodynamic_limit.flux
          (FrameworkStruct_probability_interacting_particle_hydrodynamic_limit.push k)) /\
      primary_map_probability_interacting_particle_hydrodynamic_limit ctx =
        secondary_map_probability_interacting_particle_hydrodynamic_limit ctx := by
  have hEq :
      primary_map_probability_interacting_particle_hydrodynamic_limit ctx =
      secondary_map_probability_interacting_particle_hydrodynamic_limit ctx := by
    rcases comparison_step_probability_interacting_particle_hydrodynamic_limit ctx with ⟨k, hk1, hk2⟩
    exact hk2
  refine Exists.intro
    (FrameworkStruct_probability_interacting_particle_hydrodynamic_limit.push ctx.j) ?_
  refine And.intro ?hpair hEq
  refine And.intro ?hk ?hter
  · unfold secondary_map_probability_interacting_particle_hydrodynamic_limit
    rw [FrameworkStruct_probability_interacting_particle_hydrodynamic_limit.density_flux]
  · unfold tertiary_map_probability_interacting_particle_hydrodynamic_limit
    rfl
