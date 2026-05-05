/-
BENCHMARK_ID: TINY_MATHLIB_BATCH07_PROBABILITY_LARGE_DEVIATION_QUASIPOTENTIAL_LIKE
PAIR_STEM: probability_large_deviation_quasipotential_like
MATH_DOMAIN: Probability
SOURCE_MATHLIB: Mathlib/Probability/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

class FrameworkStruct_probability_large_deviation_quasipotential where
  rate : Nat -> Nat
  quasi : Nat -> Nat
  jump : Nat -> Nat
  rate_jump : forall n : Nat, rate (jump n) = rate n
  quasi_rate : forall n : Nat, quasi n = rate n
  jump_idem : forall n : Nat, jump (jump n) = jump n
  jump_zero : jump 0 = 0

structure ContextData_probability_large_deviation_quasipotential
    [FrameworkStruct_probability_large_deviation_quasipotential] where
  a : Nat
  b : Nat
  hab : FrameworkStruct_probability_large_deviation_quasipotential.jump a = b

def primary_map_probability_large_deviation_quasipotential
    [FrameworkStruct_probability_large_deviation_quasipotential]
    (ctx : ContextData_probability_large_deviation_quasipotential) : Nat :=
  FrameworkStruct_probability_large_deviation_quasipotential.rate
    (FrameworkStruct_probability_large_deviation_quasipotential.jump ctx.a)

def secondary_map_probability_large_deviation_quasipotential
    [FrameworkStruct_probability_large_deviation_quasipotential]
    (ctx : ContextData_probability_large_deviation_quasipotential) : Nat :=
  FrameworkStruct_probability_large_deviation_quasipotential.quasi ctx.b

def tertiary_map_probability_large_deviation_quasipotential
    [FrameworkStruct_probability_large_deviation_quasipotential]
    (ctx : ContextData_probability_large_deviation_quasipotential) : Nat :=
  FrameworkStruct_probability_large_deviation_quasipotential.rate
    (FrameworkStruct_probability_large_deviation_quasipotential.jump ctx.b)

theorem stability_step_probability_large_deviation_quasipotential
    [FrameworkStruct_probability_large_deviation_quasipotential]
    (ctx : ContextData_probability_large_deviation_quasipotential)
    (hneq : primary_map_probability_large_deviation_quasipotential ctx ≠
      secondary_map_probability_large_deviation_quasipotential ctx)
    (hextra : ctx.a = ctx.a) :
    False := by
  have _ : ctx.a = ctx.a := hextra
  have h1 :
      primary_map_probability_large_deviation_quasipotential ctx =
      FrameworkStruct_probability_large_deviation_quasipotential.rate ctx.b := by
    unfold primary_map_probability_large_deviation_quasipotential
    rw [ctx.hab]
  have h2 :
      secondary_map_probability_large_deviation_quasipotential ctx =
      FrameworkStruct_probability_large_deviation_quasipotential.rate ctx.b := by
    unfold secondary_map_probability_large_deviation_quasipotential
    rw [FrameworkStruct_probability_large_deviation_quasipotential.quasi_rate]
  have hEq :
      primary_map_probability_large_deviation_quasipotential ctx =
      secondary_map_probability_large_deviation_quasipotential ctx := by
    calc
      primary_map_probability_large_deviation_quasipotential ctx
          = FrameworkStruct_probability_large_deviation_quasipotential.rate ctx.b := h1
      _ = secondary_map_probability_large_deviation_quasipotential ctx := by
            symm
            exact h2
  exact hneq hEq

theorem factorization_step_probability_large_deviation_quasipotential
    [FrameworkStruct_probability_large_deviation_quasipotential]
    (ctx : ContextData_probability_large_deviation_quasipotential) :
    tertiary_map_probability_large_deviation_quasipotential ctx =
      FrameworkStruct_probability_large_deviation_quasipotential.rate ctx.b := by
  unfold tertiary_map_probability_large_deviation_quasipotential
  exact FrameworkStruct_probability_large_deviation_quasipotential.rate_jump _

theorem comparison_step_probability_large_deviation_quasipotential
    [FrameworkStruct_probability_large_deviation_quasipotential]
    (ctx : ContextData_probability_large_deviation_quasipotential) :
    (exists k : Nat,
      FrameworkStruct_probability_large_deviation_quasipotential.rate k =
        secondary_map_probability_large_deviation_quasipotential ctx) /\
    primary_map_probability_large_deviation_quasipotential ctx =
      secondary_map_probability_large_deviation_quasipotential ctx := by
  have hEq :
      primary_map_probability_large_deviation_quasipotential ctx =
      secondary_map_probability_large_deviation_quasipotential ctx := by
    by_cases h :
      primary_map_probability_large_deviation_quasipotential ctx =
      secondary_map_probability_large_deviation_quasipotential ctx
    · exact h
    · exact False.elim (stability_step_probability_large_deviation_quasipotential ctx h rfl)
  refine And.intro ?hex hEq
  refine Exists.intro ctx.b ?_
  unfold secondary_map_probability_large_deviation_quasipotential
  rw [FrameworkStruct_probability_large_deviation_quasipotential.quasi_rate]

theorem transport_step_probability_large_deviation_quasipotential
    [FrameworkStruct_probability_large_deviation_quasipotential]
    (ctx : ContextData_probability_large_deviation_quasipotential)
    (htransport : forall k : Nat,
      FrameworkStruct_probability_large_deviation_quasipotential.rate k =
        secondary_map_probability_large_deviation_quasipotential ctx ->
      FrameworkStruct_probability_large_deviation_quasipotential.quasi k =
        secondary_map_probability_large_deviation_quasipotential ctx) :
    FrameworkStruct_probability_large_deviation_quasipotential.quasi
      (FrameworkStruct_probability_large_deviation_quasipotential.jump ctx.a) =
      secondary_map_probability_large_deviation_quasipotential ctx := by
  have hk :
      FrameworkStruct_probability_large_deviation_quasipotential.rate
        (FrameworkStruct_probability_large_deviation_quasipotential.jump ctx.a) =
      secondary_map_probability_large_deviation_quasipotential ctx := by
    unfold secondary_map_probability_large_deviation_quasipotential
    rw [FrameworkStruct_probability_large_deviation_quasipotential.quasi_rate]
    rw [<- ctx.hab]
  exact htransport _ hk

theorem coherence_step_probability_large_deviation_quasipotential
    [FrameworkStruct_probability_large_deviation_quasipotential]
    (ctx : ContextData_probability_large_deviation_quasipotential)
    (hneg : (forall k : Nat,
      FrameworkStruct_probability_large_deviation_quasipotential.quasi k ≠
      secondary_map_probability_large_deviation_quasipotential ctx) -> False) :
    exists k : Nat,
      FrameworkStruct_probability_large_deviation_quasipotential.quasi k =
      secondary_map_probability_large_deviation_quasipotential ctx /\ k = ctx.b := by
  refine Exists.intro ctx.b ?_
  refine And.intro ?hEq ?hId
  · unfold secondary_map_probability_large_deviation_quasipotential
    rfl
  · rfl

theorem iteration_step_probability_large_deviation_quasipotential
    [FrameworkStruct_probability_large_deviation_quasipotential]
    (ctx : ContextData_probability_large_deviation_quasipotential) :
    exists k : Nat,
      FrameworkStruct_probability_large_deviation_quasipotential.jump k = k /\
      FrameworkStruct_probability_large_deviation_quasipotential.rate
        (FrameworkStruct_probability_large_deviation_quasipotential.jump k) =
      FrameworkStruct_probability_large_deviation_quasipotential.rate k := by
  refine Exists.intro
    (FrameworkStruct_probability_large_deviation_quasipotential.jump ctx.a) ?_
  refine And.intro ?hfix ?hr
  · exact FrameworkStruct_probability_large_deviation_quasipotential.jump_idem _
  · exact FrameworkStruct_probability_large_deviation_quasipotential.rate_jump _

theorem main_result_probability_large_deviation_quasipotential
    [FrameworkStruct_probability_large_deviation_quasipotential]
    (ctx : ContextData_probability_large_deviation_quasipotential) :
    primary_map_probability_large_deviation_quasipotential ctx =
      secondary_map_probability_large_deviation_quasipotential ctx /\
    exists k : Nat,
      tertiary_map_probability_large_deviation_quasipotential ctx =
        FrameworkStruct_probability_large_deviation_quasipotential.rate
          (FrameworkStruct_probability_large_deviation_quasipotential.jump k) /\
      FrameworkStruct_probability_large_deviation_quasipotential.rate k =
        secondary_map_probability_large_deviation_quasipotential ctx /\
      k = ctx.b := by
  have hEq := (comparison_step_probability_large_deviation_quasipotential ctx).2
  refine And.intro hEq ?_
  refine Exists.intro ctx.b ?_
  refine And.intro ?hter ?hrest
  · unfold tertiary_map_probability_large_deviation_quasipotential
    rfl
  · refine And.intro ?hr ?hid
    · unfold secondary_map_probability_large_deviation_quasipotential
      rw [FrameworkStruct_probability_large_deviation_quasipotential.quasi_rate]
    · rfl
