/-
BENCHMARK_ID: TINY_MATHLIB_BATCH07_PROBABILITY_STOCHASTIC_CONTROL_DYNAMIC_PROGRAMMING_REFINED_LIKE
PAIR_STEM: probability_stochastic_control_dynamic_programming_refined_like
MATH_DOMAIN: Probability
SOURCE_MATHLIB: Mathlib/Probability/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class FrameworkStruct_probability_stochastic_control_dynamic_programming_refined (Omega : Type u) where
  cost : Omega -> Nat
  value : Omega -> Nat
  advance : Omega -> Omega
  mix : Omega -> Omega -> Omega
  origin : Omega
  mix_origin_left : forall x : Omega, mix origin x = x
  mix_origin_right : forall x : Omega, mix x origin = x
  value_advance : forall x : Omega, value (advance x) = value x
  cost_mix : forall x y : Omega, cost (mix x y) = cost x + cost y
  bellman_like : forall x y : Omega, value (mix x y) = cost x + value y
  value_mix_symm : forall x y : Omega, value (mix x y) = value (mix y x)

structure ContextData_probability_stochastic_control_dynamic_programming_refined
    (Omega : Type u) [FrameworkStruct_probability_stochastic_control_dynamic_programming_refined Omega] where
  state : Omega
  policy : Omega
  compat :
    FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.advance
      (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.mix state policy) =
    FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.mix
      (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.advance state)
      (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.advance policy)

def primary_map_probability_stochastic_control_dynamic_programming_refined
    {Omega : Type u} [FrameworkStruct_probability_stochastic_control_dynamic_programming_refined Omega]
    (ctx : ContextData_probability_stochastic_control_dynamic_programming_refined Omega) : Nat :=
  FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value
    (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.mix ctx.state ctx.policy)

def secondary_map_probability_stochastic_control_dynamic_programming_refined
    {Omega : Type u} [FrameworkStruct_probability_stochastic_control_dynamic_programming_refined Omega]
    (ctx : ContextData_probability_stochastic_control_dynamic_programming_refined Omega) : Nat :=
  FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.cost
    (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.mix ctx.state ctx.policy)

def tertiary_map_probability_stochastic_control_dynamic_programming_refined
    {Omega : Type u} [FrameworkStruct_probability_stochastic_control_dynamic_programming_refined Omega]
    (ctx : ContextData_probability_stochastic_control_dynamic_programming_refined Omega) : Nat :=
  FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value
    (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.advance
      (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.mix ctx.state
        (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.advance ctx.policy)))

theorem stability_step_probability_stochastic_control_dynamic_programming_refined
    {Omega : Type u} [FrameworkStruct_probability_stochastic_control_dynamic_programming_refined Omega]
    (ctx : ContextData_probability_stochastic_control_dynamic_programming_refined Omega)
    (hneq :
      primary_map_probability_stochastic_control_dynamic_programming_refined ctx ≠
      FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value
        (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.mix ctx.policy ctx.state)) :
    False := by
  have hsym :
      FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value
        (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.mix ctx.state ctx.policy) =
      FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value
        (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.mix ctx.policy ctx.state) :=
    FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value_mix_symm _ _
  have hEq :
      primary_map_probability_stochastic_control_dynamic_programming_refined ctx =
      FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value
        (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.mix ctx.policy ctx.state) := by
    unfold primary_map_probability_stochastic_control_dynamic_programming_refined
    exact hsym
  exact hneq hEq

theorem factorization_step_probability_stochastic_control_dynamic_programming_refined
    {Omega : Type u} [FrameworkStruct_probability_stochastic_control_dynamic_programming_refined Omega]
    (ctx : ContextData_probability_stochastic_control_dynamic_programming_refined Omega) :
    primary_map_probability_stochastic_control_dynamic_programming_refined ctx =
      FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.cost ctx.state +
      FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value ctx.policy := by
  have hbell :
      FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value
        (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.mix ctx.state ctx.policy) =
      FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.cost ctx.state +
      FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value ctx.policy :=
    FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.bellman_like _ _
  unfold primary_map_probability_stochastic_control_dynamic_programming_refined
  exact hbell

theorem comparison_step_probability_stochastic_control_dynamic_programming_refined
    {Omega : Type u} [FrameworkStruct_probability_stochastic_control_dynamic_programming_refined Omega]
    (ctx : ContextData_probability_stochastic_control_dynamic_programming_refined Omega) :
    primary_map_probability_stochastic_control_dynamic_programming_refined ctx =
      FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value
        (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.mix ctx.policy ctx.state) /\
    exists z : Omega,
      FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value z =
        primary_map_probability_stochastic_control_dynamic_programming_refined ctx /\
      FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.cost z =
        secondary_map_probability_stochastic_control_dynamic_programming_refined ctx := by
  refine And.intro ?hEq ?hEx
  · unfold primary_map_probability_stochastic_control_dynamic_programming_refined
    exact FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value_mix_symm _ _
  · refine Exists.intro
      (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.mix ctx.state ctx.policy) ?_
    refine And.intro ?h1 ?h2
    · rfl
    · unfold secondary_map_probability_stochastic_control_dynamic_programming_refined
      rfl

theorem transport_step_probability_stochastic_control_dynamic_programming_refined
    {Omega : Type u} [FrameworkStruct_probability_stochastic_control_dynamic_programming_refined Omega]
    (ctx : ContextData_probability_stochastic_control_dynamic_programming_refined Omega)
    (htransport : forall z : Omega,
      FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value z =
        primary_map_probability_stochastic_control_dynamic_programming_refined ctx ->
      FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value
        (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.advance z) =
      tertiary_map_probability_stochastic_control_dynamic_programming_refined ctx) :
    exists z : Omega,
      FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value
        (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.advance z) =
      tertiary_map_probability_stochastic_control_dynamic_programming_refined ctx := by
  refine Exists.intro
    (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.mix ctx.state
      (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.advance ctx.policy)) ?_
  have hz :
      FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value
        (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.mix ctx.state
          (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.advance ctx.policy)) =
      primary_map_probability_stochastic_control_dynamic_programming_refined ctx := by
    unfold primary_map_probability_stochastic_control_dynamic_programming_refined
    have hv :
        FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value
          (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.advance ctx.policy) =
        FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value ctx.policy :=
      FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value_advance _
    have hb1 := FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.bellman_like ctx.state
      (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.advance ctx.policy)
    have hb2 := FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.bellman_like ctx.state ctx.policy
    calc
      FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value
          (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.mix ctx.state
            (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.advance ctx.policy))
          = FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.cost ctx.state +
            FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value
              (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.advance ctx.policy) := hb1
      _ = FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.cost ctx.state +
            FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value ctx.policy := by rw [hv]
      _ = FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value
            (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.mix ctx.state ctx.policy) := by
              symm
              exact hb2
  exact htransport _ hz

theorem coherence_step_probability_stochastic_control_dynamic_programming_refined
    {Omega : Type u} [FrameworkStruct_probability_stochastic_control_dynamic_programming_refined Omega]
    (ctx : ContextData_probability_stochastic_control_dynamic_programming_refined Omega)
    (hmono : tertiary_map_probability_stochastic_control_dynamic_programming_refined ctx =
      FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value
        (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.mix
          (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.advance ctx.state)
          (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.advance ctx.policy))) :
    tertiary_map_probability_stochastic_control_dynamic_programming_refined ctx =
      primary_map_probability_stochastic_control_dynamic_programming_refined ctx := by
  unfold tertiary_map_probability_stochastic_control_dynamic_programming_refined at hmono
  have hstep :
      FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value
        (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.advance
          (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.mix ctx.state
            (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.advance ctx.policy))) =
      FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value
        (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.mix ctx.state
          (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.advance ctx.policy)) :=
    FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value_advance _
  have hcompat :
      FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.advance
        (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.mix ctx.state ctx.policy) =
      FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.mix
        (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.advance ctx.state)
        (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.advance ctx.policy) := ctx.compat
  have hvalueCompat :
      FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value
        (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.advance
          (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.mix ctx.state ctx.policy)) =
      FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value
        (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.mix
          (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.advance ctx.state)
          (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.advance ctx.policy)) := by
    rw [hcompat]
  have hfix :
      FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value
        (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.mix ctx.state ctx.policy) =
      FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value
        (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.advance
          (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.mix ctx.state ctx.policy)) := by
    symm
    exact FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value_advance _
  calc
    tertiary_map_probability_stochastic_control_dynamic_programming_refined ctx
        = FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value
            (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.mix
              (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.advance ctx.state)
              (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.advance ctx.policy)) := hmono
    _ = FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value
          (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.advance
            (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.mix ctx.state ctx.policy)) := by
              symm
              exact hvalueCompat
    _ = FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value
          (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.mix ctx.state ctx.policy) := by
            symm
            exact hfix
    _ = primary_map_probability_stochastic_control_dynamic_programming_refined ctx := by
          rfl

theorem iteration_step_probability_stochastic_control_dynamic_programming_refined
    {Omega : Type u} [FrameworkStruct_probability_stochastic_control_dynamic_programming_refined Omega]
    (ctx : ContextData_probability_stochastic_control_dynamic_programming_refined Omega) :
    exists z : Omega,
      FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value z =
        primary_map_probability_stochastic_control_dynamic_programming_refined ctx /\
      FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value
        (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.advance z) =
      FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value z /\
      FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.mix z
        FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.origin = z := by
  refine Exists.intro
    (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.mix ctx.state ctx.policy) ?_
  refine And.intro ?hVal ?hRest
  · rfl
  · refine And.intro ?hAdv ?hRight
    · exact FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value_advance _
    · exact FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.mix_origin_right _

theorem main_result_probability_stochastic_control_dynamic_programming_refined
    {Omega : Type u} [FrameworkStruct_probability_stochastic_control_dynamic_programming_refined Omega]
    (ctx : ContextData_probability_stochastic_control_dynamic_programming_refined Omega) :
    primary_map_probability_stochastic_control_dynamic_programming_refined ctx =
      FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value
        (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.mix ctx.policy ctx.state) /\
    exists z : Omega,
      FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value z =
        primary_map_probability_stochastic_control_dynamic_programming_refined ctx /\
      tertiary_map_probability_stochastic_control_dynamic_programming_refined ctx =
        FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value
          (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.advance z) := by
  have hbase :
      primary_map_probability_stochastic_control_dynamic_programming_refined ctx =
      FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value
        (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.mix ctx.policy ctx.state) :=
    (comparison_step_probability_stochastic_control_dynamic_programming_refined ctx).1
  refine And.intro hbase ?_
  refine Exists.intro
    (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.mix ctx.state
      (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.advance ctx.policy)) ?_
  refine And.intro ?hVal ?hTer
  · unfold primary_map_probability_stochastic_control_dynamic_programming_refined
    have hv :
        FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value
          (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.advance ctx.policy) =
        FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value ctx.policy :=
      FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value_advance _
    have hb1 := FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.bellman_like ctx.state
      (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.advance ctx.policy)
    have hb2 := FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.bellman_like ctx.state ctx.policy
    calc
      FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value
          (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.mix ctx.state
            (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.advance ctx.policy))
          = FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.cost ctx.state +
            FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value
              (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.advance ctx.policy) := hb1
      _ = FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.cost ctx.state +
            FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value ctx.policy := by rw [hv]
      _ = FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.value
            (FrameworkStruct_probability_stochastic_control_dynamic_programming_refined.mix ctx.state ctx.policy) := by
              symm
              exact hb2
  · unfold tertiary_map_probability_stochastic_control_dynamic_programming_refined
    rfl
