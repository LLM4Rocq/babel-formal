/-
BENCHMARK_ID: TINY_MATHLIB_BATCH07_PROBABILITY_MEAN_FIELD_PROPAGATION_CHAOS_LIKE
PAIR_STEM: probability_mean_field_propagation_chaos_like
MATH_DOMAIN: Probability
SOURCE_MATHLIB: Mathlib/Probability/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class FrameworkStruct_probability_mean_field_propagation_chaos (Omega : Type u) where
  state_val : Omega -> Nat
  evolve : Omega -> Omega
  couple : Omega -> Omega -> Omega
  root : Omega
  good : Omega -> Prop
  good_root : good root
  good_evolve : forall x : Omega, good x -> good (evolve x)
  good_couple_left : forall x y : Omega, good (couple x y) -> good x
  good_couple_right : forall x y : Omega, good (couple x y) -> good y
  good_couple_intro : forall x y : Omega, good x -> good y -> good (couple x y)

structure ContextData_probability_mean_field_propagation_chaos
    (Omega : Type u) [FrameworkStruct_probability_mean_field_propagation_chaos Omega] where
  p : Omega
  q : Omega
  hp : FrameworkStruct_probability_mean_field_propagation_chaos.good p
  hq : FrameworkStruct_probability_mean_field_propagation_chaos.good q

def primary_map_probability_mean_field_propagation_chaos
    {Omega : Type u} [FrameworkStruct_probability_mean_field_propagation_chaos Omega]
    (ctx : ContextData_probability_mean_field_propagation_chaos Omega) : Prop :=
  FrameworkStruct_probability_mean_field_propagation_chaos.good
    (FrameworkStruct_probability_mean_field_propagation_chaos.couple ctx.p ctx.q)

def secondary_map_probability_mean_field_propagation_chaos
    {Omega : Type u} [FrameworkStruct_probability_mean_field_propagation_chaos Omega]
    (ctx : ContextData_probability_mean_field_propagation_chaos Omega) : Prop :=
  FrameworkStruct_probability_mean_field_propagation_chaos.good
    (FrameworkStruct_probability_mean_field_propagation_chaos.evolve
      (FrameworkStruct_probability_mean_field_propagation_chaos.couple ctx.q ctx.p))

def tertiary_map_probability_mean_field_propagation_chaos
    {Omega : Type u} [FrameworkStruct_probability_mean_field_propagation_chaos Omega]
    (ctx : ContextData_probability_mean_field_propagation_chaos Omega) : Prop :=
  FrameworkStruct_probability_mean_field_propagation_chaos.good
    (FrameworkStruct_probability_mean_field_propagation_chaos.couple
      (FrameworkStruct_probability_mean_field_propagation_chaos.evolve ctx.p)
      (FrameworkStruct_probability_mean_field_propagation_chaos.evolve ctx.q))

theorem stability_step_probability_mean_field_propagation_chaos
    {Omega : Type u} [FrameworkStruct_probability_mean_field_propagation_chaos Omega]
    (ctx : ContextData_probability_mean_field_propagation_chaos Omega)
    (hcontr : primary_map_probability_mean_field_propagation_chaos ctx -> False) :
    False := by
  have hp' :
      FrameworkStruct_probability_mean_field_propagation_chaos.good ctx.p := ctx.hp
  have hq' :
      FrameworkStruct_probability_mean_field_propagation_chaos.good ctx.q := ctx.hq
  have hpair :
      primary_map_probability_mean_field_propagation_chaos ctx := by
    unfold primary_map_probability_mean_field_propagation_chaos
    exact FrameworkStruct_probability_mean_field_propagation_chaos.good_couple_intro _ _ hp' hq'
  exact hcontr hpair

theorem factorization_step_probability_mean_field_propagation_chaos
    {Omega : Type u} [FrameworkStruct_probability_mean_field_propagation_chaos Omega]
    (ctx : ContextData_probability_mean_field_propagation_chaos Omega)
    (hprim : primary_map_probability_mean_field_propagation_chaos ctx) :
    FrameworkStruct_probability_mean_field_propagation_chaos.good ctx.p /\
      FrameworkStruct_probability_mean_field_propagation_chaos.good ctx.q := by
  have hleft :
      FrameworkStruct_probability_mean_field_propagation_chaos.good ctx.p := by
    unfold primary_map_probability_mean_field_propagation_chaos at hprim
    exact FrameworkStruct_probability_mean_field_propagation_chaos.good_couple_left _ _ hprim
  have hright :
      FrameworkStruct_probability_mean_field_propagation_chaos.good ctx.q := by
    unfold primary_map_probability_mean_field_propagation_chaos at hprim
    exact FrameworkStruct_probability_mean_field_propagation_chaos.good_couple_right _ _ hprim
  exact And.intro hleft hright

theorem comparison_step_probability_mean_field_propagation_chaos
    {Omega : Type u} [FrameworkStruct_probability_mean_field_propagation_chaos Omega]
    (ctx : ContextData_probability_mean_field_propagation_chaos Omega) :
    primary_map_probability_mean_field_propagation_chaos ctx /\
      exists z : Omega, FrameworkStruct_probability_mean_field_propagation_chaos.good z := by
  have hprim :
      primary_map_probability_mean_field_propagation_chaos ctx := by
    unfold primary_map_probability_mean_field_propagation_chaos
    exact FrameworkStruct_probability_mean_field_propagation_chaos.good_couple_intro _ _ ctx.hp ctx.hq
  refine And.intro hprim ?_
  refine Exists.intro
    (FrameworkStruct_probability_mean_field_propagation_chaos.couple ctx.p ctx.q) ?_
  exact hprim

theorem transport_step_probability_mean_field_propagation_chaos
    {Omega : Type u} [FrameworkStruct_probability_mean_field_propagation_chaos Omega]
    (ctx : ContextData_probability_mean_field_propagation_chaos Omega)
    (hstep : forall z : Omega,
      FrameworkStruct_probability_mean_field_propagation_chaos.good z ->
      FrameworkStruct_probability_mean_field_propagation_chaos.good
        (FrameworkStruct_probability_mean_field_propagation_chaos.evolve z))
    (hprim : primary_map_probability_mean_field_propagation_chaos ctx) :
    secondary_map_probability_mean_field_propagation_chaos ctx := by
  unfold secondary_map_probability_mean_field_propagation_chaos
  have hqp :
      FrameworkStruct_probability_mean_field_propagation_chaos.good
        (FrameworkStruct_probability_mean_field_propagation_chaos.couple ctx.q ctx.p) :=
    FrameworkStruct_probability_mean_field_propagation_chaos.good_couple_intro _ _ ctx.hq ctx.hp
  exact hstep _ hqp

theorem coherence_step_probability_mean_field_propagation_chaos
    {Omega : Type u} [FrameworkStruct_probability_mean_field_propagation_chaos Omega]
    (ctx : ContextData_probability_mean_field_propagation_chaos Omega)
    (hnot : secondary_map_probability_mean_field_propagation_chaos ctx -> False)
    (hstep : forall z : Omega,
      FrameworkStruct_probability_mean_field_propagation_chaos.good z ->
      FrameworkStruct_probability_mean_field_propagation_chaos.good
        (FrameworkStruct_probability_mean_field_propagation_chaos.evolve z)) :
    False := by
  have hsec : secondary_map_probability_mean_field_propagation_chaos ctx :=
    transport_step_probability_mean_field_propagation_chaos ctx hstep
      (FrameworkStruct_probability_mean_field_propagation_chaos.good_couple_intro _ _ ctx.hp ctx.hq)
  exact hnot hsec

theorem iteration_step_probability_mean_field_propagation_chaos
    {Omega : Type u} [FrameworkStruct_probability_mean_field_propagation_chaos Omega]
    (ctx : ContextData_probability_mean_field_propagation_chaos Omega) :
    exists z : Omega,
      FrameworkStruct_probability_mean_field_propagation_chaos.good z /\
      (FrameworkStruct_probability_mean_field_propagation_chaos.good
          (FrameworkStruct_probability_mean_field_propagation_chaos.evolve z) ->
        secondary_map_probability_mean_field_propagation_chaos ctx) := by
  refine Exists.intro
    (FrameworkStruct_probability_mean_field_propagation_chaos.couple ctx.q ctx.p) ?_
  refine And.intro ?hgood ?himpl
  · exact FrameworkStruct_probability_mean_field_propagation_chaos.good_couple_intro _ _ ctx.hq ctx.hp
  · intro _
    unfold secondary_map_probability_mean_field_propagation_chaos
    exact FrameworkStruct_probability_mean_field_propagation_chaos.good_evolve _
      (FrameworkStruct_probability_mean_field_propagation_chaos.good_couple_intro _ _ ctx.hq ctx.hp)

theorem main_result_probability_mean_field_propagation_chaos
    {Omega : Type u} [FrameworkStruct_probability_mean_field_propagation_chaos Omega]
    (ctx : ContextData_probability_mean_field_propagation_chaos Omega) :
    primary_map_probability_mean_field_propagation_chaos ctx /\
    exists z : Omega,
      FrameworkStruct_probability_mean_field_propagation_chaos.good z /\
      (FrameworkStruct_probability_mean_field_propagation_chaos.good
          (FrameworkStruct_probability_mean_field_propagation_chaos.evolve z) /\
        secondary_map_probability_mean_field_propagation_chaos ctx) := by
  have hprim :
      primary_map_probability_mean_field_propagation_chaos ctx := by
    unfold primary_map_probability_mean_field_propagation_chaos
    exact FrameworkStruct_probability_mean_field_propagation_chaos.good_couple_intro _ _ ctx.hp ctx.hq
  refine And.intro hprim ?_
  refine Exists.intro
    (FrameworkStruct_probability_mean_field_propagation_chaos.couple ctx.q ctx.p) ?_
  refine And.intro ?hgood ?hpair
  · exact FrameworkStruct_probability_mean_field_propagation_chaos.good_couple_intro _ _ ctx.hq ctx.hp
  · refine And.intro ?hevo ?hsec
    · exact FrameworkStruct_probability_mean_field_propagation_chaos.good_evolve _
        (FrameworkStruct_probability_mean_field_propagation_chaos.good_couple_intro _ _ ctx.hq ctx.hp)
    · unfold secondary_map_probability_mean_field_propagation_chaos
      exact FrameworkStruct_probability_mean_field_propagation_chaos.good_evolve _
        (FrameworkStruct_probability_mean_field_propagation_chaos.good_couple_intro _ _ ctx.hq ctx.hp)
