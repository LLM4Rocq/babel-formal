/-
BENCHMARK_ID: TINY_MATHLIB_BATCH07_PROBABILITY_MALLIAVIN_INTEGRATION_PARTS_ADVANCED_LIKE
PAIR_STEM: probability_malliavin_integration_parts_advanced_like
MATH_DOMAIN: Probability
SOURCE_MATHLIB: Mathlib/Probability/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class FrameworkStruct_probability_malliavin_integration_parts_advanced (Omega : Type u) where
  deriv : Omega -> Nat
  weight : Omega -> Nat
  pair : Omega -> Omega -> Omega
  neutral : Omega
  pair_neutral_left : forall x : Omega, pair neutral x = x
  pair_neutral_right : forall x : Omega, pair x neutral = x
  pair_score_swap : forall x y : Omega, deriv (pair x y) = deriv (pair y x)
  deriv_weight : forall x : Omega, deriv x = weight x
  weight_pair : forall x y : Omega, weight (pair x y) = weight x + weight y

structure ContextData_probability_malliavin_integration_parts_advanced
    (Omega : Type u) [FrameworkStruct_probability_malliavin_integration_parts_advanced Omega] where
  x : Omega
  y : Omega
  hxy : FrameworkStruct_probability_malliavin_integration_parts_advanced.pair x y =
    FrameworkStruct_probability_malliavin_integration_parts_advanced.pair y x

def primary_map_probability_malliavin_integration_parts_advanced
    {Omega : Type u} [FrameworkStruct_probability_malliavin_integration_parts_advanced Omega]
    (ctx : ContextData_probability_malliavin_integration_parts_advanced Omega) : Nat :=
  FrameworkStruct_probability_malliavin_integration_parts_advanced.deriv
    (FrameworkStruct_probability_malliavin_integration_parts_advanced.pair ctx.x ctx.y)

def secondary_map_probability_malliavin_integration_parts_advanced
    {Omega : Type u} [FrameworkStruct_probability_malliavin_integration_parts_advanced Omega]
    (ctx : ContextData_probability_malliavin_integration_parts_advanced Omega) : Nat :=
  FrameworkStruct_probability_malliavin_integration_parts_advanced.deriv
    (FrameworkStruct_probability_malliavin_integration_parts_advanced.pair ctx.y ctx.x)

def tertiary_map_probability_malliavin_integration_parts_advanced
    {Omega : Type u} [FrameworkStruct_probability_malliavin_integration_parts_advanced Omega]
    (ctx : ContextData_probability_malliavin_integration_parts_advanced Omega) : Nat :=
  FrameworkStruct_probability_malliavin_integration_parts_advanced.weight
    (FrameworkStruct_probability_malliavin_integration_parts_advanced.pair
      (FrameworkStruct_probability_malliavin_integration_parts_advanced.pair ctx.x ctx.y)
      (FrameworkStruct_probability_malliavin_integration_parts_advanced.pair ctx.y ctx.x))

theorem stability_step_probability_malliavin_integration_parts_advanced
    {Omega : Type u} [FrameworkStruct_probability_malliavin_integration_parts_advanced Omega]
    (ctx : ContextData_probability_malliavin_integration_parts_advanced Omega)
    (hneq : primary_map_probability_malliavin_integration_parts_advanced ctx ≠
      secondary_map_probability_malliavin_integration_parts_advanced ctx) :
    False := by
  have hcomm :
      FrameworkStruct_probability_malliavin_integration_parts_advanced.pair ctx.x ctx.y =
      FrameworkStruct_probability_malliavin_integration_parts_advanced.pair ctx.y ctx.x :=
    ctx.hxy
  have hleft :
      primary_map_probability_malliavin_integration_parts_advanced ctx =
      FrameworkStruct_probability_malliavin_integration_parts_advanced.deriv
        (FrameworkStruct_probability_malliavin_integration_parts_advanced.pair ctx.y ctx.x) := by
    unfold primary_map_probability_malliavin_integration_parts_advanced
    rw [hcomm]
  have hright :
      FrameworkStruct_probability_malliavin_integration_parts_advanced.deriv
        (FrameworkStruct_probability_malliavin_integration_parts_advanced.pair ctx.y ctx.x) =
      secondary_map_probability_malliavin_integration_parts_advanced ctx := by
    rfl
  have hEq :
      primary_map_probability_malliavin_integration_parts_advanced ctx =
      secondary_map_probability_malliavin_integration_parts_advanced ctx := by
    calc
      primary_map_probability_malliavin_integration_parts_advanced ctx
          = FrameworkStruct_probability_malliavin_integration_parts_advanced.deriv
              (FrameworkStruct_probability_malliavin_integration_parts_advanced.pair ctx.y ctx.x) := hleft
      _ = secondary_map_probability_malliavin_integration_parts_advanced ctx := hright
  exact hneq hEq

theorem factorization_step_probability_malliavin_integration_parts_advanced
    {Omega : Type u} [FrameworkStruct_probability_malliavin_integration_parts_advanced Omega]
    (ctx : ContextData_probability_malliavin_integration_parts_advanced Omega) :
    tertiary_map_probability_malliavin_integration_parts_advanced ctx =
      primary_map_probability_malliavin_integration_parts_advanced ctx +
      secondary_map_probability_malliavin_integration_parts_advanced ctx := by
  have hw :
      FrameworkStruct_probability_malliavin_integration_parts_advanced.weight
        (FrameworkStruct_probability_malliavin_integration_parts_advanced.pair
          (FrameworkStruct_probability_malliavin_integration_parts_advanced.pair ctx.x ctx.y)
          (FrameworkStruct_probability_malliavin_integration_parts_advanced.pair ctx.y ctx.x)) =
      FrameworkStruct_probability_malliavin_integration_parts_advanced.weight
          (FrameworkStruct_probability_malliavin_integration_parts_advanced.pair ctx.x ctx.y) +
        FrameworkStruct_probability_malliavin_integration_parts_advanced.weight
          (FrameworkStruct_probability_malliavin_integration_parts_advanced.pair ctx.y ctx.x) :=
    FrameworkStruct_probability_malliavin_integration_parts_advanced.weight_pair _ _
  have hd1 :
      FrameworkStruct_probability_malliavin_integration_parts_advanced.deriv
          (FrameworkStruct_probability_malliavin_integration_parts_advanced.pair ctx.x ctx.y) =
      FrameworkStruct_probability_malliavin_integration_parts_advanced.weight
          (FrameworkStruct_probability_malliavin_integration_parts_advanced.pair ctx.x ctx.y) :=
    FrameworkStruct_probability_malliavin_integration_parts_advanced.deriv_weight _
  have hd2 :
      FrameworkStruct_probability_malliavin_integration_parts_advanced.deriv
          (FrameworkStruct_probability_malliavin_integration_parts_advanced.pair ctx.y ctx.x) =
      FrameworkStruct_probability_malliavin_integration_parts_advanced.weight
          (FrameworkStruct_probability_malliavin_integration_parts_advanced.pair ctx.y ctx.x) :=
    FrameworkStruct_probability_malliavin_integration_parts_advanced.deriv_weight _
  calc
    tertiary_map_probability_malliavin_integration_parts_advanced ctx
        = FrameworkStruct_probability_malliavin_integration_parts_advanced.weight
            (FrameworkStruct_probability_malliavin_integration_parts_advanced.pair
              (FrameworkStruct_probability_malliavin_integration_parts_advanced.pair ctx.x ctx.y)
              (FrameworkStruct_probability_malliavin_integration_parts_advanced.pair ctx.y ctx.x)) := rfl
    _ = FrameworkStruct_probability_malliavin_integration_parts_advanced.weight
          (FrameworkStruct_probability_malliavin_integration_parts_advanced.pair ctx.x ctx.y) +
        FrameworkStruct_probability_malliavin_integration_parts_advanced.weight
          (FrameworkStruct_probability_malliavin_integration_parts_advanced.pair ctx.y ctx.x) := hw
    _ = FrameworkStruct_probability_malliavin_integration_parts_advanced.deriv
          (FrameworkStruct_probability_malliavin_integration_parts_advanced.pair ctx.x ctx.y) +
        FrameworkStruct_probability_malliavin_integration_parts_advanced.deriv
          (FrameworkStruct_probability_malliavin_integration_parts_advanced.pair ctx.y ctx.x) := by
            rw [<- hd1, <- hd2]
    _ = primary_map_probability_malliavin_integration_parts_advanced ctx +
        secondary_map_probability_malliavin_integration_parts_advanced ctx := by
          rfl

theorem comparison_step_probability_malliavin_integration_parts_advanced
    {Omega : Type u} [FrameworkStruct_probability_malliavin_integration_parts_advanced Omega]
    (ctx : ContextData_probability_malliavin_integration_parts_advanced Omega) :
    primary_map_probability_malliavin_integration_parts_advanced ctx =
      secondary_map_probability_malliavin_integration_parts_advanced ctx /\
      exists z : Omega,
        FrameworkStruct_probability_malliavin_integration_parts_advanced.deriv z =
        primary_map_probability_malliavin_integration_parts_advanced ctx := by
  refine And.intro ?hEq ?hEx
  · have hswap := FrameworkStruct_probability_malliavin_integration_parts_advanced.pair_score_swap ctx.x ctx.y
    calc
      primary_map_probability_malliavin_integration_parts_advanced ctx
          = FrameworkStruct_probability_malliavin_integration_parts_advanced.deriv
              (FrameworkStruct_probability_malliavin_integration_parts_advanced.pair ctx.x ctx.y) := rfl
      _ = FrameworkStruct_probability_malliavin_integration_parts_advanced.deriv
            (FrameworkStruct_probability_malliavin_integration_parts_advanced.pair ctx.y ctx.x) := hswap
      _ = secondary_map_probability_malliavin_integration_parts_advanced ctx := by rfl
  · refine Exists.intro
      (FrameworkStruct_probability_malliavin_integration_parts_advanced.pair ctx.x ctx.y) ?_
    rfl

theorem transport_step_probability_malliavin_integration_parts_advanced
    {Omega : Type u} [FrameworkStruct_probability_malliavin_integration_parts_advanced Omega]
    (ctx : ContextData_probability_malliavin_integration_parts_advanced Omega)
    (hLift : forall z : Omega,
      FrameworkStruct_probability_malliavin_integration_parts_advanced.deriv z =
        primary_map_probability_malliavin_integration_parts_advanced ctx ->
      FrameworkStruct_probability_malliavin_integration_parts_advanced.deriv z =
        secondary_map_probability_malliavin_integration_parts_advanced ctx) :
    exists z : Omega,
      FrameworkStruct_probability_malliavin_integration_parts_advanced.deriv z =
      secondary_map_probability_malliavin_integration_parts_advanced ctx := by
  refine Exists.intro
    (FrameworkStruct_probability_malliavin_integration_parts_advanced.pair ctx.x ctx.y) ?_
  have hz :
      FrameworkStruct_probability_malliavin_integration_parts_advanced.deriv
          (FrameworkStruct_probability_malliavin_integration_parts_advanced.pair ctx.x ctx.y) =
      primary_map_probability_malliavin_integration_parts_advanced ctx := by
    rfl
  exact hLift _ hz

theorem coherence_step_probability_malliavin_integration_parts_advanced
    {Omega : Type u} [FrameworkStruct_probability_malliavin_integration_parts_advanced Omega]
    (ctx : ContextData_probability_malliavin_integration_parts_advanced Omega)
    (hbad : (forall z : Omega,
      FrameworkStruct_probability_malliavin_integration_parts_advanced.deriv z ≠
        secondary_map_probability_malliavin_integration_parts_advanced ctx) -> False) :
    exists z : Omega,
      FrameworkStruct_probability_malliavin_integration_parts_advanced.deriv z =
        secondary_map_probability_malliavin_integration_parts_advanced ctx := by
  have huse : (forall z : Omega,
      FrameworkStruct_probability_malliavin_integration_parts_advanced.deriv z ≠
        secondary_map_probability_malliavin_integration_parts_advanced ctx) -> False := hbad
  have _ : False -> False := by
    intro hF
    exact hF
  refine Exists.intro
    (FrameworkStruct_probability_malliavin_integration_parts_advanced.pair ctx.y ctx.x) ?_
  rfl

theorem iteration_step_probability_malliavin_integration_parts_advanced
    {Omega : Type u} [FrameworkStruct_probability_malliavin_integration_parts_advanced Omega]
    (ctx : ContextData_probability_malliavin_integration_parts_advanced Omega) :
    exists z : Omega,
      FrameworkStruct_probability_malliavin_integration_parts_advanced.pair
        FrameworkStruct_probability_malliavin_integration_parts_advanced.neutral z = z /\
      FrameworkStruct_probability_malliavin_integration_parts_advanced.deriv z =
        primary_map_probability_malliavin_integration_parts_advanced ctx := by
  refine Exists.intro
    (FrameworkStruct_probability_malliavin_integration_parts_advanced.pair ctx.x ctx.y) ?_
  refine And.intro ?hN ?hD
  · exact FrameworkStruct_probability_malliavin_integration_parts_advanced.pair_neutral_left _
  · rfl

theorem main_result_probability_malliavin_integration_parts_advanced
    {Omega : Type u} [FrameworkStruct_probability_malliavin_integration_parts_advanced Omega]
    (ctx : ContextData_probability_malliavin_integration_parts_advanced Omega) :
    primary_map_probability_malliavin_integration_parts_advanced ctx =
      secondary_map_probability_malliavin_integration_parts_advanced ctx /\
    exists z : Omega,
      FrameworkStruct_probability_malliavin_integration_parts_advanced.deriv z =
        primary_map_probability_malliavin_integration_parts_advanced ctx /\
      tertiary_map_probability_malliavin_integration_parts_advanced ctx =
        FrameworkStruct_probability_malliavin_integration_parts_advanced.deriv
          (FrameworkStruct_probability_malliavin_integration_parts_advanced.pair z z) := by
  refine And.intro
    (comparison_step_probability_malliavin_integration_parts_advanced ctx).1 ?_
  refine Exists.intro
    (FrameworkStruct_probability_malliavin_integration_parts_advanced.pair ctx.x ctx.y) ?_
  refine And.intro ?hDer ?hTer
  · rfl
  · unfold tertiary_map_probability_malliavin_integration_parts_advanced
    rw [ctx.hxy]
    symm
    exact FrameworkStruct_probability_malliavin_integration_parts_advanced.deriv_weight _
