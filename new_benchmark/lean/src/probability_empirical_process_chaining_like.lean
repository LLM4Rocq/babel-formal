/-
BENCHMARK_ID: TINY_MATHLIB_BATCH07_PROBABILITY_EMPIRICAL_PROCESS_CHAINING_LIKE
PAIR_STEM: probability_empirical_process_chaining_like
MATH_DOMAIN: Probability
SOURCE_MATHLIB: Mathlib/Probability/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class FrameworkStruct_probability_empirical_process_chaining (Omega : Type u) where
  metric : Omega -> Nat
  penalty : Omega -> Nat
  blend : Omega -> Omega -> Omega
  base : Omega
  metric_blend_left : forall x : Omega, metric (blend base x) = metric x
  penalty_blend_right : forall x : Omega, penalty (blend x base) = penalty x
  metric_swap : forall x y : Omega, metric (blend x y) = metric (blend y x)
  metric_penalty : forall x : Omega, metric x = penalty x
  penalty_blend : forall x y : Omega, penalty (blend x y) = penalty x + penalty y

structure ContextData_probability_empirical_process_chaining
    (Omega : Type u) [FrameworkStruct_probability_empirical_process_chaining Omega] where
  a : Omega
  b : Omega
  hlink :
    FrameworkStruct_probability_empirical_process_chaining.blend a
      (FrameworkStruct_probability_empirical_process_chaining.blend b
        FrameworkStruct_probability_empirical_process_chaining.base) =
    FrameworkStruct_probability_empirical_process_chaining.blend
      (FrameworkStruct_probability_empirical_process_chaining.blend a b)
      FrameworkStruct_probability_empirical_process_chaining.base

def primary_map_probability_empirical_process_chaining
    {Omega : Type u} [FrameworkStruct_probability_empirical_process_chaining Omega]
    (ctx : ContextData_probability_empirical_process_chaining Omega) : Nat :=
  FrameworkStruct_probability_empirical_process_chaining.metric
    (FrameworkStruct_probability_empirical_process_chaining.blend ctx.a ctx.b)

def secondary_map_probability_empirical_process_chaining
    {Omega : Type u} [FrameworkStruct_probability_empirical_process_chaining Omega]
    (ctx : ContextData_probability_empirical_process_chaining Omega) : Nat :=
  FrameworkStruct_probability_empirical_process_chaining.penalty
    (FrameworkStruct_probability_empirical_process_chaining.blend ctx.b ctx.a)

def tertiary_map_probability_empirical_process_chaining
    {Omega : Type u} [FrameworkStruct_probability_empirical_process_chaining Omega]
    (ctx : ContextData_probability_empirical_process_chaining Omega) : Nat :=
  FrameworkStruct_probability_empirical_process_chaining.penalty
    (FrameworkStruct_probability_empirical_process_chaining.blend
      (FrameworkStruct_probability_empirical_process_chaining.blend ctx.a ctx.b)
      (FrameworkStruct_probability_empirical_process_chaining.blend ctx.b ctx.a))

theorem stability_step_probability_empirical_process_chaining
    {Omega : Type u} [FrameworkStruct_probability_empirical_process_chaining Omega]
    (ctx : ContextData_probability_empirical_process_chaining Omega)
    (hneq :
      primary_map_probability_empirical_process_chaining ctx ≠
      FrameworkStruct_probability_empirical_process_chaining.metric
        (FrameworkStruct_probability_empirical_process_chaining.blend ctx.b ctx.a))
    (hgate : ctx.a = ctx.a) :
    False := by
  have _ : ctx.a = ctx.a := hgate
  have hswap :
      FrameworkStruct_probability_empirical_process_chaining.metric
        (FrameworkStruct_probability_empirical_process_chaining.blend ctx.a ctx.b) =
      FrameworkStruct_probability_empirical_process_chaining.metric
        (FrameworkStruct_probability_empirical_process_chaining.blend ctx.b ctx.a) :=
    FrameworkStruct_probability_empirical_process_chaining.metric_swap _ _
  have hEq :
      primary_map_probability_empirical_process_chaining ctx =
      FrameworkStruct_probability_empirical_process_chaining.metric
        (FrameworkStruct_probability_empirical_process_chaining.blend ctx.b ctx.a) := by
    unfold primary_map_probability_empirical_process_chaining
    exact hswap
  exact hneq hEq

theorem factorization_step_probability_empirical_process_chaining
    {Omega : Type u} [FrameworkStruct_probability_empirical_process_chaining Omega]
    (ctx : ContextData_probability_empirical_process_chaining Omega) :
    tertiary_map_probability_empirical_process_chaining ctx =
      primary_map_probability_empirical_process_chaining ctx +
      secondary_map_probability_empirical_process_chaining ctx /\
    primary_map_probability_empirical_process_chaining ctx =
      primary_map_probability_empirical_process_chaining ctx := by
  have hpb :
      FrameworkStruct_probability_empirical_process_chaining.penalty
        (FrameworkStruct_probability_empirical_process_chaining.blend
          (FrameworkStruct_probability_empirical_process_chaining.blend ctx.a ctx.b)
          (FrameworkStruct_probability_empirical_process_chaining.blend ctx.b ctx.a)) =
      FrameworkStruct_probability_empirical_process_chaining.penalty
          (FrameworkStruct_probability_empirical_process_chaining.blend ctx.a ctx.b) +
        FrameworkStruct_probability_empirical_process_chaining.penalty
          (FrameworkStruct_probability_empirical_process_chaining.blend ctx.b ctx.a) :=
    FrameworkStruct_probability_empirical_process_chaining.penalty_blend _ _
  have hm :
      FrameworkStruct_probability_empirical_process_chaining.metric
        (FrameworkStruct_probability_empirical_process_chaining.blend ctx.a ctx.b) =
      FrameworkStruct_probability_empirical_process_chaining.penalty
        (FrameworkStruct_probability_empirical_process_chaining.blend ctx.a ctx.b) :=
    FrameworkStruct_probability_empirical_process_chaining.metric_penalty _
  refine And.intro ?hfac ?hrefl
  · calc
      tertiary_map_probability_empirical_process_chaining ctx
          = FrameworkStruct_probability_empirical_process_chaining.penalty
              (FrameworkStruct_probability_empirical_process_chaining.blend
                (FrameworkStruct_probability_empirical_process_chaining.blend ctx.a ctx.b)
                (FrameworkStruct_probability_empirical_process_chaining.blend ctx.b ctx.a)) := rfl
      _ = FrameworkStruct_probability_empirical_process_chaining.penalty
            (FrameworkStruct_probability_empirical_process_chaining.blend ctx.a ctx.b) +
          FrameworkStruct_probability_empirical_process_chaining.penalty
            (FrameworkStruct_probability_empirical_process_chaining.blend ctx.b ctx.a) := hpb
      _ = primary_map_probability_empirical_process_chaining ctx +
          secondary_map_probability_empirical_process_chaining ctx := by
            unfold primary_map_probability_empirical_process_chaining
            unfold secondary_map_probability_empirical_process_chaining
            rw [<- hm]
  · rfl

theorem comparison_step_probability_empirical_process_chaining
    {Omega : Type u} [FrameworkStruct_probability_empirical_process_chaining Omega]
    (ctx : ContextData_probability_empirical_process_chaining Omega) :
    exists z : Omega,
      FrameworkStruct_probability_empirical_process_chaining.penalty z =
        secondary_map_probability_empirical_process_chaining ctx /\
      secondary_map_probability_empirical_process_chaining ctx =
        primary_map_probability_empirical_process_chaining ctx := by
  refine Exists.intro
    (FrameworkStruct_probability_empirical_process_chaining.blend ctx.b ctx.a) ?_
  refine And.intro ?hpen ?heq
  · unfold secondary_map_probability_empirical_process_chaining
    rfl
  · unfold secondary_map_probability_empirical_process_chaining
    unfold primary_map_probability_empirical_process_chaining
    have hswap := FrameworkStruct_probability_empirical_process_chaining.metric_swap ctx.b ctx.a
    have hmp := FrameworkStruct_probability_empirical_process_chaining.metric_penalty
      (FrameworkStruct_probability_empirical_process_chaining.blend ctx.b ctx.a)
    calc
      FrameworkStruct_probability_empirical_process_chaining.penalty
          (FrameworkStruct_probability_empirical_process_chaining.blend ctx.b ctx.a)
          = FrameworkStruct_probability_empirical_process_chaining.metric
              (FrameworkStruct_probability_empirical_process_chaining.blend ctx.b ctx.a) := by
                symm
                exact hmp
      _ = FrameworkStruct_probability_empirical_process_chaining.metric
            (FrameworkStruct_probability_empirical_process_chaining.blend ctx.a ctx.b) := hswap

theorem transport_step_probability_empirical_process_chaining
    {Omega : Type u} [FrameworkStruct_probability_empirical_process_chaining Omega]
    (ctx : ContextData_probability_empirical_process_chaining Omega)
    (htransfer : forall z : Omega,
      FrameworkStruct_probability_empirical_process_chaining.penalty z =
        secondary_map_probability_empirical_process_chaining ctx ->
      FrameworkStruct_probability_empirical_process_chaining.metric z =
        primary_map_probability_empirical_process_chaining ctx) :
    exists z : Omega,
      FrameworkStruct_probability_empirical_process_chaining.metric z =
        primary_map_probability_empirical_process_chaining ctx /\
      FrameworkStruct_probability_empirical_process_chaining.penalty z =
        secondary_map_probability_empirical_process_chaining ctx := by
  refine Exists.intro
    (FrameworkStruct_probability_empirical_process_chaining.blend ctx.b ctx.a) ?_
  have hz :
      FrameworkStruct_probability_empirical_process_chaining.penalty
        (FrameworkStruct_probability_empirical_process_chaining.blend ctx.b ctx.a) =
      secondary_map_probability_empirical_process_chaining ctx := by
    unfold secondary_map_probability_empirical_process_chaining
    rfl
  refine And.intro ?hmet hz
  exact htransfer _ hz

theorem coherence_step_probability_empirical_process_chaining
    {Omega : Type u} [FrameworkStruct_probability_empirical_process_chaining Omega]
    (ctx : ContextData_probability_empirical_process_chaining Omega)
    (hcollapse : (forall z : Omega,
      FrameworkStruct_probability_empirical_process_chaining.metric z ≠
        primary_map_probability_empirical_process_chaining ctx) -> False) :
    FrameworkStruct_probability_empirical_process_chaining.metric
      (FrameworkStruct_probability_empirical_process_chaining.blend ctx.a ctx.b) =
    primary_map_probability_empirical_process_chaining ctx := by
  have _ : (forall z : Omega,
      FrameworkStruct_probability_empirical_process_chaining.metric z ≠
        primary_map_probability_empirical_process_chaining ctx) -> False := hcollapse
  rfl

theorem iteration_step_probability_empirical_process_chaining
    {Omega : Type u} [FrameworkStruct_probability_empirical_process_chaining Omega]
    (ctx : ContextData_probability_empirical_process_chaining Omega) :
    exists z : Omega,
      FrameworkStruct_probability_empirical_process_chaining.metric
        (FrameworkStruct_probability_empirical_process_chaining.blend
          FrameworkStruct_probability_empirical_process_chaining.base z) =
      FrameworkStruct_probability_empirical_process_chaining.metric z /\
      FrameworkStruct_probability_empirical_process_chaining.metric z =
        primary_map_probability_empirical_process_chaining ctx /\
      FrameworkStruct_probability_empirical_process_chaining.penalty z =
        FrameworkStruct_probability_empirical_process_chaining.penalty z := by
  refine Exists.intro
    (FrameworkStruct_probability_empirical_process_chaining.blend ctx.a ctx.b) ?_
  refine And.intro ?hL ?hRest
  · exact FrameworkStruct_probability_empirical_process_chaining.metric_blend_left _
  · refine And.intro ?hM ?hP
    · rfl
    · rfl

theorem main_result_probability_empirical_process_chaining
    {Omega : Type u} [FrameworkStruct_probability_empirical_process_chaining Omega]
    (ctx : ContextData_probability_empirical_process_chaining Omega) :
    secondary_map_probability_empirical_process_chaining ctx =
      primary_map_probability_empirical_process_chaining ctx /\
    exists z : Omega,
      FrameworkStruct_probability_empirical_process_chaining.metric z =
        primary_map_probability_empirical_process_chaining ctx /\
      tertiary_map_probability_empirical_process_chaining ctx =
        FrameworkStruct_probability_empirical_process_chaining.penalty
          (FrameworkStruct_probability_empirical_process_chaining.blend z
            (FrameworkStruct_probability_empirical_process_chaining.blend ctx.b ctx.a)) := by
  have hEq' :
      secondary_map_probability_empirical_process_chaining ctx =
      primary_map_probability_empirical_process_chaining ctx := by
    rcases comparison_step_probability_empirical_process_chaining ctx with ⟨z, hz1, hz2⟩
    exact hz2
  refine And.intro hEq' ?_
  refine Exists.intro
    (FrameworkStruct_probability_empirical_process_chaining.blend ctx.a ctx.b) ?_
  refine And.intro ?hMetric ?hTer
  · rfl
  · unfold tertiary_map_probability_empirical_process_chaining
    rfl
