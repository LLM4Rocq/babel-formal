/-
BENCHMARK_ID: TINY_MATHLIB_BATCH07_PROBABILITY_SKOROKHOD_REPRESENTATION_REFINED_LIKE
PAIR_STEM: probability_skorokhod_representation_refined_like
MATH_DOMAIN: Probability
SOURCE_MATHLIB: Mathlib/Probability/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

class FrameworkStruct_probability_skorokhod_representation_refined where
  repr : Nat -> Nat
  lift : Nat -> Nat
  good : Nat -> Prop
  good_lift : forall n : Nat, good (lift n)
  repr_lift : forall n : Nat, repr (lift n) = repr n
  lift_repr : forall n : Nat, lift (repr n) = lift n
  repr_idem : forall n : Nat, repr (repr n) = repr n

structure ContextData_probability_skorokhod_representation_refined
    [FrameworkStruct_probability_skorokhod_representation_refined] where
  n : Nat
  m : Nat
  hnm : FrameworkStruct_probability_skorokhod_representation_refined.repr n =
    FrameworkStruct_probability_skorokhod_representation_refined.repr m

def primary_map_probability_skorokhod_representation_refined
    [FrameworkStruct_probability_skorokhod_representation_refined]
    (ctx : ContextData_probability_skorokhod_representation_refined) : Nat :=
  FrameworkStruct_probability_skorokhod_representation_refined.repr
    (FrameworkStruct_probability_skorokhod_representation_refined.lift ctx.n)

def secondary_map_probability_skorokhod_representation_refined
    [FrameworkStruct_probability_skorokhod_representation_refined]
    (ctx : ContextData_probability_skorokhod_representation_refined) : Nat :=
  FrameworkStruct_probability_skorokhod_representation_refined.repr ctx.m

def tertiary_map_probability_skorokhod_representation_refined
    [FrameworkStruct_probability_skorokhod_representation_refined]
    (ctx : ContextData_probability_skorokhod_representation_refined) : Prop :=
  FrameworkStruct_probability_skorokhod_representation_refined.good
    (FrameworkStruct_probability_skorokhod_representation_refined.lift
      (FrameworkStruct_probability_skorokhod_representation_refined.repr ctx.m))

theorem stability_step_probability_skorokhod_representation_refined
    [FrameworkStruct_probability_skorokhod_representation_refined]
    (ctx : ContextData_probability_skorokhod_representation_refined)
    (hneq : secondary_map_probability_skorokhod_representation_refined ctx ≠
      primary_map_probability_skorokhod_representation_refined ctx -> False) :
    primary_map_probability_skorokhod_representation_refined ctx =
      secondary_map_probability_skorokhod_representation_refined ctx := by
  by_cases hEq :
      primary_map_probability_skorokhod_representation_refined ctx =
      secondary_map_probability_skorokhod_representation_refined ctx
  · exact hEq
  · have hne : secondary_map_probability_skorokhod_representation_refined ctx ≠
        primary_map_probability_skorokhod_representation_refined ctx := by
      intro hs
      exact hEq hs.symm
    exact False.elim (hneq hne)

theorem factorization_step_probability_skorokhod_representation_refined
    [FrameworkStruct_probability_skorokhod_representation_refined]
    (ctx : ContextData_probability_skorokhod_representation_refined) :
    tertiary_map_probability_skorokhod_representation_refined ctx := by
  unfold tertiary_map_probability_skorokhod_representation_refined
  exact FrameworkStruct_probability_skorokhod_representation_refined.good_lift _

theorem comparison_step_probability_skorokhod_representation_refined
    [FrameworkStruct_probability_skorokhod_representation_refined]
    (ctx : ContextData_probability_skorokhod_representation_refined) :
    primary_map_probability_skorokhod_representation_refined ctx =
      secondary_map_probability_skorokhod_representation_refined ctx /\
      tertiary_map_probability_skorokhod_representation_refined ctx /\
      exists k : Nat,
        FrameworkStruct_probability_skorokhod_representation_refined.repr k =
        secondary_map_probability_skorokhod_representation_refined ctx := by
  have hEq :
      primary_map_probability_skorokhod_representation_refined ctx =
      secondary_map_probability_skorokhod_representation_refined ctx := by
    have hleft :
        primary_map_probability_skorokhod_representation_refined ctx =
        FrameworkStruct_probability_skorokhod_representation_refined.repr ctx.n := by
      unfold primary_map_probability_skorokhod_representation_refined
      exact FrameworkStruct_probability_skorokhod_representation_refined.repr_lift _
    have hright :
        FrameworkStruct_probability_skorokhod_representation_refined.repr ctx.n =
        secondary_map_probability_skorokhod_representation_refined ctx := by
      unfold secondary_map_probability_skorokhod_representation_refined
      exact ctx.hnm
    calc
      primary_map_probability_skorokhod_representation_refined ctx
          = FrameworkStruct_probability_skorokhod_representation_refined.repr ctx.n := hleft
      _ = secondary_map_probability_skorokhod_representation_refined ctx := hright
  have hTer : tertiary_map_probability_skorokhod_representation_refined ctx :=
    factorization_step_probability_skorokhod_representation_refined ctx
  refine And.intro hEq ?_
  refine And.intro hTer ?_
  refine Exists.intro ctx.m ?_
  unfold secondary_map_probability_skorokhod_representation_refined
  rfl

theorem transport_step_probability_skorokhod_representation_refined
    [FrameworkStruct_probability_skorokhod_representation_refined]
    (ctx : ContextData_probability_skorokhod_representation_refined)
    (htr : forall k : Nat,
      FrameworkStruct_probability_skorokhod_representation_refined.repr k =
        secondary_map_probability_skorokhod_representation_refined ctx ->
      FrameworkStruct_probability_skorokhod_representation_refined.good
        (FrameworkStruct_probability_skorokhod_representation_refined.lift k)) :
    tertiary_map_probability_skorokhod_representation_refined ctx := by
  unfold tertiary_map_probability_skorokhod_representation_refined
  have hk :
      FrameworkStruct_probability_skorokhod_representation_refined.repr ctx.m =
      secondary_map_probability_skorokhod_representation_refined ctx := by
    unfold secondary_map_probability_skorokhod_representation_refined
    rfl
  have hgood :
      FrameworkStruct_probability_skorokhod_representation_refined.good
        (FrameworkStruct_probability_skorokhod_representation_refined.lift ctx.m) := htr _ hk
  rw [FrameworkStruct_probability_skorokhod_representation_refined.lift_repr ctx.m]
  exact hgood

theorem coherence_step_probability_skorokhod_representation_refined
    [FrameworkStruct_probability_skorokhod_representation_refined]
    (ctx : ContextData_probability_skorokhod_representation_refined)
    (hbad : (tertiary_map_probability_skorokhod_representation_refined ctx -> False) -> False) :
    tertiary_map_probability_skorokhod_representation_refined ctx := by
  have hter : tertiary_map_probability_skorokhod_representation_refined ctx :=
    factorization_step_probability_skorokhod_representation_refined ctx
  exact hter

theorem iteration_step_probability_skorokhod_representation_refined
    [FrameworkStruct_probability_skorokhod_representation_refined]
    (ctx : ContextData_probability_skorokhod_representation_refined) :
    exists k : Nat,
      FrameworkStruct_probability_skorokhod_representation_refined.repr k =
        primary_map_probability_skorokhod_representation_refined ctx /\
      FrameworkStruct_probability_skorokhod_representation_refined.good
        (FrameworkStruct_probability_skorokhod_representation_refined.lift k) := by
  refine Exists.intro
    (FrameworkStruct_probability_skorokhod_representation_refined.lift ctx.n) ?_
  refine And.intro ?hrepr ?hgood
  · unfold primary_map_probability_skorokhod_representation_refined
    rfl
  · exact FrameworkStruct_probability_skorokhod_representation_refined.good_lift _

theorem main_result_probability_skorokhod_representation_refined
    [FrameworkStruct_probability_skorokhod_representation_refined]
    (ctx : ContextData_probability_skorokhod_representation_refined) :
    primary_map_probability_skorokhod_representation_refined ctx =
      secondary_map_probability_skorokhod_representation_refined ctx /\
    exists k : Nat,
      FrameworkStruct_probability_skorokhod_representation_refined.repr k =
        primary_map_probability_skorokhod_representation_refined ctx /\
      (tertiary_map_probability_skorokhod_representation_refined ctx /\
        FrameworkStruct_probability_skorokhod_representation_refined.good
          (FrameworkStruct_probability_skorokhod_representation_refined.lift k)) := by
  have hEq := (comparison_step_probability_skorokhod_representation_refined ctx).1
  refine And.intro hEq ?_
  refine Exists.intro
    (FrameworkStruct_probability_skorokhod_representation_refined.lift ctx.n) ?_
  refine And.intro ?hrepr ?hpack
  · unfold primary_map_probability_skorokhod_representation_refined
    rfl
  · refine And.intro ?hter ?hgood
    · exact factorization_step_probability_skorokhod_representation_refined ctx
    · exact FrameworkStruct_probability_skorokhod_representation_refined.good_lift _
