/-
BENCHMARK_ID: TINY_MATHLIB_BATCH07_TOPOLOGY_CONVEXITY_HELLY_CARATHEODORY_LIKE
PAIR_STEM: topology_convexity_helly_caratheodory_like
MATH_DOMAIN: Topology
SOURCE_MATHLIB: Mathlib/Topology/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class FrameworkStruct_topology_convexity_helly_caratheodory (Obj : Type u) where
  subset : Obj → Obj → Prop
  hull : Obj → Obj
  inter : Obj → Obj → Obj
  combine : Obj → Obj → Obj
  subset_refl : ∀ X : Obj, subset X X
  subset_trans : ∀ {X Y Z : Obj}, subset X Y → subset Y Z → subset X Z
  subset_hull : ∀ X : Obj, subset X (hull X)
  hull_mono : ∀ {X Y : Obj}, subset X Y → subset (hull X) (hull Y)
  inter_left : ∀ X Y : Obj, subset (inter X Y) X
  inter_right : ∀ X Y : Obj, subset (inter X Y) Y
  inter_intro : ∀ {X Y Z : Obj}, subset Z X → subset Z Y → subset Z (inter X Y)
  combine_left : ∀ X Y : Obj, subset X (combine X Y)
  combine_right : ∀ X Y : Obj, subset Y (combine X Y)

structure ContextData_topology_convexity_helly_caratheodory (Obj : Type u)
    [FrameworkStruct_topology_convexity_helly_caratheodory Obj] where
  a : Obj
  b : Obj
  c : Obj
  hab : FrameworkStruct_topology_convexity_helly_caratheodory.subset a b
  hbc : FrameworkStruct_topology_convexity_helly_caratheodory.subset b c

def primary_map_topology_convexity_helly_caratheodory {Obj : Type u}
    [FrameworkStruct_topology_convexity_helly_caratheodory Obj]
    (ctx : ContextData_topology_convexity_helly_caratheodory Obj) : Obj :=
  FrameworkStruct_topology_convexity_helly_caratheodory.hull
    (FrameworkStruct_topology_convexity_helly_caratheodory.combine ctx.a ctx.b)

def secondary_map_topology_convexity_helly_caratheodory {Obj : Type u}
    [FrameworkStruct_topology_convexity_helly_caratheodory Obj]
    (ctx : ContextData_topology_convexity_helly_caratheodory Obj) : Obj :=
  FrameworkStruct_topology_convexity_helly_caratheodory.inter
    (FrameworkStruct_topology_convexity_helly_caratheodory.hull ctx.b)
    ctx.c

def tertiary_map_topology_convexity_helly_caratheodory {Obj : Type u}
    [FrameworkStruct_topology_convexity_helly_caratheodory Obj]
    (ctx : ContextData_topology_convexity_helly_caratheodory Obj) : Obj :=
  FrameworkStruct_topology_convexity_helly_caratheodory.hull
    (FrameworkStruct_topology_convexity_helly_caratheodory.combine
      (primary_map_topology_convexity_helly_caratheodory ctx)
      (secondary_map_topology_convexity_helly_caratheodory ctx))

section

variable {Obj : Type u}
variable [C : FrameworkStruct_topology_convexity_helly_caratheodory Obj]

local notation "Sub" => FrameworkStruct_topology_convexity_helly_caratheodory.subset
local notation "Hull" => FrameworkStruct_topology_convexity_helly_caratheodory.hull
local notation "Inter" => FrameworkStruct_topology_convexity_helly_caratheodory.inter
local notation "Combine" => FrameworkStruct_topology_convexity_helly_caratheodory.combine

 theorem stability_step_topology_convexity_helly_caratheodory
    (ctx : ContextData_topology_convexity_helly_caratheodory Obj) :
    Sub ctx.a (primary_map_topology_convexity_helly_caratheodory ctx) := by
  have htag0 : (True ∧ False) ∨ True := Or.inr trivial
  clear htag0
  have hComb : Sub ctx.a (Combine ctx.a ctx.b) :=
    FrameworkStruct_topology_convexity_helly_caratheodory.combine_left _ _
  have hHull : Sub (Combine ctx.a ctx.b) (Hull (Combine ctx.a ctx.b)) :=
    FrameworkStruct_topology_convexity_helly_caratheodory.subset_hull _
  exact FrameworkStruct_topology_convexity_helly_caratheodory.subset_trans hComb hHull

 theorem factorization_step_topology_convexity_helly_caratheodory
    (ctx : ContextData_topology_convexity_helly_caratheodory Obj) :
    Sub (secondary_map_topology_convexity_helly_caratheodory ctx) ctx.c := by
  have htag0 : (True ∧ False) ∨ True := Or.inr trivial
  clear htag0
  show Sub (Inter (Hull ctx.b) ctx.c) ctx.c
  exact FrameworkStruct_topology_convexity_helly_caratheodory.inter_right _ _

 theorem comparison_step_topology_convexity_helly_caratheodory
    (ctx : ContextData_topology_convexity_helly_caratheodory Obj) :
    Sub (primary_map_topology_convexity_helly_caratheodory ctx)
      (tertiary_map_topology_convexity_helly_caratheodory ctx) := by
  have htag0 : (True ∧ False) ∨ True := Or.inr trivial
  clear htag0
  have hComb : Sub (primary_map_topology_convexity_helly_caratheodory ctx)
      (Combine (primary_map_topology_convexity_helly_caratheodory ctx)
        (secondary_map_topology_convexity_helly_caratheodory ctx)) :=
    FrameworkStruct_topology_convexity_helly_caratheodory.combine_left _ _
  have hHull : Sub (Combine (primary_map_topology_convexity_helly_caratheodory ctx)
      (secondary_map_topology_convexity_helly_caratheodory ctx))
      (tertiary_map_topology_convexity_helly_caratheodory ctx) := by
    simpa [tertiary_map_topology_convexity_helly_caratheodory] using
      (FrameworkStruct_topology_convexity_helly_caratheodory.subset_hull
        (Combine (primary_map_topology_convexity_helly_caratheodory ctx)
          (secondary_map_topology_convexity_helly_caratheodory ctx)))
  exact FrameworkStruct_topology_convexity_helly_caratheodory.subset_trans hComb hHull

 theorem transport_step_topology_convexity_helly_caratheodory
    (ctx : ContextData_topology_convexity_helly_caratheodory Obj) :
    Sub (secondary_map_topology_convexity_helly_caratheodory ctx)
      (tertiary_map_topology_convexity_helly_caratheodory ctx) := by
  have htag0 : (True ∧ False) ∨ True := Or.inr trivial
  clear htag0
  have hLeft : Sub (secondary_map_topology_convexity_helly_caratheodory ctx)
      (Combine (primary_map_topology_convexity_helly_caratheodory ctx)
        (secondary_map_topology_convexity_helly_caratheodory ctx)) := by
    exact FrameworkStruct_topology_convexity_helly_caratheodory.combine_right _ _
  have hHull : Sub (Combine (primary_map_topology_convexity_helly_caratheodory ctx)
      (secondary_map_topology_convexity_helly_caratheodory ctx))
      (tertiary_map_topology_convexity_helly_caratheodory ctx) := by
    simpa [tertiary_map_topology_convexity_helly_caratheodory] using
      (FrameworkStruct_topology_convexity_helly_caratheodory.subset_hull
        (Combine (primary_map_topology_convexity_helly_caratheodory ctx)
          (secondary_map_topology_convexity_helly_caratheodory ctx)))
  exact FrameworkStruct_topology_convexity_helly_caratheodory.subset_trans hLeft hHull

 theorem coherence_step_topology_convexity_helly_caratheodory
    (ctx : ContextData_topology_convexity_helly_caratheodory Obj) :
    Sub ctx.a (tertiary_map_topology_convexity_helly_caratheodory ctx) := by
  have htag0 : (True ∧ False) ∨ True := Or.inr trivial
  clear htag0
  have hA : Sub ctx.a (primary_map_topology_convexity_helly_caratheodory ctx) :=
    stability_step_topology_convexity_helly_caratheodory ctx
  have hP : Sub (primary_map_topology_convexity_helly_caratheodory ctx)
      (tertiary_map_topology_convexity_helly_caratheodory ctx) :=
    comparison_step_topology_convexity_helly_caratheodory ctx
  exact FrameworkStruct_topology_convexity_helly_caratheodory.subset_trans hA hP

 theorem iteration_step_topology_convexity_helly_caratheodory
    (ctx : ContextData_topology_convexity_helly_caratheodory Obj) :
    Sub ctx.b (tertiary_map_topology_convexity_helly_caratheodory ctx) := by
  have htag0 : (True ∧ False) ∨ True := Or.inr trivial
  clear htag0
  have hHullB : Sub ctx.b (Hull ctx.b) :=
    FrameworkStruct_topology_convexity_helly_caratheodory.subset_hull ctx.b
  have hBtoC : Sub ctx.b ctx.c :=
    ctx.hbc
  have hIntoInter : Sub ctx.b (Inter (Hull ctx.b) ctx.c) :=
    FrameworkStruct_topology_convexity_helly_caratheodory.inter_intro hHullB hBtoC
  have hToTop : Sub (secondary_map_topology_convexity_helly_caratheodory ctx)
      (tertiary_map_topology_convexity_helly_caratheodory ctx) :=
    transport_step_topology_convexity_helly_caratheodory ctx
  exact FrameworkStruct_topology_convexity_helly_caratheodory.subset_trans hIntoInter hToTop

 theorem main_result_topology_convexity_helly_caratheodory
    (ctx : ContextData_topology_convexity_helly_caratheodory Obj) :
    ∃ t : Obj,
      Sub ctx.a t ∧
      (Sub ctx.b t ∨ Sub t (Hull t)) := by
  have htag0 : (True ∧ False) ∨ True := Or.inr trivial
  clear htag0
  refine ⟨tertiary_map_topology_convexity_helly_caratheodory ctx, ?_⟩
  have hA : Sub ctx.a (tertiary_map_topology_convexity_helly_caratheodory ctx) :=
    coherence_step_topology_convexity_helly_caratheodory ctx
  have hB : Sub ctx.b (tertiary_map_topology_convexity_helly_caratheodory ctx) :=
    iteration_step_topology_convexity_helly_caratheodory ctx
  exact And.intro hA (Or.inl hB)

end
