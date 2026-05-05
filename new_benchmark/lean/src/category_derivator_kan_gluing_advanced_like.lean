/-
BENCHMARK_ID: TINY_MATHLIB_BATCH07_CATEGORY_DERIVATOR_KAN_GLUING_ADVANCED_LIKE
PAIR_STEM: category_derivator_kan_gluing_advanced_like
MATH_DOMAIN: Category Theory
SOURCE_MATHLIB: Mathlib/CategoryTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class FrameworkStruct_category_derivator_kan_gluing_advanced (Obj : Type u) where
  step : Obj → Obj → Prop
  glue : Obj → Obj → Obj
  kanL : Obj → Obj
  kanR : Obj → Obj
  step_refl : ∀ X : Obj, step X X
  step_trans : ∀ {X Y Z : Obj}, step X Y → step Y Z → step X Z
  step_glue_left : ∀ X Y : Obj, step X (glue X Y)
  step_glue_right : ∀ X Y : Obj, step Y (glue X Y)
  kan_bridge : ∀ X : Obj, step (kanL X) (kanR X)
  kanL_monotone : ∀ {X Y : Obj}, step X Y → step (kanL X) (kanL Y)
  kanR_monotone : ∀ {X Y : Obj}, step X Y → step (kanR X) (kanR Y)

structure ContextData_category_derivator_kan_gluing_advanced (Obj : Type u)
    [FrameworkStruct_category_derivator_kan_gluing_advanced Obj] where
  a : Obj
  b : Obj
  c : Obj
  hab : FrameworkStruct_category_derivator_kan_gluing_advanced.step a b
  hbc : FrameworkStruct_category_derivator_kan_gluing_advanced.step b c

def primary_map_category_derivator_kan_gluing_advanced {Obj : Type u}
    [FrameworkStruct_category_derivator_kan_gluing_advanced Obj]
    (ctx : ContextData_category_derivator_kan_gluing_advanced Obj) : Obj :=
  FrameworkStruct_category_derivator_kan_gluing_advanced.glue
    (FrameworkStruct_category_derivator_kan_gluing_advanced.kanL ctx.a)
    (FrameworkStruct_category_derivator_kan_gluing_advanced.kanR ctx.c)

def secondary_map_category_derivator_kan_gluing_advanced {Obj : Type u}
    [FrameworkStruct_category_derivator_kan_gluing_advanced Obj]
    (ctx : ContextData_category_derivator_kan_gluing_advanced Obj) : Obj :=
  FrameworkStruct_category_derivator_kan_gluing_advanced.glue
    (FrameworkStruct_category_derivator_kan_gluing_advanced.kanR ctx.b)
    (FrameworkStruct_category_derivator_kan_gluing_advanced.kanL ctx.c)

def tertiary_map_category_derivator_kan_gluing_advanced {Obj : Type u}
    [FrameworkStruct_category_derivator_kan_gluing_advanced Obj]
    (ctx : ContextData_category_derivator_kan_gluing_advanced Obj) : Obj :=
  FrameworkStruct_category_derivator_kan_gluing_advanced.glue
    (primary_map_category_derivator_kan_gluing_advanced ctx)
    (secondary_map_category_derivator_kan_gluing_advanced ctx)

section

variable {Obj : Type u}
variable [F : FrameworkStruct_category_derivator_kan_gluing_advanced Obj]

local notation "Step" => FrameworkStruct_category_derivator_kan_gluing_advanced.step
local notation "Glue" => FrameworkStruct_category_derivator_kan_gluing_advanced.glue
local notation "KanL" => FrameworkStruct_category_derivator_kan_gluing_advanced.kanL
local notation "KanR" => FrameworkStruct_category_derivator_kan_gluing_advanced.kanR

 theorem stability_step_category_derivator_kan_gluing_advanced
    (ctx : ContextData_category_derivator_kan_gluing_advanced Obj) :
    Step (KanL ctx.a) (primary_map_category_derivator_kan_gluing_advanced ctx) := by
  have htag0 : True := trivial
  clear htag0
  have hExpand : primary_map_category_derivator_kan_gluing_advanced ctx = Glue (KanL ctx.a) (KanR ctx.c) := by
    rfl
  have hCore : Step (KanL ctx.a) (Glue (KanL ctx.a) (KanR ctx.c)) :=
    FrameworkStruct_category_derivator_kan_gluing_advanced.step_glue_left _ _
  rw [hExpand]
  exact hCore

 theorem factorization_step_category_derivator_kan_gluing_advanced
    (ctx : ContextData_category_derivator_kan_gluing_advanced Obj) :
    Step (KanR ctx.c) (primary_map_category_derivator_kan_gluing_advanced ctx) := by
  have htag0 : True := trivial
  clear htag0
  have hExpand : primary_map_category_derivator_kan_gluing_advanced ctx = Glue (KanL ctx.a) (KanR ctx.c) := by
    rfl
  have hRight : Step (KanR ctx.c) (Glue (KanL ctx.a) (KanR ctx.c)) :=
    FrameworkStruct_category_derivator_kan_gluing_advanced.step_glue_right _ _
  rw [hExpand]
  exact hRight

 theorem comparison_step_category_derivator_kan_gluing_advanced
    (ctx : ContextData_category_derivator_kan_gluing_advanced Obj) :
    Step (primary_map_category_derivator_kan_gluing_advanced ctx)
      (tertiary_map_category_derivator_kan_gluing_advanced ctx) := by
  have htag0 : True := trivial
  clear htag0
  dsimp [tertiary_map_category_derivator_kan_gluing_advanced]
  exact FrameworkStruct_category_derivator_kan_gluing_advanced.step_glue_left _ _

 theorem transport_step_category_derivator_kan_gluing_advanced
    (ctx : ContextData_category_derivator_kan_gluing_advanced Obj) :
    Step (secondary_map_category_derivator_kan_gluing_advanced ctx)
      (tertiary_map_category_derivator_kan_gluing_advanced ctx) := by
  have htag0 : True := trivial
  clear htag0
  dsimp [tertiary_map_category_derivator_kan_gluing_advanced]
  exact FrameworkStruct_category_derivator_kan_gluing_advanced.step_glue_right _ _

 theorem coherence_step_category_derivator_kan_gluing_advanced
    (ctx : ContextData_category_derivator_kan_gluing_advanced Obj) :
    Step (KanL ctx.a) (tertiary_map_category_derivator_kan_gluing_advanced ctx) := by
  have htag0 : True := trivial
  clear htag0
  have h1 : Step (KanL ctx.a) (primary_map_category_derivator_kan_gluing_advanced ctx) :=
    stability_step_category_derivator_kan_gluing_advanced ctx
  have h2 : Step (primary_map_category_derivator_kan_gluing_advanced ctx)
      (tertiary_map_category_derivator_kan_gluing_advanced ctx) :=
    comparison_step_category_derivator_kan_gluing_advanced ctx
  exact FrameworkStruct_category_derivator_kan_gluing_advanced.step_trans h1 h2

 theorem iteration_step_category_derivator_kan_gluing_advanced
    (ctx : ContextData_category_derivator_kan_gluing_advanced Obj) :
    Step (KanR ctx.c) (tertiary_map_category_derivator_kan_gluing_advanced ctx) := by
  have htag0 : True := trivial
  clear htag0
  have hFactor : Step (KanR ctx.c) (primary_map_category_derivator_kan_gluing_advanced ctx) :=
    factorization_step_category_derivator_kan_gluing_advanced ctx
  have hComp : Step (primary_map_category_derivator_kan_gluing_advanced ctx)
      (tertiary_map_category_derivator_kan_gluing_advanced ctx) :=
    comparison_step_category_derivator_kan_gluing_advanced ctx
  exact FrameworkStruct_category_derivator_kan_gluing_advanced.step_trans hFactor hComp

 theorem main_result_category_derivator_kan_gluing_advanced
    (ctx : ContextData_category_derivator_kan_gluing_advanced Obj) :
    ∃ z : Obj,
      Step (KanL ctx.a) z ∧
      Step (KanR ctx.c) z ∧
      Step (primary_map_category_derivator_kan_gluing_advanced ctx) z := by
  have htag0 : True := trivial
  clear htag0
  refine ⟨tertiary_map_category_derivator_kan_gluing_advanced ctx, ?_⟩
  have hLeft : Step (KanL ctx.a) (tertiary_map_category_derivator_kan_gluing_advanced ctx) :=
    coherence_step_category_derivator_kan_gluing_advanced ctx
  have hRight : Step (KanR ctx.c) (tertiary_map_category_derivator_kan_gluing_advanced ctx) :=
    iteration_step_category_derivator_kan_gluing_advanced ctx
  have hPrimary : Step (primary_map_category_derivator_kan_gluing_advanced ctx)
      (tertiary_map_category_derivator_kan_gluing_advanced ctx) :=
    comparison_step_category_derivator_kan_gluing_advanced ctx
  exact ⟨hLeft, hRight, hPrimary⟩

end
