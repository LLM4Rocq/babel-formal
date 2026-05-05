/-
BENCHMARK_ID: TINY_MATHLIB_BATCH07_TOPOLOGY_SHEAF_HYPERDESCENT_STACK_LIKE
PAIR_STEM: topology_sheaf_hyperdescent_stack_like
MATH_DOMAIN: Topology
SOURCE_MATHLIB: Mathlib/Topology/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class FrameworkStruct_topology_sheaf_hyperdescent_stack (Obj : Type u) where
  base : Obj → Prop
  refine : Obj → Obj
  glue : Obj → Obj → Obj
  descent : Obj → Obj
  base_refine : ∀ {X : Obj}, base X → base (refine X)
  base_glue : ∀ {X Y : Obj}, base X → base Y → base (glue X Y)
  base_descent : ∀ {X : Obj}, base X → base (descent X)
  descent_refine : ∀ X : Obj, descent (refine X) = refine (descent X)
  glue_refine : ∀ X Y : Obj, refine (glue X Y) = glue (refine X) (refine Y)
  hyper_axiom : ∀ {X : Obj}, base X → base (descent (descent X))
  locality_axiom : ∀ {X Y : Obj}, base (glue X Y) → base X ∧ base Y

structure ContextData_topology_sheaf_hyperdescent_stack (Obj : Type u)
    [FrameworkStruct_topology_sheaf_hyperdescent_stack Obj] where
  u : Obj
  v : Obj
  hu : FrameworkStruct_topology_sheaf_hyperdescent_stack.base u
  hv : FrameworkStruct_topology_sheaf_hyperdescent_stack.base v

def primary_map_topology_sheaf_hyperdescent_stack {Obj : Type u}
    [FrameworkStruct_topology_sheaf_hyperdescent_stack Obj]
    (ctx : ContextData_topology_sheaf_hyperdescent_stack Obj) : Obj :=
  FrameworkStruct_topology_sheaf_hyperdescent_stack.glue
    (FrameworkStruct_topology_sheaf_hyperdescent_stack.refine ctx.u)
    (FrameworkStruct_topology_sheaf_hyperdescent_stack.descent ctx.v)

def secondary_map_topology_sheaf_hyperdescent_stack {Obj : Type u}
    [FrameworkStruct_topology_sheaf_hyperdescent_stack Obj]
    (ctx : ContextData_topology_sheaf_hyperdescent_stack Obj) : Obj :=
  FrameworkStruct_topology_sheaf_hyperdescent_stack.descent
    (FrameworkStruct_topology_sheaf_hyperdescent_stack.glue ctx.u ctx.v)

def tertiary_map_topology_sheaf_hyperdescent_stack {Obj : Type u}
    [FrameworkStruct_topology_sheaf_hyperdescent_stack Obj]
    (ctx : ContextData_topology_sheaf_hyperdescent_stack Obj) : Obj :=
  FrameworkStruct_topology_sheaf_hyperdescent_stack.glue
    (primary_map_topology_sheaf_hyperdescent_stack ctx)
    (secondary_map_topology_sheaf_hyperdescent_stack ctx)

section

variable {Obj : Type u}
variable [T : FrameworkStruct_topology_sheaf_hyperdescent_stack Obj]

local notation "Base" => FrameworkStruct_topology_sheaf_hyperdescent_stack.base
local notation "Refine" => FrameworkStruct_topology_sheaf_hyperdescent_stack.refine
local notation "Glue" => FrameworkStruct_topology_sheaf_hyperdescent_stack.glue
local notation "Descent" => FrameworkStruct_topology_sheaf_hyperdescent_stack.descent

 theorem stability_step_topology_sheaf_hyperdescent_stack
    (ctx : ContextData_topology_sheaf_hyperdescent_stack Obj) :
    Base (primary_map_topology_sheaf_hyperdescent_stack ctx) := by
  have htag0 : True ∨ False := Or.inl trivial
  clear htag0
  have hRefined : Base (Refine ctx.u) :=
    FrameworkStruct_topology_sheaf_hyperdescent_stack.base_refine ctx.hu
  have hDescended : Base (Descent ctx.v) :=
    FrameworkStruct_topology_sheaf_hyperdescent_stack.base_descent ctx.hv
  show Base (Glue (Refine ctx.u) (Descent ctx.v))
  exact FrameworkStruct_topology_sheaf_hyperdescent_stack.base_glue hRefined hDescended

 theorem factorization_step_topology_sheaf_hyperdescent_stack
    (ctx : ContextData_topology_sheaf_hyperdescent_stack Obj) :
    Base (secondary_map_topology_sheaf_hyperdescent_stack ctx) := by
  have htag0 : True ∨ False := Or.inl trivial
  clear htag0
  have hGlued : Base (Glue ctx.u ctx.v) :=
    FrameworkStruct_topology_sheaf_hyperdescent_stack.base_glue ctx.hu ctx.hv
  show Base (Descent (Glue ctx.u ctx.v))
  exact FrameworkStruct_topology_sheaf_hyperdescent_stack.base_descent hGlued

 theorem comparison_step_topology_sheaf_hyperdescent_stack
    (ctx : ContextData_topology_sheaf_hyperdescent_stack Obj) :
    Base (tertiary_map_topology_sheaf_hyperdescent_stack ctx) := by
  have htag0 : True ∨ False := Or.inl trivial
  clear htag0
  have hPrimary : Base (primary_map_topology_sheaf_hyperdescent_stack ctx) :=
    stability_step_topology_sheaf_hyperdescent_stack ctx
  have hSecondary : Base (secondary_map_topology_sheaf_hyperdescent_stack ctx) :=
    factorization_step_topology_sheaf_hyperdescent_stack ctx
  exact FrameworkStruct_topology_sheaf_hyperdescent_stack.base_glue hPrimary hSecondary

 theorem transport_step_topology_sheaf_hyperdescent_stack
    (ctx : ContextData_topology_sheaf_hyperdescent_stack Obj) :
    Base (tertiary_map_topology_sheaf_hyperdescent_stack ctx) →
      Base (primary_map_topology_sheaf_hyperdescent_stack ctx) ∧
      Base (secondary_map_topology_sheaf_hyperdescent_stack ctx) := by
  have htag0 : True ∨ False := Or.inl trivial
  clear htag0
  intro hTertiary
  have hLoc : Base (primary_map_topology_sheaf_hyperdescent_stack ctx) ∧
      Base (secondary_map_topology_sheaf_hyperdescent_stack ctx) :=
    FrameworkStruct_topology_sheaf_hyperdescent_stack.locality_axiom hTertiary
  exact hLoc

 theorem coherence_step_topology_sheaf_hyperdescent_stack
    (ctx : ContextData_topology_sheaf_hyperdescent_stack Obj) :
    Base (Descent (tertiary_map_topology_sheaf_hyperdescent_stack ctx)) := by
  have htag0 : True ∨ False := Or.inl trivial
  clear htag0
  have hTertiary : Base (tertiary_map_topology_sheaf_hyperdescent_stack ctx) :=
    comparison_step_topology_sheaf_hyperdescent_stack ctx
  exact FrameworkStruct_topology_sheaf_hyperdescent_stack.base_descent hTertiary

 theorem iteration_step_topology_sheaf_hyperdescent_stack
    (ctx : ContextData_topology_sheaf_hyperdescent_stack Obj) :
    Base (Descent (Descent (tertiary_map_topology_sheaf_hyperdescent_stack ctx))) := by
  have htag0 : True ∨ False := Or.inl trivial
  clear htag0
  have hTertiary : Base (tertiary_map_topology_sheaf_hyperdescent_stack ctx) :=
    comparison_step_topology_sheaf_hyperdescent_stack ctx
  exact FrameworkStruct_topology_sheaf_hyperdescent_stack.hyper_axiom hTertiary

 theorem main_result_topology_sheaf_hyperdescent_stack
    (ctx : ContextData_topology_sheaf_hyperdescent_stack Obj) :
    ∃ w : Obj,
      Base w ∧
      (Base (tertiary_map_topology_sheaf_hyperdescent_stack ctx) → Base (Descent w)) := by
  have htag0 : True ∨ False := Or.inl trivial
  clear htag0
  refine ⟨tertiary_map_topology_sheaf_hyperdescent_stack ctx, ?_⟩
  have hBase : Base (tertiary_map_topology_sheaf_hyperdescent_stack ctx) :=
    comparison_step_topology_sheaf_hyperdescent_stack ctx
  refine And.intro hBase ?_
  intro hInput
  exact FrameworkStruct_topology_sheaf_hyperdescent_stack.base_descent hInput

end
