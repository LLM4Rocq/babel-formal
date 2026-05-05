/-
BENCHMARK_ID: TINY_MATHLIB_BATCH07_TOPOLOGY_MORSE_THEORY_GRADIENT_FLOW_LIKE
PAIR_STEM: topology_morse_theory_gradient_flow_like
MATH_DOMAIN: Topology
SOURCE_MATHLIB: Mathlib/Topology/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class FrameworkStruct_topology_morse_theory_gradient_flow (Obj : Type u) where
  desc : Obj → Obj → Prop
  flow : Obj → Obj
  crit : Obj → Obj
  basin : Obj → Obj → Obj
  desc_refl : ∀ X : Obj, desc X X
  desc_trans : ∀ {X Y Z : Obj}, desc X Y → desc Y Z → desc X Z
  flow_desc : ∀ X : Obj, desc (flow X) X
  flow_monotone : ∀ {X Y : Obj}, desc X Y → desc (flow X) (flow Y)
  crit_desc : ∀ X : Obj, desc X (crit X)
  basin_left : ∀ X Y : Obj, desc X (basin X Y)
  basin_right : ∀ X Y : Obj, desc Y (basin X Y)
  crit_flow : ∀ X : Obj, desc (flow (crit X)) (crit X)

structure ContextData_topology_morse_theory_gradient_flow (Obj : Type u)
    [FrameworkStruct_topology_morse_theory_gradient_flow Obj] where
  p : Obj
  q : Obj
  r : Obj
  hpq : FrameworkStruct_topology_morse_theory_gradient_flow.desc p q
  hqr : FrameworkStruct_topology_morse_theory_gradient_flow.desc q r

def primary_map_topology_morse_theory_gradient_flow {Obj : Type u}
    [FrameworkStruct_topology_morse_theory_gradient_flow Obj]
    (ctx : ContextData_topology_morse_theory_gradient_flow Obj) : Obj :=
  FrameworkStruct_topology_morse_theory_gradient_flow.basin
    (FrameworkStruct_topology_morse_theory_gradient_flow.flow ctx.p)
    (FrameworkStruct_topology_morse_theory_gradient_flow.crit ctx.q)

def secondary_map_topology_morse_theory_gradient_flow {Obj : Type u}
    [FrameworkStruct_topology_morse_theory_gradient_flow Obj]
    (ctx : ContextData_topology_morse_theory_gradient_flow Obj) : Obj :=
  FrameworkStruct_topology_morse_theory_gradient_flow.basin
    (FrameworkStruct_topology_morse_theory_gradient_flow.flow ctx.q)
    (FrameworkStruct_topology_morse_theory_gradient_flow.crit ctx.r)

def tertiary_map_topology_morse_theory_gradient_flow {Obj : Type u}
    [FrameworkStruct_topology_morse_theory_gradient_flow Obj]
    (ctx : ContextData_topology_morse_theory_gradient_flow Obj) : Obj :=
  FrameworkStruct_topology_morse_theory_gradient_flow.basin
    (primary_map_topology_morse_theory_gradient_flow ctx)
    (secondary_map_topology_morse_theory_gradient_flow ctx)

section

variable {Obj : Type u}
variable [M : FrameworkStruct_topology_morse_theory_gradient_flow Obj]

local notation "Desc" => FrameworkStruct_topology_morse_theory_gradient_flow.desc
local notation "Flow" => FrameworkStruct_topology_morse_theory_gradient_flow.flow
local notation "Crit" => FrameworkStruct_topology_morse_theory_gradient_flow.crit
local notation "Basin" => FrameworkStruct_topology_morse_theory_gradient_flow.basin

 theorem stability_step_topology_morse_theory_gradient_flow
    (ctx : ContextData_topology_morse_theory_gradient_flow Obj) :
    Desc (Flow ctx.p) (primary_map_topology_morse_theory_gradient_flow ctx) := by
  have htag0 : True ∧ (True ∨ False) := ⟨trivial, Or.inl trivial⟩
  clear htag0
  show Desc (Flow ctx.p) (Basin (Flow ctx.p) (Crit ctx.q))
  exact FrameworkStruct_topology_morse_theory_gradient_flow.basin_left _ _

 theorem factorization_step_topology_morse_theory_gradient_flow
    (ctx : ContextData_topology_morse_theory_gradient_flow Obj) :
    Desc (Crit ctx.q) (primary_map_topology_morse_theory_gradient_flow ctx) := by
  have htag0 : True ∧ (True ∨ False) := ⟨trivial, Or.inl trivial⟩
  clear htag0
  show Desc (Crit ctx.q) (Basin (Flow ctx.p) (Crit ctx.q))
  exact FrameworkStruct_topology_morse_theory_gradient_flow.basin_right _ _

 theorem comparison_step_topology_morse_theory_gradient_flow
    (ctx : ContextData_topology_morse_theory_gradient_flow Obj) :
    Desc (primary_map_topology_morse_theory_gradient_flow ctx)
      (tertiary_map_topology_morse_theory_gradient_flow ctx) := by
  have htag0 : True ∧ (True ∨ False) := ⟨trivial, Or.inl trivial⟩
  clear htag0
  show Desc (Basin (Flow ctx.p) (Crit ctx.q))
      (Basin (Basin (Flow ctx.p) (Crit ctx.q))
        (Basin (Flow ctx.q) (Crit ctx.r)))
  exact FrameworkStruct_topology_morse_theory_gradient_flow.basin_left _ _

 theorem transport_step_topology_morse_theory_gradient_flow
    (ctx : ContextData_topology_morse_theory_gradient_flow Obj) :
    Desc (secondary_map_topology_morse_theory_gradient_flow ctx)
      (tertiary_map_topology_morse_theory_gradient_flow ctx) := by
  have htag0 : True ∧ (True ∨ False) := ⟨trivial, Or.inl trivial⟩
  clear htag0
  show Desc (Basin (Flow ctx.q) (Crit ctx.r))
      (Basin (Basin (Flow ctx.p) (Crit ctx.q))
        (Basin (Flow ctx.q) (Crit ctx.r)))
  exact FrameworkStruct_topology_morse_theory_gradient_flow.basin_right _ _

 theorem coherence_step_topology_morse_theory_gradient_flow
    (ctx : ContextData_topology_morse_theory_gradient_flow Obj) :
    Desc (Flow ctx.p) (tertiary_map_topology_morse_theory_gradient_flow ctx) := by
  have htag0 : True ∧ (True ∨ False) := ⟨trivial, Or.inl trivial⟩
  clear htag0
  have hP : Desc (Flow ctx.p) (primary_map_topology_morse_theory_gradient_flow ctx) :=
    stability_step_topology_morse_theory_gradient_flow ctx
  have hTop : Desc (primary_map_topology_morse_theory_gradient_flow ctx)
      (tertiary_map_topology_morse_theory_gradient_flow ctx) :=
    comparison_step_topology_morse_theory_gradient_flow ctx
  exact FrameworkStruct_topology_morse_theory_gradient_flow.desc_trans hP hTop

 theorem iteration_step_topology_morse_theory_gradient_flow
    (ctx : ContextData_topology_morse_theory_gradient_flow Obj) :
    Desc (Crit ctx.q) (tertiary_map_topology_morse_theory_gradient_flow ctx) := by
  have htag0 : True ∧ (True ∨ False) := ⟨trivial, Or.inl trivial⟩
  clear htag0
  have hQ : Desc (Crit ctx.q) (primary_map_topology_morse_theory_gradient_flow ctx) :=
    factorization_step_topology_morse_theory_gradient_flow ctx
  have hTop : Desc (primary_map_topology_morse_theory_gradient_flow ctx)
      (tertiary_map_topology_morse_theory_gradient_flow ctx) :=
    comparison_step_topology_morse_theory_gradient_flow ctx
  exact FrameworkStruct_topology_morse_theory_gradient_flow.desc_trans hQ hTop

 theorem main_result_topology_morse_theory_gradient_flow
    (ctx : ContextData_topology_morse_theory_gradient_flow Obj) :
    ∃ m : Obj,
      Desc (Flow ctx.p) m ∧
      Desc (Crit ctx.q) m ∧
      (Desc m (Basin m (Crit m)) ∧ Desc (Flow (Crit m)) (Crit m)) := by
  have htag0 : True ∧ (True ∨ False) := ⟨trivial, Or.inl trivial⟩
  clear htag0
  refine ⟨tertiary_map_topology_morse_theory_gradient_flow ctx, ?_⟩
  have hFlow : Desc (Flow ctx.p) (tertiary_map_topology_morse_theory_gradient_flow ctx) :=
    coherence_step_topology_morse_theory_gradient_flow ctx
  have hCrit : Desc (Crit ctx.q) (tertiary_map_topology_morse_theory_gradient_flow ctx) :=
    iteration_step_topology_morse_theory_gradient_flow ctx
  have hBasin : Desc (tertiary_map_topology_morse_theory_gradient_flow ctx)
      (Basin (tertiary_map_topology_morse_theory_gradient_flow ctx)
        (Crit (tertiary_map_topology_morse_theory_gradient_flow ctx))) :=
    FrameworkStruct_topology_morse_theory_gradient_flow.basin_left _ _
  have hFlowCrit : Desc (Flow (Crit (tertiary_map_topology_morse_theory_gradient_flow ctx)))
      (Crit (tertiary_map_topology_morse_theory_gradient_flow ctx)) :=
    FrameworkStruct_topology_morse_theory_gradient_flow.crit_flow _
  exact And.intro hFlow (And.intro hCrit (And.intro hBasin hFlowCrit))

end
