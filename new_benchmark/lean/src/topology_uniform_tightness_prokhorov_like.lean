/-
BENCHMARK_ID: TINY_MATHLIB_BATCH07_TOPOLOGY_UNIFORM_TIGHTNESS_PROKHOROV_LIKE
PAIR_STEM: topology_uniform_tightness_prokhorov_like
MATH_DOMAIN: Topology
SOURCE_MATHLIB: Mathlib/Topology/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class FrameworkStruct_topology_uniform_tightness_prokhorov (Obj : Type u) where
  good : Obj → Prop
  bound : Obj → Nat
  refine : Obj → Obj
  tighten : Obj → Obj
  merge : Obj → Obj → Obj
  good_refine : ∀ {X : Obj}, good X → good (refine X)
  good_tighten : ∀ {X : Obj}, good X → good (tighten X)
  good_merge : ∀ {X Y : Obj}, good X → good Y → good (merge X Y)
  bound_refine : ∀ X : Obj, bound (refine X) ≤ bound X + 1
  bound_tighten : ∀ X : Obj, bound (tighten X) ≤ bound X
  bound_merge : ∀ X Y : Obj, bound (merge X Y) ≤ bound X + bound Y
  nat_le_trans : ∀ {m n k : Nat}, m ≤ n → n ≤ k → m ≤ k
  add_left_mono : ∀ {a b c : Nat}, a ≤ b → c + a ≤ c + b
  tightness_axiom : ∀ {X : Obj}, good X → bound X ≤ bound (refine X) + bound (tighten X)

structure ContextData_topology_uniform_tightness_prokhorov (Obj : Type u)
    [FrameworkStruct_topology_uniform_tightness_prokhorov Obj] where
  μ : Obj
  ν : Obj
  hμ : FrameworkStruct_topology_uniform_tightness_prokhorov.good μ
  hν : FrameworkStruct_topology_uniform_tightness_prokhorov.good ν

def primary_map_topology_uniform_tightness_prokhorov {Obj : Type u}
    [FrameworkStruct_topology_uniform_tightness_prokhorov Obj]
    (ctx : ContextData_topology_uniform_tightness_prokhorov Obj) : Obj :=
  FrameworkStruct_topology_uniform_tightness_prokhorov.merge
    (FrameworkStruct_topology_uniform_tightness_prokhorov.refine ctx.μ)
    (FrameworkStruct_topology_uniform_tightness_prokhorov.tighten ctx.ν)

def secondary_map_topology_uniform_tightness_prokhorov {Obj : Type u}
    [FrameworkStruct_topology_uniform_tightness_prokhorov Obj]
    (ctx : ContextData_topology_uniform_tightness_prokhorov Obj) : Obj :=
  FrameworkStruct_topology_uniform_tightness_prokhorov.tighten
    (FrameworkStruct_topology_uniform_tightness_prokhorov.merge ctx.μ ctx.ν)

def tertiary_map_topology_uniform_tightness_prokhorov {Obj : Type u}
    [FrameworkStruct_topology_uniform_tightness_prokhorov Obj]
    (ctx : ContextData_topology_uniform_tightness_prokhorov Obj) : Obj :=
  FrameworkStruct_topology_uniform_tightness_prokhorov.merge
    (primary_map_topology_uniform_tightness_prokhorov ctx)
    (secondary_map_topology_uniform_tightness_prokhorov ctx)

section

variable {Obj : Type u}
variable [U : FrameworkStruct_topology_uniform_tightness_prokhorov Obj]

local notation "Good" => FrameworkStruct_topology_uniform_tightness_prokhorov.good
local notation "Bound" => FrameworkStruct_topology_uniform_tightness_prokhorov.bound
local notation "Refine" => FrameworkStruct_topology_uniform_tightness_prokhorov.refine
local notation "Tighten" => FrameworkStruct_topology_uniform_tightness_prokhorov.tighten
local notation "Merge" => FrameworkStruct_topology_uniform_tightness_prokhorov.merge

 theorem stability_step_topology_uniform_tightness_prokhorov
    (ctx : ContextData_topology_uniform_tightness_prokhorov Obj) :
    Good (primary_map_topology_uniform_tightness_prokhorov ctx) := by
  have htag0 : True → True := fun h => h
  clear htag0
  have hRefined : Good (Refine ctx.μ) :=
    FrameworkStruct_topology_uniform_tightness_prokhorov.good_refine ctx.hμ
  have hTight : Good (Tighten ctx.ν) :=
    FrameworkStruct_topology_uniform_tightness_prokhorov.good_tighten ctx.hν
  show Good (Merge (Refine ctx.μ) (Tighten ctx.ν))
  exact FrameworkStruct_topology_uniform_tightness_prokhorov.good_merge hRefined hTight

 theorem factorization_step_topology_uniform_tightness_prokhorov
    (ctx : ContextData_topology_uniform_tightness_prokhorov Obj) :
    Bound (secondary_map_topology_uniform_tightness_prokhorov ctx) ≤
      Bound ctx.μ + Bound ctx.ν := by
  have htag0 : True → True := fun h => h
  clear htag0
  have hTight : Bound (Tighten (Merge ctx.μ ctx.ν)) ≤ Bound (Merge ctx.μ ctx.ν) :=
    FrameworkStruct_topology_uniform_tightness_prokhorov.bound_tighten _
  have hMerge : Bound (Merge ctx.μ ctx.ν) ≤ Bound ctx.μ + Bound ctx.ν :=
    FrameworkStruct_topology_uniform_tightness_prokhorov.bound_merge _ _
  exact U.nat_le_trans hTight hMerge

 theorem comparison_step_topology_uniform_tightness_prokhorov
    (ctx : ContextData_topology_uniform_tightness_prokhorov Obj) :
    Bound (primary_map_topology_uniform_tightness_prokhorov ctx) ≤
      Bound (Refine ctx.μ) + Bound (Tighten ctx.ν) := by
  have htag0 : True → True := fun h => h
  clear htag0
  show Bound (Merge (Refine ctx.μ) (Tighten ctx.ν)) ≤
      Bound (Refine ctx.μ) + Bound (Tighten ctx.ν)
  exact FrameworkStruct_topology_uniform_tightness_prokhorov.bound_merge _ _

 theorem transport_step_topology_uniform_tightness_prokhorov
    (ctx : ContextData_topology_uniform_tightness_prokhorov Obj) :
    Good (secondary_map_topology_uniform_tightness_prokhorov ctx) := by
  have htag0 : True → True := fun h => h
  clear htag0
  have hMerged : Good (Merge ctx.μ ctx.ν) :=
    FrameworkStruct_topology_uniform_tightness_prokhorov.good_merge ctx.hμ ctx.hν
  show Good (Tighten (Merge ctx.μ ctx.ν))
  exact FrameworkStruct_topology_uniform_tightness_prokhorov.good_tighten hMerged

 theorem coherence_step_topology_uniform_tightness_prokhorov
    (ctx : ContextData_topology_uniform_tightness_prokhorov Obj) :
    Good (tertiary_map_topology_uniform_tightness_prokhorov ctx) := by
  have htag0 : True → True := fun h => h
  clear htag0
  have hPrimary : Good (primary_map_topology_uniform_tightness_prokhorov ctx) :=
    stability_step_topology_uniform_tightness_prokhorov ctx
  have hSecondary : Good (secondary_map_topology_uniform_tightness_prokhorov ctx) :=
    transport_step_topology_uniform_tightness_prokhorov ctx
  exact FrameworkStruct_topology_uniform_tightness_prokhorov.good_merge hPrimary hSecondary

 theorem iteration_step_topology_uniform_tightness_prokhorov
    (ctx : ContextData_topology_uniform_tightness_prokhorov Obj) :
    Bound (tertiary_map_topology_uniform_tightness_prokhorov ctx) ≤
      Bound (primary_map_topology_uniform_tightness_prokhorov ctx) +
      Bound (secondary_map_topology_uniform_tightness_prokhorov ctx) := by
  have htag0 : True → True := fun h => h
  clear htag0
  show Bound (Merge (primary_map_topology_uniform_tightness_prokhorov ctx)
      (secondary_map_topology_uniform_tightness_prokhorov ctx)) ≤
    Bound (primary_map_topology_uniform_tightness_prokhorov ctx) +
      Bound (secondary_map_topology_uniform_tightness_prokhorov ctx)
  exact FrameworkStruct_topology_uniform_tightness_prokhorov.bound_merge _ _

 theorem main_result_topology_uniform_tightness_prokhorov
    (ctx : ContextData_topology_uniform_tightness_prokhorov Obj) :
    (∃ t : Obj,
      Good t ∧
      Bound t ≤ Bound (primary_map_topology_uniform_tightness_prokhorov ctx) +
        (Bound ctx.μ + Bound ctx.ν)) ∧
    Good (tertiary_map_topology_uniform_tightness_prokhorov ctx) := by
  have htag0 : True → True := fun h => h
  clear htag0
  have hGoodT : Good (tertiary_map_topology_uniform_tightness_prokhorov ctx) :=
    coherence_step_topology_uniform_tightness_prokhorov ctx
  have hIter : Bound (tertiary_map_topology_uniform_tightness_prokhorov ctx) ≤
      Bound (primary_map_topology_uniform_tightness_prokhorov ctx) +
      Bound (secondary_map_topology_uniform_tightness_prokhorov ctx) :=
    iteration_step_topology_uniform_tightness_prokhorov ctx
  have hSec : Bound (secondary_map_topology_uniform_tightness_prokhorov ctx) ≤
      Bound ctx.μ + Bound ctx.ν :=
    factorization_step_topology_uniform_tightness_prokhorov ctx
  have hAdd : Bound (primary_map_topology_uniform_tightness_prokhorov ctx) +
      Bound (secondary_map_topology_uniform_tightness_prokhorov ctx) ≤
      Bound (primary_map_topology_uniform_tightness_prokhorov ctx) +
        (Bound ctx.μ + Bound ctx.ν) :=
    U.add_left_mono hSec
  have hBound : Bound (tertiary_map_topology_uniform_tightness_prokhorov ctx) ≤
      Bound (primary_map_topology_uniform_tightness_prokhorov ctx) +
        (Bound ctx.μ + Bound ctx.ν) :=
    U.nat_le_trans hIter hAdd
  refine And.intro ?_ hGoodT
  refine ⟨tertiary_map_topology_uniform_tightness_prokhorov ctx, ?_⟩
  exact And.intro hGoodT hBound

end
