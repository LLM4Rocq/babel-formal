/-
BENCHMARK_ID: TINY_MATHLIB_BATCH07_TOPOLOGY_ETALE_FUNDAMENTAL_GROUPOID_LIKE
PAIR_STEM: topology_etale_fundamental_groupoid_like
MATH_DOMAIN: Topology
SOURCE_MATHLIB: Mathlib/Topology/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class FrameworkStruct_topology_etale_fundamental_groupoid (Point : Type u) where
  path : Point → Point → Prop
  lift : Point → Point
  concat : Point → Point → Point
  basepoint : Point
  path_refl : ∀ p : Point, path p p
  path_symm : ∀ {p q : Point}, path p q → path q p
  path_trans : ∀ {p q r : Point}, path p q → path q r → path p r
  path_lift : ∀ {p q : Point}, path p q → path (lift p) (lift q)
  path_to_concat_left : ∀ p q : Point, path p (concat p q)
  path_to_concat_right : ∀ p q : Point, path q (concat p q)
  concat_base_left : ∀ p : Point, concat basepoint p = p

structure ContextData_topology_etale_fundamental_groupoid (Point : Type u)
    [FrameworkStruct_topology_etale_fundamental_groupoid Point] where
  p : Point
  q : Point
  r : Point
  hpq : FrameworkStruct_topology_etale_fundamental_groupoid.path p q
  hqr : FrameworkStruct_topology_etale_fundamental_groupoid.path q r

def primary_map_topology_etale_fundamental_groupoid {Point : Type u}
    [FrameworkStruct_topology_etale_fundamental_groupoid Point]
    (ctx : ContextData_topology_etale_fundamental_groupoid Point) : Point :=
  FrameworkStruct_topology_etale_fundamental_groupoid.concat ctx.p ctx.q

def secondary_map_topology_etale_fundamental_groupoid {Point : Type u}
    [FrameworkStruct_topology_etale_fundamental_groupoid Point]
    (ctx : ContextData_topology_etale_fundamental_groupoid Point) : Point :=
  FrameworkStruct_topology_etale_fundamental_groupoid.lift ctx.r

def tertiary_map_topology_etale_fundamental_groupoid {Point : Type u}
    [FrameworkStruct_topology_etale_fundamental_groupoid Point]
    (ctx : ContextData_topology_etale_fundamental_groupoid Point) : Point :=
  FrameworkStruct_topology_etale_fundamental_groupoid.concat
    (primary_map_topology_etale_fundamental_groupoid ctx)
    (secondary_map_topology_etale_fundamental_groupoid ctx)

section

variable {Point : Type u}
variable [E : FrameworkStruct_topology_etale_fundamental_groupoid Point]

local notation "Path" => FrameworkStruct_topology_etale_fundamental_groupoid.path
local notation "Lift" => FrameworkStruct_topology_etale_fundamental_groupoid.lift
local notation "Concat" => FrameworkStruct_topology_etale_fundamental_groupoid.concat

 theorem stability_step_topology_etale_fundamental_groupoid
    (ctx : ContextData_topology_etale_fundamental_groupoid Point) :
    Path ctx.p (primary_map_topology_etale_fundamental_groupoid ctx) := by
  have htag0 : False ∨ True := Or.inr trivial
  clear htag0
  show Path ctx.p (Concat ctx.p ctx.q)
  exact FrameworkStruct_topology_etale_fundamental_groupoid.path_to_concat_left _ _

 theorem factorization_step_topology_etale_fundamental_groupoid
    (ctx : ContextData_topology_etale_fundamental_groupoid Point) :
    Path ctx.q (primary_map_topology_etale_fundamental_groupoid ctx) := by
  have htag0 : False ∨ True := Or.inr trivial
  clear htag0
  show Path ctx.q (Concat ctx.p ctx.q)
  exact FrameworkStruct_topology_etale_fundamental_groupoid.path_to_concat_right _ _

 theorem comparison_step_topology_etale_fundamental_groupoid
    (ctx : ContextData_topology_etale_fundamental_groupoid Point) :
    Path (primary_map_topology_etale_fundamental_groupoid ctx)
      (tertiary_map_topology_etale_fundamental_groupoid ctx) := by
  have htag0 : False ∨ True := Or.inr trivial
  clear htag0
  show Path (Concat ctx.p ctx.q)
      (Concat (Concat ctx.p ctx.q) (Lift ctx.r))
  exact FrameworkStruct_topology_etale_fundamental_groupoid.path_to_concat_left _ _

 theorem transport_step_topology_etale_fundamental_groupoid
    (ctx : ContextData_topology_etale_fundamental_groupoid Point) :
    Path (secondary_map_topology_etale_fundamental_groupoid ctx)
      (tertiary_map_topology_etale_fundamental_groupoid ctx) := by
  have htag0 : False ∨ True := Or.inr trivial
  clear htag0
  show Path (Lift ctx.r) (Concat (Concat ctx.p ctx.q) (Lift ctx.r))
  exact FrameworkStruct_topology_etale_fundamental_groupoid.path_to_concat_right _ _

 theorem coherence_step_topology_etale_fundamental_groupoid
    (ctx : ContextData_topology_etale_fundamental_groupoid Point) :
    Path ctx.p (tertiary_map_topology_etale_fundamental_groupoid ctx) := by
  have htag0 : False ∨ True := Or.inr trivial
  clear htag0
  have hp : Path ctx.p (primary_map_topology_etale_fundamental_groupoid ctx) :=
    stability_step_topology_etale_fundamental_groupoid ctx
  have hq : Path (primary_map_topology_etale_fundamental_groupoid ctx)
      (tertiary_map_topology_etale_fundamental_groupoid ctx) :=
    comparison_step_topology_etale_fundamental_groupoid ctx
  exact FrameworkStruct_topology_etale_fundamental_groupoid.path_trans hp hq

 theorem iteration_step_topology_etale_fundamental_groupoid
    (ctx : ContextData_topology_etale_fundamental_groupoid Point) :
    Path ctx.q (tertiary_map_topology_etale_fundamental_groupoid ctx) := by
  have htag0 : False ∨ True := Or.inr trivial
  clear htag0
  have hp : Path ctx.q (primary_map_topology_etale_fundamental_groupoid ctx) :=
    factorization_step_topology_etale_fundamental_groupoid ctx
  have hq : Path (primary_map_topology_etale_fundamental_groupoid ctx)
      (tertiary_map_topology_etale_fundamental_groupoid ctx) :=
    comparison_step_topology_etale_fundamental_groupoid ctx
  exact FrameworkStruct_topology_etale_fundamental_groupoid.path_trans hp hq

 theorem main_result_topology_etale_fundamental_groupoid
    (ctx : ContextData_topology_etale_fundamental_groupoid Point) :
    (Path ctx.p (tertiary_map_topology_etale_fundamental_groupoid ctx) ∧
      Path ctx.q (tertiary_map_topology_etale_fundamental_groupoid ctx)) ∧
    ∃ t : Point,
      Path (tertiary_map_topology_etale_fundamental_groupoid ctx) t ∧
      Path t (tertiary_map_topology_etale_fundamental_groupoid ctx) := by
  have htag0 : False ∨ True := Or.inr trivial
  clear htag0
  have hp : Path ctx.p (tertiary_map_topology_etale_fundamental_groupoid ctx) :=
    coherence_step_topology_etale_fundamental_groupoid ctx
  have hq : Path ctx.q (tertiary_map_topology_etale_fundamental_groupoid ctx) :=
    iteration_step_topology_etale_fundamental_groupoid ctx
  refine And.intro (And.intro hp hq) ?_
  refine ⟨tertiary_map_topology_etale_fundamental_groupoid ctx, ?_⟩
  have hrr : Path (tertiary_map_topology_etale_fundamental_groupoid ctx)
      (tertiary_map_topology_etale_fundamental_groupoid ctx) :=
    FrameworkStruct_topology_etale_fundamental_groupoid.path_refl _
  exact And.intro hrr hrr

end
