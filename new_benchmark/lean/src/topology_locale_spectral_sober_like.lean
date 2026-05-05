/-
BENCHMARK_ID: TINY_MATHLIB_BATCH07_TOPOLOGY_LOCALE_SPECTRAL_SOBER_LIKE
PAIR_STEM: topology_locale_spectral_sober_like
MATH_DOMAIN: Topology
SOURCE_MATHLIB: Mathlib/Topology/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class FrameworkStruct_topology_locale_spectral_sober (Obj : Type u) where
  le : Obj → Obj → Prop
  join : Obj → Obj → Obj
  closure : Obj → Obj
  spectral : Obj → Obj
  le_refl : ∀ X : Obj, le X X
  le_trans : ∀ {X Y Z : Obj}, le X Y → le Y Z → le X Z
  join_left : ∀ X Y : Obj, le X (join X Y)
  join_right : ∀ X Y : Obj, le Y (join X Y)
  closure_extensive : ∀ X : Obj, le X (closure X)
  closure_monotone : ∀ {X Y : Obj}, le X Y → le (closure X) (closure Y)
  closure_idem : ∀ X : Obj, le (closure (closure X)) (closure X)
  spectral_bridge : ∀ X : Obj, le (closure X) (spectral (closure X))

structure ContextData_topology_locale_spectral_sober (Obj : Type u)
    [FrameworkStruct_topology_locale_spectral_sober Obj] where
  u : Obj
  v : Obj
  w : Obj
  huv : FrameworkStruct_topology_locale_spectral_sober.le u v
  hvw : FrameworkStruct_topology_locale_spectral_sober.le v w

def primary_map_topology_locale_spectral_sober {Obj : Type u}
    [FrameworkStruct_topology_locale_spectral_sober Obj]
    (ctx : ContextData_topology_locale_spectral_sober Obj) : Obj :=
  FrameworkStruct_topology_locale_spectral_sober.closure
    (FrameworkStruct_topology_locale_spectral_sober.join ctx.u ctx.v)

def secondary_map_topology_locale_spectral_sober {Obj : Type u}
    [FrameworkStruct_topology_locale_spectral_sober Obj]
    (ctx : ContextData_topology_locale_spectral_sober Obj) : Obj :=
  FrameworkStruct_topology_locale_spectral_sober.spectral
    (FrameworkStruct_topology_locale_spectral_sober.closure ctx.w)

def tertiary_map_topology_locale_spectral_sober {Obj : Type u}
    [FrameworkStruct_topology_locale_spectral_sober Obj]
    (ctx : ContextData_topology_locale_spectral_sober Obj) : Obj :=
  FrameworkStruct_topology_locale_spectral_sober.closure
    (FrameworkStruct_topology_locale_spectral_sober.join
      (primary_map_topology_locale_spectral_sober ctx)
      (secondary_map_topology_locale_spectral_sober ctx))

section

variable {Obj : Type u}
variable [L : FrameworkStruct_topology_locale_spectral_sober Obj]

local notation "Le" => FrameworkStruct_topology_locale_spectral_sober.le
local notation "Join" => FrameworkStruct_topology_locale_spectral_sober.join
local notation "Closure" => FrameworkStruct_topology_locale_spectral_sober.closure
local notation "Spectral" => FrameworkStruct_topology_locale_spectral_sober.spectral

 theorem stability_step_topology_locale_spectral_sober
    (ctx : ContextData_topology_locale_spectral_sober Obj) :
    Le ctx.u (primary_map_topology_locale_spectral_sober ctx) := by
  have htag0 : False → False := fun h => h
  clear htag0
  have huJoin : Le ctx.u (Join ctx.u ctx.v) :=
    FrameworkStruct_topology_locale_spectral_sober.join_left _ _
  have hJoinClose : Le (Join ctx.u ctx.v) (Closure (Join ctx.u ctx.v)) :=
    FrameworkStruct_topology_locale_spectral_sober.closure_extensive _
  exact FrameworkStruct_topology_locale_spectral_sober.le_trans huJoin hJoinClose

 theorem factorization_step_topology_locale_spectral_sober
    (ctx : ContextData_topology_locale_spectral_sober Obj) :
    Le (secondary_map_topology_locale_spectral_sober ctx)
      (tertiary_map_topology_locale_spectral_sober ctx) := by
  have htag0 : False → False := fun h => h
  clear htag0
  have hRight : Le (secondary_map_topology_locale_spectral_sober ctx)
      (Join (primary_map_topology_locale_spectral_sober ctx)
        (secondary_map_topology_locale_spectral_sober ctx)) :=
    FrameworkStruct_topology_locale_spectral_sober.join_right _ _
  have hClose : Le (Join (primary_map_topology_locale_spectral_sober ctx)
      (secondary_map_topology_locale_spectral_sober ctx))
      (Closure (Join (primary_map_topology_locale_spectral_sober ctx)
        (secondary_map_topology_locale_spectral_sober ctx))) :=
    FrameworkStruct_topology_locale_spectral_sober.closure_extensive _
  exact FrameworkStruct_topology_locale_spectral_sober.le_trans hRight hClose

 theorem comparison_step_topology_locale_spectral_sober
    (ctx : ContextData_topology_locale_spectral_sober Obj) :
    Le (primary_map_topology_locale_spectral_sober ctx)
      (tertiary_map_topology_locale_spectral_sober ctx) := by
  have htag0 : False → False := fun h => h
  clear htag0
  have hLeft : Le (primary_map_topology_locale_spectral_sober ctx)
      (Join (primary_map_topology_locale_spectral_sober ctx)
        (secondary_map_topology_locale_spectral_sober ctx)) :=
    FrameworkStruct_topology_locale_spectral_sober.join_left _ _
  have hClose : Le (Join (primary_map_topology_locale_spectral_sober ctx)
      (secondary_map_topology_locale_spectral_sober ctx))
      (tertiary_map_topology_locale_spectral_sober ctx) := by
    simpa [tertiary_map_topology_locale_spectral_sober] using
      (FrameworkStruct_topology_locale_spectral_sober.closure_extensive
        (Join (primary_map_topology_locale_spectral_sober ctx)
          (secondary_map_topology_locale_spectral_sober ctx)))
  exact FrameworkStruct_topology_locale_spectral_sober.le_trans hLeft hClose

 theorem transport_step_topology_locale_spectral_sober
    (ctx : ContextData_topology_locale_spectral_sober Obj) :
    Le (FrameworkStruct_topology_locale_spectral_sober.closure ctx.w)
      (secondary_map_topology_locale_spectral_sober ctx) := by
  have htag0 : False → False := fun h => h
  clear htag0
  show Le (Closure ctx.w) (Spectral (Closure ctx.w))
  exact FrameworkStruct_topology_locale_spectral_sober.spectral_bridge ctx.w

 theorem coherence_step_topology_locale_spectral_sober
    (ctx : ContextData_topology_locale_spectral_sober Obj) :
    Le ctx.u (tertiary_map_topology_locale_spectral_sober ctx) := by
  have htag0 : False → False := fun h => h
  clear htag0
  have hStable : Le ctx.u (primary_map_topology_locale_spectral_sober ctx) :=
    stability_step_topology_locale_spectral_sober ctx
  have hCompare : Le (primary_map_topology_locale_spectral_sober ctx)
      (tertiary_map_topology_locale_spectral_sober ctx) :=
    comparison_step_topology_locale_spectral_sober ctx
  exact FrameworkStruct_topology_locale_spectral_sober.le_trans hStable hCompare

 theorem iteration_step_topology_locale_spectral_sober
    (ctx : ContextData_topology_locale_spectral_sober Obj) :
    Le (Closure (Closure ctx.w)) (tertiary_map_topology_locale_spectral_sober ctx) := by
  have htag0 : False → False := fun h => h
  clear htag0
  have hIdem : Le (Closure (Closure ctx.w)) (Closure ctx.w) :=
    FrameworkStruct_topology_locale_spectral_sober.closure_idem ctx.w
  have hBridge : Le (Closure ctx.w) (secondary_map_topology_locale_spectral_sober ctx) :=
    transport_step_topology_locale_spectral_sober ctx
  have hTop : Le (secondary_map_topology_locale_spectral_sober ctx)
      (tertiary_map_topology_locale_spectral_sober ctx) :=
    factorization_step_topology_locale_spectral_sober ctx
  have hMid : Le (Closure (Closure ctx.w)) (secondary_map_topology_locale_spectral_sober ctx) :=
    FrameworkStruct_topology_locale_spectral_sober.le_trans hIdem hBridge
  exact FrameworkStruct_topology_locale_spectral_sober.le_trans hMid hTop

 theorem main_result_topology_locale_spectral_sober
    (ctx : ContextData_topology_locale_spectral_sober Obj) :
    ∃ t : Obj,
      (Le ctx.u t ∧ Le (Closure ctx.w) t) ∧
      Le t (Closure t) := by
  have htag0 : False → False := fun h => h
  clear htag0
  refine ⟨tertiary_map_topology_locale_spectral_sober ctx, ?_⟩
  have hu : Le ctx.u (tertiary_map_topology_locale_spectral_sober ctx) :=
    coherence_step_topology_locale_spectral_sober ctx
  have hw1 : Le (Closure ctx.w) (secondary_map_topology_locale_spectral_sober ctx) :=
    transport_step_topology_locale_spectral_sober ctx
  have hw2 : Le (secondary_map_topology_locale_spectral_sober ctx)
      (tertiary_map_topology_locale_spectral_sober ctx) :=
    factorization_step_topology_locale_spectral_sober ctx
  have hw : Le (Closure ctx.w) (tertiary_map_topology_locale_spectral_sober ctx) :=
    FrameworkStruct_topology_locale_spectral_sober.le_trans hw1 hw2
  have hClose : Le (tertiary_map_topology_locale_spectral_sober ctx)
      (Closure (tertiary_map_topology_locale_spectral_sober ctx)) :=
    FrameworkStruct_topology_locale_spectral_sober.closure_extensive _
  exact And.intro (And.intro hu hw) hClose

end
