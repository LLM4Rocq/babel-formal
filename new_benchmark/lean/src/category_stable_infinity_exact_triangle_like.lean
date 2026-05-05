/-
BENCHMARK_ID: TINY_MATHLIB_BATCH07_CATEGORY_STABLE_INFINITY_EXACT_TRIANGLE_LIKE
PAIR_STEM: category_stable_infinity_exact_triangle_like
MATH_DOMAIN: Category Theory
SOURCE_MATHLIB: Mathlib/CategoryTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class FrameworkStruct_category_stable_infinity_exact_triangle (Obj : Type u) where
  rel : Obj → Obj → Prop
  shift : Obj → Obj
  cone : Obj → Obj
  fiber : Obj → Obj
  rel_refl : ∀ X : Obj, rel X X
  rel_trans : ∀ {X Y Z : Obj}, rel X Y → rel Y Z → rel X Z
  rel_shift : ∀ {X Y : Obj}, rel X Y → rel (shift X) (shift Y)
  rel_cone : ∀ {X Y : Obj}, rel X Y → rel (cone X) (cone Y)
  rel_fiber : ∀ {X Y : Obj}, rel X Y → rel (fiber X) (fiber Y)
  triangle_left : ∀ X : Obj, rel (cone X) (shift X)
  triangle_right : ∀ X : Obj, rel (shift X) (fiber X)

structure ContextData_category_stable_infinity_exact_triangle (Obj : Type u)
    [FrameworkStruct_category_stable_infinity_exact_triangle Obj] where
  x : Obj
  y : Obj
  z : Obj
  hxy : FrameworkStruct_category_stable_infinity_exact_triangle.rel x y
  hyz : FrameworkStruct_category_stable_infinity_exact_triangle.rel y z

def primary_map_category_stable_infinity_exact_triangle {Obj : Type u}
    [FrameworkStruct_category_stable_infinity_exact_triangle Obj]
    (ctx : ContextData_category_stable_infinity_exact_triangle Obj) : Obj :=
  FrameworkStruct_category_stable_infinity_exact_triangle.cone ctx.x

def secondary_map_category_stable_infinity_exact_triangle {Obj : Type u}
    [FrameworkStruct_category_stable_infinity_exact_triangle Obj]
    (ctx : ContextData_category_stable_infinity_exact_triangle Obj) : Obj :=
  FrameworkStruct_category_stable_infinity_exact_triangle.shift ctx.z

def tertiary_map_category_stable_infinity_exact_triangle {Obj : Type u}
    [FrameworkStruct_category_stable_infinity_exact_triangle Obj]
    (ctx : ContextData_category_stable_infinity_exact_triangle Obj) : Obj :=
  FrameworkStruct_category_stable_infinity_exact_triangle.fiber ctx.z

section

variable {Obj : Type u}
variable [S : FrameworkStruct_category_stable_infinity_exact_triangle Obj]

local notation "Rel" => FrameworkStruct_category_stable_infinity_exact_triangle.rel
local notation "Shift" => FrameworkStruct_category_stable_infinity_exact_triangle.shift
local notation "Cone" => FrameworkStruct_category_stable_infinity_exact_triangle.cone
local notation "Fiber" => FrameworkStruct_category_stable_infinity_exact_triangle.fiber

 theorem stability_step_category_stable_infinity_exact_triangle
    (ctx : ContextData_category_stable_infinity_exact_triangle Obj) :
    Rel (primary_map_category_stable_infinity_exact_triangle ctx)
      (Shift ctx.x) := by
  have htag0 : True ∧ True := ⟨trivial, trivial⟩
  clear htag0
  show Rel (Cone ctx.x) (Shift ctx.x)
  exact FrameworkStruct_category_stable_infinity_exact_triangle.triangle_left _

 theorem factorization_step_category_stable_infinity_exact_triangle
    (ctx : ContextData_category_stable_infinity_exact_triangle Obj) :
    Rel (Shift ctx.x) (secondary_map_category_stable_infinity_exact_triangle ctx) := by
  have htag0 : True ∧ True := ⟨trivial, trivial⟩
  clear htag0
  have hxyShift : Rel (Shift ctx.x) (Shift ctx.y) :=
    FrameworkStruct_category_stable_infinity_exact_triangle.rel_shift ctx.hxy
  have hyzShift : Rel (Shift ctx.y) (Shift ctx.z) :=
    FrameworkStruct_category_stable_infinity_exact_triangle.rel_shift ctx.hyz
  have hcomp : Rel (Shift ctx.x) (Shift ctx.z) :=
    FrameworkStruct_category_stable_infinity_exact_triangle.rel_trans hxyShift hyzShift
  simpa [secondary_map_category_stable_infinity_exact_triangle] using hcomp

 theorem comparison_step_category_stable_infinity_exact_triangle
    (ctx : ContextData_category_stable_infinity_exact_triangle Obj) :
    Rel (primary_map_category_stable_infinity_exact_triangle ctx)
      (tertiary_map_category_stable_infinity_exact_triangle ctx) := by
  have htag0 : True ∧ True := ⟨trivial, trivial⟩
  clear htag0
  have h1 : Rel (Cone ctx.x) (Shift ctx.x) :=
    FrameworkStruct_category_stable_infinity_exact_triangle.triangle_left ctx.x
  have h2 : Rel (Shift ctx.x) (secondary_map_category_stable_infinity_exact_triangle ctx) :=
    factorization_step_category_stable_infinity_exact_triangle ctx
  have h3 : Rel (secondary_map_category_stable_infinity_exact_triangle ctx)
      (tertiary_map_category_stable_infinity_exact_triangle ctx) :=
    by
      show Rel (Shift ctx.z) (Fiber ctx.z)
      exact FrameworkStruct_category_stable_infinity_exact_triangle.triangle_right ctx.z
  have h12 : Rel (Cone ctx.x) (secondary_map_category_stable_infinity_exact_triangle ctx) :=
    FrameworkStruct_category_stable_infinity_exact_triangle.rel_trans h1 h2
  have h123 : Rel (Cone ctx.x) (tertiary_map_category_stable_infinity_exact_triangle ctx) :=
    FrameworkStruct_category_stable_infinity_exact_triangle.rel_trans h12 h3
  simpa [primary_map_category_stable_infinity_exact_triangle,
    tertiary_map_category_stable_infinity_exact_triangle] using h123

  theorem transport_step_category_stable_infinity_exact_triangle
    (ctx : ContextData_category_stable_infinity_exact_triangle Obj) :
    Rel (secondary_map_category_stable_infinity_exact_triangle ctx)
      (tertiary_map_category_stable_infinity_exact_triangle ctx) := by
  have htag0 : True ∧ True := ⟨trivial, trivial⟩
  clear htag0
  show Rel (Shift ctx.z) (Fiber ctx.z)
  exact FrameworkStruct_category_stable_infinity_exact_triangle.triangle_right ctx.z

 theorem coherence_step_category_stable_infinity_exact_triangle
    (ctx : ContextData_category_stable_infinity_exact_triangle Obj) :
    Rel (primary_map_category_stable_infinity_exact_triangle ctx)
      (secondary_map_category_stable_infinity_exact_triangle ctx) := by
  have htag0 : True ∧ True := ⟨trivial, trivial⟩
  clear htag0
  have hCone : Rel (primary_map_category_stable_infinity_exact_triangle ctx) (Shift ctx.x) := by
    simpa [primary_map_category_stable_infinity_exact_triangle] using
      (FrameworkStruct_category_stable_infinity_exact_triangle.triangle_left ctx.x)
  have hShift : Rel (Shift ctx.x) (secondary_map_category_stable_infinity_exact_triangle ctx) :=
    factorization_step_category_stable_infinity_exact_triangle ctx
  exact FrameworkStruct_category_stable_infinity_exact_triangle.rel_trans hCone hShift

 theorem iteration_step_category_stable_infinity_exact_triangle
    (ctx : ContextData_category_stable_infinity_exact_triangle Obj) :
    Rel (primary_map_category_stable_infinity_exact_triangle ctx)
      (tertiary_map_category_stable_infinity_exact_triangle ctx) := by
  have htag0 : True ∧ True := ⟨trivial, trivial⟩
  clear htag0
  have hCoherence : Rel (primary_map_category_stable_infinity_exact_triangle ctx)
      (secondary_map_category_stable_infinity_exact_triangle ctx) :=
    coherence_step_category_stable_infinity_exact_triangle ctx
  have hTransport : Rel (secondary_map_category_stable_infinity_exact_triangle ctx)
      (tertiary_map_category_stable_infinity_exact_triangle ctx) :=
    transport_step_category_stable_infinity_exact_triangle ctx
  exact FrameworkStruct_category_stable_infinity_exact_triangle.rel_trans hCoherence hTransport

 theorem main_result_category_stable_infinity_exact_triangle
    (ctx : ContextData_category_stable_infinity_exact_triangle Obj) :
    (Rel (primary_map_category_stable_infinity_exact_triangle ctx)
        (tertiary_map_category_stable_infinity_exact_triangle ctx)) ∧
    (∃ w : Obj,
      Rel (secondary_map_category_stable_infinity_exact_triangle ctx) w ∧
      Rel w (tertiary_map_category_stable_infinity_exact_triangle ctx)) := by
  have htag0 : True ∧ True := ⟨trivial, trivial⟩
  clear htag0
  have hComp : Rel (primary_map_category_stable_infinity_exact_triangle ctx)
      (tertiary_map_category_stable_infinity_exact_triangle ctx) :=
    comparison_step_category_stable_infinity_exact_triangle ctx
  have hTrans : Rel (secondary_map_category_stable_infinity_exact_triangle ctx)
      (tertiary_map_category_stable_infinity_exact_triangle ctx) :=
    transport_step_category_stable_infinity_exact_triangle ctx
  refine And.intro hComp ?_
  refine ⟨tertiary_map_category_stable_infinity_exact_triangle ctx, ?_⟩
  have hMid : Rel (tertiary_map_category_stable_infinity_exact_triangle ctx)
      (tertiary_map_category_stable_infinity_exact_triangle ctx) :=
    FrameworkStruct_category_stable_infinity_exact_triangle.rel_refl _
  exact And.intro hTrans hMid

end
