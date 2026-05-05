/-
BENCHMARK_ID: TINY_MATHLIB_BATCH07_CATEGORY_MODEL_INFINITY_LOCALIZATION_LIKE
PAIR_STEM: category_model_infinity_localization_like
MATH_DOMAIN: Category Theory
SOURCE_MATHLIB: Mathlib/CategoryTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

namespace CategoryModelInfinityLocalizationLike

class FrameworkStruct_category_model_infinity_localization (α : Type u) where
  objComp : α → α → α
  localize : α → α
  witness : α → α
  comp_assoc : ∀ a b c : α,
    objComp (objComp a b) c = objComp a (objComp b c)
  loc_idem : ∀ a : α, localize (localize a) = localize a
  loc_comp : ∀ a b : α,
    localize (objComp a b) = objComp (localize a) (localize b)
  witness_loc : ∀ a : α, witness (localize a) = localize (witness a)
  witness_comp : ∀ a b : α,
    witness (objComp a b) = objComp (witness a) (witness b)
  loc_witness : ∀ a : α, localize (witness a) = witness (localize a)

structure ContextData_category_model_infinity_localization
    (α : Type u) [FrameworkStruct_category_model_infinity_localization α] where
  f : α
  g : α
  h : α
  k : α

variable {α : Type u}
variable [F : FrameworkStruct_category_model_infinity_localization α]

def primary_map_category_model_infinity_localization
    (d : ContextData_category_model_infinity_localization α) : α :=
  F.localize (F.objComp d.f d.g)

def secondary_map_category_model_infinity_localization
    (d : ContextData_category_model_infinity_localization α) : α :=
  F.objComp (F.localize d.f) (F.localize d.g)

def tertiary_map_category_model_infinity_localization
    (d : ContextData_category_model_infinity_localization α) : α :=
  F.localize
    (F.objComp
      (secondary_map_category_model_infinity_localization d)
      (F.witness d.h))

theorem stability_step_category_model_infinity_localization
    (d : ContextData_category_model_infinity_localization α) :
    primary_map_category_model_infinity_localization d =
      secondary_map_category_model_infinity_localization d := by
  calc
    primary_map_category_model_infinity_localization d
        = F.localize (F.objComp d.f d.g) := by
          rfl
    _ = F.objComp (F.localize d.f) (F.localize d.g) := by
          exact F.loc_comp _ _
    _ = secondary_map_category_model_infinity_localization d := by
          rfl

theorem factorization_step_category_model_infinity_localization
    (d : ContextData_category_model_infinity_localization α) :
    F.localize (secondary_map_category_model_infinity_localization d) =
      secondary_map_category_model_infinity_localization d := by
  calc
    F.localize (secondary_map_category_model_infinity_localization d)
        = F.localize (F.objComp (F.localize d.f) (F.localize d.g)) := by
          rfl
    _ = F.objComp (F.localize (F.localize d.f)) (F.localize (F.localize d.g)) := by
          exact F.loc_comp _ _
    _ = F.objComp (F.localize d.f) (F.localize (F.localize d.g)) := by
          rw [F.loc_idem d.f]
    _ = F.objComp (F.localize d.f) (F.localize d.g) := by
          rw [F.loc_idem d.g]
    _ = secondary_map_category_model_infinity_localization d := by
          rfl

theorem comparison_step_category_model_infinity_localization
    (d : ContextData_category_model_infinity_localization α) :
    F.localize
      (F.objComp
        (primary_map_category_model_infinity_localization d)
        (F.witness d.h))
    = tertiary_map_category_model_infinity_localization d := by
  have hstable := stability_step_category_model_infinity_localization d
  calc
    F.localize
      (F.objComp
        (primary_map_category_model_infinity_localization d)
        (F.witness d.h))
        = F.localize
            (F.objComp
              (secondary_map_category_model_infinity_localization d)
              (F.witness d.h)) := by
          rw [hstable]
    _ = tertiary_map_category_model_infinity_localization d := by
          rfl

theorem transport_step_category_model_infinity_localization
    (d : ContextData_category_model_infinity_localization α)
    (hf : d.f = d.k) (hg : d.g = d.h) :
    primary_map_category_model_infinity_localization d =
      F.localize (F.objComp d.k d.h) := by
  calc
    primary_map_category_model_infinity_localization d
        = F.localize (F.objComp d.f d.g) := by
          rfl
    _ = F.localize (F.objComp d.k d.g) := by
          rw [hf]
    _ = F.localize (F.objComp d.k d.h) := by
          rw [hg]

theorem coherence_step_category_model_infinity_localization
    (d : ContextData_category_model_infinity_localization α) :
    F.objComp
      (primary_map_category_model_infinity_localization d)
      (F.objComp (F.witness d.h) (F.witness d.k))
    =
    F.objComp
      (F.objComp
        (primary_map_category_model_infinity_localization d)
        (F.witness d.h))
      (F.witness d.k) := by
  symm
  exact F.comp_assoc _ _ _

theorem iteration_step_category_model_infinity_localization
    (d : ContextData_category_model_infinity_localization α) :
    (F.witness
      (F.localize (tertiary_map_category_model_infinity_localization d))
    =
    F.localize
      (F.witness (tertiary_map_category_model_infinity_localization d))) ∧
    True := by
  refine And.intro ?_ True.intro
  exact F.witness_loc _

theorem main_result_category_model_infinity_localization
    (d : ContextData_category_model_infinity_localization α) :
    ∃ m : α,
      F.localize m = tertiary_map_category_model_infinity_localization d ∧
      m = F.objComp
        (secondary_map_category_model_infinity_localization d)
        (F.witness d.h) := by
  refine ⟨F.objComp
    (secondary_map_category_model_infinity_localization d)
    (F.witness d.h), ?_⟩
  constructor
  · rfl
  · rfl

end CategoryModelInfinityLocalizationLike
