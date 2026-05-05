/-
BENCHMARK_ID: TINY_MATHLIB_BATCH07_CATEGORY_BICATEGORY_PSEUDOFUNCTOR_COHERENCE_LIKE
PAIR_STEM: category_bicategory_pseudofunctor_coherence_like
MATH_DOMAIN: Category Theory
SOURCE_MATHLIB: Mathlib/CategoryTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

namespace CategoryBicategoryPseudofunctorCoherenceLike

class FrameworkStruct_category_bicategory_pseudofunctor_coherence (α : Type u) where
  comp : α → α → α
  leftWhisker : α → α
  rightWhisker : α → α
  comp_assoc : ∀ a b c : α, comp (comp a b) c = comp a (comp b c)
  leftWhisker_comp : ∀ a b : α,
    leftWhisker (comp a b) = comp (leftWhisker a) (leftWhisker b)
  rightWhisker_comp : ∀ a b : α,
    rightWhisker (comp a b) = comp (rightWhisker a) (rightWhisker b)
  whisker_interchange : ∀ a : α,
    leftWhisker (rightWhisker a) = rightWhisker (leftWhisker a)
  leftWhisker_idem : ∀ a : α, leftWhisker (leftWhisker a) = leftWhisker a
  rightWhisker_idem : ∀ a : α, rightWhisker (rightWhisker a) = rightWhisker a

structure ContextData_category_bicategory_pseudofunctor_coherence
    (α : Type u) [FrameworkStruct_category_bicategory_pseudofunctor_coherence α] where
  η : α
  θ : α
  ι : α
  κ : α

variable {α : Type u}
variable [F : FrameworkStruct_category_bicategory_pseudofunctor_coherence α]

def primary_map_category_bicategory_pseudofunctor_coherence
    (d : ContextData_category_bicategory_pseudofunctor_coherence α) : α :=
  F.comp (F.leftWhisker d.η) (F.rightWhisker d.θ)

def secondary_map_category_bicategory_pseudofunctor_coherence
    (d : ContextData_category_bicategory_pseudofunctor_coherence α) : α :=
  F.comp (F.rightWhisker d.θ) (F.leftWhisker d.ι)

def tertiary_map_category_bicategory_pseudofunctor_coherence
    (d : ContextData_category_bicategory_pseudofunctor_coherence α) : α :=
  F.comp
    (F.comp
      (primary_map_category_bicategory_pseudofunctor_coherence d)
      (secondary_map_category_bicategory_pseudofunctor_coherence d))
    (F.leftWhisker d.κ)

theorem stability_step_category_bicategory_pseudofunctor_coherence
    (d : ContextData_category_bicategory_pseudofunctor_coherence α) :
    F.leftWhisker (primary_map_category_bicategory_pseudofunctor_coherence d) =
      F.comp (F.leftWhisker (F.leftWhisker d.η))
        (F.leftWhisker (F.rightWhisker d.θ)) := by
  calc
    F.leftWhisker (primary_map_category_bicategory_pseudofunctor_coherence d)
        = F.leftWhisker (F.comp (F.leftWhisker d.η) (F.rightWhisker d.θ)) := by
          rfl
    _ = F.comp (F.leftWhisker (F.leftWhisker d.η))
          (F.leftWhisker (F.rightWhisker d.θ)) := by
          exact F.leftWhisker_comp _ _

theorem factorization_step_category_bicategory_pseudofunctor_coherence
    (d : ContextData_category_bicategory_pseudofunctor_coherence α) :
    F.rightWhisker (secondary_map_category_bicategory_pseudofunctor_coherence d) =
      F.comp (F.rightWhisker (F.rightWhisker d.θ))
        (F.rightWhisker (F.leftWhisker d.ι)) := by
  calc
    F.rightWhisker (secondary_map_category_bicategory_pseudofunctor_coherence d)
        = F.rightWhisker (F.comp (F.rightWhisker d.θ) (F.leftWhisker d.ι)) := by
          rfl
    _ = F.comp (F.rightWhisker (F.rightWhisker d.θ))
          (F.rightWhisker (F.leftWhisker d.ι)) := by
          exact F.rightWhisker_comp _ _

theorem comparison_step_category_bicategory_pseudofunctor_coherence
    (d : ContextData_category_bicategory_pseudofunctor_coherence α) :
    F.comp
        (F.leftWhisker (primary_map_category_bicategory_pseudofunctor_coherence d))
        (F.rightWhisker (secondary_map_category_bicategory_pseudofunctor_coherence d))
      =
    F.comp
      (F.comp (F.leftWhisker (F.leftWhisker d.η))
        (F.leftWhisker (F.rightWhisker d.θ)))
      (F.comp (F.rightWhisker (F.rightWhisker d.θ))
        (F.rightWhisker (F.leftWhisker d.ι))) := by
  have hleft := stability_step_category_bicategory_pseudofunctor_coherence d
  have hright := factorization_step_category_bicategory_pseudofunctor_coherence d
  calc
    F.comp
        (F.leftWhisker (primary_map_category_bicategory_pseudofunctor_coherence d))
        (F.rightWhisker (secondary_map_category_bicategory_pseudofunctor_coherence d))
        = F.comp
            (F.comp (F.leftWhisker (F.leftWhisker d.η))
              (F.leftWhisker (F.rightWhisker d.θ)))
            (F.rightWhisker
              (secondary_map_category_bicategory_pseudofunctor_coherence d)) := by
          rw [hleft]
    _ = F.comp
          (F.comp (F.leftWhisker (F.leftWhisker d.η))
            (F.leftWhisker (F.rightWhisker d.θ)))
          (F.comp (F.rightWhisker (F.rightWhisker d.θ))
            (F.rightWhisker (F.leftWhisker d.ι))) := by
          rw [hright]

theorem transport_step_category_bicategory_pseudofunctor_coherence
    (d : ContextData_category_bicategory_pseudofunctor_coherence α)
    (hη : d.η = d.κ) (hθ : d.θ = d.ι) :
    F.leftWhisker (primary_map_category_bicategory_pseudofunctor_coherence d) =
      F.comp (F.leftWhisker (F.leftWhisker d.κ))
        (F.leftWhisker (F.rightWhisker d.ι)) := by
  have hstable := stability_step_category_bicategory_pseudofunctor_coherence d
  calc
    F.leftWhisker (primary_map_category_bicategory_pseudofunctor_coherence d)
        = F.comp (F.leftWhisker (F.leftWhisker d.η))
            (F.leftWhisker (F.rightWhisker d.θ)) := by
          exact hstable
    _ = F.comp (F.leftWhisker (F.leftWhisker d.κ))
          (F.leftWhisker (F.rightWhisker d.θ)) := by
          rw [hη]
    _ = F.comp (F.leftWhisker (F.leftWhisker d.κ))
          (F.leftWhisker (F.rightWhisker d.ι)) := by
          rw [hθ]

theorem coherence_step_category_bicategory_pseudofunctor_coherence
    (d : ContextData_category_bicategory_pseudofunctor_coherence α) :
    F.comp
      (primary_map_category_bicategory_pseudofunctor_coherence d)
      (F.comp
        (secondary_map_category_bicategory_pseudofunctor_coherence d)
        (F.leftWhisker d.κ))
      = tertiary_map_category_bicategory_pseudofunctor_coherence d := by
  calc
    F.comp
      (primary_map_category_bicategory_pseudofunctor_coherence d)
      (F.comp
        (secondary_map_category_bicategory_pseudofunctor_coherence d)
        (F.leftWhisker d.κ))
        = F.comp
            (F.comp
              (primary_map_category_bicategory_pseudofunctor_coherence d)
              (secondary_map_category_bicategory_pseudofunctor_coherence d))
            (F.leftWhisker d.κ) := by
          symm
          exact F.comp_assoc _ _ _
    _ = tertiary_map_category_bicategory_pseudofunctor_coherence d := by
          rfl

theorem iteration_step_category_bicategory_pseudofunctor_coherence
    (d : ContextData_category_bicategory_pseudofunctor_coherence α) :
    F.rightWhisker (F.leftWhisker
      (tertiary_map_category_bicategory_pseudofunctor_coherence d)) =
    F.leftWhisker (F.rightWhisker
      (tertiary_map_category_bicategory_pseudofunctor_coherence d)) := by
  have hswap := F.whisker_interchange
    (tertiary_map_category_bicategory_pseudofunctor_coherence d)
  calc
    F.rightWhisker (F.leftWhisker
      (tertiary_map_category_bicategory_pseudofunctor_coherence d))
        = F.leftWhisker (F.rightWhisker
            (tertiary_map_category_bicategory_pseudofunctor_coherence d)) := by
          exact hswap.symm

theorem main_result_category_bicategory_pseudofunctor_coherence
    (d : ContextData_category_bicategory_pseudofunctor_coherence α) :
    F.rightWhisker (F.leftWhisker
      (F.comp
        (primary_map_category_bicategory_pseudofunctor_coherence d)
        (secondary_map_category_bicategory_pseudofunctor_coherence d)))
    =
    F.leftWhisker (F.rightWhisker
      (F.comp
        (primary_map_category_bicategory_pseudofunctor_coherence d)
        (secondary_map_category_bicategory_pseudofunctor_coherence d))) := by
  have hcore := F.whisker_interchange
    (F.comp
      (primary_map_category_bicategory_pseudofunctor_coherence d)
      (secondary_map_category_bicategory_pseudofunctor_coherence d))
  calc
    F.rightWhisker (F.leftWhisker
      (F.comp
        (primary_map_category_bicategory_pseudofunctor_coherence d)
        (secondary_map_category_bicategory_pseudofunctor_coherence d)))
        = F.leftWhisker (F.rightWhisker
            (F.comp
              (primary_map_category_bicategory_pseudofunctor_coherence d)
              (secondary_map_category_bicategory_pseudofunctor_coherence d))) := by
          exact hcore.symm

end CategoryBicategoryPseudofunctorCoherenceLike
