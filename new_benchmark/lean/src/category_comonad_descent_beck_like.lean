/-
BENCHMARK_ID: TINY_MATHLIB_BATCH07_CATEGORY_COMONAD_DESCENT_BECK_LIKE
PAIR_STEM: category_comonad_descent_beck_like
MATH_DOMAIN: Category Theory
SOURCE_MATHLIB: Mathlib/CategoryTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

namespace CategoryComonadDescentBeckLike

class FrameworkStruct_category_comonad_descent_beck (α : Type u) where
  merge : α → α → α
  extend : α → α
  reduce : α → α
  merge_assoc : ∀ a b c : α, merge (merge a b) c = merge a (merge b c)
  extend_merge : ∀ a b : α, extend (merge a b) = merge (extend a) (extend b)
  reduce_merge : ∀ a b : α, reduce (merge a b) = merge (reduce a) (reduce b)
  extend_idem : ∀ a : α, extend (extend a) = extend a
  reduce_idem : ∀ a : α, reduce (reduce a) = reduce a
  reduce_extend_comm : ∀ a : α, reduce (extend a) = extend (reduce a)
  beck_swap : ∀ a b : α, merge (extend a) (reduce b) = merge (reduce b) (extend a)

structure ContextData_category_comonad_descent_beck
    (α : Type u) [FrameworkStruct_category_comonad_descent_beck α] where
  P : α
  Q : α
  R : α
  S : α

variable {α : Type u}
variable [F : FrameworkStruct_category_comonad_descent_beck α]

def primary_map_category_comonad_descent_beck
    (d : ContextData_category_comonad_descent_beck α) : α :=
  F.merge (F.extend d.P) (F.reduce d.Q)

def secondary_map_category_comonad_descent_beck
    (d : ContextData_category_comonad_descent_beck α) : α :=
  F.merge (F.extend d.R) (F.reduce d.S)

def tertiary_map_category_comonad_descent_beck
    (d : ContextData_category_comonad_descent_beck α) : α :=
  F.extend
    (F.merge
      (primary_map_category_comonad_descent_beck d)
      (secondary_map_category_comonad_descent_beck d))

theorem stability_step_category_comonad_descent_beck
    (d : ContextData_category_comonad_descent_beck α) :
    ∃ y : α,
      y = F.reduce (primary_map_category_comonad_descent_beck d) ∧
      y = F.merge (F.reduce (F.extend d.P)) (F.reduce (F.reduce d.Q)) := by
  refine ⟨F.reduce (primary_map_category_comonad_descent_beck d), ?_, ?_⟩
  · rfl
  · calc
      F.reduce (primary_map_category_comonad_descent_beck d)
          = F.reduce (F.merge (F.extend d.P) (F.reduce d.Q)) := by
            rfl
      _ = F.merge (F.reduce (F.extend d.P)) (F.reduce (F.reduce d.Q)) := by
            exact F.reduce_merge _ _

theorem factorization_step_category_comonad_descent_beck
    (d : ContextData_category_comonad_descent_beck α) :
    d.R = d.R →
    F.extend (secondary_map_category_comonad_descent_beck d) =
      F.merge (F.extend (F.extend d.R)) (F.extend (F.reduce d.S)) := by
  intro hR
  have hkeep : d.R = d.R := hR
  calc
    F.extend (secondary_map_category_comonad_descent_beck d)
        = F.extend (F.merge (F.extend d.R) (F.reduce d.S)) := by
          rfl
    _ = F.merge (F.extend (F.extend d.R)) (F.extend (F.reduce d.S)) := by
          exact F.extend_merge _ _

theorem comparison_step_category_comonad_descent_beck
    (d : ContextData_category_comonad_descent_beck α) :
    (F.merge
      (F.reduce (primary_map_category_comonad_descent_beck d))
      (F.extend (secondary_map_category_comonad_descent_beck d))
    =
    F.merge
      (F.merge (F.reduce (F.extend d.P)) (F.reduce (F.reduce d.Q)))
      (F.merge (F.extend (F.extend d.R)) (F.extend (F.reduce d.S)))) ∧
    True := by
  have hf := factorization_step_category_comonad_descent_beck d rfl
  constructor
  · calc
      F.merge
        (F.reduce (primary_map_category_comonad_descent_beck d))
        (F.extend (secondary_map_category_comonad_descent_beck d))
          = F.merge
              (F.reduce (F.merge (F.extend d.P) (F.reduce d.Q)))
              (F.extend (secondary_map_category_comonad_descent_beck d)) := by
            rfl
      _ = F.merge
            (F.merge (F.reduce (F.extend d.P)) (F.reduce (F.reduce d.Q)))
            (F.extend (secondary_map_category_comonad_descent_beck d)) := by
            rw [F.reduce_merge _ _]
      _ = F.merge
            (F.merge (F.reduce (F.extend d.P)) (F.reduce (F.reduce d.Q)))
            (F.merge (F.extend (F.extend d.R)) (F.extend (F.reduce d.S))) := by
            rw [hf]
  · exact True.intro

theorem transport_step_category_comonad_descent_beck
    (d : ContextData_category_comonad_descent_beck α)
    (hP : d.P = d.R) (hQ : d.Q = d.S) :
    primary_map_category_comonad_descent_beck d =
      secondary_map_category_comonad_descent_beck d := by
  calc
    primary_map_category_comonad_descent_beck d
        = F.merge (F.extend d.P) (F.reduce d.Q) := by
          rfl
    _ = F.merge (F.extend d.R) (F.reduce d.Q) := by
          rw [hP]
    _ = F.merge (F.extend d.R) (F.reduce d.S) := by
          rw [hQ]
    _ = secondary_map_category_comonad_descent_beck d := by
          rfl

theorem coherence_step_category_comonad_descent_beck
    (d : ContextData_category_comonad_descent_beck α) :
    (tertiary_map_category_comonad_descent_beck d =
      F.merge
        (F.extend (primary_map_category_comonad_descent_beck d))
        (F.extend (secondary_map_category_comonad_descent_beck d))) ↔
    True := by
  constructor
  · intro h
    exact True.intro
  · intro hTrue
    have : True := hTrue
    calc
      tertiary_map_category_comonad_descent_beck d
          = F.extend
              (F.merge
                (primary_map_category_comonad_descent_beck d)
                (secondary_map_category_comonad_descent_beck d)) := by
            rfl
      _ = F.merge
            (F.extend (primary_map_category_comonad_descent_beck d))
            (F.extend (secondary_map_category_comonad_descent_beck d)) := by
            exact F.extend_merge _ _

theorem iteration_step_category_comonad_descent_beck
    (d : ContextData_category_comonad_descent_beck α) :
    ∃ z : α,
      z = F.reduce (F.extend (tertiary_map_category_comonad_descent_beck d)) ∧
      z = F.extend (F.reduce (tertiary_map_category_comonad_descent_beck d)) := by
  refine ⟨F.reduce (F.extend (tertiary_map_category_comonad_descent_beck d)), ?_, ?_⟩
  · rfl
  · exact F.reduce_extend_comm _

theorem main_result_category_comonad_descent_beck
    (d : ContextData_category_comonad_descent_beck α) :
    F.merge (F.extend d.P) (F.reduce d.Q) =
      F.merge (F.reduce d.Q) (F.extend d.P) ∧
    ∃ t : α, t = tertiary_map_category_comonad_descent_beck d := by
  constructor
  · exact F.beck_swap _ _
  · refine ⟨tertiary_map_category_comonad_descent_beck d, ?_⟩
    rfl

end CategoryComonadDescentBeckLike
