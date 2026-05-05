/-
BENCHMARK_ID: TINY_MATHLIB_BATCH07_CATEGORY_ENRICHED_END_COEND_FUBINI_LIKE
PAIR_STEM: category_enriched_end_coend_fubini_like
MATH_DOMAIN: Category Theory
SOURCE_MATHLIB: Mathlib/CategoryTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 11
-/

universe u

namespace CategoryEnrichedEndCoendFubiniLike

class FrameworkStruct_category_enriched_end_coend_fubini (α : Type u) where
  tensor : α → α → α
  endOp : α → α
  coendOp : α → α
  tensor_assoc : ∀ a b c : α, tensor (tensor a b) c = tensor a (tensor b c)
  end_tensor : ∀ a b : α, endOp (tensor a b) = tensor (endOp a) (endOp b)
  coend_tensor : ∀ a b : α, coendOp (tensor a b) = tensor (coendOp a) (coendOp b)
  end_idem : ∀ a : α, endOp (endOp a) = endOp a
  coend_idem : ∀ a : α, coendOp (coendOp a) = coendOp a
  end_coend_comm : ∀ a : α, endOp (coendOp a) = coendOp (endOp a)
  fubini_swap : ∀ a b : α, tensor (endOp a) (coendOp b) = tensor (coendOp b) (endOp a)

structure ContextData_category_enriched_end_coend_fubini
    (α : Type u) [FrameworkStruct_category_enriched_end_coend_fubini α] where
  U : α
  V : α
  W : α
  Z : α

variable {α : Type u}
variable [F : FrameworkStruct_category_enriched_end_coend_fubini α]

def primary_map_category_enriched_end_coend_fubini
    (d : ContextData_category_enriched_end_coend_fubini α) : α :=
  F.tensor (F.endOp d.U) (F.coendOp d.V)

def secondary_map_category_enriched_end_coend_fubini
    (d : ContextData_category_enriched_end_coend_fubini α) : α :=
  F.tensor (F.coendOp d.V) (F.endOp d.W)

def tertiary_map_category_enriched_end_coend_fubini
    (d : ContextData_category_enriched_end_coend_fubini α) : α :=
  F.tensor
    (F.tensor
      (primary_map_category_enriched_end_coend_fubini d)
      (secondary_map_category_enriched_end_coend_fubini d))
    (F.endOp d.Z)

theorem stability_step_category_enriched_end_coend_fubini
    (d : ContextData_category_enriched_end_coend_fubini α) :
    d.U = d.U →
    (F.endOp (primary_map_category_enriched_end_coend_fubini d) =
      F.tensor (F.endOp (F.endOp d.U)) (F.endOp (F.coendOp d.V))) ∧
    True := by
  intro hUU
  have hkeep : d.U = d.U := hUU
  constructor
  · calc
      F.endOp (primary_map_category_enriched_end_coend_fubini d)
          = F.endOp (F.tensor (F.endOp d.U) (F.coendOp d.V)) := by
            rfl
      _ = F.tensor (F.endOp (F.endOp d.U)) (F.endOp (F.coendOp d.V)) := by
            exact F.end_tensor _ _
  · exact True.intro

theorem factorization_step_category_enriched_end_coend_fubini
    (d : ContextData_category_enriched_end_coend_fubini α) :
    (F.coendOp (secondary_map_category_enriched_end_coend_fubini d) =
      F.tensor (F.coendOp (F.coendOp d.V)) (F.coendOp (F.endOp d.W))) ∧
      d.V = d.V := by
  constructor
  · calc
      F.coendOp (secondary_map_category_enriched_end_coend_fubini d)
          = F.coendOp (F.tensor (F.coendOp d.V) (F.endOp d.W)) := by
            rfl
      _ = F.tensor (F.coendOp (F.coendOp d.V)) (F.coendOp (F.endOp d.W)) := by
            exact F.coend_tensor _ _
  · rfl

theorem comparison_step_category_enriched_end_coend_fubini
    (d : ContextData_category_enriched_end_coend_fubini α) :
    ∃ x : α,
      x =
        F.tensor
          (F.endOp (primary_map_category_enriched_end_coend_fubini d))
          (F.coendOp (secondary_map_category_enriched_end_coend_fubini d)) ∧
      x =
        F.tensor
          (F.tensor (F.endOp (F.endOp d.U)) (F.endOp (F.coendOp d.V)))
          (F.tensor (F.coendOp (F.coendOp d.V)) (F.coendOp (F.endOp d.W))) := by
  have hs := stability_step_category_enriched_end_coend_fubini d rfl
  have hf := factorization_step_category_enriched_end_coend_fubini d
  refine ⟨F.tensor
    (F.endOp (primary_map_category_enriched_end_coend_fubini d))
    (F.coendOp (secondary_map_category_enriched_end_coend_fubini d)), ?_, ?_⟩
  · rfl
  · calc
      F.tensor
        (F.endOp (primary_map_category_enriched_end_coend_fubini d))
        (F.coendOp (secondary_map_category_enriched_end_coend_fubini d))
          = F.tensor
              (F.tensor (F.endOp (F.endOp d.U)) (F.endOp (F.coendOp d.V)))
              (F.coendOp (secondary_map_category_enriched_end_coend_fubini d)) := by
            rw [hs.1]
      _ = F.tensor
            (F.tensor (F.endOp (F.endOp d.U)) (F.endOp (F.coendOp d.V)))
            (F.tensor (F.coendOp (F.coendOp d.V)) (F.coendOp (F.endOp d.W))) := by
            rw [hf.1]

theorem transport_step_category_enriched_end_coend_fubini
    (d : ContextData_category_enriched_end_coend_fubini α)
    (hU : d.U = d.W) (hV : d.V = d.Z) :
    ∃ p : α,
      p = F.tensor (F.endOp d.U) (F.coendOp d.V) ∧
      p = F.tensor (F.endOp d.W) (F.coendOp d.Z) := by
  refine ⟨F.tensor (F.endOp d.U) (F.coendOp d.V), ?_, ?_⟩
  · rfl
  · calc
      F.tensor (F.endOp d.U) (F.coendOp d.V)
          = F.tensor (F.endOp d.W) (F.coendOp d.V) := by
            rw [hU]
      _ = F.tensor (F.endOp d.W) (F.coendOp d.Z) := by
            rw [hV]

theorem iteration_step_category_enriched_end_coend_fubini
    (d : ContextData_category_enriched_end_coend_fubini α) :
    True →
    F.endOp (F.coendOp (primary_map_category_enriched_end_coend_fubini d)) =
      F.coendOp (F.endOp (primary_map_category_enriched_end_coend_fubini d)) := by
  intro hTrue
  have : True := hTrue
  calc
    F.endOp (F.coendOp (primary_map_category_enriched_end_coend_fubini d))
        = F.coendOp (F.endOp (primary_map_category_enriched_end_coend_fubini d)) := by
          exact F.end_coend_comm _

theorem main_result_category_enriched_end_coend_fubini
    (d : ContextData_category_enriched_end_coend_fubini α) :
    ∃ t : α,
      t = tertiary_map_category_enriched_end_coend_fubini d ∧
      F.tensor (F.endOp d.U) (F.coendOp d.V) =
        F.tensor (F.coendOp d.V) (F.endOp d.U) := by
  refine ⟨tertiary_map_category_enriched_end_coend_fubini d, ?_, ?_⟩
  · rfl
  · exact F.fubini_swap _ _

end CategoryEnrichedEndCoendFubiniLike
