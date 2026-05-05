/-
BENCHMARK_ID: TINY_MATHLIB_BATCH07_CATEGORY_TRACED_MONOIDAL_FEEDBACK_LIKE
PAIR_STEM: category_traced_monoidal_feedback_like
MATH_DOMAIN: Category Theory
SOURCE_MATHLIB: Mathlib/CategoryTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

namespace CategoryTracedMonoidalFeedbackLike

class FrameworkStruct_category_traced_monoidal_feedback (α : Type u) where
  tensor : α → α → α
  traceOp : α → α
  feedbackOp : α → α
  tensor_assoc : ∀ a b c : α, tensor (tensor a b) c = tensor a (tensor b c)
  trace_tensor : ∀ a b : α,
    traceOp (tensor a b) = tensor (traceOp a) (traceOp b)
  feedback_tensor : ∀ a b : α,
    feedbackOp (tensor a b) = tensor (feedbackOp a) (feedbackOp b)
  trace_feedback_comm : ∀ a : α,
    traceOp (feedbackOp a) = feedbackOp (traceOp a)
  trace_idem : ∀ a : α, traceOp (traceOp a) = traceOp a
  feedback_idem : ∀ a : α, feedbackOp (feedbackOp a) = feedbackOp a
  braid : ∀ a b : α, tensor a b = tensor b a

structure ContextData_category_traced_monoidal_feedback
    (α : Type u) [FrameworkStruct_category_traced_monoidal_feedback α] where
  M : α
  N : α
  P : α
  Q : α

variable {α : Type u}
variable [F : FrameworkStruct_category_traced_monoidal_feedback α]

def primary_map_category_traced_monoidal_feedback
    (d : ContextData_category_traced_monoidal_feedback α) : α :=
  F.tensor (F.traceOp d.M) (F.feedbackOp d.N)

def secondary_map_category_traced_monoidal_feedback
    (d : ContextData_category_traced_monoidal_feedback α) : α :=
  F.tensor (F.feedbackOp d.P) (F.traceOp d.Q)

def tertiary_map_category_traced_monoidal_feedback
    (d : ContextData_category_traced_monoidal_feedback α) : α :=
  F.feedbackOp
    (F.tensor
      (primary_map_category_traced_monoidal_feedback d)
      (secondary_map_category_traced_monoidal_feedback d))

theorem stability_step_category_traced_monoidal_feedback
    (d : ContextData_category_traced_monoidal_feedback α) :
    False ∨
      F.traceOp (primary_map_category_traced_monoidal_feedback d) =
        F.tensor (F.traceOp (F.traceOp d.M)) (F.traceOp (F.feedbackOp d.N)) := by
  right
  calc
    F.traceOp (primary_map_category_traced_monoidal_feedback d)
        = F.traceOp (F.tensor (F.traceOp d.M) (F.feedbackOp d.N)) := by
          rfl
    _ = F.tensor (F.traceOp (F.traceOp d.M)) (F.traceOp (F.feedbackOp d.N)) := by
          exact F.trace_tensor _ _

theorem factorization_step_category_traced_monoidal_feedback
    (d : ContextData_category_traced_monoidal_feedback α) :
    (F.feedbackOp (secondary_map_category_traced_monoidal_feedback d) =
      F.tensor (F.feedbackOp (F.feedbackOp d.P)) (F.feedbackOp (F.traceOp d.Q))) ↔
    True := by
  constructor
  · intro hEq
    exact True.intro
  · intro hTrue
    have : True := hTrue
    calc
      F.feedbackOp (secondary_map_category_traced_monoidal_feedback d)
          = F.feedbackOp (F.tensor (F.feedbackOp d.P) (F.traceOp d.Q)) := by
            rfl
      _ = F.tensor (F.feedbackOp (F.feedbackOp d.P)) (F.feedbackOp (F.traceOp d.Q)) := by
            exact F.feedback_tensor _ _

theorem comparison_step_category_traced_monoidal_feedback
    (d : ContextData_category_traced_monoidal_feedback α) :
    True ∧
    (F.tensor
      (F.traceOp (primary_map_category_traced_monoidal_feedback d))
      (F.feedbackOp (secondary_map_category_traced_monoidal_feedback d))
    =
    F.tensor
      (F.tensor (F.traceOp (F.traceOp d.M)) (F.traceOp (F.feedbackOp d.N)))
      (F.tensor (F.feedbackOp (F.feedbackOp d.P)) (F.feedbackOp (F.traceOp d.Q)))) := by
  constructor
  · exact True.intro
  · calc
      F.tensor
        (F.traceOp (primary_map_category_traced_monoidal_feedback d))
        (F.feedbackOp (secondary_map_category_traced_monoidal_feedback d))
          = F.tensor
              (F.traceOp (F.tensor (F.traceOp d.M) (F.feedbackOp d.N)))
              (F.feedbackOp (secondary_map_category_traced_monoidal_feedback d)) := by
            rfl
      _ = F.tensor
            (F.tensor (F.traceOp (F.traceOp d.M)) (F.traceOp (F.feedbackOp d.N)))
            (F.feedbackOp (secondary_map_category_traced_monoidal_feedback d)) := by
            rw [F.trace_tensor _ _]
      _ = F.tensor
            (F.tensor (F.traceOp (F.traceOp d.M)) (F.traceOp (F.feedbackOp d.N)))
            (F.feedbackOp (F.tensor (F.feedbackOp d.P) (F.traceOp d.Q))) := by
            rfl
      _ = F.tensor
            (F.tensor (F.traceOp (F.traceOp d.M)) (F.traceOp (F.feedbackOp d.N)))
            (F.tensor (F.feedbackOp (F.feedbackOp d.P)) (F.feedbackOp (F.traceOp d.Q))) := by
            rw [F.feedback_tensor _ _]

theorem transport_step_category_traced_monoidal_feedback
    (d : ContextData_category_traced_monoidal_feedback α)
    (hM : d.M = d.P) (hQ : d.Q = d.N) :
    (F.tensor (F.traceOp d.M) (F.feedbackOp d.Q) =
      F.tensor (F.traceOp d.P) (F.feedbackOp d.N)) ∧
    d.Q = d.Q := by
  constructor
  · calc
      F.tensor (F.traceOp d.M) (F.feedbackOp d.Q)
          = F.tensor (F.traceOp d.P) (F.feedbackOp d.Q) := by
            rw [hM]
      _ = F.tensor (F.traceOp d.P) (F.feedbackOp d.N) := by
            rw [hQ]
  · rfl

theorem coherence_step_category_traced_monoidal_feedback
    (d : ContextData_category_traced_monoidal_feedback α) :
    True →
    tertiary_map_category_traced_monoidal_feedback d =
      F.tensor
        (F.feedbackOp (primary_map_category_traced_monoidal_feedback d))
        (F.feedbackOp (secondary_map_category_traced_monoidal_feedback d)) := by
  intro hTrue
  have : True := hTrue
  calc
    tertiary_map_category_traced_monoidal_feedback d
        = F.feedbackOp
            (F.tensor
              (primary_map_category_traced_monoidal_feedback d)
              (secondary_map_category_traced_monoidal_feedback d)) := by
          rfl
    _ = F.tensor
          (F.feedbackOp (primary_map_category_traced_monoidal_feedback d))
          (F.feedbackOp (secondary_map_category_traced_monoidal_feedback d)) := by
          exact F.feedback_tensor _ _

theorem iteration_step_category_traced_monoidal_feedback
    (d : ContextData_category_traced_monoidal_feedback α) :
    False ∨
    F.traceOp (F.feedbackOp (tertiary_map_category_traced_monoidal_feedback d)) =
      F.feedbackOp (F.traceOp (tertiary_map_category_traced_monoidal_feedback d)) := by
  right
  calc
    F.traceOp (F.feedbackOp (tertiary_map_category_traced_monoidal_feedback d))
        = F.feedbackOp (F.traceOp (tertiary_map_category_traced_monoidal_feedback d)) := by
          exact F.trace_feedback_comm _

theorem main_result_category_traced_monoidal_feedback
    (d : ContextData_category_traced_monoidal_feedback α) :
    F.tensor
      (primary_map_category_traced_monoidal_feedback d)
      (secondary_map_category_traced_monoidal_feedback d)
    =
    F.tensor
      (secondary_map_category_traced_monoidal_feedback d)
      (primary_map_category_traced_monoidal_feedback d)
    ∧ ∃ t : α, t = tertiary_map_category_traced_monoidal_feedback d := by
  constructor
  · exact F.braid _ _
  · refine ⟨tertiary_map_category_traced_monoidal_feedback d, ?_⟩
    rfl

end CategoryTracedMonoidalFeedbackLike
