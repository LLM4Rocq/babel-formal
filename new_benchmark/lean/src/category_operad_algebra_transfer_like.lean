/-
BENCHMARK_ID: TINY_MATHLIB_BATCH07_CATEGORY_OPERAD_ALGEBRA_TRANSFER_LIKE
PAIR_STEM: category_operad_algebra_transfer_like
MATH_DOMAIN: Category Theory
SOURCE_MATHLIB: Mathlib/CategoryTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

namespace CategoryOperadAlgebraTransferLike

class FrameworkStruct_category_operad_algebra_transfer (α : Type u) where
  operadComp : α → α → α
  actOn : α → α → α
  liftOp : α → α
  restrictOp : α → α
  comp_assoc : ∀ a b c : α,
    operadComp (operadComp a b) c = operadComp a (operadComp b c)
  action_assoc : ∀ o1 o2 a : α,
    actOn (operadComp o1 o2) a = actOn o1 (actOn o2 a)
  lift_act : ∀ o a : α,
    liftOp (actOn o a) = actOn (liftOp o) (liftOp a)
  restrict_act : ∀ o a : α,
    restrictOp (actOn o a) = actOn (restrictOp o) (restrictOp a)
  lift_idem : ∀ a : α, liftOp (liftOp a) = liftOp a
  restrict_idem : ∀ a : α, restrictOp (restrictOp a) = restrictOp a
  lift_restrict_comm : ∀ a : α, liftOp (restrictOp a) = restrictOp (liftOp a)

structure ContextData_category_operad_algebra_transfer
    (α : Type u) [FrameworkStruct_category_operad_algebra_transfer α] where
  O1 : α
  O2 : α
  A1 : α
  A2 : α

variable {α : Type u}
variable [F : FrameworkStruct_category_operad_algebra_transfer α]

def primary_map_category_operad_algebra_transfer
    (d : ContextData_category_operad_algebra_transfer α) : α :=
  F.actOn (F.liftOp d.O1) d.A1

def secondary_map_category_operad_algebra_transfer
    (d : ContextData_category_operad_algebra_transfer α) : α :=
  F.actOn (F.restrictOp d.O2) d.A2

def tertiary_map_category_operad_algebra_transfer
    (d : ContextData_category_operad_algebra_transfer α) : α :=
  F.actOn
    (F.operadComp (F.liftOp d.O1) (F.restrictOp d.O2))
    (F.liftOp d.A1)

theorem stability_step_category_operad_algebra_transfer
    (d : ContextData_category_operad_algebra_transfer α) :
    True →
    F.liftOp (primary_map_category_operad_algebra_transfer d) =
      F.actOn (F.liftOp (F.liftOp d.O1)) (F.liftOp d.A1) := by
  intro hTrue
  have : True := hTrue
  calc
    F.liftOp (primary_map_category_operad_algebra_transfer d)
        = F.liftOp (F.actOn (F.liftOp d.O1) d.A1) := by
          rfl
    _ = F.actOn (F.liftOp (F.liftOp d.O1)) (F.liftOp d.A1) := by
          exact F.lift_act _ _

theorem factorization_step_category_operad_algebra_transfer
    (d : ContextData_category_operad_algebra_transfer α) :
    (F.restrictOp (secondary_map_category_operad_algebra_transfer d) =
      F.actOn (F.restrictOp (F.restrictOp d.O2)) (F.restrictOp d.A2)) ∧
    True := by
  constructor
  · calc
      F.restrictOp (secondary_map_category_operad_algebra_transfer d)
          = F.restrictOp (F.actOn (F.restrictOp d.O2) d.A2) := by
            rfl
      _ = F.actOn (F.restrictOp (F.restrictOp d.O2)) (F.restrictOp d.A2) := by
            exact F.restrict_act _ _
  · exact True.intro

theorem comparison_step_category_operad_algebra_transfer
    (d : ContextData_category_operad_algebra_transfer α) :
    F.actOn
      (F.operadComp (F.liftOp d.O1) (F.restrictOp d.O2))
      (F.actOn (F.liftOp d.A1) (F.restrictOp d.A2))
    =
    F.actOn
      (F.liftOp d.O1)
      (F.actOn
        (F.restrictOp d.O2)
        (F.actOn (F.liftOp d.A1) (F.restrictOp d.A2))) := by
  exact F.action_assoc _ _ _

theorem transport_step_category_operad_algebra_transfer
    (d : ContextData_category_operad_algebra_transfer α)
    (hO : d.O1 = F.restrictOp d.O2) (hA : d.A1 = d.A2) :
    primary_map_category_operad_algebra_transfer d =
      F.actOn (F.liftOp (F.restrictOp d.O2)) d.A2 := by
  calc
    primary_map_category_operad_algebra_transfer d
        = F.actOn (F.liftOp d.O1) d.A1 := by
          rfl
    _ = F.actOn (F.liftOp (F.restrictOp d.O2)) d.A1 := by
          rw [hO]
    _ = F.actOn (F.liftOp (F.restrictOp d.O2)) d.A2 := by
          rw [hA]

theorem coherence_step_category_operad_algebra_transfer
    (d : ContextData_category_operad_algebra_transfer α) :
    tertiary_map_category_operad_algebra_transfer d =
      F.actOn
        (F.liftOp d.O1)
        (F.actOn (F.restrictOp d.O2) (F.liftOp d.A1)) := by
  calc
    tertiary_map_category_operad_algebra_transfer d
        = F.actOn
            (F.operadComp (F.liftOp d.O1) (F.restrictOp d.O2))
            (F.liftOp d.A1) := by
          rfl
    _ = F.actOn
          (F.liftOp d.O1)
          (F.actOn (F.restrictOp d.O2) (F.liftOp d.A1)) := by
          exact F.action_assoc _ _ _

theorem iteration_step_category_operad_algebra_transfer
    (d : ContextData_category_operad_algebra_transfer α) :
    ¬ False →
    F.liftOp (F.restrictOp (tertiary_map_category_operad_algebra_transfer d)) =
      F.restrictOp (F.liftOp (tertiary_map_category_operad_algebra_transfer d)) := by
  intro hnf
  have : ¬ False := hnf
  calc
    F.liftOp (F.restrictOp (tertiary_map_category_operad_algebra_transfer d))
        = F.restrictOp (F.liftOp (tertiary_map_category_operad_algebra_transfer d)) := by
          exact F.lift_restrict_comm _

theorem main_result_category_operad_algebra_transfer
    (d : ContextData_category_operad_algebra_transfer α) :
    ∃ x : α,
      x = tertiary_map_category_operad_algebra_transfer d ∧
      F.actOn (F.liftOp d.O1) d.A1 = primary_map_category_operad_algebra_transfer d := by
  refine ⟨tertiary_map_category_operad_algebra_transfer d, ?_, ?_⟩
  · rfl
  · rfl

end CategoryOperadAlgebraTransferLike
