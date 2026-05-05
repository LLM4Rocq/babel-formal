/-
BENCHMARK_ID: TINY_MATHLIB_BATCH03_ALG_GROUP_CANCEL
PAIR_STEM: algebra_group_cancel
MATH_DOMAIN: Algebra
SOURCE_MATHLIB: Mathlib/Algebra/Group/Basic
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class GroupLike (α : Type u) where
  one : α
  mul : α → α → α
  inv : α → α
  mul_assoc : ∀ a b c : α, mul (mul a b) c = mul a (mul b c)
  one_mul : ∀ a : α, mul one a = a
  mul_one : ∀ a : α, mul a one = a
  inv_mul_axiom : ∀ a : α, mul (inv a) a = one

infixl:70 " * " => GroupLike.mul
postfix:max "⁻¹" => GroupLike.inv

theorem mul_left_cancel {α : Type u} [GroupLike α]
    (a b c : α) (h : a * b = a * c) : b = c := by
  calc
    b = GroupLike.one (α := α) * b := by
      symm
      exact GroupLike.one_mul (α := α) b
    _ = (a⁻¹ * a) * b := by
      rw [GroupLike.inv_mul_axiom (α := α) a]
    _ = a⁻¹ * (a * b) := by
      rw [GroupLike.mul_assoc (α := α) (a⁻¹) a b]
    _ = a⁻¹ * (a * c) := by
      rw [h]
    _ = (a⁻¹ * a) * c := by
      rw [← GroupLike.mul_assoc (α := α) (a⁻¹) a c]
    _ = GroupLike.one (α := α) * c := by
      rw [GroupLike.inv_mul_axiom (α := α) a]
    _ = c := by
      rw [GroupLike.one_mul (α := α) c]

theorem mul_right_cancel {α : Type u} [GroupLike α]
    (a b c : α) (h : b * a = c * a) : b = c := by
  have h_inv : a * a⁻¹ = GroupLike.one (α := α) := by
    have h_aux : a⁻¹ * (a * a⁻¹) = a⁻¹ * GroupLike.one (α := α) := by
      calc
        a⁻¹ * (a * a⁻¹) = (a⁻¹ * a) * a⁻¹ := by
          rw [← GroupLike.mul_assoc (α := α) (a⁻¹) a (a⁻¹)]
        _ = GroupLike.one (α := α) * a⁻¹ := by
          rw [GroupLike.inv_mul_axiom (α := α) a]
        _ = a⁻¹ := by
          rw [GroupLike.one_mul (α := α) (a⁻¹)]
        _ = a⁻¹ * GroupLike.one (α := α) := by
          rw [GroupLike.mul_one (α := α) (a⁻¹)]
    exact mul_left_cancel (a := a⁻¹) (b := a * a⁻¹) (c := GroupLike.one (α := α)) h_aux
  calc
    b = b * GroupLike.one (α := α) := by
      symm
      exact GroupLike.mul_one (α := α) b
    _ = b * (a * a⁻¹) := by
      rw [h_inv]
    _ = (b * a) * a⁻¹ := by
      rw [← GroupLike.mul_assoc (α := α) b a (a⁻¹)]
    _ = (c * a) * a⁻¹ := by
      rw [h]
    _ = c * (a * a⁻¹) := by
      rw [← GroupLike.mul_assoc (α := α) c a (a⁻¹)]
    _ = c * GroupLike.one (α := α) := by
      rw [h_inv]
    _ = c := by
      rw [GroupLike.mul_one (α := α) c]

theorem inv_inv {α : Type u} [GroupLike α] (a : α) : (a⁻¹)⁻¹ = a := by
  have h_inv : a * a⁻¹ = GroupLike.one (α := α) := by
    have h_cancel : a⁻¹ * (a * a⁻¹) = a⁻¹ * GroupLike.one (α := α) := by
      calc
        a⁻¹ * (a * a⁻¹) = (a⁻¹ * a) * a⁻¹ := by
          rw [← GroupLike.mul_assoc (α := α) (a⁻¹) a (a⁻¹)]
        _ = GroupLike.one (α := α) * a⁻¹ := by
          rw [GroupLike.inv_mul_axiom (α := α) a]
        _ = a⁻¹ := by
          rw [GroupLike.one_mul (α := α) (a⁻¹)]
        _ = a⁻¹ * GroupLike.one (α := α) := by
          rw [GroupLike.mul_one (α := α) (a⁻¹)]
    exact mul_left_cancel (a := a⁻¹) (b := a * a⁻¹) (c := GroupLike.one (α := α)) h_cancel
  have h_aux : (a⁻¹)⁻¹ * a⁻¹ = a * a⁻¹ := by
    calc
      (a⁻¹)⁻¹ * a⁻¹ = GroupLike.one (α := α) := by
        exact GroupLike.inv_mul_axiom (α := α) (a⁻¹)
      _ = a * a⁻¹ := by
        symm
        exact h_inv
  exact mul_right_cancel (a := a⁻¹) (b := (a⁻¹)⁻¹) (c := a) h_aux

theorem inv_mul {α : Type u} [GroupLike α] (a b : α) : a⁻¹ * (a * b) = b := by
  calc
    a⁻¹ * (a * b) = (a⁻¹ * a) * b := by
      rw [← GroupLike.mul_assoc (α := α) (a⁻¹) a b]
    _ = GroupLike.one (α := α) * b := by
      rw [GroupLike.inv_mul_axiom (α := α) a]
    _ = b := by
      rw [GroupLike.one_mul (α := α) b]

theorem mul_inv_cancel {α : Type u} [GroupLike α] (a : α) : a * a⁻¹ = GroupLike.one (α := α) := by
  have h_aux : a⁻¹ * (a * a⁻¹) = a⁻¹ * GroupLike.one (α := α) := by
    calc
      a⁻¹ * (a * a⁻¹) = a⁻¹ := by
        rw [inv_mul (a := a) (b := a⁻¹)]
      _ = a⁻¹ * GroupLike.one (α := α) := by
        rw [GroupLike.mul_one (α := α) (a⁻¹)]
  exact mul_left_cancel (a := a⁻¹) (b := a * a⁻¹) (c := GroupLike.one (α := α)) h_aux

theorem inv_mul_cancel {α : Type u} [GroupLike α] (a : α) : a⁻¹ * a = GroupLike.one (α := α) := by
  calc
    a⁻¹ * a = a⁻¹ * (a * GroupLike.one (α := α)) := by
      rw [GroupLike.mul_one (α := α) a]
    _ = GroupLike.one (α := α) := by
      rw [inv_mul (a := a) (b := GroupLike.one (α := α))]

theorem eq_inv_of_mul_eq_one {α : Type u} [GroupLike α]
    (a b : α) (h : a * b = GroupLike.one (α := α)) : a = b⁻¹ := by
  have h_aux : a * b = b⁻¹ * b := by
    calc
      a * b = GroupLike.one (α := α) := h
      _ = b⁻¹ * b := by
        symm
        exact inv_mul_cancel (a := b)
  exact mul_right_cancel (a := b) (b := a) (c := b⁻¹) h_aux

theorem eq_inv_of_mul_eq_one_left {α : Type u} [GroupLike α]
    (a b : α) (h : b * a = GroupLike.one (α := α)) : a = b⁻¹ := by
  have h_swap : b = a⁻¹ := eq_inv_of_mul_eq_one (a := b) (b := a) h
  calc
    a = (a⁻¹)⁻¹ := by
      symm
      exact inv_inv (a := a)
    _ = b⁻¹ := by
      rw [← h_swap]

theorem inv_eq_of_mul_eq_one {α : Type u} [GroupLike α]
    (a b : α) (h : a * b = GroupLike.one (α := α)) : a⁻¹ = b := by
  have h_aux : a * a⁻¹ = a * b := by
    calc
      a * a⁻¹ = GroupLike.one (α := α) := mul_inv_cancel (a := a)
      _ = a * b := by
        symm
        exact h
  exact mul_left_cancel (a := a) (b := a⁻¹) (c := b) h_aux

theorem inv_eq_of_eq_inv {α : Type u} [GroupLike α]
    (a b : α) (h : a = b⁻¹) : a⁻¹ = b := by
  have h_mul : a * b = GroupLike.one (α := α) := by
    calc
      a * b = b⁻¹ * b := by
        rw [h]
      _ = GroupLike.one (α := α) := by
        exact inv_mul_cancel (a := b)
  exact inv_eq_of_mul_eq_one (a := a) (b := b) h_mul

theorem mul_eq_one_of_eq_inv {α : Type u} [GroupLike α]
    (a b : α) (h : a = b⁻¹) : a * b = GroupLike.one (α := α) := by
  calc
    a * b = b⁻¹ * b := by
      rw [h]
    _ = GroupLike.one (α := α) := by
      exact inv_mul_cancel (a := b)
