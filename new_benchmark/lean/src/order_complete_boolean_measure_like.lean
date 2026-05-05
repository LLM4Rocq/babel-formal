/-
BENCHMARK_ID: TINY_MATHLIB_BATCH05_ORDER_COMPLETE_BOOLEAN_MEASURE_LIKE
PAIR_STEM: order_complete_boolean_measure_like
MATH_DOMAIN: Order Theory
SOURCE_MATHLIB: Mathlib/Order/CompleteBooleanAlgebra
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class OrderStruct_complete_boolean_measure (α : Type u) where
  le : α → α → Prop
  lower : α → α
  upper : α → α
  measure : α → Nat
  le_refl : ∀ x : α, le x x
  le_trans : ∀ {x y z : α}, le x y → le y z → le x z
  lower_mono_axiom : ∀ {x y : α}, le x y → le (lower x) (lower y)
  upper_mono_axiom : ∀ {x y : α}, le x y → le (upper x) (upper y)
  lower_extensive_axiom : ∀ x : α, le x (lower x)
  upper_reductive_axiom : ∀ x : α, le (upper x) x
  absorb_axiom : ∀ x : α, lower (upper x) = upper x
  fixed_axiom : ∀ x : α, lower x = x ↔ upper x = x
  duality_axiom : ∀ x : α, measure (lower x) + measure (upper x) = measure x + measure x

def LowerOp_complete_boolean_measure {α : Type u}
    [S : OrderStruct_complete_boolean_measure α] (x : α) : α :=
  S.lower x

def UpperOp_complete_boolean_measure {α : Type u}
    [S : OrderStruct_complete_boolean_measure α] (x : α) : α :=
  S.upper x

def FixedSet_complete_boolean_measure {α : Type u}
    [S : OrderStruct_complete_boolean_measure α] (x : α) : Prop :=
  LowerOp_complete_boolean_measure x = x ∧ UpperOp_complete_boolean_measure x = x

def IterStep_complete_boolean_measure {α : Type u}
    [S : OrderStruct_complete_boolean_measure α] (x : α) : α :=
  LowerOp_complete_boolean_measure (UpperOp_complete_boolean_measure x)

theorem lower_mono_complete_boolean_measure {α : Type u}
    [S : OrderStruct_complete_boolean_measure α]
    {x y : α} (hxy : S.le x y) :
    S.le (LowerOp_complete_boolean_measure x) (LowerOp_complete_boolean_measure y) := by
  have hLower : S.le (S.lower x) (S.lower y) := S.lower_mono_axiom hxy
  have hx : LowerOp_complete_boolean_measure x = S.lower x := by
    rfl
  have hy : LowerOp_complete_boolean_measure y = S.lower y := by
    rfl
  rw [hx, hy]
  exact hLower

theorem upper_mono_complete_boolean_measure {α : Type u}
    [S : OrderStruct_complete_boolean_measure α]
    {x y : α} (hxy : S.le x y) :
    S.le (UpperOp_complete_boolean_measure x) (UpperOp_complete_boolean_measure y) := by
  have hUpper : S.le (S.upper x) (S.upper y) := S.upper_mono_axiom hxy
  have hx : UpperOp_complete_boolean_measure x = S.upper x := by
    rfl
  have hy : UpperOp_complete_boolean_measure y = S.upper y := by
    rfl
  have hRewrite :
      S.le (UpperOp_complete_boolean_measure x) (UpperOp_complete_boolean_measure y)
        = S.le (S.upper x) (S.upper y) := by
    rw [hx, hy]
  rw [hRewrite]
  exact hUpper

theorem lower_extensive_complete_boolean_measure {α : Type u}
    [S : OrderStruct_complete_boolean_measure α]
    (x : α) :
    S.le x (LowerOp_complete_boolean_measure x) := by
  have hExt : S.le x (S.lower x) := S.lower_extensive_axiom x
  have hLower : LowerOp_complete_boolean_measure x = S.lower x := by
    rfl
  rw [hLower]
  exact hExt

theorem upper_reductive_complete_boolean_measure {α : Type u}
    [S : OrderStruct_complete_boolean_measure α]
    (x : α) :
    S.le (UpperOp_complete_boolean_measure x) x := by
  have hRed : S.le (S.upper x) x := S.upper_reductive_axiom x
  have hUpper : UpperOp_complete_boolean_measure x = S.upper x := by
    rfl
  rw [hUpper]
  exact hRed

theorem iteration_stable_complete_boolean_measure {α : Type u}
    [S : OrderStruct_complete_boolean_measure α]
    (x : α) :
    IterStep_complete_boolean_measure x = UpperOp_complete_boolean_measure x := by
  have hAbs : S.lower (S.upper x) = S.upper x := S.absorb_axiom x
  have hIter : IterStep_complete_boolean_measure x = S.lower (S.upper x) := by
    rfl
  have hUpper : UpperOp_complete_boolean_measure x = S.upper x := by
    rfl
  calc
    IterStep_complete_boolean_measure x = S.lower (S.upper x) := hIter
    _ = S.upper x := hAbs
    _ = UpperOp_complete_boolean_measure x := by
      symm
      exact hUpper

theorem fixedpoint_characterization_complete_boolean_measure {α : Type u}
    [S : OrderStruct_complete_boolean_measure α]
    (x : α) :
    FixedSet_complete_boolean_measure x ↔
      (LowerOp_complete_boolean_measure x = x ∧ IterStep_complete_boolean_measure x = x) := by
  constructor
  · intro hFixed
    rcases hFixed with ⟨hLowerEq, hUpperEq⟩
    have hIterUpper : IterStep_complete_boolean_measure x = UpperOp_complete_boolean_measure x :=
      iteration_stable_complete_boolean_measure x
    have hIterEq : IterStep_complete_boolean_measure x = x := by
      calc
        IterStep_complete_boolean_measure x = UpperOp_complete_boolean_measure x := hIterUpper
        _ = x := hUpperEq
    exact ⟨hLowerEq, hIterEq⟩
  · intro hData
    rcases hData with ⟨hLowerEq, hIterEq⟩
    have hIterUpper : IterStep_complete_boolean_measure x = UpperOp_complete_boolean_measure x :=
      iteration_stable_complete_boolean_measure x
    have hUpperEq : UpperOp_complete_boolean_measure x = x := by
      calc
        UpperOp_complete_boolean_measure x = IterStep_complete_boolean_measure x := by
          symm
          exact hIterUpper
        _ = x := hIterEq
    exact ⟨hLowerEq, hUpperEq⟩

theorem duality_bridge_complete_boolean_measure {α : Type u}
    [S : OrderStruct_complete_boolean_measure α]
    (x : α)
    (hFixed : FixedSet_complete_boolean_measure x) :
    S.measure (LowerOp_complete_boolean_measure x) =
      S.measure (UpperOp_complete_boolean_measure x) := by
  rcases hFixed with ⟨hLowerEq, hUpperEq⟩
  have hDual : S.measure (S.lower x) + S.measure (S.upper x) = S.measure x + S.measure x :=
    S.duality_axiom x
  have hLowerRaw : S.lower x = x := by
    simpa [LowerOp_complete_boolean_measure] using hLowerEq
  have hUpperRaw : S.upper x = x := by
    simpa [UpperOp_complete_boolean_measure] using hUpperEq
  have hLowerVal : S.measure (LowerOp_complete_boolean_measure x) = S.measure x := by
    rw [hLowerEq]
  have hUpperVal : S.measure (UpperOp_complete_boolean_measure x) = S.measure x := by
    rw [hUpperEq]
  have hCheck : S.measure x + S.measure x = S.measure x + S.measure x := by
    calc
      S.measure x + S.measure x = S.measure (S.lower x) + S.measure (S.upper x) := by
        symm
        exact hDual
      _ = S.measure x + S.measure x := by
        rw [hLowerRaw, hUpperRaw]
  calc
    S.measure (LowerOp_complete_boolean_measure x) = S.measure x := hLowerVal
    _ = S.measure (UpperOp_complete_boolean_measure x) := by
      symm
      exact hUpperVal
