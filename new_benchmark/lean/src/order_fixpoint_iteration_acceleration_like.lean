/-
BENCHMARK_ID: TINY_MATHLIB_BATCH05_ORDER_FIXPOINT_ITERATION_ACCELERATION_LIKE
PAIR_STEM: order_fixpoint_iteration_acceleration_like
MATH_DOMAIN: Order Theory
SOURCE_MATHLIB: Mathlib/Order/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class OrderStruct_fixpoint_iteration_acceleration (α : Type u) where
  le : α -> α -> Prop
  le_refl : forall x : α, le x x
  le_trans : forall {x y z : α}, le x y -> le y z -> le x z
  lower : α -> α
  upper : α -> α
  lower_mono_axiom : forall {x y : α}, le x y -> le (lower x) (lower y)
  upper_mono_axiom : forall {x y : α}, le x y -> le (upper x) (upper y)
  lower_extensive_axiom : forall x : α, le x (lower x)
  upper_reductive_axiom : forall x : α, le (upper x) x
  bridge_axiom : forall x : α, le (lower (upper x)) (upper (lower x))


def LowerOp_fixpoint_iteration_acceleration {α : Type u}
    [OrderStruct_fixpoint_iteration_acceleration α] : α -> α :=
  OrderStruct_fixpoint_iteration_acceleration.lower

def UpperOp_fixpoint_iteration_acceleration {α : Type u}
    [OrderStruct_fixpoint_iteration_acceleration α] : α -> α :=
  OrderStruct_fixpoint_iteration_acceleration.upper

def FixedSet_fixpoint_iteration_acceleration {α : Type u}
    [OrderStruct_fixpoint_iteration_acceleration α] (x : α) : Prop :=
  LowerOp_fixpoint_iteration_acceleration x = x ∧
    UpperOp_fixpoint_iteration_acceleration x = x

def IterStep_fixpoint_iteration_acceleration {α : Type u}
    [OrderStruct_fixpoint_iteration_acceleration α] (x : α) : α :=
  LowerOp_fixpoint_iteration_acceleration (UpperOp_fixpoint_iteration_acceleration x)

theorem lower_mono_fixpoint_iteration_acceleration {α : Type u}
    [OrderStruct_fixpoint_iteration_acceleration α]
    {x y : α}
    (hxy : OrderStruct_fixpoint_iteration_acceleration.le x y) :
    OrderStruct_fixpoint_iteration_acceleration.le
      (LowerOp_fixpoint_iteration_acceleration x)
      (LowerOp_fixpoint_iteration_acceleration y) := by
  have hraw :
      OrderStruct_fixpoint_iteration_acceleration.le
        (OrderStruct_fixpoint_iteration_acceleration.lower x)
        (OrderStruct_fixpoint_iteration_acceleration.lower y) :=
    OrderStruct_fixpoint_iteration_acceleration.lower_mono_axiom hxy
  simpa [LowerOp_fixpoint_iteration_acceleration] using hraw

theorem upper_mono_fixpoint_iteration_acceleration {α : Type u}
    [OrderStruct_fixpoint_iteration_acceleration α]
    {x y : α}
    (hxy : OrderStruct_fixpoint_iteration_acceleration.le x y) :
    OrderStruct_fixpoint_iteration_acceleration.le
      (UpperOp_fixpoint_iteration_acceleration x)
      (UpperOp_fixpoint_iteration_acceleration y) := by
  have hraw :
      OrderStruct_fixpoint_iteration_acceleration.le
        (OrderStruct_fixpoint_iteration_acceleration.upper x)
        (OrderStruct_fixpoint_iteration_acceleration.upper y) :=
    OrderStruct_fixpoint_iteration_acceleration.upper_mono_axiom hxy
  simpa [UpperOp_fixpoint_iteration_acceleration] using hraw

theorem lower_extensive_fixpoint_iteration_acceleration {α : Type u}
    [OrderStruct_fixpoint_iteration_acceleration α]
    (x : α) :
    OrderStruct_fixpoint_iteration_acceleration.le x
      (LowerOp_fixpoint_iteration_acceleration x) ∧
    OrderStruct_fixpoint_iteration_acceleration.le
      (LowerOp_fixpoint_iteration_acceleration x)
      (LowerOp_fixpoint_iteration_acceleration x) := by
  have hbase :
      OrderStruct_fixpoint_iteration_acceleration.le x
        (OrderStruct_fixpoint_iteration_acceleration.lower x) :=
    OrderStruct_fixpoint_iteration_acceleration.lower_extensive_axiom x
  have hself :
      OrderStruct_fixpoint_iteration_acceleration.le
        (LowerOp_fixpoint_iteration_acceleration x)
        (LowerOp_fixpoint_iteration_acceleration x) :=
    OrderStruct_fixpoint_iteration_acceleration.le_refl _
  refine ⟨?_, hself⟩
  simpa [LowerOp_fixpoint_iteration_acceleration] using hbase

theorem upper_reductive_fixpoint_iteration_acceleration {α : Type u}
    [OrderStruct_fixpoint_iteration_acceleration α]
    (x : α) :
    OrderStruct_fixpoint_iteration_acceleration.le
      (UpperOp_fixpoint_iteration_acceleration x) x := by
  have hbase :
      OrderStruct_fixpoint_iteration_acceleration.le
        (OrderStruct_fixpoint_iteration_acceleration.upper x) x :=
    OrderStruct_fixpoint_iteration_acceleration.upper_reductive_axiom x
  simpa [UpperOp_fixpoint_iteration_acceleration] using hbase

theorem iteration_stable_fixpoint_iteration_acceleration {α : Type u}
    [OrderStruct_fixpoint_iteration_acceleration α]
    (x : α) :
    OrderStruct_fixpoint_iteration_acceleration.le
      (IterStep_fixpoint_iteration_acceleration x)
      (UpperOp_fixpoint_iteration_acceleration
        (LowerOp_fixpoint_iteration_acceleration x)) ∧
    OrderStruct_fixpoint_iteration_acceleration.le
      (IterStep_fixpoint_iteration_acceleration x)
      (IterStep_fixpoint_iteration_acceleration x) := by
  have hbridge :
      OrderStruct_fixpoint_iteration_acceleration.le
        (OrderStruct_fixpoint_iteration_acceleration.lower
          (OrderStruct_fixpoint_iteration_acceleration.upper x))
        (OrderStruct_fixpoint_iteration_acceleration.upper
          (OrderStruct_fixpoint_iteration_acceleration.lower x)) :=
    OrderStruct_fixpoint_iteration_acceleration.bridge_axiom x
  have hstable :
      OrderStruct_fixpoint_iteration_acceleration.le
        (IterStep_fixpoint_iteration_acceleration x)
        (UpperOp_fixpoint_iteration_acceleration
          (LowerOp_fixpoint_iteration_acceleration x)) := by
    simpa [IterStep_fixpoint_iteration_acceleration,
      LowerOp_fixpoint_iteration_acceleration,
      UpperOp_fixpoint_iteration_acceleration] using hbridge
  have hself :
      OrderStruct_fixpoint_iteration_acceleration.le
        (IterStep_fixpoint_iteration_acceleration x)
        (IterStep_fixpoint_iteration_acceleration x) :=
    OrderStruct_fixpoint_iteration_acceleration.le_refl _
  exact ⟨hstable, hself⟩

theorem fixedpoint_characterization_fixpoint_iteration_acceleration {α : Type u}
    [OrderStruct_fixpoint_iteration_acceleration α]
    {x : α}
    (hfix : FixedSet_fixpoint_iteration_acceleration x) :
    IterStep_fixpoint_iteration_acceleration x = x ∧
    OrderStruct_fixpoint_iteration_acceleration.le
      (IterStep_fixpoint_iteration_acceleration x)
      (IterStep_fixpoint_iteration_acceleration x) := by
  rcases hfix with ⟨hlow, hupp⟩
  have hEq : IterStep_fixpoint_iteration_acceleration x = x := by
    calc
      IterStep_fixpoint_iteration_acceleration x
          = LowerOp_fixpoint_iteration_acceleration
              (UpperOp_fixpoint_iteration_acceleration x) := rfl
      _ = LowerOp_fixpoint_iteration_acceleration x := by rw [hupp]
      _ = x := hlow
  have hSelf :
      OrderStruct_fixpoint_iteration_acceleration.le
        (IterStep_fixpoint_iteration_acceleration x)
        (IterStep_fixpoint_iteration_acceleration x) :=
    OrderStruct_fixpoint_iteration_acceleration.le_refl _
  exact ⟨hEq, hSelf⟩

theorem duality_bridge_fixpoint_iteration_acceleration {α : Type u}
    [OrderStruct_fixpoint_iteration_acceleration α]
    (x : α) :
    ∃ y : α,
      y = IterStep_fixpoint_iteration_acceleration x ∧
      OrderStruct_fixpoint_iteration_acceleration.le
        y
        (UpperOp_fixpoint_iteration_acceleration
          (LowerOp_fixpoint_iteration_acceleration x)) := by
  refine ⟨IterStep_fixpoint_iteration_acceleration x, rfl, ?_⟩
  have hPair := iteration_stable_fixpoint_iteration_acceleration (α := α) x
  exact hPair.1
