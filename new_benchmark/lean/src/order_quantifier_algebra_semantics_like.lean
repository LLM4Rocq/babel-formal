/-
BENCHMARK_ID: TINY_MATHLIB_BATCH05_ORDER_QUANTIFIER_ALGEBRA_SEMANTICS_LIKE
PAIR_STEM: order_quantifier_algebra_semantics_like
MATH_DOMAIN: Order Theory
SOURCE_MATHLIB: Mathlib/Order/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class OrderStruct_quantifier_algebra_semantics (A : Type u) where
  le : A -> A -> Prop
  le_refl : forall x : A, le x x
  le_trans : forall {x y z : A}, le x y -> le y z -> le x z
  lower : A -> A
  upper : A -> A
  lower_mono_axiom : forall {x y : A}, le x y -> le (lower x) (lower y)
  upper_mono_axiom : forall {x y : A}, le x y -> le (upper x) (upper y)
  lower_extensive_axiom : forall x : A, le x (lower x)
  upper_reductive_axiom : forall x : A, le (upper x) x
  lower_idem_axiom : forall x : A, lower (lower x) = lower x
  upper_idem_axiom : forall x : A, upper (upper x) = upper x
  adjoint_axiom : forall x y : A, le (lower x) y <-> le x (upper y)

infix:50 " ⊑ " => OrderStruct_quantifier_algebra_semantics.le

def LowerOp_quantifier_algebra_semantics
    {A : Type u} [OrderStruct_quantifier_algebra_semantics A] (x : A) : A :=
  OrderStruct_quantifier_algebra_semantics.lower x

def UpperOp_quantifier_algebra_semantics
    {A : Type u} [OrderStruct_quantifier_algebra_semantics A] (x : A) : A :=
  OrderStruct_quantifier_algebra_semantics.upper x

def FixedSet_quantifier_algebra_semantics
    {A : Type u} [OrderStruct_quantifier_algebra_semantics A] (x : A) : Prop :=
  LowerOp_quantifier_algebra_semantics x = x /\
  UpperOp_quantifier_algebra_semantics x = x

def IterStep_quantifier_algebra_semantics
    {A : Type u} [OrderStruct_quantifier_algebra_semantics A] (x : A) : A :=
  LowerOp_quantifier_algebra_semantics (LowerOp_quantifier_algebra_semantics x)

theorem lower_mono_quantifier_algebra_semantics
    {A : Type u} [OrderStruct_quantifier_algebra_semantics A]
    {x y : A} (hxy : x ⊑ y) :
    LowerOp_quantifier_algebra_semantics x ⊑
      LowerOp_quantifier_algebra_semantics y := by
  have hbase : OrderStruct_quantifier_algebra_semantics.lower x ⊑
      OrderStruct_quantifier_algebra_semantics.lower y :=
    OrderStruct_quantifier_algebra_semantics.lower_mono_axiom hxy
  simpa [LowerOp_quantifier_algebra_semantics] using hbase

theorem upper_mono_quantifier_algebra_semantics
    {A : Type u} [OrderStruct_quantifier_algebra_semantics A]
    {x y : A} (hxy : x ⊑ y) :
    UpperOp_quantifier_algebra_semantics x ⊑
      UpperOp_quantifier_algebra_semantics y := by
  have hbase : OrderStruct_quantifier_algebra_semantics.upper x ⊑
      OrderStruct_quantifier_algebra_semantics.upper y :=
    OrderStruct_quantifier_algebra_semantics.upper_mono_axiom hxy
  simpa [UpperOp_quantifier_algebra_semantics] using hbase

theorem lower_extensive_quantifier_algebra_semantics
    {A : Type u} [OrderStruct_quantifier_algebra_semantics A]
    (x : A) :
    x ⊑ LowerOp_quantifier_algebra_semantics x := by
  have hraw : x ⊑ OrderStruct_quantifier_algebra_semantics.lower x :=
    OrderStruct_quantifier_algebra_semantics.lower_extensive_axiom x
  simpa [LowerOp_quantifier_algebra_semantics] using hraw

theorem upper_reductive_quantifier_algebra_semantics
    {A : Type u} [OrderStruct_quantifier_algebra_semantics A]
    (x : A) :
    UpperOp_quantifier_algebra_semantics x ⊑ x := by
  have hraw : OrderStruct_quantifier_algebra_semantics.upper x ⊑ x :=
    OrderStruct_quantifier_algebra_semantics.upper_reductive_axiom x
  simpa [UpperOp_quantifier_algebra_semantics] using hraw

theorem iteration_stable_quantifier_algebra_semantics
    {A : Type u} [OrderStruct_quantifier_algebra_semantics A]
    (x : A) :
    IterStep_quantifier_algebra_semantics (IterStep_quantifier_algebra_semantics x) =
      IterStep_quantifier_algebra_semantics x /\
    LowerOp_quantifier_algebra_semantics (IterStep_quantifier_algebra_semantics x) =
      IterStep_quantifier_algebra_semantics x := by
  have hIdemOuter :
      LowerOp_quantifier_algebra_semantics
          (LowerOp_quantifier_algebra_semantics
            (LowerOp_quantifier_algebra_semantics
              (LowerOp_quantifier_algebra_semantics x))) =
        LowerOp_quantifier_algebra_semantics
          (LowerOp_quantifier_algebra_semantics x) := by
    change OrderStruct_quantifier_algebra_semantics.lower
        (OrderStruct_quantifier_algebra_semantics.lower
          (OrderStruct_quantifier_algebra_semantics.lower
            (OrderStruct_quantifier_algebra_semantics.lower x))) =
      OrderStruct_quantifier_algebra_semantics.lower
        (OrderStruct_quantifier_algebra_semantics.lower x)
    rw [OrderStruct_quantifier_algebra_semantics.lower_idem_axiom]
    rw [OrderStruct_quantifier_algebra_semantics.lower_idem_axiom]
  have hIdemInner :
      LowerOp_quantifier_algebra_semantics
        (IterStep_quantifier_algebra_semantics x) =
      IterStep_quantifier_algebra_semantics x := by
    change OrderStruct_quantifier_algebra_semantics.lower
        (OrderStruct_quantifier_algebra_semantics.lower
          (OrderStruct_quantifier_algebra_semantics.lower x)) =
      OrderStruct_quantifier_algebra_semantics.lower
        (OrderStruct_quantifier_algebra_semantics.lower x)
    rw [OrderStruct_quantifier_algebra_semantics.lower_idem_axiom]
  exact ⟨hIdemOuter, hIdemInner⟩

theorem fixedpoint_characterization_quantifier_algebra_semantics
    {A : Type u} [OrderStruct_quantifier_algebra_semantics A]
    {x : A} (hfix : FixedSet_quantifier_algebra_semantics x) :
    IterStep_quantifier_algebra_semantics x = x /\
    UpperOp_quantifier_algebra_semantics x = x := by
  rcases hfix with ⟨hlower, hupper⟩
  have hIter : IterStep_quantifier_algebra_semantics x =
      LowerOp_quantifier_algebra_semantics (LowerOp_quantifier_algebra_semantics x) := by
    rfl
  have hMain : IterStep_quantifier_algebra_semantics x = x := by
    calc
      IterStep_quantifier_algebra_semantics x =
          LowerOp_quantifier_algebra_semantics (LowerOp_quantifier_algebra_semantics x) := hIter
      _ = LowerOp_quantifier_algebra_semantics x := by
            change OrderStruct_quantifier_algebra_semantics.lower
              (OrderStruct_quantifier_algebra_semantics.lower x) =
              OrderStruct_quantifier_algebra_semantics.lower x
            exact OrderStruct_quantifier_algebra_semantics.lower_idem_axiom x
      _ = x := hlower
  exact ⟨hMain, hupper⟩

theorem duality_bridge_quantifier_algebra_semantics
    {A : Type u} [OrderStruct_quantifier_algebra_semantics A]
    (x y : A) :
    (LowerOp_quantifier_algebra_semantics x ⊑ y) ↔
      (x ⊑ UpperOp_quantifier_algebra_semantics y) := by
  have hAdj : OrderStruct_quantifier_algebra_semantics.le
      (OrderStruct_quantifier_algebra_semantics.lower x) y <->
      OrderStruct_quantifier_algebra_semantics.le x
        (OrderStruct_quantifier_algebra_semantics.upper y) :=
    OrderStruct_quantifier_algebra_semantics.adjoint_axiom x y
  constructor
  · intro hxy
    have hNorm :
        OrderStruct_quantifier_algebra_semantics.le
          (OrderStruct_quantifier_algebra_semantics.lower x) y := by
      simpa [LowerOp_quantifier_algebra_semantics] using hxy
    have hOut :
        OrderStruct_quantifier_algebra_semantics.le x
          (OrderStruct_quantifier_algebra_semantics.upper y) :=
      (Iff.mp hAdj) hNorm
    simpa [UpperOp_quantifier_algebra_semantics] using hOut
  · intro hxy
    have hNorm :
        OrderStruct_quantifier_algebra_semantics.le x
          (OrderStruct_quantifier_algebra_semantics.upper y) := by
      simpa [UpperOp_quantifier_algebra_semantics] using hxy
    have hOut :
        OrderStruct_quantifier_algebra_semantics.le
          (OrderStruct_quantifier_algebra_semantics.lower x) y :=
      (Iff.mpr hAdj) hNorm
    simpa [LowerOp_quantifier_algebra_semantics] using hOut
