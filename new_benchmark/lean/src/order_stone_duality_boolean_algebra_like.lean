/-
BENCHMARK_ID: TINY_MATHLIB_BATCH05_ORDER_STONE_DUALITY_BOOLEAN_ALGEBRA_LIKE
PAIR_STEM: order_stone_duality_boolean_algebra_like
MATH_DOMAIN: Order Theory
SOURCE_MATHLIB: Mathlib/Order/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class OrderStruct_stone_duality_boolean (A : Type u) where
  le : A -> A -> Prop
  le_refl : forall x : A, le x x
  le_trans : forall {x y z : A}, le x y -> le y z -> le x z
  compl : A -> A
  lower : A -> A
  upper : A -> A
  lower_mono_axiom : forall {x y : A}, le x y -> le (lower x) (lower y)
  upper_mono_axiom : forall {x y : A}, le x y -> le (upper x) (upper y)
  lower_extensive_axiom : forall x : A, le x (lower x)
  upper_reductive_axiom : forall x : A, le (upper x) x
  stone_bridge_axiom : forall x y : A, le (lower x) y <-> le x (upper y)
  deMorgan_axiom : forall x : A, lower (compl x) = compl (upper x)

infix:50 " <=b " => OrderStruct_stone_duality_boolean.le

def LowerOp_stone_duality_boolean {A : Type u}
    [OrderStruct_stone_duality_boolean A] (x : A) : A :=
  OrderStruct_stone_duality_boolean.lower x

def UpperOp_stone_duality_boolean {A : Type u}
    [OrderStruct_stone_duality_boolean A] (x : A) : A :=
  OrderStruct_stone_duality_boolean.upper x

def FixedSet_stone_duality_boolean {A : Type u}
    [OrderStruct_stone_duality_boolean A] (x : A) : Prop :=
  LowerOp_stone_duality_boolean x = x /\
    UpperOp_stone_duality_boolean x = x

def IterStep_stone_duality_boolean {A : Type u}
    [OrderStruct_stone_duality_boolean A] (x : A) : A :=
  LowerOp_stone_duality_boolean (UpperOp_stone_duality_boolean x)

theorem lower_mono_stone_duality_boolean {A : Type u}
    [OrderStruct_stone_duality_boolean A]
    {x y : A} (hxy : x <=b y) :
    LowerOp_stone_duality_boolean x <=b LowerOp_stone_duality_boolean y := by
  have hRaw :
      OrderStruct_stone_duality_boolean.lower x <=b
        OrderStruct_stone_duality_boolean.lower y :=
    OrderStruct_stone_duality_boolean.lower_mono_axiom hxy
  calc
    LowerOp_stone_duality_boolean x
        = OrderStruct_stone_duality_boolean.lower x := by
          rfl
    _ <=b OrderStruct_stone_duality_boolean.lower y := hRaw
    _ = LowerOp_stone_duality_boolean y := by
          rfl

theorem upper_mono_stone_duality_boolean {A : Type u}
    [OrderStruct_stone_duality_boolean A]
    {x y : A} (hxy : x <=b y) :
    UpperOp_stone_duality_boolean x <=b UpperOp_stone_duality_boolean y := by
  have hRaw :
      OrderStruct_stone_duality_boolean.upper x <=b
        OrderStruct_stone_duality_boolean.upper y :=
    OrderStruct_stone_duality_boolean.upper_mono_axiom hxy
  calc
    UpperOp_stone_duality_boolean x
        = OrderStruct_stone_duality_boolean.upper x := by
          rfl
    _ <=b OrderStruct_stone_duality_boolean.upper y := hRaw
    _ = UpperOp_stone_duality_boolean y := by
          rfl

theorem lower_extensive_stone_duality_boolean {A : Type u}
    [OrderStruct_stone_duality_boolean A]
    (x : A) :
    x <=b LowerOp_stone_duality_boolean x := by
  have hRaw : x <=b OrderStruct_stone_duality_boolean.lower x :=
    OrderStruct_stone_duality_boolean.lower_extensive_axiom x
  calc
    x <=b OrderStruct_stone_duality_boolean.lower x := hRaw
    _ = LowerOp_stone_duality_boolean x := by
          rfl

theorem upper_reductive_stone_duality_boolean {A : Type u}
    [OrderStruct_stone_duality_boolean A]
    (x : A) :
    UpperOp_stone_duality_boolean x <=b x := by
  have hRaw : OrderStruct_stone_duality_boolean.upper x <=b x :=
    OrderStruct_stone_duality_boolean.upper_reductive_axiom x
  calc
    UpperOp_stone_duality_boolean x
        = OrderStruct_stone_duality_boolean.upper x := by
          rfl
    _ <=b x := hRaw

theorem iteration_stable_stone_duality_boolean {A : Type u}
    [OrderStruct_stone_duality_boolean A]
    (hLowerIdem :
      forall x : A,
        LowerOp_stone_duality_boolean (LowerOp_stone_duality_boolean x) =
          LowerOp_stone_duality_boolean x)
    (hUpperIdem :
      forall x : A,
        UpperOp_stone_duality_boolean (UpperOp_stone_duality_boolean x) =
          UpperOp_stone_duality_boolean x)
    (hComm :
      forall x : A,
        UpperOp_stone_duality_boolean (LowerOp_stone_duality_boolean x) =
          LowerOp_stone_duality_boolean (UpperOp_stone_duality_boolean x))
    (x : A) :
    IterStep_stone_duality_boolean (IterStep_stone_duality_boolean x) =
      IterStep_stone_duality_boolean x := by
  have hCommX :
      UpperOp_stone_duality_boolean
          (LowerOp_stone_duality_boolean (UpperOp_stone_duality_boolean x)) =
        LowerOp_stone_duality_boolean
          (UpperOp_stone_duality_boolean (UpperOp_stone_duality_boolean x)) :=
    hComm (UpperOp_stone_duality_boolean x)
  have hLift :
      LowerOp_stone_duality_boolean
          (UpperOp_stone_duality_boolean
            (LowerOp_stone_duality_boolean (UpperOp_stone_duality_boolean x))) =
        LowerOp_stone_duality_boolean
          (LowerOp_stone_duality_boolean
            (UpperOp_stone_duality_boolean (UpperOp_stone_duality_boolean x))) := by
    rw [hCommX]
  have hCollapseLower :
      LowerOp_stone_duality_boolean
          (LowerOp_stone_duality_boolean
            (UpperOp_stone_duality_boolean (UpperOp_stone_duality_boolean x))) =
        LowerOp_stone_duality_boolean
          (UpperOp_stone_duality_boolean (UpperOp_stone_duality_boolean x)) :=
    hLowerIdem (UpperOp_stone_duality_boolean (UpperOp_stone_duality_boolean x))
  have hCollapseUpper :
      UpperOp_stone_duality_boolean (UpperOp_stone_duality_boolean x) =
        UpperOp_stone_duality_boolean x :=
    hUpperIdem x
  calc
    IterStep_stone_duality_boolean (IterStep_stone_duality_boolean x)
        = LowerOp_stone_duality_boolean
            (UpperOp_stone_duality_boolean
              (LowerOp_stone_duality_boolean (UpperOp_stone_duality_boolean x))) := by
          rfl
    _ = LowerOp_stone_duality_boolean
          (LowerOp_stone_duality_boolean
            (UpperOp_stone_duality_boolean (UpperOp_stone_duality_boolean x))) := hLift
    _ = LowerOp_stone_duality_boolean
          (UpperOp_stone_duality_boolean (UpperOp_stone_duality_boolean x)) := hCollapseLower
    _ = LowerOp_stone_duality_boolean (UpperOp_stone_duality_boolean x) := by
          rw [hCollapseUpper]
    _ = IterStep_stone_duality_boolean x := by
          rfl

theorem fixedpoint_characterization_stone_duality_boolean {A : Type u}
    [OrderStruct_stone_duality_boolean A]
    (x : A) (hFix : FixedSet_stone_duality_boolean x) :
    IterStep_stone_duality_boolean x = x := by
  have hLower : LowerOp_stone_duality_boolean x = x := hFix.1
  have hUpper : UpperOp_stone_duality_boolean x = x := hFix.2
  calc
    IterStep_stone_duality_boolean x
        = LowerOp_stone_duality_boolean (UpperOp_stone_duality_boolean x) := by
          rfl
    _ = LowerOp_stone_duality_boolean x := by
          rw [hUpper]
    _ = x := hLower

theorem duality_bridge_stone_duality_boolean {A : Type u}
    [OrderStruct_stone_duality_boolean A]
    (x y : A) :
    LowerOp_stone_duality_boolean x <=b y <->
      x <=b UpperOp_stone_duality_boolean y := by
  have hRaw :
      OrderStruct_stone_duality_boolean.lower x <=b y <->
        x <=b OrderStruct_stone_duality_boolean.upper y :=
    OrderStruct_stone_duality_boolean.stone_bridge_axiom x y
  constructor
  · intro hxy
    have hLeft :
        OrderStruct_stone_duality_boolean.lower x <=b y := by
      calc
        OrderStruct_stone_duality_boolean.lower x
            = LowerOp_stone_duality_boolean x := by
              rfl
        _ <=b y := hxy
    have hTo : x <=b OrderStruct_stone_duality_boolean.upper y :=
      (Iff.mp hRaw) hLeft
    calc
      x <=b OrderStruct_stone_duality_boolean.upper y := hTo
      _ = UpperOp_stone_duality_boolean y := by
            rfl
  · intro hxy
    have hRight : x <=b OrderStruct_stone_duality_boolean.upper y := by
      calc
        x <=b UpperOp_stone_duality_boolean y := hxy
        _ = OrderStruct_stone_duality_boolean.upper y := by
              rfl
    have hTo : OrderStruct_stone_duality_boolean.lower x <=b y :=
      (Iff.mpr hRaw) hRight
    calc
      LowerOp_stone_duality_boolean x
          = OrderStruct_stone_duality_boolean.lower x := by
            rfl
      _ <=b y := hTo
