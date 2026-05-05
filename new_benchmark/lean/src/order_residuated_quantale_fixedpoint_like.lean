/-
BENCHMARK_ID: TINY_MATHLIB_BATCH04_ORDER_RESIDUATED_QUANTALE_FIXEDPOINT_LIKE
PAIR_STEM: order_residuated_quantale_fixedpoint_like
MATH_DOMAIN: Order Theory
SOURCE_MATHLIB: Mathlib/Order/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 17
-/

universe u

class QuantaleLike (Q : Type u) where
  le : Q -> Q -> Prop
  le_refl : forall a : Q, le a a
  le_trans : forall {a b c : Q}, le a b -> le b c -> le a c
  mul : Q -> Q -> Q
  lres : Q -> Q -> Q
  rres : Q -> Q -> Q
  resid_left : forall a b c : Q, le (mul a b) c <-> le b (lres a c)
  resid_right : forall a b c : Q, le (mul a b) c <-> le a (rres b c)

infix:50 " <= " => QuantaleLike.le
infixl:70 " *q " => QuantaleLike.mul

def leftRes {Q : Type u} [QuantaleLike Q] (a c : Q) : Q :=
  QuantaleLike.lres a c

def rightRes {Q : Type u} [QuantaleLike Q] (b c : Q) : Q :=
  QuantaleLike.rres b c

def monotone {Q : Type u} [QuantaleLike Q] (f : Q -> Q) : Prop :=
  forall {x y : Q}, x <= y -> f x <= f y

def closureOp {Q : Type u} [QuantaleLike Q] (a x : Q) : Q :=
  leftRes a (a *q x)

def interiorOp {Q : Type u} [QuantaleLike Q] (a x : Q) : Q :=
  (rightRes a x) *q a

theorem residuation_left {Q : Type u} [QuantaleLike Q] (a b c : Q) :
    a *q b <= c <-> b <= leftRes a c := by
  have hRaw : a *q b <= c <-> b <= QuantaleLike.lres a c :=
    QuantaleLike.resid_left a b c
  have hDef : leftRes a c = QuantaleLike.lres a c := by
    rfl
  constructor
  · intro hMul
    have hTo : b <= QuantaleLike.lres a c := (Iff.mp hRaw) hMul
    calc
      b <= QuantaleLike.lres a c := hTo
      _ = leftRes a c := by
            rw [hDef]
  · intro hRes
    have hBack : b <= QuantaleLike.lres a c := by
      calc
        b <= leftRes a c := hRes
        _ = QuantaleLike.lres a c := hDef
    exact (Iff.mpr hRaw) hBack

theorem residuation_right {Q : Type u} [QuantaleLike Q] (a b c : Q) :
    a *q b <= c <-> a <= rightRes b c := by
  have hRaw : a *q b <= c <-> a <= QuantaleLike.rres b c :=
    QuantaleLike.resid_right a b c
  have hDef : rightRes b c = QuantaleLike.rres b c := by
    rfl
  constructor
  · intro hMul
    have hTo : a <= QuantaleLike.rres b c := (Iff.mp hRaw) hMul
    calc
      a <= QuantaleLike.rres b c := hTo
      _ = rightRes b c := by
            rw [hDef]
  · intro hRes
    have hBack : a <= QuantaleLike.rres b c := by
      calc
        a <= rightRes b c := hRes
        _ = QuantaleLike.rres b c := hDef
    exact (Iff.mpr hRaw) hBack

theorem closure_extensive {Q : Type u} [QuantaleLike Q] (a x : Q) :
    x <= closureOp a x := by
  have hDiag : a *q x <= a *q x :=
    QuantaleLike.le_refl (a *q x)
  have hRes : x <= leftRes a (a *q x) :=
    (Iff.mp (residuation_left a x (a *q x))) hDiag
  calc
    x <= leftRes a (a *q x) := hRes
    _ = closureOp a x := by
          rfl

theorem interior_reductive {Q : Type u} [QuantaleLike Q] (a x : Q) :
    interiorOp a x <= x := by
  have hRefl : rightRes a x <= rightRes a x :=
    QuantaleLike.le_refl (rightRes a x)
  have hRaw : (rightRes a x) *q a <= x :=
    (Iff.mpr (residuation_right (rightRes a x) a x)) hRefl
  calc
    interiorOp a x = (rightRes a x) *q a := by
      rfl
    _ <= x := hRaw

theorem fixedpoint_transfer_left {Q : Type u} [QuantaleLike Q]
    (a : Q) (f : Q -> Q) (hf : monotone f)
    (hcl_mono : forall {u v : Q}, u <= v -> closureOp a u <= closureOp a v)
    {x : Q} (hfix : f x <= x) :
    closureOp a (f x) <= closureOp a x := by
  have hSelf : x <= x := QuantaleLike.le_refl x
  have hMonoSelf : f x <= f x := hf hSelf
  have hAnchor : closureOp a (f x) <= closureOp a (f x) :=
    hcl_mono hMonoSelf
  have hDirect : closureOp a (f x) <= closureOp a x :=
    hcl_mono hfix
  exact QuantaleLike.le_trans hAnchor hDirect

theorem fixedpoint_transfer_right {Q : Type u} [QuantaleLike Q]
    (a : Q) (f : Q -> Q) (hf : monotone f)
    (hint_mono : forall {u v : Q}, u <= v -> interiorOp a u <= interiorOp a v)
    {x : Q} (hfix : x <= f x) :
    interiorOp a x <= interiorOp a (f (f x)) := by
  have hFirst : interiorOp a x <= interiorOp a (f x) :=
    hint_mono hfix
  have hNext : f x <= f (f x) :=
    hf hfix
  have hSecond : interiorOp a (f x) <= interiorOp a (f (f x)) :=
    hint_mono hNext
  exact QuantaleLike.le_trans hFirst hSecond

theorem closure_mul_upper {Q : Type u} [QuantaleLike Q] (a x : Q) :
    a *q closureOp a x <= a *q x := by
  have hResSelf : closureOp a x <= leftRes a (a *q x) := by
    calc
      closureOp a x = leftRes a (a *q x) := by
        rfl
      _ <= leftRes a (a *q x) :=
        QuantaleLike.le_refl (leftRes a (a *q x))
  exact (Iff.mpr (residuation_left a (closureOp a x) (a *q x))) hResSelf

theorem closure_le_of_mul_le {Q : Type u} [QuantaleLike Q] (a x y : Q)
    (hMul : a *q y <= a *q x) :
    y <= closureOp a x := by
  have hRes : y <= leftRes a (a *q x) :=
    (Iff.mp (residuation_left a y (a *q x))) hMul
  calc
    y <= leftRes a (a *q x) := hRes
    _ = closureOp a x := by
          rfl

theorem interior_le_of_rightRes_le {Q : Type u} [QuantaleLike Q] (a x y : Q)
    (hRes : rightRes a x <= rightRes a y) :
    interiorOp a x <= y := by
  have hMul : (rightRes a x) *q a <= y :=
    (Iff.mpr (residuation_right (rightRes a x) a y)) hRes
  calc
    interiorOp a x = (rightRes a x) *q a := by
      rfl
    _ <= y := hMul

theorem fixedpoint_transfer_left_twice {Q : Type u} [QuantaleLike Q]
    (a : Q) (f : Q -> Q) (hf : monotone f)
    (hcl_mono : forall {u v : Q}, u <= v -> closureOp a u <= closureOp a v)
    {x : Q} (hfix : f x <= x) :
    closureOp a (f (f x)) <= closureOp a x := by
  have hStep1 : f (f x) <= f x :=
    hf hfix
  have hLift1 : closureOp a (f (f x)) <= closureOp a (f x) :=
    hcl_mono hStep1
  have hLift2 : closureOp a (f x) <= closureOp a x :=
    hcl_mono hfix
  have hSelf : closureOp a x <= closureOp a x :=
    QuantaleLike.le_refl (closureOp a x)
  have hTail : closureOp a (f x) <= closureOp a x :=
    QuantaleLike.le_trans hLift2 hSelf
  exact QuantaleLike.le_trans hLift1 hTail

theorem fixedpoint_transfer_right_triple {Q : Type u} [QuantaleLike Q]
    (a : Q) (f : Q -> Q) (hf : monotone f)
    (hint_mono : forall {u v : Q}, u <= v -> interiorOp a u <= interiorOp a v)
    {x : Q} (hfix : x <= f x) :
    interiorOp a x <= interiorOp a (f (f (f x))) := by
  have hFirst : interiorOp a x <= interiorOp a (f x) :=
    hint_mono hfix
  have hSecondArg : f x <= f (f x) :=
    hf hfix
  have hSecond : interiorOp a (f x) <= interiorOp a (f (f x)) :=
    hint_mono hSecondArg
  have hThirdArg : f (f x) <= f (f (f x)) :=
    hf hSecondArg
  have hThird : interiorOp a (f (f x)) <= interiorOp a (f (f (f x))) :=
    hint_mono hThirdArg
  have hFirstSecond : interiorOp a x <= interiorOp a (f (f x)) :=
    QuantaleLike.le_trans hFirst hSecond
  exact QuantaleLike.le_trans hFirstSecond hThird
