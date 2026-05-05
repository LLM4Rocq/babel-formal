/-
BENCHMARK_ID: TINY_MATHLIB_BATCH05_PROBABILITY_GIRSANOV_TRANSFORM_LIKE
PAIR_STEM: probability_girsanov_transform_like
MATH_DOMAIN: Probability Theory
SOURCE_MATHLIB: Mathlib/Probability/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class ProbStruct_girsanov_transform (Omega : Type u) where
  Expect : (Omega -> Nat) -> Nat
  CondExpect : Nat -> (Omega -> Nat) -> (Omega -> Nat)
  cond_tower_axiom :
    forall n m : Nat, n <= m -> forall X : Omega -> Nat,
      Expect (CondExpect n (CondExpect m X)) = Expect (CondExpect n X)
  cond_idem_axiom :
    forall n : Nat, forall X : Omega -> Nat,
      CondExpect n (CondExpect n X) = CondExpect n X
  expect_add_axiom :
    forall X Y : Omega -> Nat,
      Expect (fun omega => X omega + Y omega) = Expect X + Expect Y
  expect_mono_axiom :
    forall X Y : Omega -> Nat,
      (forall omega : Omega, X omega <= Y omega) -> Expect X <= Expect Y

def Filtration_girsanov_transform (Omega : Type u) : Type u :=
  Nat -> (Omega -> Prop) -> Prop

def MartingaleStep_girsanov_transform {Omega : Type u}
    [ProbStruct_girsanov_transform Omega]
    (X : Nat -> Omega -> Nat) : Prop :=
  forall n : Nat,
    ProbStruct_girsanov_transform.Expect (X (Nat.succ n)) =
      ProbStruct_girsanov_transform.Expect (X n)

def DriftShift_girsanov_transform {Omega : Type u}
    (X theta : Nat -> Omega -> Nat) : Nat -> Omega -> Nat :=
  fun n omega => X n omega + theta n omega

def RateFunc_girsanov_transform {Omega : Type u}
    [ProbStruct_girsanov_transform Omega]
    (theta : Nat -> Omega -> Nat) : Nat -> Nat :=
  fun n => ProbStruct_girsanov_transform.Expect (theta n)

theorem adaptivity_rule_girsanov_transform {Omega : Type u}
    (F : Filtration_girsanov_transform Omega)
    (X theta : Nat -> Omega -> Nat)
    (hX : forall n : Nat, F n (fun omega => X n omega = X n omega))
    (hTheta : forall n : Nat, F n (fun omega => theta n omega = theta n omega))
    (hInter : forall n : Nat, forall s t : Omega -> Prop,
      F n s -> F n t -> F n (fun omega => s omega /\ t omega))
    (hSuperset : forall n : Nat, forall s t : Omega -> Prop,
      F n s -> (forall omega : Omega, s omega -> t omega) -> F n t) :
    forall n : Nat,
      F n (fun omega =>
        DriftShift_girsanov_transform X theta n omega =
          DriftShift_girsanov_transform X theta n omega) := by
  intro n
  have hLocalX : F n (fun omega => X n omega = X n omega) := hX n
  have hLocalTheta : F n (fun omega => theta n omega = theta n omega) := hTheta n
  have hBoth :
      F n (fun omega =>
        (X n omega = X n omega) /\ (theta n omega = theta n omega)) :=
    hInter n (fun omega => X n omega = X n omega)
      (fun omega => theta n omega = theta n omega) hLocalX hLocalTheta
  have hCast :
      forall omega : Omega,
        ((X n omega = X n omega) /\ (theta n omega = theta n omega)) ->
          DriftShift_girsanov_transform X theta n omega =
            DriftShift_girsanov_transform X theta n omega := by
    intro omega hPair
    rcases hPair with ⟨hEqX, hEqTheta⟩
    have hStep : X n omega + theta n omega = X n omega + theta n omega := by
      rw [hEqX, hEqTheta]
    simpa [DriftShift_girsanov_transform] using hStep
  exact hSuperset n
    (fun omega => (X n omega = X n omega) /\ (theta n omega = theta n omega))
    (fun omega =>
      DriftShift_girsanov_transform X theta n omega =
        DriftShift_girsanov_transform X theta n omega)
    hBoth hCast

theorem tower_property_girsanov_transform {Omega : Type u}
    [ProbStruct_girsanov_transform Omega]
    (n m : Nat) (hnm : n <= m) (X : Omega -> Nat) :
    ProbStruct_girsanov_transform.Expect
      (ProbStruct_girsanov_transform.CondExpect n
        (ProbStruct_girsanov_transform.CondExpect m X)) =
      ProbStruct_girsanov_transform.Expect
        (ProbStruct_girsanov_transform.CondExpect n X) := by
  have hTower :
      ProbStruct_girsanov_transform.Expect
          (ProbStruct_girsanov_transform.CondExpect n
            (ProbStruct_girsanov_transform.CondExpect m X)) =
        ProbStruct_girsanov_transform.Expect
          (ProbStruct_girsanov_transform.CondExpect n X) :=
    ProbStruct_girsanov_transform.cond_tower_axiom n m hnm X
  have hLeft :
      ProbStruct_girsanov_transform.Expect
        (ProbStruct_girsanov_transform.CondExpect n
          (ProbStruct_girsanov_transform.CondExpect m X)) =
      ProbStruct_girsanov_transform.Expect
        (ProbStruct_girsanov_transform.CondExpect n
          (ProbStruct_girsanov_transform.CondExpect m X)) := by
    rfl
  calc
    ProbStruct_girsanov_transform.Expect
        (ProbStruct_girsanov_transform.CondExpect n
          (ProbStruct_girsanov_transform.CondExpect m X))
        = ProbStruct_girsanov_transform.Expect
            (ProbStruct_girsanov_transform.CondExpect n
              (ProbStruct_girsanov_transform.CondExpect m X)) := hLeft
    _ = ProbStruct_girsanov_transform.Expect
          (ProbStruct_girsanov_transform.CondExpect n X) := hTower

theorem change_measure_step_girsanov_transform {Omega : Type u}
    [ProbStruct_girsanov_transform Omega]
    (X theta : Nat -> Omega -> Nat)
    (hMart : MartingaleStep_girsanov_transform X)
    (hZero : forall n : Nat,
      ProbStruct_girsanov_transform.Expect (theta n) = 0)
    (n : Nat) :
    ProbStruct_girsanov_transform.Expect
      (DriftShift_girsanov_transform X theta n) =
      ProbStruct_girsanov_transform.Expect (X (Nat.succ n)) := by
  have hAdd :
      ProbStruct_girsanov_transform.Expect
          (DriftShift_girsanov_transform X theta n) =
        ProbStruct_girsanov_transform.Expect (X n) +
          ProbStruct_girsanov_transform.Expect (theta n) := by
    simpa [DriftShift_girsanov_transform] using
      (ProbStruct_girsanov_transform.expect_add_axiom (X n) (theta n))
  have hZeroN : ProbStruct_girsanov_transform.Expect (theta n) = 0 := hZero n
  have hStep :
      ProbStruct_girsanov_transform.Expect (X (Nat.succ n)) =
        ProbStruct_girsanov_transform.Expect (X n) :=
    hMart n
  calc
    ProbStruct_girsanov_transform.Expect
        (DriftShift_girsanov_transform X theta n)
        = ProbStruct_girsanov_transform.Expect (X n) +
            ProbStruct_girsanov_transform.Expect (theta n) := hAdd
    _ = ProbStruct_girsanov_transform.Expect (X n) + 0 := by
          rw [hZeroN]
    _ = ProbStruct_girsanov_transform.Expect (X n) := by
          rw [Nat.add_zero]
    _ = ProbStruct_girsanov_transform.Expect (X (Nat.succ n)) := by
          symm
          exact hStep

theorem stopping_control_girsanov_transform {Omega : Type u}
    [ProbStruct_girsanov_transform Omega]
    (X : Nat -> Omega -> Nat)
    (tau n : Nat) (htau : tau <= n)
    (hMonotone :
      forall a b : Nat, a <= b ->
        ProbStruct_girsanov_transform.Expect (X a) <=
          ProbStruct_girsanov_transform.Expect (X b)) :
    ProbStruct_girsanov_transform.Expect (X (Nat.min tau n)) <=
      ProbStruct_girsanov_transform.Expect (X n) := by
  have hMinEq : Nat.min tau n = tau := Nat.min_eq_left htau
  have hTauBound :
      ProbStruct_girsanov_transform.Expect (X tau) <=
        ProbStruct_girsanov_transform.Expect (X n) :=
    hMonotone tau n htau
  calc
    ProbStruct_girsanov_transform.Expect (X (Nat.min tau n))
        = ProbStruct_girsanov_transform.Expect (X tau) := by
          rw [hMinEq]
    _ <= ProbStruct_girsanov_transform.Expect (X n) := hTauBound

theorem ld_upper_bound_girsanov_transform {Omega : Type u}
    [ProbStruct_girsanov_transform Omega]
    (X theta : Nat -> Omega -> Nat)
    (hProc : forall n : Nat,
      ProbStruct_girsanov_transform.Expect (X n) <=
        ProbStruct_girsanov_transform.Expect (X (Nat.succ n)))
    (hRate : forall n : Nat,
      ProbStruct_girsanov_transform.Expect (theta n) <=
        RateFunc_girsanov_transform theta (Nat.succ n))
    (hShift : forall n : Nat,
      ProbStruct_girsanov_transform.Expect (DriftShift_girsanov_transform X theta n) =
        ProbStruct_girsanov_transform.Expect (X n) +
          ProbStruct_girsanov_transform.Expect (theta n))
    (n : Nat) :
    ProbStruct_girsanov_transform.Expect (DriftShift_girsanov_transform X theta n) <=
      ProbStruct_girsanov_transform.Expect (X (Nat.succ n)) +
        RateFunc_girsanov_transform theta (Nat.succ n) := by
  have hLeft :
      ProbStruct_girsanov_transform.Expect (X n) <=
        ProbStruct_girsanov_transform.Expect (X (Nat.succ n)) :=
    hProc n
  have hRight :
      ProbStruct_girsanov_transform.Expect (theta n) <=
        RateFunc_girsanov_transform theta (Nat.succ n) :=
    hRate n
  have hAdd :
      ProbStruct_girsanov_transform.Expect (X n) +
        ProbStruct_girsanov_transform.Expect (theta n) <=
      ProbStruct_girsanov_transform.Expect (X (Nat.succ n)) +
        RateFunc_girsanov_transform theta (Nat.succ n) :=
    Nat.add_le_add hLeft hRight
  calc
    ProbStruct_girsanov_transform.Expect (DriftShift_girsanov_transform X theta n)
        = ProbStruct_girsanov_transform.Expect (X n) +
            ProbStruct_girsanov_transform.Expect (theta n) := hShift n
    _ <= ProbStruct_girsanov_transform.Expect (X (Nat.succ n)) +
          RateFunc_girsanov_transform theta (Nat.succ n) := hAdd

theorem coupling_estimate_girsanov_transform {Omega : Type u}
    [ProbStruct_girsanov_transform Omega]
    (X Y theta phi : Nat -> Omega -> Nat)
    (hCouple : forall n : Nat,
      ProbStruct_girsanov_transform.Expect (X n) <=
        ProbStruct_girsanov_transform.Expect (Y n))
    (hRate : forall n : Nat,
      RateFunc_girsanov_transform theta n <= RateFunc_girsanov_transform phi n)
    (hShiftX : forall n : Nat,
      ProbStruct_girsanov_transform.Expect (DriftShift_girsanov_transform X theta n) =
        ProbStruct_girsanov_transform.Expect (X n) + RateFunc_girsanov_transform theta n)
    (hShiftY : forall n : Nat,
      ProbStruct_girsanov_transform.Expect (DriftShift_girsanov_transform Y phi n) =
        ProbStruct_girsanov_transform.Expect (Y n) + RateFunc_girsanov_transform phi n)
    (n : Nat) :
    ProbStruct_girsanov_transform.Expect (DriftShift_girsanov_transform X theta n) <=
      ProbStruct_girsanov_transform.Expect (DriftShift_girsanov_transform Y phi n) := by
  have hBase :
      ProbStruct_girsanov_transform.Expect (X n) + RateFunc_girsanov_transform theta n <=
        ProbStruct_girsanov_transform.Expect (Y n) + RateFunc_girsanov_transform phi n :=
    Nat.add_le_add (hCouple n) (hRate n)
  calc
    ProbStruct_girsanov_transform.Expect (DriftShift_girsanov_transform X theta n)
        = ProbStruct_girsanov_transform.Expect (X n) + RateFunc_girsanov_transform theta n :=
          hShiftX n
    _ <= ProbStruct_girsanov_transform.Expect (Y n) + RateFunc_girsanov_transform phi n := hBase
    _ = ProbStruct_girsanov_transform.Expect (DriftShift_girsanov_transform Y phi n) := by
          symm
          exact hShiftY n

theorem stochastic_fubini_rule_girsanov_transform {Omega : Type u}
    [ProbStruct_girsanov_transform Omega]
    (A B C : Omega -> Nat) :
    ProbStruct_girsanov_transform.Expect (fun omega => A omega + (B omega + C omega)) =
      ProbStruct_girsanov_transform.Expect (fun omega => (A omega + B omega) + C omega) := by
  have hAssocPoint :
      (fun omega => A omega + (B omega + C omega)) =
      (fun omega => (A omega + B omega) + C omega) := by
    funext omega
    exact Eq.symm (Nat.add_assoc (A omega) (B omega) (C omega))
  have hLeftSplit :
      ProbStruct_girsanov_transform.Expect (fun omega => A omega + (B omega + C omega)) =
      ProbStruct_girsanov_transform.Expect A +
        ProbStruct_girsanov_transform.Expect (fun omega => B omega + C omega) :=
    ProbStruct_girsanov_transform.expect_add_axiom A (fun omega => B omega + C omega)
  have hRightSplit :
      ProbStruct_girsanov_transform.Expect (fun omega => (A omega + B omega) + C omega) =
      ProbStruct_girsanov_transform.Expect (fun omega => A omega + B omega) +
        ProbStruct_girsanov_transform.Expect C :=
    ProbStruct_girsanov_transform.expect_add_axiom (fun omega => A omega + B omega) C
  have hMidSplit :
      ProbStruct_girsanov_transform.Expect (fun omega => A omega + B omega) =
      ProbStruct_girsanov_transform.Expect A + ProbStruct_girsanov_transform.Expect B :=
    ProbStruct_girsanov_transform.expect_add_axiom A B
  have hTailSplit :
      ProbStruct_girsanov_transform.Expect (fun omega => B omega + C omega) =
      ProbStruct_girsanov_transform.Expect B + ProbStruct_girsanov_transform.Expect C :=
    ProbStruct_girsanov_transform.expect_add_axiom B C
  calc
    ProbStruct_girsanov_transform.Expect (fun omega => A omega + (B omega + C omega))
        = ProbStruct_girsanov_transform.Expect A +
            ProbStruct_girsanov_transform.Expect (fun omega => B omega + C omega) := hLeftSplit
    _ = ProbStruct_girsanov_transform.Expect A +
          (ProbStruct_girsanov_transform.Expect B + ProbStruct_girsanov_transform.Expect C) := by
          rw [hTailSplit]
    _ = (ProbStruct_girsanov_transform.Expect A + ProbStruct_girsanov_transform.Expect B) +
          ProbStruct_girsanov_transform.Expect C := by
          rw [Nat.add_assoc]
    _ = ProbStruct_girsanov_transform.Expect (fun omega => A omega + B omega) +
          ProbStruct_girsanov_transform.Expect C := by
          rw [<- hMidSplit]
    _ = ProbStruct_girsanov_transform.Expect (fun omega => (A omega + B omega) + C omega) := by
          symm
          exact hRightSplit
