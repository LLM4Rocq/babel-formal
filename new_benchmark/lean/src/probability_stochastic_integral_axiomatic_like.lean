/-
BENCHMARK_ID: TINY_MATHLIB_BATCH05_PROBABILITY_STOCHASTIC_INTEGRAL_AXIOMATIC_LIKE
PAIR_STEM: probability_stochastic_integral_axiomatic_like
MATH_DOMAIN: Probability Theory
SOURCE_MATHLIB: Mathlib/Probability/StochasticIntegral
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u v

class ProbStruct_stochastic_integral (Ω : Type u) where
  Event : Type v
  filtration : Nat → Event → Prop
  integral : (Ω → Nat) → Nat
  condExp : Nat → (Ω → Nat) → Ω → Nat
  driftShift : Nat → (Ω → Nat) → Ω → Nat
  stopShift : Nat → (Ω → Nat) → Ω → Nat
  rate : Nat → Nat
  filtration_mono_axiom : ∀ {n m : Nat} {A : Event}, n ≤ m → filtration n A → filtration m A
  tower_axiom : ∀ {n m : Nat} (X : Ω → Nat), n ≤ m → condExp n (condExp m X) = condExp n X
  change_measure_axiom : ∀ (n : Nat) (X : Ω → Nat),
    integral (driftShift n X) ≤ integral X + rate n
  stopping_axiom : ∀ (n : Nat) (X : Ω → Nat),
    integral (stopShift n X) ≤ integral (driftShift n X)
  coupling_axiom : ∀ (n : Nat) (X Y : Ω → Nat),
    integral (driftShift n X) ≤ rate n →
    integral (driftShift n Y) ≤ rate n →
    integral (fun ω => driftShift n X ω + driftShift n Y ω) ≤ rate n + rate n
  fubini_axiom : ∀ (n : Nat) (X : Ω → Nat),
    integral (condExp n (driftShift n X)) = integral (driftShift n X)
  add_right_mono_axiom : ∀ a b c : Nat, a ≤ b → a + c ≤ b + c
  nat_le_trans_axiom : ∀ a b c : Nat, a ≤ b → b ≤ c → a ≤ c

def Filtration_stochastic_integral {Ω : Type u}
    (P : ProbStruct_stochastic_integral Ω) (n : Nat) : P.Event → Prop :=
  P.filtration n

def MartingaleStep_stochastic_integral {Ω : Type u}
    (P : ProbStruct_stochastic_integral Ω) (n : Nat) (X : Ω → Nat) : Ω → Nat :=
  P.condExp n X

def DriftShift_stochastic_integral {Ω : Type u}
    (P : ProbStruct_stochastic_integral Ω) (n : Nat) (X : Ω → Nat) : Ω → Nat :=
  P.driftShift n X

def RateFunc_stochastic_integral {Ω : Type u}
    (P : ProbStruct_stochastic_integral Ω) (n : Nat) : Nat :=
  P.rate n

theorem adaptivity_rule_stochastic_integral {Ω : Type u}
    (P : ProbStruct_stochastic_integral Ω)
    {n m : Nat} (hnm : n ≤ m) {A : P.Event}
    (hA : Filtration_stochastic_integral P n A) :
    Filtration_stochastic_integral P m A := by
  have hMono : P.filtration m A := P.filtration_mono_axiom hnm hA
  have hTarget : Filtration_stochastic_integral P m A = P.filtration m A := by
    rfl
  rw [hTarget]
  exact hMono

theorem tower_property_stochastic_integral {Ω : Type u}
    (P : ProbStruct_stochastic_integral Ω)
    {n m : Nat} (X : Ω → Nat) (hnm : n ≤ m) :
    MartingaleStep_stochastic_integral P n (MartingaleStep_stochastic_integral P m X)
      = MartingaleStep_stochastic_integral P n X := by
  have hTower : P.condExp n (P.condExp m X) = P.condExp n X := P.tower_axiom X hnm
  have hLeft :
      MartingaleStep_stochastic_integral P n (MartingaleStep_stochastic_integral P m X)
        = P.condExp n (P.condExp m X) := by
    rfl
  have hRight : MartingaleStep_stochastic_integral P n X = P.condExp n X := by
    rfl
  calc
    MartingaleStep_stochastic_integral P n (MartingaleStep_stochastic_integral P m X)
        = P.condExp n (P.condExp m X) := hLeft
    _ = P.condExp n X := hTower
    _ = MartingaleStep_stochastic_integral P n X := by
      symm
      exact hRight

theorem change_measure_step_stochastic_integral {Ω : Type u}
    (P : ProbStruct_stochastic_integral Ω)
    (n : Nat) (X : Ω → Nat) :
    P.integral (DriftShift_stochastic_integral P n X)
      ≤ P.integral X + RateFunc_stochastic_integral P n := by
  have hChange : P.integral (P.driftShift n X) ≤ P.integral X + P.rate n :=
    P.change_measure_axiom n X
  have hDrift : DriftShift_stochastic_integral P n X = P.driftShift n X := by
    rfl
  have hRate : RateFunc_stochastic_integral P n = P.rate n := by
    rfl
  rw [hDrift, hRate]
  exact hChange

theorem stopping_control_stochastic_integral {Ω : Type u}
    (P : ProbStruct_stochastic_integral Ω)
    (n : Nat) (X : Ω → Nat) :
    P.integral (P.stopShift n X) ≤ P.integral X + RateFunc_stochastic_integral P n := by
  have hStop : P.integral (P.stopShift n X) ≤ P.integral (P.driftShift n X) :=
    P.stopping_axiom n X
  have hChange :
      P.integral (P.driftShift n X) ≤ P.integral X + RateFunc_stochastic_integral P n := by
    simpa [RateFunc_stochastic_integral] using change_measure_step_stochastic_integral P n X
  have hTrans : P.integral (P.stopShift n X) ≤ P.integral X + RateFunc_stochastic_integral P n :=
    P.nat_le_trans_axiom (P.integral (P.stopShift n X)) (P.integral (P.driftShift n X))
      (P.integral X + RateFunc_stochastic_integral P n) hStop hChange
  exact hTrans

theorem ld_upper_bound_stochastic_integral {Ω : Type u}
    (P : ProbStruct_stochastic_integral Ω)
    (n : Nat) (X : Ω → Nat)
    (hX : P.integral X ≤ RateFunc_stochastic_integral P n) :
    P.integral (DriftShift_stochastic_integral P n X)
      ≤ RateFunc_stochastic_integral P n + RateFunc_stochastic_integral P n := by
  have hChange :
      P.integral (DriftShift_stochastic_integral P n X)
        ≤ P.integral X + RateFunc_stochastic_integral P n :=
    change_measure_step_stochastic_integral P n X
  have hAdd : P.integral X + RateFunc_stochastic_integral P n
      ≤ RateFunc_stochastic_integral P n + RateFunc_stochastic_integral P n := by
    exact P.add_right_mono_axiom (P.integral X) (RateFunc_stochastic_integral P n)
      (RateFunc_stochastic_integral P n) hX
  have hFinal :
      P.integral (DriftShift_stochastic_integral P n X)
        ≤ RateFunc_stochastic_integral P n + RateFunc_stochastic_integral P n :=
    P.nat_le_trans_axiom (P.integral (DriftShift_stochastic_integral P n X))
      (P.integral X + RateFunc_stochastic_integral P n)
      (RateFunc_stochastic_integral P n + RateFunc_stochastic_integral P n) hChange hAdd
  exact hFinal

theorem coupling_estimate_stochastic_integral {Ω : Type u}
    (P : ProbStruct_stochastic_integral Ω)
    (n : Nat) (X Y : Ω → Nat)
    (hX : P.integral (DriftShift_stochastic_integral P n X) ≤ RateFunc_stochastic_integral P n)
    (hY : P.integral (DriftShift_stochastic_integral P n Y) ≤ RateFunc_stochastic_integral P n) :
    P.integral (fun ω => DriftShift_stochastic_integral P n X ω + DriftShift_stochastic_integral P n Y ω)
      ≤ RateFunc_stochastic_integral P n + RateFunc_stochastic_integral P n := by
  have hCouple :
      P.integral (fun ω => P.driftShift n X ω + P.driftShift n Y ω)
        ≤ P.rate n + P.rate n := by
    exact P.coupling_axiom n X Y (by simpa [DriftShift_stochastic_integral, RateFunc_stochastic_integral] using hX)
      (by simpa [DriftShift_stochastic_integral, RateFunc_stochastic_integral] using hY)
  have hCoupleRate :
      P.integral (fun ω => P.driftShift n X ω + P.driftShift n Y ω)
        ≤ RateFunc_stochastic_integral P n + RateFunc_stochastic_integral P n := by
    simpa [RateFunc_stochastic_integral] using hCouple
  have hDrift :
      (fun ω => DriftShift_stochastic_integral P n X ω + DriftShift_stochastic_integral P n Y ω)
      = (fun ω => P.driftShift n X ω + P.driftShift n Y ω) := by
    funext ω
    rfl
  rw [hDrift]
  exact hCoupleRate

theorem stochastic_fubini_rule_stochastic_integral {Ω : Type u}
    (P : ProbStruct_stochastic_integral Ω)
    (n : Nat) (X : Ω → Nat) :
    P.integral (MartingaleStep_stochastic_integral P n (DriftShift_stochastic_integral P n X))
      = P.integral (DriftShift_stochastic_integral P n X) := by
  have hFubini : P.integral (P.condExp n (P.driftShift n X)) = P.integral (P.driftShift n X) :=
    P.fubini_axiom n X
  have hLeft :
      P.integral (MartingaleStep_stochastic_integral P n (DriftShift_stochastic_integral P n X))
        = P.integral (P.condExp n (P.driftShift n X)) := by
    rfl
  have hRight : P.integral (DriftShift_stochastic_integral P n X) = P.integral (P.driftShift n X) := by
    rfl
  calc
    P.integral (MartingaleStep_stochastic_integral P n (DriftShift_stochastic_integral P n X))
        = P.integral (P.condExp n (P.driftShift n X)) := hLeft
    _ = P.integral (P.driftShift n X) := hFubini
    _ = P.integral (DriftShift_stochastic_integral P n X) := by
      symm
      exact hRight
