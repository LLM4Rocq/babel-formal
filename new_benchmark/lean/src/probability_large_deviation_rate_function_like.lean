/-
BENCHMARK_ID: TINY_MATHLIB_BATCH05_PROBABILITY_LARGE_DEVIATION_RATE_FUNCTION_LIKE
PAIR_STEM: probability_large_deviation_rate_function_like
MATH_DOMAIN: Probability Theory
SOURCE_MATHLIB: Mathlib/Probability/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class ProbStruct_large_deviation_rate (V : Type u) where
  le : V -> V -> Prop
  le_refl : forall x : V, le x x
  le_trans : forall {x y z : V}, le x y -> le y z -> le x z
  add : V -> V -> V
  cond : V -> V
  drift : V -> V
  rate : V -> V
  cond_mono : forall {x y : V}, le x y -> le (cond x) (cond y)
  drift_mono : forall {x y : V}, le x y -> le (drift x) (drift y)
  rate_mono : forall {x y : V}, le x y -> le (rate x) (rate y)
  tower_axiom : forall x : V, le (cond (cond x)) (cond x)
  change_measure_axiom : forall x : V, le (drift x) (add x (rate x))
  stopping_axiom : forall x : V, le (cond x) x
  coupling_axiom : forall x y : V, le (add (rate x) (rate y)) (rate (add x y))
  fubini_axiom : forall x y : V, le (cond (add x y)) (add (cond x) (cond y))

infix:50 " ⪯ " => ProbStruct_large_deviation_rate.le
infixl:65 " ⊞ " => ProbStruct_large_deviation_rate.add

def Filtration_large_deviation_rate
    {V : Type u} [ProbStruct_large_deviation_rate V] (x : V) : V :=
  ProbStruct_large_deviation_rate.cond x

def MartingaleStep_large_deviation_rate
    {V : Type u} [ProbStruct_large_deviation_rate V] (x : V) : V :=
  Filtration_large_deviation_rate x

def DriftShift_large_deviation_rate
    {V : Type u} [ProbStruct_large_deviation_rate V] (x : V) : V :=
  ProbStruct_large_deviation_rate.drift x

def RateFunc_large_deviation_rate
    {V : Type u} [ProbStruct_large_deviation_rate V] (x : V) : V :=
  ProbStruct_large_deviation_rate.rate x

theorem adaptivity_rule_large_deviation_rate
    {V : Type u} [ProbStruct_large_deviation_rate V]
    (x : V) :
    MartingaleStep_large_deviation_rate x ⪯ x := by
  have hstop : ProbStruct_large_deviation_rate.cond x ⪯ x :=
    ProbStruct_large_deviation_rate.stopping_axiom x
  calc
    MartingaleStep_large_deviation_rate x = ProbStruct_large_deviation_rate.cond x := by
      rfl
    _ ⪯ x := hstop

theorem tower_property_large_deviation_rate
    {V : Type u} [ProbStruct_large_deviation_rate V]
    (x : V) :
    MartingaleStep_large_deviation_rate
      (MartingaleStep_large_deviation_rate x) ⪯
      MartingaleStep_large_deviation_rate x /\
    MartingaleStep_large_deviation_rate x ⪯ x := by
  have htower : ProbStruct_large_deviation_rate.le
      (ProbStruct_large_deviation_rate.cond
        (ProbStruct_large_deviation_rate.cond x))
      (ProbStruct_large_deviation_rate.cond x) :=
    ProbStruct_large_deviation_rate.tower_axiom x
  have hadapt : MartingaleStep_large_deviation_rate x ⪯ x :=
    adaptivity_rule_large_deviation_rate x
  constructor
  · calc
      MartingaleStep_large_deviation_rate
        (MartingaleStep_large_deviation_rate x)
          = ProbStruct_large_deviation_rate.cond
              (ProbStruct_large_deviation_rate.cond x) := by
                rfl
      _ ⪯ ProbStruct_large_deviation_rate.cond x := htower
      _ = MartingaleStep_large_deviation_rate x := by
            rfl
  · exact hadapt

theorem change_measure_step_large_deviation_rate
    {V : Type u} [ProbStruct_large_deviation_rate V]
    (x : V) :
    DriftShift_large_deviation_rate x ⪯
      x ⊞ (@RateFunc_large_deviation_rate V _ x) := by
  have hraw : ProbStruct_large_deviation_rate.le
      (ProbStruct_large_deviation_rate.drift x)
      (ProbStruct_large_deviation_rate.add x (ProbStruct_large_deviation_rate.rate x)) :=
    ProbStruct_large_deviation_rate.change_measure_axiom x
  calc
    DriftShift_large_deviation_rate x = ProbStruct_large_deviation_rate.drift x := by
      rfl
    _ ⪯ ProbStruct_large_deviation_rate.add x (ProbStruct_large_deviation_rate.rate x) := hraw
    _ = x ⊞ (@RateFunc_large_deviation_rate V _ x) := by
          rfl

theorem stopping_control_large_deviation_rate
    {V : Type u} [ProbStruct_large_deviation_rate V]
    (x : V) :
    ∃ z : V,
      z = DriftShift_large_deviation_rate x /\
      MartingaleStep_large_deviation_rate
        (DriftShift_large_deviation_rate x) ⪯ z /\
      z ⪯ DriftShift_large_deviation_rate x := by
  have hstop : ProbStruct_large_deviation_rate.le
      (ProbStruct_large_deviation_rate.cond (ProbStruct_large_deviation_rate.drift x))
      (ProbStruct_large_deviation_rate.drift x) :=
    ProbStruct_large_deviation_rate.stopping_axiom (ProbStruct_large_deviation_rate.drift x)
  have hreflex : DriftShift_large_deviation_rate x ⪯
      DriftShift_large_deviation_rate x :=
    ProbStruct_large_deviation_rate.le_refl (DriftShift_large_deviation_rate x)
  refine ⟨DriftShift_large_deviation_rate x, rfl, ?_, ?_⟩
  · calc
      MartingaleStep_large_deviation_rate
        (DriftShift_large_deviation_rate x)
          = ProbStruct_large_deviation_rate.cond (ProbStruct_large_deviation_rate.drift x) := by
              rfl
      _ ⪯ ProbStruct_large_deviation_rate.drift x := hstop
      _ = DriftShift_large_deviation_rate x := by
            rfl
  · exact hreflex

theorem ld_upper_bound_large_deviation_rate
    {V : Type u} [ProbStruct_large_deviation_rate V]
    (x : V) :
    MartingaleStep_large_deviation_rate
      (DriftShift_large_deviation_rate x) ⪯
      x ⊞ (@RateFunc_large_deviation_rate V _ x) := by
  have h1 : MartingaleStep_large_deviation_rate
      (DriftShift_large_deviation_rate x) ⪯
      DriftShift_large_deviation_rate x := by
    rcases stopping_control_large_deviation_rate x with ⟨z, hz, hstep, hback⟩
    subst hz
    exact hstep
  have h2 : DriftShift_large_deviation_rate x ⪯
      x ⊞ (@RateFunc_large_deviation_rate V _ x) :=
    change_measure_step_large_deviation_rate x
  exact ProbStruct_large_deviation_rate.le_trans h1 h2

theorem coupling_estimate_large_deviation_rate
    {V : Type u} [ProbStruct_large_deviation_rate V]
    (x y : V) :
    (@RateFunc_large_deviation_rate V _ x) ⊞
      (@RateFunc_large_deviation_rate V _ y) ⪯
      @RateFunc_large_deviation_rate V _ (x ⊞ y) := by
  have hcouple : ProbStruct_large_deviation_rate.le
      (ProbStruct_large_deviation_rate.add
        (ProbStruct_large_deviation_rate.rate x)
        (ProbStruct_large_deviation_rate.rate y))
      (ProbStruct_large_deviation_rate.rate
        (ProbStruct_large_deviation_rate.add x y)) :=
    ProbStruct_large_deviation_rate.coupling_axiom x y
  calc
    (@RateFunc_large_deviation_rate V _ x) ⊞ (@RateFunc_large_deviation_rate V _ y)
        = ProbStruct_large_deviation_rate.add
            (ProbStruct_large_deviation_rate.rate x)
            (ProbStruct_large_deviation_rate.rate y) := by
              rfl
    _ ⪯ ProbStruct_large_deviation_rate.rate (ProbStruct_large_deviation_rate.add x y) := hcouple
    _ = @RateFunc_large_deviation_rate V _ (x ⊞ y) := by
          rfl

theorem stochastic_fubini_rule_large_deviation_rate
    {V : Type u} [ProbStruct_large_deviation_rate V]
    (x y : V) :
    MartingaleStep_large_deviation_rate (x ⊞ y) ⪯
      (MartingaleStep_large_deviation_rate x) ⊞
        (MartingaleStep_large_deviation_rate y) := by
  have hfub : ProbStruct_large_deviation_rate.le
      (ProbStruct_large_deviation_rate.cond
        (ProbStruct_large_deviation_rate.add x y))
      (ProbStruct_large_deviation_rate.add
        (ProbStruct_large_deviation_rate.cond x)
        (ProbStruct_large_deviation_rate.cond y)) :=
    ProbStruct_large_deviation_rate.fubini_axiom x y
  calc
    MartingaleStep_large_deviation_rate (x ⊞ y)
        = ProbStruct_large_deviation_rate.cond
            (ProbStruct_large_deviation_rate.add x y) := by
              rfl
    _ ⪯ ProbStruct_large_deviation_rate.add
          (ProbStruct_large_deviation_rate.cond x)
          (ProbStruct_large_deviation_rate.cond y) := hfub
    _ = (MartingaleStep_large_deviation_rate x) ⊞
          (MartingaleStep_large_deviation_rate y) := by
          rfl
