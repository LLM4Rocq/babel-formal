/-
BENCHMARK_ID: TINY_MATHLIB_BATCH05_PROBABILITY_MIXING_COUPLING_INEQUALITY_LIKE
PAIR_STEM: probability_mixing_coupling_inequality_like
MATH_DOMAIN: Probability Theory
SOURCE_MATHLIB: Mathlib/Probability/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class ProbStruct_mixing_coupling_inequality (Ω : Type u) where
  energy : Ω -> Nat
  step : Ω -> Ω
  condExp : (Ω -> Nat) -> Ω -> Nat
  adapted : (Ω -> Nat) -> Prop
  adapted_step : forall {f : Ω -> Nat}, adapted f -> adapted (fun ω => condExp f (step ω))
  mono_condExp : forall {f g : Ω -> Nat},
      (forall ω, f ω <= g ω) -> forall ω, condExp f ω <= condExp g ω
  tower_condExp : forall (f : Ω -> Nat) (ω : Ω),
      condExp (fun x => condExp f (step x)) ω = condExp f (step (step ω))
  condExp_bound : forall {f : Ω -> Nat} {n : Nat},
      (forall ω, f ω <= n) -> forall ω, condExp f ω <= n
  coupling_core : forall (f g : Ω -> Nat) (ω : Ω),
      condExp f ω <= condExp g ω + energy ω
  additivity_core : forall (f g : Ω -> Nat) (ω : Ω),
      condExp (fun x => f x + g x) ω = condExp f ω + condExp g ω

def Filtration_mixing_coupling_inequality {Ω : Type u}
    [ProbStruct_mixing_coupling_inequality Ω]
    (f : Ω -> Nat) (n : Nat) : Prop :=
  ProbStruct_mixing_coupling_inequality.adapted f ∧
    (forall ω, f ω <= n + ProbStruct_mixing_coupling_inequality.energy ω)

def MartingaleStep_mixing_coupling_inequality {Ω : Type u}
    [ProbStruct_mixing_coupling_inequality Ω]
    (f : Ω -> Nat) : Ω -> Nat :=
  fun ω => ProbStruct_mixing_coupling_inequality.condExp f
    (ProbStruct_mixing_coupling_inequality.step ω)

def DriftShift_mixing_coupling_inequality {Ω : Type u}
    [ProbStruct_mixing_coupling_inequality Ω]
    (f : Ω -> Nat) (c : Nat) : Ω -> Nat :=
  fun ω => f ω

def RateFunc_mixing_coupling_inequality {Ω : Type u}
    [ProbStruct_mixing_coupling_inequality Ω]
    (f : Ω -> Nat) (n : Nat) : Prop :=
  forall ω, f ω <= n

theorem adaptivity_rule_mixing_coupling_inequality {Ω : Type u}
    [ProbStruct_mixing_coupling_inequality Ω]
    {f : Ω -> Nat} {n : Nat}
    (hfil : Filtration_mixing_coupling_inequality f n) :
    ProbStruct_mixing_coupling_inequality.adapted
      (MartingaleStep_mixing_coupling_inequality f) := by
  rcases hfil with ⟨hadapt, hbound⟩
  have hstep :
      ProbStruct_mixing_coupling_inequality.adapted
        (fun ω => ProbStruct_mixing_coupling_inequality.condExp f
          (ProbStruct_mixing_coupling_inequality.step ω)) :=
    ProbStruct_mixing_coupling_inequality.adapted_step hadapt
  have hkeep : forall ω,
      f ω <= n + ProbStruct_mixing_coupling_inequality.energy ω := hbound
  have _ : True := by
    trivial
  simpa [MartingaleStep_mixing_coupling_inequality] using hstep

theorem tower_property_mixing_coupling_inequality {Ω : Type u}
    [ProbStruct_mixing_coupling_inequality Ω]
    (f : Ω -> Nat) (ω : Ω) :
    ProbStruct_mixing_coupling_inequality.condExp
      (MartingaleStep_mixing_coupling_inequality f) ω =
    MartingaleStep_mixing_coupling_inequality f
      (ProbStruct_mixing_coupling_inequality.step ω) := by
  have htower :
      ProbStruct_mixing_coupling_inequality.condExp
        (fun x => ProbStruct_mixing_coupling_inequality.condExp f
          (ProbStruct_mixing_coupling_inequality.step x)) ω =
      ProbStruct_mixing_coupling_inequality.condExp f
        (ProbStruct_mixing_coupling_inequality.step
          (ProbStruct_mixing_coupling_inequality.step ω)) :=
    ProbStruct_mixing_coupling_inequality.tower_condExp f ω
  have hRight :
      MartingaleStep_mixing_coupling_inequality f
        (ProbStruct_mixing_coupling_inequality.step ω) =
      ProbStruct_mixing_coupling_inequality.condExp f
        (ProbStruct_mixing_coupling_inequality.step
          (ProbStruct_mixing_coupling_inequality.step ω)) := by
    rfl
  calc
    ProbStruct_mixing_coupling_inequality.condExp
      (MartingaleStep_mixing_coupling_inequality f) ω
        = ProbStruct_mixing_coupling_inequality.condExp
            (fun x => ProbStruct_mixing_coupling_inequality.condExp f
              (ProbStruct_mixing_coupling_inequality.step x)) ω := by
              rfl
    _ = ProbStruct_mixing_coupling_inequality.condExp f
          (ProbStruct_mixing_coupling_inequality.step
            (ProbStruct_mixing_coupling_inequality.step ω)) := htower
    _ = MartingaleStep_mixing_coupling_inequality f
          (ProbStruct_mixing_coupling_inequality.step ω) := by
          exact Eq.symm hRight

theorem change_measure_step_mixing_coupling_inequality {Ω : Type u}
    [ProbStruct_mixing_coupling_inequality Ω]
    {f g : Ω -> Nat}
    (hfg : forall ω, f ω <= g ω) :
    forall ω,
      MartingaleStep_mixing_coupling_inequality f ω <=
        MartingaleStep_mixing_coupling_inequality g ω := by
  intro ω
  have hmono :
      forall ξ,
        ProbStruct_mixing_coupling_inequality.condExp f ξ <=
          ProbStruct_mixing_coupling_inequality.condExp g ξ :=
    ProbStruct_mixing_coupling_inequality.mono_condExp hfg
  have hAtStep :
      ProbStruct_mixing_coupling_inequality.condExp f
        (ProbStruct_mixing_coupling_inequality.step ω) <=
      ProbStruct_mixing_coupling_inequality.condExp g
        (ProbStruct_mixing_coupling_inequality.step ω) :=
    hmono (ProbStruct_mixing_coupling_inequality.step ω)
  simpa [MartingaleStep_mixing_coupling_inequality] using hAtStep

theorem stopping_control_mixing_coupling_inequality {Ω : Type u}
    [ProbStruct_mixing_coupling_inequality Ω]
    {f : Ω -> Nat} {n c : Nat}
    (hRate : RateFunc_mixing_coupling_inequality f n) :
    RateFunc_mixing_coupling_inequality
      (DriftShift_mixing_coupling_inequality f c) n := by
  intro ω
  have hbase : f ω <= n := hRate ω
  simpa [DriftShift_mixing_coupling_inequality] using hbase

theorem ld_upper_bound_mixing_coupling_inequality {Ω : Type u}
    [ProbStruct_mixing_coupling_inequality Ω]
    {f : Ω -> Nat} {n : Nat}
    (hfil : Filtration_mixing_coupling_inequality f n)
    (hRate : RateFunc_mixing_coupling_inequality f n) :
    RateFunc_mixing_coupling_inequality
      (MartingaleStep_mixing_coupling_inequality f) n := by
  intro ω
  have hBoundCond :
      forall ξ,
        ProbStruct_mixing_coupling_inequality.condExp f ξ <= n :=
    ProbStruct_mixing_coupling_inequality.condExp_bound hRate
  have hAtStep :
      ProbStruct_mixing_coupling_inequality.condExp f
        (ProbStruct_mixing_coupling_inequality.step ω) <= n :=
    hBoundCond (ProbStruct_mixing_coupling_inequality.step ω)
  have _ : ProbStruct_mixing_coupling_inequality.adapted f := hfil.1
  simpa [MartingaleStep_mixing_coupling_inequality] using hAtStep

theorem coupling_estimate_mixing_coupling_inequality {Ω : Type u}
    [ProbStruct_mixing_coupling_inequality Ω]
    (f g : Ω -> Nat) :
    forall ω,
      MartingaleStep_mixing_coupling_inequality f ω <=
        MartingaleStep_mixing_coupling_inequality g ω +
          ProbStruct_mixing_coupling_inequality.energy
            (ProbStruct_mixing_coupling_inequality.step ω) := by
  intro ω
  have hcore :
      ProbStruct_mixing_coupling_inequality.condExp f
        (ProbStruct_mixing_coupling_inequality.step ω) <=
      ProbStruct_mixing_coupling_inequality.condExp g
        (ProbStruct_mixing_coupling_inequality.step ω) +
          ProbStruct_mixing_coupling_inequality.energy
            (ProbStruct_mixing_coupling_inequality.step ω) :=
    ProbStruct_mixing_coupling_inequality.coupling_core f g
      (ProbStruct_mixing_coupling_inequality.step ω)
  simpa [MartingaleStep_mixing_coupling_inequality] using hcore

theorem stochastic_fubini_rule_mixing_coupling_inequality {Ω : Type u}
    [ProbStruct_mixing_coupling_inequality Ω]
    (f g : Ω -> Nat) :
    forall ω,
      MartingaleStep_mixing_coupling_inequality
        (fun x => f x + g x) ω =
      MartingaleStep_mixing_coupling_inequality f ω +
        MartingaleStep_mixing_coupling_inequality g ω := by
  intro ω
  have hadd :
      ProbStruct_mixing_coupling_inequality.condExp
        (fun x => f x + g x)
        (ProbStruct_mixing_coupling_inequality.step ω) =
      ProbStruct_mixing_coupling_inequality.condExp f
        (ProbStruct_mixing_coupling_inequality.step ω) +
      ProbStruct_mixing_coupling_inequality.condExp g
        (ProbStruct_mixing_coupling_inequality.step ω) :=
    ProbStruct_mixing_coupling_inequality.additivity_core f g
      (ProbStruct_mixing_coupling_inequality.step ω)
  calc
    MartingaleStep_mixing_coupling_inequality
      (fun x => f x + g x) ω
        = ProbStruct_mixing_coupling_inequality.condExp
            (fun x => f x + g x)
            (ProbStruct_mixing_coupling_inequality.step ω) := by
              rfl
    _ = ProbStruct_mixing_coupling_inequality.condExp f
          (ProbStruct_mixing_coupling_inequality.step ω) +
        ProbStruct_mixing_coupling_inequality.condExp g
          (ProbStruct_mixing_coupling_inequality.step ω) := hadd
    _ = MartingaleStep_mixing_coupling_inequality f ω +
        MartingaleStep_mixing_coupling_inequality g ω := by
          rfl
