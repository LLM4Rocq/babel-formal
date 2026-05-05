/-
BENCHMARK_ID: TINY_MATHLIB_BATCH05_PROBABILITY_STOCHASTIC_FUBINI_AXIOMATIC_LIKE
PAIR_STEM: probability_stochastic_fubini_axiomatic_like
MATH_DOMAIN: Probability Theory
SOURCE_MATHLIB: Mathlib/Probability/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class ProbStruct_stochastic_fubini (Ω : Type u) where
  Expect : (Ω → Nat) → Nat
  CondExp : Nat → (Ω → Nat) → (Ω → Nat)
  Expect_mono : ∀ {f g : Ω → Nat}, (∀ ω : Ω, f ω ≤ g ω) → Expect f ≤ Expect g
  Tower_axiom : ∀ n : Nat, ∀ f : Ω → Nat, Expect (CondExp n f) = Expect f
  Cond_mono_axiom :
    ∀ n : Nat, ∀ {f g : Ω → Nat}, (∀ ω : Ω, f ω ≤ g ω) →
      (∀ ω : Ω, CondExp n f ω ≤ CondExp n g ω)
  Change_measure_axiom :
    ∀ w f : Ω → Nat, Expect f ≤ Expect (fun ω => f ω + w ω)
  Stopping_axiom : ∀ τ : Nat, ∀ f : Ω → Nat, Expect (CondExp τ f) ≤ Expect f
  Nat_le_trans_axiom : ∀ a b c : Nat, a ≤ b → b ≤ c → a ≤ c
  Fubini_swap_axiom :
    ∀ n m : Nat, ∀ f : Ω → Nat,
      Expect (CondExp n (CondExp m f)) = Expect (CondExp m (CondExp n f))

def Filtration_stochastic_fubini (Ω : Type u) : Type u :=
  Nat → (Ω → Prop) → Prop

def MartingaleStep_stochastic_fubini {Ω : Type u}
    (X : Nat → Ω → Nat) : Nat → Ω → Nat :=
  fun n ω => X (Nat.succ n) ω

def DriftShift_stochastic_fubini {Ω : Type u}
    (X D : Nat → Ω → Nat) : Nat → Ω → Nat :=
  fun n ω => X n ω + D n ω

def RateFunc_stochastic_fubini (a b : Nat → Nat) : Nat → Nat :=
  fun n => a n + b n

theorem adaptivity_rule_stochastic_fubini
    {Ω : Type u} (F : Filtration_stochastic_fubini Ω)
    (X : Nat → Ω → Nat)
    (hAdapt : ∀ n m : Nat, F n (fun ω => X m ω = X m ω)) :
    ∀ n m : Nat,
      F n
        (fun ω =>
          MartingaleStep_stochastic_fubini X m ω = MartingaleStep_stochastic_fubini X m ω) := by
  intro n
  intro m
  have hShift : F n (fun ω => X (Nat.succ m) ω = X (Nat.succ m) ω) :=
    hAdapt n (Nat.succ m)
  have hDef :
      (fun ω => MartingaleStep_stochastic_fubini X m ω = MartingaleStep_stochastic_fubini X m ω)
        = (fun ω => X (Nat.succ m) ω = X (Nat.succ m) ω) := by
    funext ω
    rfl
  rw [hDef]
  exact hShift

theorem tower_property_stochastic_fubini
    {Ω : Type u} [ProbStruct_stochastic_fubini Ω]
    (n : Nat) (f : Ω → Nat) :
    ProbStruct_stochastic_fubini.Expect
      (ProbStruct_stochastic_fubini.CondExp n f) =
      ProbStruct_stochastic_fubini.Expect f := by
  have hRaw :
      ProbStruct_stochastic_fubini.Expect (ProbStruct_stochastic_fubini.CondExp n f) =
        ProbStruct_stochastic_fubini.Expect f :=
    ProbStruct_stochastic_fubini.Tower_axiom n f
  have hLeft :
      ProbStruct_stochastic_fubini.Expect (ProbStruct_stochastic_fubini.CondExp n f) =
        ProbStruct_stochastic_fubini.Expect (fun ω => ProbStruct_stochastic_fubini.CondExp n f ω) := by
    rfl
  have hRight : ProbStruct_stochastic_fubini.Expect (fun ω => f ω) = ProbStruct_stochastic_fubini.Expect f := by
    rfl
  calc
    ProbStruct_stochastic_fubini.Expect (ProbStruct_stochastic_fubini.CondExp n f)
        = ProbStruct_stochastic_fubini.Expect (fun ω => ProbStruct_stochastic_fubini.CondExp n f ω) := hLeft
    _ = ProbStruct_stochastic_fubini.Expect f := hRaw
    _ = ProbStruct_stochastic_fubini.Expect (fun ω => f ω) := by
          symm
          exact hRight
    _ = ProbStruct_stochastic_fubini.Expect f := hRight

theorem change_measure_step_stochastic_fubini
    {Ω : Type u} [ProbStruct_stochastic_fubini Ω]
    (w f : Ω → Nat) :
    ProbStruct_stochastic_fubini.Expect f ≤
      ProbStruct_stochastic_fubini.Expect
        (DriftShift_stochastic_fubini (fun _ : Nat => f) (fun _ : Nat => w) 0) := by
  have hAxiom :
      ProbStruct_stochastic_fubini.Expect f ≤
        ProbStruct_stochastic_fubini.Expect (fun ω => f ω + w ω) :=
    ProbStruct_stochastic_fubini.Change_measure_axiom w f
  have hDrift :
      DriftShift_stochastic_fubini (fun _ : Nat => f) (fun _ : Nat => w) 0 =
        (fun ω => f ω + w ω) := by
    funext ω
    rfl
  calc
    ProbStruct_stochastic_fubini.Expect f
        ≤ ProbStruct_stochastic_fubini.Expect (fun ω => f ω + w ω) := hAxiom
    _ = ProbStruct_stochastic_fubini.Expect
          (DriftShift_stochastic_fubini (fun _ : Nat => f) (fun _ : Nat => w) 0) := by
          rw [hDrift]

theorem stopping_control_stochastic_fubini
    {Ω : Type u} [ProbStruct_stochastic_fubini Ω]
    (τ : Nat) (f : Ω → Nat) :
    ProbStruct_stochastic_fubini.Expect (ProbStruct_stochastic_fubini.CondExp τ f) ≤
      ProbStruct_stochastic_fubini.Expect f := by
  have hStop :
      ProbStruct_stochastic_fubini.Expect (ProbStruct_stochastic_fubini.CondExp τ f)
        ≤ ProbStruct_stochastic_fubini.Expect f :=
    ProbStruct_stochastic_fubini.Stopping_axiom τ f
  exact hStop

theorem ld_upper_bound_stochastic_fubini
    {Ω : Type u} [ProbStruct_stochastic_fubini Ω]
    (f : Ω → Nat) (a b : Nat → Nat) (n : Nat)
    (hBase : ProbStruct_stochastic_fubini.Expect f ≤ RateFunc_stochastic_fubini a b n)
    (hRate : RateFunc_stochastic_fubini a b n ≤ RateFunc_stochastic_fubini a b (Nat.succ n)) :
    ProbStruct_stochastic_fubini.Expect f ≤ RateFunc_stochastic_fubini a b (Nat.succ n) := by
  have hStep : ProbStruct_stochastic_fubini.Expect f ≤ RateFunc_stochastic_fubini a b n := hBase
  exact ProbStruct_stochastic_fubini.Nat_le_trans_axiom
    (Ω := Ω)
    (a := ProbStruct_stochastic_fubini.Expect f)
    (b := RateFunc_stochastic_fubini a b n)
    (c := RateFunc_stochastic_fubini a b (Nat.succ n))
    hStep hRate

theorem coupling_estimate_stochastic_fubini
    {Ω : Type u} [ProbStruct_stochastic_fubini Ω]
    (f g w : Ω → Nat)
    (hfg : ∀ ω : Ω, f ω ≤ g ω)
    (hgw : ∀ ω : Ω, g ω ≤ g ω + w ω) :
    ProbStruct_stochastic_fubini.Expect f ≤
      ProbStruct_stochastic_fubini.Expect (fun ω => g ω + w ω) := by
  have hMono₁ :
      ProbStruct_stochastic_fubini.Expect f ≤ ProbStruct_stochastic_fubini.Expect g :=
    ProbStruct_stochastic_fubini.Expect_mono hfg
  have hMono₂ :
      ProbStruct_stochastic_fubini.Expect g ≤
        ProbStruct_stochastic_fubini.Expect (fun ω => g ω + w ω) :=
    ProbStruct_stochastic_fubini.Expect_mono hgw
  exact ProbStruct_stochastic_fubini.Nat_le_trans_axiom
    (Ω := Ω)
    (a := ProbStruct_stochastic_fubini.Expect f)
    (b := ProbStruct_stochastic_fubini.Expect g)
    (c := ProbStruct_stochastic_fubini.Expect (fun ω => g ω + w ω))
    hMono₁ hMono₂

theorem stochastic_fubini_rule_stochastic_fubini
    {Ω : Type u} [ProbStruct_stochastic_fubini Ω]
    (n m : Nat) (f : Ω → Nat) :
    ProbStruct_stochastic_fubini.Expect
      (ProbStruct_stochastic_fubini.CondExp n (ProbStruct_stochastic_fubini.CondExp m f)) =
      ProbStruct_stochastic_fubini.Expect f := by
  have hSwap :
      ProbStruct_stochastic_fubini.Expect
          (ProbStruct_stochastic_fubini.CondExp n (ProbStruct_stochastic_fubini.CondExp m f)) =
        ProbStruct_stochastic_fubini.Expect
          (ProbStruct_stochastic_fubini.CondExp m (ProbStruct_stochastic_fubini.CondExp n f)) :=
    ProbStruct_stochastic_fubini.Fubini_swap_axiom n m f
  have hTower₁ :
      ProbStruct_stochastic_fubini.Expect
          (ProbStruct_stochastic_fubini.CondExp m (ProbStruct_stochastic_fubini.CondExp n f)) =
        ProbStruct_stochastic_fubini.Expect (ProbStruct_stochastic_fubini.CondExp n f) :=
    ProbStruct_stochastic_fubini.Tower_axiom m (ProbStruct_stochastic_fubini.CondExp n f)
  have hTower₂ :
      ProbStruct_stochastic_fubini.Expect (ProbStruct_stochastic_fubini.CondExp n f)
        = ProbStruct_stochastic_fubini.Expect f :=
    ProbStruct_stochastic_fubini.Tower_axiom n f
  calc
    ProbStruct_stochastic_fubini.Expect
        (ProbStruct_stochastic_fubini.CondExp n (ProbStruct_stochastic_fubini.CondExp m f))
        = ProbStruct_stochastic_fubini.Expect
            (ProbStruct_stochastic_fubini.CondExp m (ProbStruct_stochastic_fubini.CondExp n f)) := hSwap
    _ = ProbStruct_stochastic_fubini.Expect (ProbStruct_stochastic_fubini.CondExp n f) := hTower₁
    _ = ProbStruct_stochastic_fubini.Expect f := hTower₂
