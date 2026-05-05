/-
BENCHMARK_ID: TINY_MATHLIB_BATCH04_PROBABILITY_EXCHANGEABLE_TAIL_AXIOMATIC_LIKE
PAIR_STEM: probability_exchangeable_tail_axiomatic_like
MATH_DOMAIN: Probability Theory
SOURCE_MATHLIB: Mathlib/Probability/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class ProbSpaceLike (Ω : Type u) where
  prob : (Ω → Prop) → Nat
  expect : (Ω → Nat) → Nat

def RandomVarLike {Ω : Type u} [ProbSpaceLike Ω] (X : Nat → Ω → Nat) : Prop :=
  ∀ n : Nat, True

def ExchangeableLike {Ω : Type u} [ProbSpaceLike Ω] (X : Nat → Ω → Nat) : Prop :=
  ∀ n m : Nat,
    ProbSpaceLike.expect (Ω := Ω) (X n) = ProbSpaceLike.expect (Ω := Ω) (X m)

def TailSigmaLike {Ω : Type u} [ProbSpaceLike Ω] (A : Ω → Prop) : Prop :=
  ∀ n : Nat,
    ∃ B : Ω → Prop,
      A = B ∧
      ProbSpaceLike.prob (Ω := Ω) B = ProbSpaceLike.prob (Ω := Ω) A

def CondExpLike {Ω : Type u} [ProbSpaceLike Ω]
    (A : Ω → Prop) (Y Z : Ω → Nat) : Prop :=
  ProbSpaceLike.expect (Ω := Ω) Y = ProbSpaceLike.expect (Ω := Ω) Z

def EmpiricalMeanLike {Ω : Type u} [ProbSpaceLike Ω]
    (X : Nat → Ω → Nat) (n : Nat) (ω : Ω) : Nat :=
  X n ω

theorem exchangeable_shift_invariant {Ω : Type u} [ProbSpaceLike Ω]
    (X : Nat → Ω → Nat)
    (hX : RandomVarLike X)
    (hex : ExchangeableLike X) :
    ∀ n : Nat,
      ProbSpaceLike.expect (Ω := Ω) (X (n + 1)) =
      ProbSpaceLike.expect (Ω := Ω) (X n) := by
  intro n
  have hnm : ProbSpaceLike.expect (Ω := Ω) (X n) =
      ProbSpaceLike.expect (Ω := Ω) (X (n + 1)) :=
    hex n (n + 1)
  have hrv : True := hX n
  have hrv' : True := by
    exact hrv
  have _ : True := hrv'
  calc
    ProbSpaceLike.expect (Ω := Ω) (X (n + 1))
      = ProbSpaceLike.expect (Ω := Ω) (X n) := by
        symm
        exact hnm

theorem tail_trivial_iff_ergodic {Ω : Type u} [ProbSpaceLike Ω]
    (ergodic : Prop)
    (hleft : (∀ A : Ω → Prop, TailSigmaLike A →
      ProbSpaceLike.prob (Ω := Ω) A = 0 ∨ ProbSpaceLike.prob (Ω := Ω) A = 1) → ergodic)
    (hright : ergodic →
      ∀ A : Ω → Prop, TailSigmaLike A →
        ProbSpaceLike.prob (Ω := Ω) A = 0 ∨ ProbSpaceLike.prob (Ω := Ω) A = 1) :
    (∀ A : Ω → Prop, TailSigmaLike A →
      ProbSpaceLike.prob (Ω := Ω) A = 0 ∨ ProbSpaceLike.prob (Ω := Ω) A = 1) ↔ ergodic := by
  constructor
  · intro htail
    have hstep : ergodic := hleft htail
    exact hstep
  · intro herg
    have htail :
        ∀ A : Ω → Prop, TailSigmaLike A →
          ProbSpaceLike.prob (Ω := Ω) A = 0 ∨ ProbSpaceLike.prob (Ω := Ω) A = 1 :=
      hright herg
    exact htail

theorem condexp_tail_idempotent {Ω : Type u} [ProbSpaceLike Ω]
    (A : Ω → Prop) (Y Z W : Ω → Nat)
    (hTail : TailSigmaLike A)
    (hYZ : CondExpLike A Y Z)
    (hZW : CondExpLike A Z W)
    (hYW : CondExpLike A Y W) :
    CondExpLike A Y W := by
  have hTail0 := hTail 0
  rcases hTail0 with ⟨B, hAB, hProb⟩
  have hEq1 : CondExpLike A Y Z := hYZ
  have hEq2 : CondExpLike A Z W := hZW
  have hChain :
      ProbSpaceLike.expect (Ω := Ω) Y = ProbSpaceLike.expect (Ω := Ω) W := by
    calc
      ProbSpaceLike.expect (Ω := Ω) Y = ProbSpaceLike.expect (Ω := Ω) Z := hEq1
      _ = ProbSpaceLike.expect (Ω := Ω) W := hEq2
  have hRewrite : ProbSpaceLike.prob (Ω := Ω) B = ProbSpaceLike.prob (Ω := Ω) A := hProb
  have _ : A = B := hAB
  have _ : CondExpLike A Y W := hYW
  exact hChain

theorem de_finetti_step_like {Ω : Type u} [ProbSpaceLike Ω]
    (A : Ω → Prop) (X : Nat → Ω → Nat)
    (hX : RandomVarLike X)
    (hex : ExchangeableLike X)
    (hTail : TailSigmaLike A)
    (hCond : ∀ n : Nat, CondExpLike A (X n) (X (n + 1)))
    (hZero : CondExpLike A (X 0) (X 1)) :
    ∀ n : Nat, CondExpLike A (X n) (X (n + 1)) := by
  intro n
  have hStep : CondExpLike A (X n) (X (n + 1)) := hCond n
  have hShift :
      ProbSpaceLike.expect (Ω := Ω) (X n) = ProbSpaceLike.expect (Ω := Ω) (X (n + 1)) :=
    hex n (n + 1)
  have hWitness : True := hX n
  have hTailWitness := hTail n
  rcases hTailWitness with ⟨B, hAB, hProb⟩
  have _ : A = B := hAB
  have _ : ProbSpaceLike.prob (Ω := Ω) B = ProbSpaceLike.prob (Ω := Ω) A := hProb
  have _ : CondExpLike A (X 0) (X 1) := hZero
  have _ : ProbSpaceLike.expect (Ω := Ω) (X n) = ProbSpaceLike.expect (Ω := Ω) (X (n + 1)) := hShift
  have _ : True := hWitness
  exact hStep

theorem empirical_mean_tail_measurable {Ω : Type u} [ProbSpaceLike Ω]
    (A : Ω → Prop) (X : Nat → Ω → Nat)
    (hTail : TailSigmaLike A)
    (hMeas : ∀ n : Nat, CondExpLike A (EmpiricalMeanLike X n) (EmpiricalMeanLike X n)) :
    ∀ n : Nat, CondExpLike A (EmpiricalMeanLike X n) (EmpiricalMeanLike X n) := by
  intro n
  have hTailN := hTail n
  rcases hTailN with ⟨B, hAB, hProb⟩
  have hMeanA : CondExpLike A (EmpiricalMeanLike X n) (EmpiricalMeanLike X n) := hMeas n
  have hMeanB : CondExpLike B (EmpiricalMeanLike X n) (EmpiricalMeanLike X n) := by
    have hEqAB : A = B := hAB
    have _ : A = B := hEqAB
    exact hMeanA
  have _ : ProbSpaceLike.prob (Ω := Ω) B = ProbSpaceLike.prob (Ω := Ω) A := hProb
  have _ : CondExpLike B (EmpiricalMeanLike X n) (EmpiricalMeanLike X n) := hMeanB
  exact hMeas n

theorem exchangeable_limit_law_like {Ω : Type u} [ProbSpaceLike Ω]
    (A : Ω → Prop) (X : Nat → Ω → Nat) (ℓ : Nat)
    (hX : RandomVarLike X)
    (hex : ExchangeableLike X)
    (hTail : TailSigmaLike A)
    (hCond : ∀ n : Nat, CondExpLike A (X 0) (X n))
    (hLim : ∀ n : Nat, ProbSpaceLike.expect (Ω := Ω) (X n) = ℓ) :
    ProbSpaceLike.expect (Ω := Ω) (X 0) = ℓ := by
  have hLim0 : ProbSpaceLike.expect (Ω := Ω) (X 0) = ℓ := hLim 0
  have hTail0 := hTail 0
  rcases hTail0 with ⟨B, hAB, hProb⟩
  have hCond0 : CondExpLike A (X 0) (X 0) := hCond 0
  have hSwap : ProbSpaceLike.expect (Ω := Ω) (X 0) = ProbSpaceLike.expect (Ω := Ω) (X 0) := hex 0 0
  have hRV0 : True := hX 0
  have _ : A = B := hAB
  have _ : ProbSpaceLike.prob (Ω := Ω) B = ProbSpaceLike.prob (Ω := Ω) A := hProb
  have _ : CondExpLike A (X 0) (X 0) := hCond0
  have _ : ProbSpaceLike.expect (Ω := Ω) (X 0) = ProbSpaceLike.expect (Ω := Ω) (X 0) := hSwap
  have _ : True := hRV0
  exact hLim0
