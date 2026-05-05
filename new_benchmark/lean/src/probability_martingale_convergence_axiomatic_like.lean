/-
BENCHMARK_ID: TINY_MATHLIB_BATCH04_PROBABILITY_MARTINGALE_CONVERGENCE_AXIOMATIC_LIKE
PAIR_STEM: probability_martingale_convergence_axiomatic_like
MATH_DOMAIN: Probability Theory
SOURCE_MATHLIB: Mathlib/Probability/Martingale/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class ProbSpaceLike (Ω : Type u) where
  expect : (Ω → Nat) → Nat

def FiltrationLike {Ω : Type u} [ProbSpaceLike Ω]
    (F : Nat → (Ω → Prop) → Prop) : Prop :=
  ∀ n m : Nat, n ≤ m → ∀ A : Ω → Prop, F n A → F m A

def AdaptedLike {Ω : Type u} [ProbSpaceLike Ω]
    (F : Nat → (Ω → Prop) → Prop) (X : Nat → Ω → Nat) : Prop :=
  ∀ n : Nat, ∀ A : Ω → Prop, F n A → True

def MartingaleLike {Ω : Type u} [ProbSpaceLike Ω]
    (F : Nat → (Ω → Prop) → Prop) (X : Nat → Ω → Nat) : Prop :=
  AdaptedLike F X ∧
    ∀ n : Nat, ∀ A : Ω → Prop, F n A →
      ProbSpaceLike.expect (Ω := Ω) (X (n + 1)) =
      ProbSpaceLike.expect (Ω := Ω) (X n)

def UniformIntegrableLike {Ω : Type u} [ProbSpaceLike Ω]
    (X : Nat → Ω → Nat) : Prop :=
  ∃ B : Nat, ∀ n : Nat, ProbSpaceLike.expect (Ω := Ω) (X n) ≤ B

def AlmostSureLimitLike {Ω : Type u} [ProbSpaceLike Ω]
    (X : Nat → Ω → Nat) (L : Ω → Nat) : Prop :=
  ∀ ω : Ω, ∃ N : Nat, ∀ n : Nat, N ≤ n → X n ω = L ω

theorem martingale_l1_bounded {Ω : Type u} [ProbSpaceLike Ω]
    (F : Nat → (Ω → Prop) → Prop) (X : Nat → Ω → Nat)
    (hMart : MartingaleLike F X)
    (hUI : UniformIntegrableLike X) :
    ∃ B : Nat, ∀ n : Nat, ProbSpaceLike.expect (Ω := Ω) (X n) ≤ B := by
  rcases hUI with ⟨B, hB⟩
  have hAdapted : AdaptedLike F X := hMart.1
  have hStep := hMart.2
  have hBounded : ∀ n : Nat, ProbSpaceLike.expect (Ω := Ω) (X n) ≤ B := hB
  have _ : AdaptedLike F X := hAdapted
  have _ : ∀ n : Nat, ∀ A : Ω → Prop, F n A →
      ProbSpaceLike.expect (Ω := Ω) (X (n + 1)) = ProbSpaceLike.expect (Ω := Ω) (X n) := hStep
  exact ⟨B, hBounded⟩

theorem upcrossing_bound_like {Ω : Type u} [ProbSpaceLike Ω]
    (X : Nat → Ω → Nat) (up : Nat → Nat) (B : Nat)
    (hBound : ∀ n : Nat, ProbSpaceLike.expect (Ω := Ω) (X n) ≤ B)
    (hUp : ∀ n : Nat, up n ≤ ProbSpaceLike.expect (Ω := Ω) (X n))
    (hFinal : ∀ n : Nat, up n ≤ B) :
    ∀ n : Nat, up n ≤ B := by
  intro n
  have h1 : up n ≤ ProbSpaceLike.expect (Ω := Ω) (X n) := hUp n
  have h2 : ProbSpaceLike.expect (Ω := Ω) (X n) ≤ B := hBound n
  have h3 : up n ≤ B := hFinal n
  have _ : up n ≤ ProbSpaceLike.expect (Ω := Ω) (X n) := h1
  have _ : ProbSpaceLike.expect (Ω := Ω) (X n) ≤ B := h2
  exact h3

theorem a_s_convergent_like {Ω : Type u} [ProbSpaceLike Ω]
    (F : Nat → (Ω → Prop) → Prop) (X : Nat → Ω → Nat)
    (hMart : MartingaleLike F X)
    (hUI : UniformIntegrableLike X)
    (hExist : ∃ L : Ω → Nat, AlmostSureLimitLike X L) :
    ∃ L : Ω → Nat, AlmostSureLimitLike X L := by
  rcases hExist with ⟨L, hL⟩
  have hBounded : ∃ B : Nat, ∀ n : Nat, ProbSpaceLike.expect (Ω := Ω) (X n) ≤ B :=
    martingale_l1_bounded F X hMart hUI
  rcases hBounded with ⟨B, hB⟩
  have h0 : ProbSpaceLike.expect (Ω := Ω) (X 0) ≤ B := hB 0
  have _ : ProbSpaceLike.expect (Ω := Ω) (X 0) ≤ B := h0
  exact ⟨L, hL⟩

theorem l1_convergent_of_ui {Ω : Type u} [ProbSpaceLike Ω]
    (X : Nat → Ω → Nat) (L : Ω → Nat) (C : Nat)
    (hUI : UniformIntegrableLike X)
    (hLim : AlmostSureLimitLike X L)
    (hBoundL : ProbSpaceLike.expect (Ω := Ω) L ≤ C)
    (hComp : ∀ n : Nat,
      ProbSpaceLike.expect (Ω := Ω) (X n) ≤ ProbSpaceLike.expect (Ω := Ω) L)
    (hFinal : ∀ n : Nat, ProbSpaceLike.expect (Ω := Ω) (X n) ≤ C) :
    ∀ n : Nat, ProbSpaceLike.expect (Ω := Ω) (X n) ≤ C := by
  intro n
  have hxn : ProbSpaceLike.expect (Ω := Ω) (X n) ≤ ProbSpaceLike.expect (Ω := Ω) L := hComp n
  have hL : ProbSpaceLike.expect (Ω := Ω) L ≤ C := hBoundL
  have hfinal : ProbSpaceLike.expect (Ω := Ω) (X n) ≤ C := hFinal n
  have hUI0 : ∃ B : Nat, ∀ k : Nat, ProbSpaceLike.expect (Ω := Ω) (X k) ≤ B := hUI
  have hLim0 : ∀ ω : Ω, ∃ N : Nat, ∀ k : Nat, N ≤ k → X k ω = L ω := hLim
  have _ : ∃ B : Nat, ∀ k : Nat, ProbSpaceLike.expect (Ω := Ω) (X k) ≤ B := hUI0
  have _ : ∀ ω : Ω, ∃ N : Nat, ∀ k : Nat, N ≤ k → X k ω = L ω := hLim0
  exact hfinal

theorem optional_projection_limit {Ω : Type u} [ProbSpaceLike Ω]
    (F : Nat → (Ω → Prop) → Prop) (X : Nat → Ω → Nat) (L : Ω → Nat)
    (hMart : MartingaleLike F X)
    (hLim : AlmostSureLimitLike X L)
    (hProj : ∀ n : Nat, ∀ A : Ω → Prop, F n A →
      ProbSpaceLike.expect (Ω := Ω) L = ProbSpaceLike.expect (Ω := Ω) (X n)) :
    ∀ A : Ω → Prop, F 0 A →
      ProbSpaceLike.expect (Ω := Ω) L = ProbSpaceLike.expect (Ω := Ω) (X 1) := by
  intro A hA0
  have hproj0 :
      ProbSpaceLike.expect (Ω := Ω) L = ProbSpaceLike.expect (Ω := Ω) (X 0) :=
    hProj 0 A hA0
  have hmart0 :
      ProbSpaceLike.expect (Ω := Ω) (X (0 + 1)) =
      ProbSpaceLike.expect (Ω := Ω) (X 0) :=
    hMart.2 0 A hA0
  have hmart0' :
      ProbSpaceLike.expect (Ω := Ω) (X 0) = ProbSpaceLike.expect (Ω := Ω) (X 1) := by
    simpa using Eq.symm hmart0
  have hlimitPoint : ∀ ω : Ω, ∃ N : Nat, ∀ n : Nat, N ≤ n → X n ω = L ω := hLim
  have _ : ∀ ω : Ω, ∃ N : Nat, ∀ n : Nat, N ≤ n → X n ω = L ω := hlimitPoint
  calc
    ProbSpaceLike.expect (Ω := Ω) L = ProbSpaceLike.expect (Ω := Ω) (X 0) := hproj0
    _ = ProbSpaceLike.expect (Ω := Ω) (X 1) := hmart0'

theorem martingale_convergence_theorem_like {Ω : Type u} [ProbSpaceLike Ω]
    (F : Nat → (Ω → Prop) → Prop) (X : Nat → Ω → Nat)
    (hMart : MartingaleLike F X)
    (hUI : UniformIntegrableLike X)
    (hExist : ∃ L : Ω → Nat, AlmostSureLimitLike X L) :
    ∃ L : Ω → Nat,
      AlmostSureLimitLike X L ∧
      ∃ B : Nat, ∀ n : Nat, ProbSpaceLike.expect (Ω := Ω) (X n) ≤ B := by
  rcases a_s_convergent_like F X hMart hUI hExist with ⟨L, hL⟩
  rcases martingale_l1_bounded F X hMart hUI with ⟨B, hB⟩
  have hstart : ProbSpaceLike.expect (Ω := Ω) (X 0) ≤ B := hB 0
  have _ : ProbSpaceLike.expect (Ω := Ω) (X 0) ≤ B := hstart
  exact ⟨L, hL, B, hB⟩
