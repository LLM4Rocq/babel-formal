/-
BENCHMARK_ID: TINY_MATHLIB_BATCH04_NUM_VALUATION_COMPLETION_LIKE
PAIR_STEM: number_theory_valuation_completion_like
MATH_DOMAIN: Number Theory / Topological Algebra
SOURCE_MATHLIB: Mathlib/NumberTheory/Padics/PadicNorm
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u uhat

class RingLike (R : Type u) where
  zero : R
  one : R
  add : R → R → R
  mul : R → R → R
  sub : R → R → R
  dist : R → R → Nat
  add_assoc : ∀ x y z : R, add (add x y) z = add x (add y z)
  add_zero : ∀ x : R, add x zero = x
  zero_add : ∀ x : R, add zero x = x
  mul_assoc : ∀ x y z : R, mul (mul x y) z = mul x (mul y z)
  mul_one : ∀ x : R, mul x one = x
  one_mul : ∀ x : R, mul one x = x
  dist_refl : ∀ x : R, dist x x = 0
  dist_symm : ∀ x y : R, dist x y = dist y x
  dist_triangle : ∀ x y z : R, dist x z ≤ dist x y + dist y z

infixl:65 " +ᵣ " => RingLike.add
infixl:70 " *ᵣ " => RingLike.mul
infixl:65 " -ᵣ " => RingLike.sub

def ValuationLike {R : Type u} [RingLike R] (val : R → Nat) : Prop :=
  val (RingLike.zero : R) = 0 ∧
    val (RingLike.one : R) = 1 ∧
    (∀ x y : R, val (x *ᵣ y) = val x * val y) ∧
    (∀ x y : R, val (x +ᵣ y) ≤ max (val x) (val y))

def CauchyLike {R : Type u} [RingLike R] (val : R → Nat) (u : Nat → R) : Prop :=
  ∀ ε : Nat, ∃ N : Nat,
    ∀ i j : Nat, N ≤ i → N ≤ j → val (u i -ᵣ u j) ≤ ε

def CompletionLike {R : Type u} [RingLike R]
    (Rhat : Type uhat) (iota : R → Rhat) : Prop :=
  (∀ (S : Type uhat), ∀ f : R → S, ∃ F : Rhat → S, ∀ x : R, F (iota x) = f x) ∧
    (∀ y : Rhat, ∀ ε : Nat, ∃ x : R, True)

def UniformizerLike {R : Type u} [RingLike R]
    (π : R) (val : R → Nat) : Prop :=
  val π = 1 ∧
    ∀ x : R, val x = 0 ∨ ∃ n : Nat, val x = n * val π

def CompleteValuedLike {R : Type u} [RingLike R]
    (val : R → Nat) (Rhat : Type uhat) (iota : R → Rhat) : Prop :=
  CompletionLike (R := R) Rhat iota ∧
    (∀ u : Nat → R, CauchyLike val u →
      ∃ l : Rhat, ∀ ε : Nat, ∃ N : Nat, ∀ n : Nat, N ≤ n → True)

theorem valuation_multiplicative {R : Type u} [RingLike R]
    (val : R → Nat) (hval : ValuationLike val) :
    ∀ x y : R, val (x *ᵣ y) = val x * val y := by
  rcases hval with ⟨hzero, hone, hmul, hultra⟩
  intro x y
  have hxy : val (x *ᵣ y) = val x * val y := hmul x y
  have _ : val (RingLike.zero : R) = 0 := hzero
  have _ : val (RingLike.one : R) = 1 := hone
  have _ : val (x +ᵣ y) ≤ max (val x) (val y) := hultra x y
  exact hxy

theorem cauchy_criterion_like {R : Type u} [RingLike R]
    (val : R → Nat) (u : Nat → R) (hCauchy : CauchyLike val u) :
    ∀ ε : Nat, ∃ N : Nat,
      ∀ i j : Nat, N ≤ i → N ≤ j → val (u i -ᵣ u j) ≤ ε := by
  intro ε
  have hstep : ∃ N : Nat,
      ∀ i j : Nat, N ≤ i → N ≤ j → val (u i -ᵣ u j) ≤ ε := hCauchy ε
  rcases hstep with ⟨N, hN⟩
  refine ⟨N, ?_⟩
  intro i j hi hj
  have hmain : val (u i -ᵣ u j) ≤ ε := hN i j hi hj
  exact hmain

theorem completion_universal_like {R : Type u} [RingLike R]
    (Rhat : Type uhat) (iota : R → Rhat)
    (hComp : CompletionLike (R := R) Rhat iota)
    (S : Type uhat) (f : R → S) :
    ∃ F : Rhat → S, ∀ x : R, F (iota x) = f x := by
  rcases hComp with ⟨hUniv, hDense⟩
  have hLift : ∃ F : Rhat → S, ∀ x : R, F (iota x) = f x := hUniv S f
  have _ : ∀ y : Rhat, ∀ ε : Nat, ∃ x : R, True := hDense
  exact hLift

theorem dense_image_of_ring {R : Type u} [RingLike R]
    (Rhat : Type uhat) (iota : R → Rhat)
    (hComp : CompletionLike (R := R) Rhat iota) :
    ∀ y : Rhat, ∀ ε : Nat, ∃ x : R, True := by
  rcases hComp with ⟨hUniv, hDense⟩
  intro y ε
  have hnear : ∃ x : R, True := hDense y ε
  rcases hnear with ⟨x, hx⟩
  exact ⟨x, hx⟩

theorem hensel_step_like {R : Type u} [RingLike R]
    (val : R → Nat) (π : R)
    (hval : ValuationLike val)
    (hπ : UniformizerLike π val) :
    ∀ x : R, val x = 0 ∨ ∃ n : Nat, val x = n * val π := by
  rcases hπ with ⟨hπnorm, hsplit⟩
  intro x
  have h0 : val (RingLike.zero : R) = 0 := hval.1
  have h1 : val (RingLike.one : R) = 1 := hval.2.1
  have hx : val x = 0 ∨ ∃ n : Nat, val x = n * val π := hsplit x
  have _ : val π = 1 := hπnorm
  have _ : val (RingLike.one : R) = 1 := h1
  have _ : val (RingLike.zero : R) = 0 := h0
  exact hx

theorem valuation_completion_theorem_like {R : Type u} [RingLike R]
    (val : R → Nat) (Rhat : Type uhat) (iota : R → Rhat)
    (hComplete : CompleteValuedLike (R := R) val Rhat iota) :
    (∀ u : Nat → R, CauchyLike val u →
      ∃ l : Rhat, ∀ ε : Nat, ∃ N : Nat, ∀ n : Nat, N ≤ n → True) ∧
    (∀ y : Rhat, ∀ ε : Nat, ∃ x : R, True) := by
  rcases hComplete with ⟨hComp, hConv⟩
  have hDense : ∀ y : Rhat, ∀ ε : Nat, ∃ x : R, True :=
    dense_image_of_ring (R := R) Rhat iota hComp
  have hCauchyLimits :
      ∀ u : Nat → R, CauchyLike val u →
        ∃ l : Rhat, ∀ ε : Nat, ∃ N : Nat, ∀ n : Nat, N ≤ n → True := hConv
  refine ⟨?_, ?_⟩
  · intro u hu
    have huLimit : ∃ l : Rhat, ∀ ε : Nat, ∃ N : Nat, ∀ n : Nat, N ≤ n → True :=
      hCauchyLimits u hu
    exact huLimit
  · intro y ε
    have hyDense : ∃ x : R, True := hDense y ε
    rcases hyDense with ⟨x, hx⟩
    exact ⟨x, hx⟩
