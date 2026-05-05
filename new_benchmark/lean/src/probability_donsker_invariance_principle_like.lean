/-
BENCHMARK_ID: TINY_MATHLIB_BATCH06_PROBABILITY_DONSKER_INVARIANCE_PRINCIPLE_LIKE
PAIR_STEM: probability_donsker_invariance_principle_like
MATH_DOMAIN: Probability
SOURCE_MATHLIB: Mathlib/Probability/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class InvarianceStruct_donsker_principle (Ω : Type u) where
  expect : (Ω → Nat) → Nat
  finite_dimensional_axiom :
    ∀ X : Nat → Ω → Nat,
      ∀ L : Ω → Nat,
        ∀ B : Nat → Nat,
          (∀ n : Nat, expect (X n) ≤ B n) →
          (∀ n : Nat, B n ≤ expect L) →
          ∀ n : Nat, expect (X n) ≤ expect L
  tightness_axiom :
    ∀ X : Nat → Ω → Nat,
      ∀ B : Nat → Nat,
        (∀ n : Nat, expect (X n) ≤ B n) →
        (∀ n : Nat, B n ≤ B (n + 1)) →
        ∀ n : Nat, expect (X n) ≤ B (n + 1)
  kolmogorov_step_axiom :
    ∀ X : Nat → Ω → Nat,
      ∀ B : Nat → Nat,
        (∀ n : Nat, expect (X n) ≤ B (n + 1)) →
        ∀ n : Nat, expect (X n) ≤ B (n + 2)
  interpolation_axiom :
    ∀ X : Nat → Ω → Nat,
      ∀ I : Nat → Ω → Nat,
        ∀ B : Nat → Nat,
          (∀ n : Nat, expect (I n) ≤ expect (X n)) →
          (∀ n : Nat, expect (X n) ≤ B n) →
          ∀ n : Nat, expect (I n) ≤ B n
  weak_limit_axiom :
    ∀ X : Nat → Ω → Nat,
      ∀ L : Ω → Nat,
        (∀ n : Nat, expect (X n) ≤ expect L) →
        (∀ n : Nat, expect L ≤ expect (X n)) →
        ∀ n : Nat, expect (X n) = expect L
  projection_axiom :
    ∀ X : Nat → Ω → Nat,
      ∀ P : Nat → Ω → Nat,
        (∀ n : Nat, expect (P n) ≤ expect (X n)) →
        (∀ n : Nat, expect (X n) ≤ expect (P n)) →
        ∀ n : Nat, expect (P n) = expect (X n)

structure ProcessData_donsker_invariance_principle
    {Ω : Type u} [h : InvarianceStruct_donsker_principle Ω] where
  walk : Nat → Ω → Nat
  bridge : Nat → Ω → Nat
  limit : Ω → Nat
  modulus : Nat → Nat
  walk_bound : ∀ n : Nat, h.expect (walk n) ≤ modulus n
  bridge_le_walk : ∀ n : Nat, h.expect (bridge n) ≤ h.expect (walk n)
  modulus_mono : ∀ n : Nat, modulus n ≤ modulus (n + 1)

def rescaled_walk_donsker_invariance_principle
    {Ω : Type u} [h : InvarianceStruct_donsker_principle Ω]
    (P : ProcessData_donsker_invariance_principle (Ω := Ω)) :
    Nat → Ω → Nat :=
  P.walk

def brownian_limit_donsker_invariance_principle
    {Ω : Type u} [h : InvarianceStruct_donsker_principle Ω]
    (P : ProcessData_donsker_invariance_principle (Ω := Ω)) :
    Ω → Nat :=
  P.limit

def modulus_control_donsker_invariance_principle
    {Ω : Type u} [h : InvarianceStruct_donsker_principle Ω]
    (P : ProcessData_donsker_invariance_principle (Ω := Ω)) :
    Nat → Nat :=
  P.modulus

theorem finite_dimensional_convergence_donsker_invariance_principle
    {Ω : Type u} [h : InvarianceStruct_donsker_principle Ω]
    (P : ProcessData_donsker_invariance_principle (Ω := Ω))
    (hCap :
      ∀ n : Nat,
        modulus_control_donsker_invariance_principle (Ω := Ω) P n ≤
          h.expect (brownian_limit_donsker_invariance_principle (Ω := Ω) P)) :
    ∀ n : Nat,
      h.expect (rescaled_walk_donsker_invariance_principle (Ω := Ω) P n) ≤
        h.expect (brownian_limit_donsker_invariance_principle (Ω := Ω) P) := by
  have hWalkBound :
      ∀ n : Nat,
        h.expect (rescaled_walk_donsker_invariance_principle (Ω := Ω) P n) ≤
          modulus_control_donsker_invariance_principle (Ω := Ω) P n := by
    intro n
    simpa [rescaled_walk_donsker_invariance_principle, modulus_control_donsker_invariance_principle] using
      P.walk_bound n
  have hUpperBound :
      ∀ n : Nat,
        modulus_control_donsker_invariance_principle (Ω := Ω) P n ≤
          h.expect (brownian_limit_donsker_invariance_principle (Ω := Ω) P) := hCap
  have hRaw :
      ∀ n : Nat,
        h.expect (P.walk n) ≤ h.expect P.limit :=
    h.finite_dimensional_axiom P.walk P.limit P.modulus
      (by
        intro n
        exact P.walk_bound n)
      (by
        intro n
        exact hCap n)
  have _ : ∀ n : Nat,
      h.expect (rescaled_walk_donsker_invariance_principle (Ω := Ω) P n) ≤
        modulus_control_donsker_invariance_principle (Ω := Ω) P n := hWalkBound
  have _ : ∀ n : Nat,
      modulus_control_donsker_invariance_principle (Ω := Ω) P n ≤
        h.expect (brownian_limit_donsker_invariance_principle (Ω := Ω) P) := hUpperBound
  intro n
  simpa [rescaled_walk_donsker_invariance_principle, brownian_limit_donsker_invariance_principle] using hRaw n

theorem tightness_criterion_donsker_invariance_principle
    {Ω : Type u} [h : InvarianceStruct_donsker_principle Ω]
    (P : ProcessData_donsker_invariance_principle (Ω := Ω)) :
    ∀ n : Nat,
      h.expect (rescaled_walk_donsker_invariance_principle (Ω := Ω) P n) ≤
        modulus_control_donsker_invariance_principle (Ω := Ω) P (n + 1) := by
  have hBoundRaw : ∀ n : Nat, h.expect (P.walk n) ≤ P.modulus n := P.walk_bound
  have hMonoRaw : ∀ n : Nat, P.modulus n ≤ P.modulus (n + 1) := P.modulus_mono
  have hTightRaw : ∀ n : Nat, h.expect (P.walk n) ≤ P.modulus (n + 1) :=
    h.tightness_axiom P.walk P.modulus hBoundRaw hMonoRaw
  have hRewrite :
      ∀ n : Nat,
        h.expect (rescaled_walk_donsker_invariance_principle (Ω := Ω) P n) ≤
          modulus_control_donsker_invariance_principle (Ω := Ω) P (n + 1) := by
    intro n
    simpa [rescaled_walk_donsker_invariance_principle, modulus_control_donsker_invariance_principle] using
      hTightRaw n
  intro n
  exact hRewrite n

theorem kolmogorov_bound_step_donsker_invariance_principle
    {Ω : Type u} [h : InvarianceStruct_donsker_principle Ω]
    (P : ProcessData_donsker_invariance_principle (Ω := Ω)) :
    ∀ n : Nat,
      h.expect (rescaled_walk_donsker_invariance_principle (Ω := Ω) P n) ≤
        modulus_control_donsker_invariance_principle (Ω := Ω) P (n + 2) := by
  have hTight :
      ∀ n : Nat,
        h.expect (rescaled_walk_donsker_invariance_principle (Ω := Ω) P n) ≤
          modulus_control_donsker_invariance_principle (Ω := Ω) P (n + 1) :=
    tightness_criterion_donsker_invariance_principle (Ω := Ω) P
  have hStepRaw : ∀ n : Nat, h.expect (P.walk n) ≤ P.modulus (n + 2) :=
    h.kolmogorov_step_axiom P.walk P.modulus (by
      intro n
      simpa [rescaled_walk_donsker_invariance_principle, modulus_control_donsker_invariance_principle] using hTight n)
  have hStep :
      ∀ n : Nat,
        h.expect (rescaled_walk_donsker_invariance_principle (Ω := Ω) P n) ≤
          modulus_control_donsker_invariance_principle (Ω := Ω) P (n + 2) := by
    intro n
    simpa [rescaled_walk_donsker_invariance_principle, modulus_control_donsker_invariance_principle] using
      hStepRaw n
  intro n
  exact hStep n

theorem interpolation_error_donsker_invariance_principle
    {Ω : Type u} [h : InvarianceStruct_donsker_principle Ω]
    (P : ProcessData_donsker_invariance_principle (Ω := Ω)) :
    ∀ n : Nat,
      h.expect (P.bridge n) ≤
        modulus_control_donsker_invariance_principle (Ω := Ω) P n := by
  have hBridgeLeWalk :
      ∀ n : Nat,
        h.expect (P.bridge n) ≤ h.expect (rescaled_walk_donsker_invariance_principle (Ω := Ω) P n) := by
    intro n
    simpa [rescaled_walk_donsker_invariance_principle] using P.bridge_le_walk n
  have hWalkLeBound :
      ∀ n : Nat,
        h.expect (rescaled_walk_donsker_invariance_principle (Ω := Ω) P n) ≤
          modulus_control_donsker_invariance_principle (Ω := Ω) P n := by
    intro n
    simpa [rescaled_walk_donsker_invariance_principle, modulus_control_donsker_invariance_principle] using
      P.walk_bound n
  have hRaw : ∀ n : Nat, h.expect (P.bridge n) ≤ P.modulus n :=
    h.interpolation_axiom P.walk P.bridge P.modulus
      (by
        intro n
        exact P.bridge_le_walk n)
      (by
        intro n
        exact P.walk_bound n)
  have _ : ∀ n : Nat,
      h.expect (P.bridge n) ≤ h.expect (rescaled_walk_donsker_invariance_principle (Ω := Ω) P n) := hBridgeLeWalk
  have _ : ∀ n : Nat,
      h.expect (rescaled_walk_donsker_invariance_principle (Ω := Ω) P n) ≤
        modulus_control_donsker_invariance_principle (Ω := Ω) P n := hWalkLeBound
  intro n
  simpa [modulus_control_donsker_invariance_principle] using hRaw n

theorem weak_limit_identification_donsker_invariance_principle
    {Ω : Type u} [h : InvarianceStruct_donsker_principle Ω]
    (P : ProcessData_donsker_invariance_principle (Ω := Ω))
    (hCap :
      ∀ n : Nat,
        modulus_control_donsker_invariance_principle (Ω := Ω) P n ≤
          h.expect (brownian_limit_donsker_invariance_principle (Ω := Ω) P))
    (hLower :
      ∀ n : Nat,
        h.expect (brownian_limit_donsker_invariance_principle (Ω := Ω) P) ≤
          h.expect (rescaled_walk_donsker_invariance_principle (Ω := Ω) P n)) :
    ∀ n : Nat,
      h.expect (rescaled_walk_donsker_invariance_principle (Ω := Ω) P n) =
        h.expect (brownian_limit_donsker_invariance_principle (Ω := Ω) P) := by
  have hUpper :
      ∀ n : Nat,
        h.expect (rescaled_walk_donsker_invariance_principle (Ω := Ω) P n) ≤
          h.expect (brownian_limit_donsker_invariance_principle (Ω := Ω) P) :=
    finite_dimensional_convergence_donsker_invariance_principle (Ω := Ω) P hCap
  have hUpperRaw : ∀ n : Nat, h.expect (P.walk n) ≤ h.expect P.limit := by
    intro n
    simpa [rescaled_walk_donsker_invariance_principle, brownian_limit_donsker_invariance_principle] using hUpper n
  have hLowerRaw : ∀ n : Nat, h.expect P.limit ≤ h.expect (P.walk n) := by
    intro n
    simpa [rescaled_walk_donsker_invariance_principle, brownian_limit_donsker_invariance_principle] using hLower n
  have hEqRaw : ∀ n : Nat, h.expect (P.walk n) = h.expect P.limit :=
    h.weak_limit_axiom P.walk P.limit hUpperRaw hLowerRaw
  intro n
  simpa [rescaled_walk_donsker_invariance_principle, brownian_limit_donsker_invariance_principle] using hEqRaw n

theorem martingale_projection_step_donsker_invariance_principle
    {Ω : Type u} [h : InvarianceStruct_donsker_principle Ω]
    (P : ProcessData_donsker_invariance_principle (Ω := Ω))
    (Q : Nat → Ω → Nat)
    (hProjLe :
      ∀ n : Nat,
        h.expect (Q n) ≤ h.expect (rescaled_walk_donsker_invariance_principle (Ω := Ω) P n))
    (hProjGe :
      ∀ n : Nat,
        h.expect (rescaled_walk_donsker_invariance_principle (Ω := Ω) P n) ≤ h.expect (Q n)) :
    ∀ n : Nat,
      h.expect (Q n) = h.expect (rescaled_walk_donsker_invariance_principle (Ω := Ω) P n) := by
  have hProjLeRaw : ∀ n : Nat, h.expect (Q n) ≤ h.expect (P.walk n) := by
    intro n
    simpa [rescaled_walk_donsker_invariance_principle] using hProjLe n
  have hProjGeRaw : ∀ n : Nat, h.expect (P.walk n) ≤ h.expect (Q n) := by
    intro n
    simpa [rescaled_walk_donsker_invariance_principle] using hProjGe n
  have hEqRaw : ∀ n : Nat, h.expect (Q n) = h.expect (P.walk n) :=
    h.projection_axiom P.walk Q hProjLeRaw hProjGeRaw
  have hPack :
      ∀ n : Nat,
        h.expect (Q n) = h.expect (rescaled_walk_donsker_invariance_principle (Ω := Ω) P n) := by
    intro n
    simpa [rescaled_walk_donsker_invariance_principle] using hEqRaw n
  intro n
  exact hPack n

theorem invariance_principle_final_donsker_invariance_principle
    {Ω : Type u} [h : InvarianceStruct_donsker_principle Ω]
    (P : ProcessData_donsker_invariance_principle (Ω := Ω))
    (Q : Nat → Ω → Nat)
    (hCap :
      ∀ n : Nat,
        modulus_control_donsker_invariance_principle (Ω := Ω) P n ≤
          h.expect (brownian_limit_donsker_invariance_principle (Ω := Ω) P))
    (hLower :
      ∀ n : Nat,
        h.expect (brownian_limit_donsker_invariance_principle (Ω := Ω) P) ≤
          h.expect (rescaled_walk_donsker_invariance_principle (Ω := Ω) P n))
    (hProjLe :
      ∀ n : Nat,
        h.expect (Q n) ≤ h.expect (rescaled_walk_donsker_invariance_principle (Ω := Ω) P n))
    (hProjGe :
      ∀ n : Nat,
        h.expect (rescaled_walk_donsker_invariance_principle (Ω := Ω) P n) ≤ h.expect (Q n)) :
    ∀ n : Nat,
      h.expect (Q n) = h.expect (brownian_limit_donsker_invariance_principle (Ω := Ω) P) := by
  have hWeak :
      ∀ n : Nat,
        h.expect (rescaled_walk_donsker_invariance_principle (Ω := Ω) P n) =
          h.expect (brownian_limit_donsker_invariance_principle (Ω := Ω) P) :=
    weak_limit_identification_donsker_invariance_principle (Ω := Ω) P hCap hLower
  have hProj :
      ∀ n : Nat,
        h.expect (Q n) = h.expect (rescaled_walk_donsker_invariance_principle (Ω := Ω) P n) :=
    martingale_projection_step_donsker_invariance_principle (Ω := Ω) P Q hProjLe hProjGe
  intro n
  have hEq1 : h.expect (Q n) = h.expect (rescaled_walk_donsker_invariance_principle (Ω := Ω) P n) :=
    hProj n
  have hEq2 :
      h.expect (rescaled_walk_donsker_invariance_principle (Ω := Ω) P n) =
        h.expect (brownian_limit_donsker_invariance_principle (Ω := Ω) P) :=
    hWeak n
  calc
    h.expect (Q n) =
        h.expect (rescaled_walk_donsker_invariance_principle (Ω := Ω) P n) := hEq1
    _ =
        h.expect (brownian_limit_donsker_invariance_principle (Ω := Ω) P) := hEq2
