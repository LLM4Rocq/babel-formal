/-
BENCHMARK_ID: TINY_MATHLIB_BATCH05_NUMBER_THEORY_IWASAWA_CONTROL_AXIOMATIC_LIKE
PAIR_STEM: number_theory_iwasawa_control_axiomatic_like
MATH_DOMAIN: Number Theory
SOURCE_MATHLIB: Mathlib/NumberTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u v

class NumberStruct_theory_iwasawa_control (K : Type u) where
  Rep : Type v
  selmer : Rep → Prop
  height : Rep → Nat
  dense : Rep → Prop
  modular_lift_axiom : ∀ r : Rep, dense r → selmer r
  control_axiom : ∀ r : Rep, selmer r → dense r
  hodge_axiom : ∀ r : Rep, selmer r → height r ≤ height r + 1
  height_axiom : ∀ r : Rep, dense r → height r ≤ height r + height r
  chebotarev_axiom : ∀ r : Rep, dense r → ∃ n : Nat, n = height r
  transfer_axiom : ∀ r : Rep, selmer r → dense r → selmer r ∧ dense r
  finiteness_axiom : ∀ r : Rep, selmer r → ∃ m : Nat, height r ≤ m

def GaloisRep_theory_iwasawa_control
    {K : Type u} [h : NumberStruct_theory_iwasawa_control K] : Type v :=
  h.Rep

def SelmerObj_theory_iwasawa_control
    {K : Type u} [h : NumberStruct_theory_iwasawa_control K] :
    GaloisRep_theory_iwasawa_control (K := K) → Prop :=
  h.selmer

def HeightObj_theory_iwasawa_control
    {K : Type u} [h : NumberStruct_theory_iwasawa_control K] :
    GaloisRep_theory_iwasawa_control (K := K) → Nat :=
  h.height

def DensityObj_theory_iwasawa_control
    {K : Type u} [h : NumberStruct_theory_iwasawa_control K] :
    GaloisRep_theory_iwasawa_control (K := K) → Prop :=
  h.dense

theorem modular_lift_step_theory_iwasawa_control
    {K : Type u} [h : NumberStruct_theory_iwasawa_control K]
    (r : GaloisRep_theory_iwasawa_control (K := K))
    (hDense : DensityObj_theory_iwasawa_control (K := K) r) :
    SelmerObj_theory_iwasawa_control (K := K) r := by
  have hLift : h.selmer r := h.modular_lift_axiom r hDense
  have hPack : SelmerObj_theory_iwasawa_control (K := K) r := hLift
  exact hPack

theorem control_theorem_step_theory_iwasawa_control
    {K : Type u} [h : NumberStruct_theory_iwasawa_control K]
    (r : GaloisRep_theory_iwasawa_control (K := K))
    (hSel : SelmerObj_theory_iwasawa_control (K := K) r) :
    DensityObj_theory_iwasawa_control (K := K) r := by
  have hDense : h.dense r := h.control_axiom r hSel
  have hPack : DensityObj_theory_iwasawa_control (K := K) r := hDense
  exact hPack

theorem hodge_filtration_step_theory_iwasawa_control
    {K : Type u} [h : NumberStruct_theory_iwasawa_control K]
    (r : GaloisRep_theory_iwasawa_control (K := K))
    (hSel : SelmerObj_theory_iwasawa_control (K := K) r) :
    HeightObj_theory_iwasawa_control (K := K) r ≤
      HeightObj_theory_iwasawa_control (K := K) r + 1 := by
  have hBound : h.height r ≤ h.height r + 1 := h.hodge_axiom r hSel
  have hPack :
      HeightObj_theory_iwasawa_control (K := K) r ≤
        HeightObj_theory_iwasawa_control (K := K) r + 1 := hBound
  exact hPack

theorem height_inequality_step_theory_iwasawa_control
    {K : Type u} [h : NumberStruct_theory_iwasawa_control K]
    (r : GaloisRep_theory_iwasawa_control (K := K))
    (hSel : SelmerObj_theory_iwasawa_control (K := K) r) :
    HeightObj_theory_iwasawa_control (K := K) r ≤
      HeightObj_theory_iwasawa_control (K := K) r +
      HeightObj_theory_iwasawa_control (K := K) r := by
  have hDense : DensityObj_theory_iwasawa_control (K := K) r :=
    control_theorem_step_theory_iwasawa_control (K := K) r hSel
  have hHodge :
      HeightObj_theory_iwasawa_control (K := K) r ≤
        HeightObj_theory_iwasawa_control (K := K) r + 1 :=
    hodge_filtration_step_theory_iwasawa_control (K := K) r hSel
  have hBase : h.height r ≤ h.height r + h.height r := h.height_axiom r hDense
  have hPack :
      HeightObj_theory_iwasawa_control (K := K) r ≤
        HeightObj_theory_iwasawa_control (K := K) r +
        HeightObj_theory_iwasawa_control (K := K) r := hBase
  have _ :
      HeightObj_theory_iwasawa_control (K := K) r ≤
        HeightObj_theory_iwasawa_control (K := K) r + 1 := hHodge
  exact hPack

theorem chebotarev_count_step_theory_iwasawa_control
    {K : Type u} [h : NumberStruct_theory_iwasawa_control K]
    (r : GaloisRep_theory_iwasawa_control (K := K))
    (hSel : SelmerObj_theory_iwasawa_control (K := K) r) :
    ∃ n : Nat,
      n = HeightObj_theory_iwasawa_control (K := K) r := by
  have hDense : DensityObj_theory_iwasawa_control (K := K) r :=
    control_theorem_step_theory_iwasawa_control (K := K) r hSel
  rcases h.chebotarev_axiom r hDense with ⟨n, hn⟩
  have hPack : n = HeightObj_theory_iwasawa_control (K := K) r := hn
  exact ⟨n, hPack⟩

theorem arithmetic_transfer_theory_iwasawa_control
    {K : Type u} [h : NumberStruct_theory_iwasawa_control K]
    (r : GaloisRep_theory_iwasawa_control (K := K))
    (hSel : SelmerObj_theory_iwasawa_control (K := K) r) :
    SelmerObj_theory_iwasawa_control (K := K) r ∧
    DensityObj_theory_iwasawa_control (K := K) r := by
  have hDense : DensityObj_theory_iwasawa_control (K := K) r :=
    control_theorem_step_theory_iwasawa_control (K := K) r hSel
  have hPair : h.selmer r ∧ h.dense r := h.transfer_axiom r hSel hDense
  have hSelOut : SelmerObj_theory_iwasawa_control (K := K) r := hPair.left
  have hDenseOut : DensityObj_theory_iwasawa_control (K := K) r := hPair.right
  exact And.intro hSelOut hDenseOut

theorem global_finiteness_theory_iwasawa_control
    {K : Type u} [h : NumberStruct_theory_iwasawa_control K]
    (r : GaloisRep_theory_iwasawa_control (K := K))
    (hDense : DensityObj_theory_iwasawa_control (K := K) r) :
    ∃ m : Nat,
      HeightObj_theory_iwasawa_control (K := K) r ≤ m ∧
      DensityObj_theory_iwasawa_control (K := K) r := by
  have hSel : SelmerObj_theory_iwasawa_control (K := K) r :=
    modular_lift_step_theory_iwasawa_control (K := K) r hDense
  rcases h.finiteness_axiom r hSel with ⟨m, hm⟩
  have hBound : HeightObj_theory_iwasawa_control (K := K) r ≤ m := hm
  have hKeepDense : DensityObj_theory_iwasawa_control (K := K) r := hDense
  exact ⟨m, And.intro hBound hKeepDense⟩
