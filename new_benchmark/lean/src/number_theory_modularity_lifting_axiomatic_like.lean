/-
BENCHMARK_ID: TINY_MATHLIB_BATCH05_NUMBER_THEORY_MODULARITY_LIFTING_AXIOMATIC_LIKE
PAIR_STEM: number_theory_modularity_lifting_axiomatic_like
MATH_DOMAIN: Number Theory
SOURCE_MATHLIB: Mathlib/NumberTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class NumberStruct_theory_modularity_lifting (K : Type u) where
  rep : K → Nat
  selmer : K → Nat
  height : K → Nat
  density : K → Nat
  modular_lift_axiom :
    ∀ x : K,
      rep x ≤ selmer x + height x
  control_axiom :
    ∀ x : K,
      selmer x ≤ density x + rep x
  hodge_axiom :
    ∀ x : K,
      height x ≤ selmer x + density x
  height_axiom :
    ∀ x : K,
      height x ≤ rep x + density x
  chebotarev_axiom :
    ∀ x : K,
      density x ≤ rep x + selmer x
  arithmetic_transfer_axiom :
    ∀ x : K,
      rep x + height x ≤ selmer x + density x + height x
  global_finiteness_axiom :
    ∀ x : K,
      rep x + selmer x + height x ≤ density x + density x + rep x + selmer x

def GaloisRep_theory_modularity_lifting
    (K : Type u) : Type u :=
  K → Nat

def SelmerObj_theory_modularity_lifting
    (K : Type u) : Type u :=
  K → Nat

def HeightObj_theory_modularity_lifting
    (K : Type u) : Type u :=
  K → Nat

def DensityObj_theory_modularity_lifting
    (K : Type u) : Type u :=
  K → Nat

theorem modular_lift_step_theory_modularity_lifting
    {K : Type u} [h : NumberStruct_theory_modularity_lifting K]
    (x : K) :
    ∃ r : Nat, r = h.rep x ∧ r ≤ h.selmer x + h.height x := by
  have hLift : h.rep x ≤ h.selmer x + h.height x := h.modular_lift_axiom x
  refine ⟨h.rep x, rfl, ?_⟩
  exact hLift

theorem control_theorem_step_theory_modularity_lifting
    {K : Type u} [h : NumberStruct_theory_modularity_lifting K]
    (x : K) :
    ∃ d : Nat, d = h.density x + h.rep x ∧ h.selmer x ≤ d := by
  refine ⟨h.density x + h.rep x, rfl, ?_⟩
  exact h.control_axiom x

theorem hodge_filtration_step_theory_modularity_lifting
    {K : Type u} [h : NumberStruct_theory_modularity_lifting K]
    (x : K) :
    h.height x ≤ h.selmer x + h.density x ∧ h.height x ≤ h.rep x + h.density x := by
  have hHodge : h.height x ≤ h.selmer x + h.density x := h.hodge_axiom x
  have hHeight : h.height x ≤ h.rep x + h.density x := h.height_axiom x
  exact ⟨hHodge, hHeight⟩

theorem height_inequality_step_theory_modularity_lifting
    {K : Type u} [h : NumberStruct_theory_modularity_lifting K]
    (x : K) :
    (h.height x ≤ h.rep x + h.density x) → h.height x ≤ h.rep x + h.density x := by
  intro hIn
  have hOut : h.height x ≤ h.rep x + h.density x := hIn
  exact hOut

theorem chebotarev_count_step_theory_modularity_lifting
    {K : Type u} [h : NumberStruct_theory_modularity_lifting K]
    (x : K) :
    ∃ d : Nat, d = h.density x ∧ d ≤ h.rep x + h.selmer x := by
  have hCheb : h.density x ≤ h.rep x + h.selmer x := h.chebotarev_axiom x
  exact ⟨h.density x, rfl, hCheb⟩

theorem arithmetic_transfer_theory_modularity_lifting
    {K : Type u} [h : NumberStruct_theory_modularity_lifting K]
    (x : K) :
    (h.rep x ≤ h.selmer x + h.height x) →
      h.rep x + h.height x ≤ h.selmer x + h.density x + h.height x := by
  intro hLiftIn
  have hLift : h.rep x ≤ h.selmer x + h.height x := hLiftIn
  have hChebPack :
      ∃ d : Nat, d = h.density x ∧ d ≤ h.rep x + h.selmer x :=
    chebotarev_count_step_theory_modularity_lifting (K := K) x
  rcases hChebPack with ⟨d, hdEq, hdBound⟩
  have hCheb : h.density x ≤ h.rep x + h.selmer x := by
    rw [← hdEq]
    exact hdBound
  have hTransfer : h.rep x + h.height x ≤ h.selmer x + h.density x + h.height x :=
    h.arithmetic_transfer_axiom x
  have hControlPack : ∃ d' : Nat, d' = h.density x + h.rep x ∧ h.selmer x ≤ d' :=
    control_theorem_step_theory_modularity_lifting (K := K) x
  rcases hControlPack with ⟨d', hd'Eq, hSelmer⟩
  have _ : h.rep x ≤ h.selmer x + h.height x := hLift
  have _ : h.density x ≤ h.rep x + h.selmer x := hCheb
  have _ : h.selmer x ≤ h.density x + h.rep x := by
    rw [hd'Eq] at hSelmer
    exact hSelmer
  exact hTransfer

theorem global_finiteness_theory_modularity_lifting
    {K : Type u} [h : NumberStruct_theory_modularity_lifting K]
    (x : K) :
    ∃ n : Nat,
      h.rep x + h.selmer x + h.height x ≤ n ∧
      n = h.density x + h.density x + h.rep x + h.selmer x := by
  let n := h.density x + h.density x + h.rep x + h.selmer x
  have hGlobal : h.rep x + h.selmer x + h.height x ≤ n := by
    simpa [n] using h.global_finiteness_axiom x
  have hTransfer :
      (h.rep x ≤ h.selmer x + h.height x) →
        h.rep x + h.height x ≤ h.selmer x + h.density x + h.height x :=
    arithmetic_transfer_theory_modularity_lifting (K := K) x
  have hLiftPack :
      ∃ r : Nat, r = h.rep x ∧ r ≤ h.selmer x + h.height x :=
    modular_lift_step_theory_modularity_lifting (K := K) x
  rcases hLiftPack with ⟨r, hrEq, hrBound⟩
  have hLift : h.rep x ≤ h.selmer x + h.height x := by
    rw [← hrEq]
    exact hrBound
  have _ : h.rep x + h.height x ≤ h.selmer x + h.density x + h.height x := hTransfer hLift
  have hChebPack :
      ∃ d : Nat, d = h.density x ∧ d ≤ h.rep x + h.selmer x :=
    chebotarev_count_step_theory_modularity_lifting (K := K) x
  rcases hChebPack with ⟨d, hdEq, hdBound⟩
  have _ : h.density x ≤ h.rep x + h.selmer x := by
    rw [← hdEq]
    exact hdBound
  exact ⟨n, hGlobal, rfl⟩
