/-
BENCHMARK_ID: TINY_MATHLIB_BATCH05_NUMBER_THEORY_ARAKELOV_HEIGHT_INEQUALITY_LIKE
PAIR_STEM: number_theory_arakelov_height_inequality_like
MATH_DOMAIN: Number Theory
SOURCE_MATHLIB: Mathlib/NumberTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class NumberStruct_theory_arakelov_height (K : Type u) where
  rep : K → Nat
  selmer : K → Nat
  height : K → Nat
  density : K → Nat
  nat_le_trans : ∀ a b c : Nat, a ≤ b → b ≤ c → a ≤ c
  nat_add_mono : ∀ a b c d : Nat, a ≤ b → c ≤ d → a + c ≤ b + d
  nat_le_refl : ∀ a : Nat, a ≤ a
  modular_lift_axiom :
    ∀ x : K,
      rep x + rep x ≤ selmer x + height x + density x
  control_axiom :
    ∀ x : K,
      selmer x + height x ≤ density x + rep x + height x
  hodge_axiom :
    ∀ x : K,
      height x + selmer x ≤ selmer x + density x + selmer x
  height_axiom :
    ∀ x : K,
      height x + rep x ≤ density x + selmer x + rep x
  chebotarev_axiom :
    ∀ x : K,
      density x + height x ≤ rep x + selmer x + height x
  arithmetic_transfer_axiom :
    ∀ x : K,
      rep x + height x ≤ selmer x + density x + height x
  global_finiteness_axiom :
    ∀ x : K,
      rep x + selmer x + height x ≤ density x + density x + rep x + selmer x

def GaloisRep_theory_arakelov_height (K : Type u) : Type u :=
  K → Nat

def SelmerObj_theory_arakelov_height (K : Type u) : Type u :=
  K → Nat

def HeightObj_theory_arakelov_height (K : Type u) : Type u :=
  K → Nat

def DensityObj_theory_arakelov_height (K : Type u) : Type u :=
  K → Nat

theorem modular_lift_step_theory_arakelov_height
    {K : Type u} [h : NumberStruct_theory_arakelov_height K]
    (x : K) :
    ∃ b : Nat, b = h.selmer x + h.height x + h.density x ∧ h.rep x + h.rep x ≤ b := by
  refine ⟨h.selmer x + h.height x + h.density x, rfl, ?_⟩
  exact h.modular_lift_axiom x

theorem control_theorem_step_theory_arakelov_height
    {K : Type u} [h : NumberStruct_theory_arakelov_height K]
    (x : K) :
    h.selmer x + h.height x ≤ h.density x + h.rep x + h.height x ∧
    h.selmer x + h.height x ≤ h.density x + h.rep x + h.height x := by
  have hCtrl : h.selmer x + h.height x ≤ h.density x + h.rep x + h.height x :=
    h.control_axiom x
  constructor
  · exact hCtrl
  · exact hCtrl

theorem hodge_filtration_step_theory_arakelov_height
    {K : Type u} [h : NumberStruct_theory_arakelov_height K]
    (x : K) :
    (h.selmer x + h.height x ≤ h.density x + h.rep x + h.height x) →
    h.height x + h.selmer x ≤ h.selmer x + h.density x + h.selmer x := by
  intro hcontrol
  have hUse : h.selmer x + h.height x ≤ h.density x + h.rep x + h.height x := hcontrol
  have hHodge : h.height x + h.selmer x ≤ h.selmer x + h.density x + h.selmer x :=
    h.hodge_axiom x
  have _ : h.selmer x + h.height x ≤ h.density x + h.rep x + h.height x := hUse
  exact hHodge

theorem height_inequality_step_theory_arakelov_height
    {K : Type u} [h : NumberStruct_theory_arakelov_height K]
    (x : K) :
    ∃ t : Nat, t = h.density x + h.selmer x + h.rep x ∧ h.height x + h.rep x ≤ t := by
  refine ⟨h.density x + h.selmer x + h.rep x, rfl, ?_⟩
  exact h.height_axiom x

theorem chebotarev_count_step_theory_arakelov_height
    {K : Type u} [h : NumberStruct_theory_arakelov_height K]
    (x : K) :
    h.density x + h.height x ≤ h.rep x + h.selmer x + h.height x ∧
    h.rep x + h.rep x ≤ h.selmer x + h.height x + h.density x := by
  have hCheb : h.density x + h.height x ≤ h.rep x + h.selmer x + h.height x :=
    h.chebotarev_axiom x
  rcases modular_lift_step_theory_arakelov_height (K := K) x with ⟨b, hbEq, hbLe⟩
  have hLift : h.rep x + h.rep x ≤ h.selmer x + h.height x + h.density x := by
    rw [hbEq] at hbLe
    exact hbLe
  exact ⟨hCheb, hLift⟩

theorem arithmetic_transfer_theory_arakelov_height
    {K : Type u} [h : NumberStruct_theory_arakelov_height K]
    (x : K) :
    ∃ t : Nat, t = h.selmer x + h.density x + h.height x ∧ h.rep x + h.height x ≤ t := by
  have hBase : h.rep x + h.height x ≤ h.selmer x + h.density x + h.height x :=
    h.arithmetic_transfer_axiom x
  have hSelf : h.rep x + h.height x ≤ h.rep x + h.height x :=
    h.nat_le_refl (h.rep x + h.height x)
  have hChain : h.rep x + h.height x ≤ h.selmer x + h.density x + h.height x :=
    h.nat_le_trans (h.rep x + h.height x) (h.rep x + h.height x)
      (h.selmer x + h.density x + h.height x) hSelf hBase
  refine ⟨h.selmer x + h.density x + h.height x, rfl, hChain⟩

theorem global_finiteness_theory_arakelov_height
    {K : Type u} [h : NumberStruct_theory_arakelov_height K]
    (x : K) :
    h.rep x + h.selmer x + h.height x ≤
      h.density x + h.density x + h.rep x + h.selmer x := by
  have hGlobal :
      h.rep x + h.selmer x + h.height x ≤
        h.density x + h.density x + h.rep x + h.selmer x :=
    h.global_finiteness_axiom x
  have hAux :
      ∃ t : Nat, t = h.selmer x + h.density x + h.height x ∧ h.rep x + h.height x ≤ t :=
    arithmetic_transfer_theory_arakelov_height (K := K) x
  rcases hAux with ⟨t, ht, hle⟩
  have _ : h.rep x + h.height x ≤ t := hle
  have _ : t = h.selmer x + h.density x + h.height x := ht
  exact hGlobal
