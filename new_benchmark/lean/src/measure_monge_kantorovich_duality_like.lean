/-
BENCHMARK_ID: TINY_MATHLIB_BATCH06_MEASURE_MONGE_KANTOROVICH_DUALITY_LIKE
PAIR_STEM: measure_monge_kantorovich_duality_like
MATH_DOMAIN: Measure Theory
SOURCE_MATHLIB: Mathlib/MeasureTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u v w

class TransportStruct_monge_kantorovich_duality
    (X : Type u) (Y : Type v) (C : Type w) where
  feasible : (X → Y → Prop) → Prop
  dual_admissible : (X → Prop) → (Y → Prop) → Prop
  primal_value : (X → Y → Prop) → C
  dual_value : (X → Prop) → (Y → Prop) → C
  c_transform_raw : (X → Prop) → (Y → Prop)
  le : C → C → Prop
  le_refl : ∀ c : C, le c c
  le_trans : ∀ a b c : C, le a b → le b c → le a c
  primal_bound_axiom :
    ∀ π : X → Y → Prop,
      ∀ φ : X → Prop,
        ∀ ψ : Y → Prop,
          feasible π →
          dual_admissible φ ψ →
          le (dual_value φ ψ) (primal_value π)
  c_transform_admissible_axiom :
    ∀ φ : X → Prop,
      dual_admissible φ (c_transform_raw φ)
  c_transform_dominates_axiom :
    ∀ φ : X → Prop,
      ∀ ψ : Y → Prop,
        dual_admissible φ ψ →
        le (dual_value φ ψ) (dual_value φ (c_transform_raw φ))
  tightness_axiom :
    ∀ π : X → Y → Prop,
      feasible π →
      ∃ πc : X → Y → Prop,
        feasible πc ∧
        le (primal_value πc) (primal_value π)
  minimax_axiom :
    ∀ φ : X → Prop,
      dual_admissible φ (c_transform_raw φ) →
      ∃ π : X → Y → Prop,
        feasible π ∧
        le (dual_value φ (c_transform_raw φ)) (primal_value π)
  optimal_plan_axiom :
    ∀ φ : X → Prop,
      dual_admissible φ (c_transform_raw φ) →
      ∃ π : X → Y → Prop,
        feasible π ∧
        le (primal_value π) (dual_value φ (c_transform_raw φ))
  slackness_axiom :
    ∀ π : X → Y → Prop,
      ∀ φ : X → Prop,
        feasible π →
        dual_admissible φ (c_transform_raw φ) →
        le (dual_value φ (c_transform_raw φ)) (primal_value π) →
        le (primal_value π) (dual_value φ (c_transform_raw φ))

structure CouplingData_monge_kantorovich_duality
    {X : Type u} {Y : Type v} {C : Type w}
    [h : TransportStruct_monge_kantorovich_duality X Y C] where
  plan : X → Y → Prop
  plan_feasible : h.feasible plan
  left_support : X → Prop
  right_support : Y → Prop
  support_compatible :
    ∀ x : X,
      ∀ y : Y,
        plan x y →
        left_support x ∧ right_support y

def primal_cost_monge_kantorovich_duality
    {X : Type u} {Y : Type v} {C : Type w}
    [h : TransportStruct_monge_kantorovich_duality X Y C]
    (Γ : CouplingData_monge_kantorovich_duality (X := X) (Y := Y) (C := C)) : C :=
  h.primal_value Γ.plan

def dual_potential_monge_kantorovich_duality
    {X : Type u} {Y : Type v} {C : Type w}
    [h : TransportStruct_monge_kantorovich_duality X Y C]
    (φ : X → Prop) (ψ : Y → Prop) : C :=
  h.dual_value φ ψ

def c_transform_monge_kantorovich_duality
    {X : Type u} {Y : Type v} {C : Type w}
    [h : TransportStruct_monge_kantorovich_duality X Y C]
    (φ : X → Prop) : Y → Prop :=
  h.c_transform_raw φ

theorem primal_bound_dual_monge_kantorovich_duality
    {X : Type u} {Y : Type v} {C : Type w}
    [h : TransportStruct_monge_kantorovich_duality X Y C]
    (Γ : CouplingData_monge_kantorovich_duality (X := X) (Y := Y) (C := C))
    (φ : X → Prop) (ψ : Y → Prop)
    (hDual : h.dual_admissible φ ψ) :
    h.le
      (dual_potential_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ ψ)
      (primal_cost_monge_kantorovich_duality (X := X) (Y := Y) (C := C) Γ) := by
  have hFeasiblePlan : h.feasible Γ.plan := Γ.plan_feasible
  have hSupportFrame :
      ∀ x : X, ∀ y : Y, Γ.plan x y → Γ.left_support x ∧ Γ.right_support y :=
    Γ.support_compatible
  have hRawBound : h.le (h.dual_value φ ψ) (h.primal_value Γ.plan) :=
    h.primal_bound_axiom Γ.plan φ ψ hFeasiblePlan hDual
  have hLeftProjection : ∀ x : X, ∀ y : Y, Γ.plan x y → Γ.left_support x := by
    intro x y hxy
    exact (hSupportFrame x y hxy).1
  have _ : ∀ x : X, ∀ y : Y, Γ.plan x y → Γ.left_support x := hLeftProjection
  have hDualAlias :
      dual_potential_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ ψ =
        h.dual_value φ ψ := rfl
  have hPrimalAlias :
      primal_cost_monge_kantorovich_duality (X := X) (Y := Y) (C := C) Γ =
        h.primal_value Γ.plan := rfl
  have _ : h.dual_value φ ψ = h.dual_value φ ψ := rfl
  simpa [dual_potential_monge_kantorovich_duality, primal_cost_monge_kantorovich_duality] using hRawBound

theorem dual_admissible_closure_monge_kantorovich_duality
    {X : Type u} {Y : Type v} {C : Type w}
    [h : TransportStruct_monge_kantorovich_duality X Y C]
    (φ : X → Prop) (ψ : Y → Prop)
    (hDual : h.dual_admissible φ ψ) :
    h.dual_admissible φ (c_transform_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ) ∧
    h.le
      (dual_potential_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ ψ)
      (dual_potential_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ
        (c_transform_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ)) := by
  have hClosure :
      h.dual_admissible φ
        (c_transform_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ) :=
    h.c_transform_admissible_axiom φ
  have hDomination :
      h.le (h.dual_value φ ψ) (h.dual_value φ (h.c_transform_raw φ)) :=
    h.c_transform_dominates_axiom φ ψ hDual
  have hRawRight :
      h.le
        (dual_potential_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ ψ)
        (dual_potential_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ
          (c_transform_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ)) := by
    simpa [dual_potential_monge_kantorovich_duality, c_transform_monge_kantorovich_duality] using hDomination
  have hRawLeft :
      h.dual_admissible φ
        (c_transform_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ) := hClosure
  exact And.intro hRawLeft hRawRight

theorem tightness_compactness_step_monge_kantorovich_duality
    {X : Type u} {Y : Type v} {C : Type w}
    [h : TransportStruct_monge_kantorovich_duality X Y C]
    (Γ : CouplingData_monge_kantorovich_duality (X := X) (Y := Y) (C := C))
    (φ : X → Prop) (ψ : Y → Prop)
    (hDual : h.dual_admissible φ ψ) :
    ∃ Γc : CouplingData_monge_kantorovich_duality (X := X) (Y := Y) (C := C),
      h.le
        (primal_cost_monge_kantorovich_duality (X := X) (Y := Y) (C := C) Γc)
        (primal_cost_monge_kantorovich_duality (X := X) (Y := Y) (C := C) Γ) ∧
      h.le
        (dual_potential_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ ψ)
        (primal_cost_monge_kantorovich_duality (X := X) (Y := Y) (C := C) Γc) := by
  have hTight :
      ∃ πc : X → Y → Prop,
        h.feasible πc ∧
        h.le (h.primal_value πc) (h.primal_value Γ.plan) :=
    h.tightness_axiom Γ.plan Γ.plan_feasible
  rcases hTight with ⟨πc, hFeasibleC, hCostCompare⟩
  let Γc : CouplingData_monge_kantorovich_duality (X := X) (Y := Y) (C := C) :=
    { plan := πc
      plan_feasible := hFeasibleC
      left_support := fun _ => True
      right_support := fun _ => True
      support_compatible := by
        intro x y hxy
        exact And.intro trivial trivial }
  have hDualBoundRaw : h.le (h.dual_value φ ψ) (h.primal_value πc) :=
    h.primal_bound_axiom πc φ ψ hFeasibleC hDual
  have hDualBound :
      h.le
        (dual_potential_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ ψ)
        (primal_cost_monge_kantorovich_duality (X := X) (Y := Y) (C := C) Γc) := by
    simpa [dual_potential_monge_kantorovich_duality, primal_cost_monge_kantorovich_duality, Γc] using hDualBoundRaw
  have hPrimalCompare :
      h.le
        (primal_cost_monge_kantorovich_duality (X := X) (Y := Y) (C := C) Γc)
        (primal_cost_monge_kantorovich_duality (X := X) (Y := Y) (C := C) Γ) := by
    simpa [primal_cost_monge_kantorovich_duality, Γc] using hCostCompare
  have hPack :
      h.le
        (primal_cost_monge_kantorovich_duality (X := X) (Y := Y) (C := C) Γc)
        (primal_cost_monge_kantorovich_duality (X := X) (Y := Y) (C := C) Γ) ∧
      h.le
        (dual_potential_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ ψ)
        (primal_cost_monge_kantorovich_duality (X := X) (Y := Y) (C := C) Γc) :=
    And.intro hPrimalCompare hDualBound
  exact ⟨Γc, hPack⟩

theorem minimax_exchange_step_monge_kantorovich_duality
    {X : Type u} {Y : Type v} {C : Type w}
    [h : TransportStruct_monge_kantorovich_duality X Y C]
    (φ : X → Prop) (ψ : Y → Prop)
    (hDual : h.dual_admissible φ ψ) :
    ∃ Γ : CouplingData_monge_kantorovich_duality (X := X) (Y := Y) (C := C),
      h.le
        (dual_potential_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ ψ)
        (primal_cost_monge_kantorovich_duality (X := X) (Y := Y) (C := C) Γ) ∧
      h.le
        (dual_potential_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ
          (c_transform_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ))
        (primal_cost_monge_kantorovich_duality (X := X) (Y := Y) (C := C) Γ) := by
  have hCTAdmissible :
      h.dual_admissible φ
        (c_transform_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ) :=
    h.c_transform_admissible_axiom φ
  have hMinimaxRaw :
      ∃ π : X → Y → Prop,
        h.feasible π ∧
        h.le (h.dual_value φ (h.c_transform_raw φ)) (h.primal_value π) :=
    h.minimax_axiom φ hCTAdmissible
  rcases hMinimaxRaw with ⟨π, hπFeasible, hCTToPrimal⟩
  have hDualToCT : h.le (h.dual_value φ ψ) (h.dual_value φ (h.c_transform_raw φ)) :=
    h.c_transform_dominates_axiom φ ψ hDual
  have hDualToPrimal : h.le (h.dual_value φ ψ) (h.primal_value π) :=
    h.le_trans (h.dual_value φ ψ) (h.dual_value φ (h.c_transform_raw φ)) (h.primal_value π)
      hDualToCT hCTToPrimal
  let Γ : CouplingData_monge_kantorovich_duality (X := X) (Y := Y) (C := C) :=
    { plan := π
      plan_feasible := hπFeasible
      left_support := fun _ => True
      right_support := fun _ => True
      support_compatible := by
        intro x y hxy
        exact And.intro trivial trivial }
  have hFirst :
      h.le
        (dual_potential_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ ψ)
        (primal_cost_monge_kantorovich_duality (X := X) (Y := Y) (C := C) Γ) := by
    simpa [dual_potential_monge_kantorovich_duality, primal_cost_monge_kantorovich_duality, Γ] using hDualToPrimal
  have hSecond :
      h.le
        (dual_potential_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ
          (c_transform_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ))
        (primal_cost_monge_kantorovich_duality (X := X) (Y := Y) (C := C) Γ) := by
    simpa [dual_potential_monge_kantorovich_duality, c_transform_monge_kantorovich_duality,
      primal_cost_monge_kantorovich_duality, Γ] using hCTToPrimal
  exact ⟨Γ, And.intro hFirst hSecond⟩

theorem optimal_plan_existence_monge_kantorovich_duality
    {X : Type u} {Y : Type v} {C : Type w}
    [h : TransportStruct_monge_kantorovich_duality X Y C]
    (φ : X → Prop)
    (hCTAdmissible :
      h.dual_admissible φ
        (c_transform_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ)) :
    ∃ Γ : CouplingData_monge_kantorovich_duality (X := X) (Y := Y) (C := C),
      h.le
        (primal_cost_monge_kantorovich_duality (X := X) (Y := Y) (C := C) Γ)
        (dual_potential_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ
          (c_transform_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ)) := by
  have hRaw :
      ∃ π : X → Y → Prop,
        h.feasible π ∧
        h.le (h.primal_value π) (h.dual_value φ (h.c_transform_raw φ)) :=
    h.optimal_plan_axiom φ hCTAdmissible
  rcases hRaw with ⟨π, hπFeasible, hPrimalLeDual⟩
  let Γ : CouplingData_monge_kantorovich_duality (X := X) (Y := Y) (C := C) :=
    { plan := π
      plan_feasible := hπFeasible
      left_support := fun _ => True
      right_support := fun _ => True
      support_compatible := by
        intro x y hxy
        exact And.intro trivial trivial }
  have hPack :
      h.le
        (primal_cost_monge_kantorovich_duality (X := X) (Y := Y) (C := C) Γ)
        (dual_potential_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ
          (c_transform_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ)) := by
    simpa [primal_cost_monge_kantorovich_duality, dual_potential_monge_kantorovich_duality,
      c_transform_monge_kantorovich_duality, Γ] using hPrimalLeDual
  exact ⟨Γ, hPack⟩

theorem complementary_slackness_monge_kantorovich_duality
    {X : Type u} {Y : Type v} {C : Type w}
    [h : TransportStruct_monge_kantorovich_duality X Y C]
    (Γ : CouplingData_monge_kantorovich_duality (X := X) (Y := Y) (C := C))
    (φ : X → Prop)
    (hCTAdmissible :
      h.dual_admissible φ
        (c_transform_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ))
    (hLower :
      h.le
        (dual_potential_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ
          (c_transform_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ))
        (primal_cost_monge_kantorovich_duality (X := X) (Y := Y) (C := C) Γ)) :
    h.le
      (primal_cost_monge_kantorovich_duality (X := X) (Y := Y) (C := C) Γ)
      (dual_potential_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ
        (c_transform_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ)) ∧
    h.le
      (dual_potential_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ
        (c_transform_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ))
      (primal_cost_monge_kantorovich_duality (X := X) (Y := Y) (C := C) Γ) := by
  have hPlanFeasible : h.feasible Γ.plan := Γ.plan_feasible
  have hUpperRaw :
      h.le (h.primal_value Γ.plan) (h.dual_value φ (h.c_transform_raw φ)) :=
    h.slackness_axiom Γ.plan φ hPlanFeasible hCTAdmissible (by
      simpa [dual_potential_monge_kantorovich_duality, c_transform_monge_kantorovich_duality,
        primal_cost_monge_kantorovich_duality] using hLower)
  have hUpper :
      h.le
        (primal_cost_monge_kantorovich_duality (X := X) (Y := Y) (C := C) Γ)
        (dual_potential_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ
          (c_transform_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ)) := by
    simpa [primal_cost_monge_kantorovich_duality, dual_potential_monge_kantorovich_duality,
      c_transform_monge_kantorovich_duality] using hUpperRaw
  have hKeepLower :
      h.le
        (dual_potential_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ
          (c_transform_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ))
        (primal_cost_monge_kantorovich_duality (X := X) (Y := Y) (C := C) Γ) := hLower
  exact And.intro hUpper hKeepLower

theorem strong_duality_monge_kantorovich_duality
    {X : Type u} {Y : Type v} {C : Type w}
    [h : TransportStruct_monge_kantorovich_duality X Y C]
    (φ : X → Prop)
    (hCTAdmissible :
      h.dual_admissible φ
        (c_transform_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ)) :
    ∃ Γ : CouplingData_monge_kantorovich_duality (X := X) (Y := Y) (C := C),
      h.le
        (dual_potential_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ
          (c_transform_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ))
        (primal_cost_monge_kantorovich_duality (X := X) (Y := Y) (C := C) Γ) ∧
      h.le
        (primal_cost_monge_kantorovich_duality (X := X) (Y := Y) (C := C) Γ)
        (dual_potential_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ
          (c_transform_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ)) := by
  rcases optimal_plan_existence_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ hCTAdmissible with
    ⟨Γopt, hPrimalLeDual⟩
  have hDualLePrimal :
      h.le
        (dual_potential_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ
          (c_transform_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ))
        (primal_cost_monge_kantorovich_duality (X := X) (Y := Y) (C := C) Γopt) :=
    primal_bound_dual_monge_kantorovich_duality
      (X := X) (Y := Y) (C := C)
      Γopt φ (c_transform_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ) hCTAdmissible
  have hSlack :
      h.le
        (primal_cost_monge_kantorovich_duality (X := X) (Y := Y) (C := C) Γopt)
        (dual_potential_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ
          (c_transform_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ)) ∧
      h.le
        (dual_potential_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ
          (c_transform_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ))
        (primal_cost_monge_kantorovich_duality (X := X) (Y := Y) (C := C) Γopt) :=
    complementary_slackness_monge_kantorovich_duality
      (X := X) (Y := Y) (C := C) Γopt φ hCTAdmissible hDualLePrimal
  have hUpper :
      h.le
        (primal_cost_monge_kantorovich_duality (X := X) (Y := Y) (C := C) Γopt)
        (dual_potential_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ
          (c_transform_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ)) := hSlack.1
  have hLower :
      h.le
        (dual_potential_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ
          (c_transform_monge_kantorovich_duality (X := X) (Y := Y) (C := C) φ))
        (primal_cost_monge_kantorovich_duality (X := X) (Y := Y) (C := C) Γopt) := hSlack.2
  exact ⟨Γopt, And.intro hLower hUpper⟩
