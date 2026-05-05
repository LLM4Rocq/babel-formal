/-
BENCHMARK_ID: TINY_MATHLIB_BATCH06_ANALYSIS_HORMANDER_L2_ESTIMATE_LIKE
PAIR_STEM: analysis_hormander_l2_estimate_like
MATH_DOMAIN: Analysis
SOURCE_MATHLIB: Mathlib/Analysis/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class HormanderStruct_l2_estimate (E : Type u) where
  l2norm : E → Nat
  rhs : E → Nat
  curvature : E → Nat
  error : E → Nat
  adjoint : E → E
  solver : E → E
  le_trans_nat : ∀ a b c : Nat, a ≤ b → b ≤ c → a ≤ c
  add_le_add_left_nat : ∀ a b c : Nat, a ≤ b → c + a ≤ c + b
  add_le_add_right_nat : ∀ a b c : Nat, a ≤ b → a + c ≤ b + c
  add_le_add_nat : ∀ a b c d : Nat, a ≤ b → c ≤ d → a + c ≤ b + d
  le_add_right_nat : ∀ a b : Nat, a ≤ a + b
  le_add_left_nat : ∀ a b : Nat, b ≤ a + b
  bochner_axiom :
    ∀ x : E, l2norm (adjoint x) + curvature x ≤ l2norm x + rhs x
  weighted_cauchy_axiom :
    ∀ x : E, rhs x ≤ l2norm x + error x
  coercive_axiom :
    ∀ x : E, l2norm (solver x) ≤ rhs x + error x
  weak_solution_axiom :
    ∀ x : E, error (solver x) ≤ error x + curvature x
  minimizer_axiom :
    ∀ x : E, l2norm (adjoint (solver x)) ≤ l2norm (solver x) + curvature x
  final_axiom :
    ∀ x : E, l2norm (solver x) + l2norm (adjoint (solver x)) ≤ rhs x + rhs x + curvature x

structure WeightData_hormander_l2_estimate (E : Type u) [h : HormanderStruct_l2_estimate E] where
  state : E
  weight : Nat
  weight_pos : 0 < weight
  rhs_le_weight : h.rhs state ≤ weight
  error_le_curved : h.error state ≤ h.curvature state + weight

def adjoint_operator_hormander_l2_estimate
    {E : Type u} [h : HormanderStruct_l2_estimate E]
    (d : WeightData_hormander_l2_estimate E) : E :=
  h.adjoint d.state

def curvature_form_hormander_l2_estimate
    {E : Type u} [h : HormanderStruct_l2_estimate E]
    (d : WeightData_hormander_l2_estimate E) : Nat :=
  h.curvature d.state + d.weight

def solution_operator_hormander_l2_estimate
    {E : Type u} [h : HormanderStruct_l2_estimate E]
    (d : WeightData_hormander_l2_estimate E) : E :=
  h.solver d.state

theorem bochner_identity_step_hormander_l2_estimate
    {E : Type u} [h : HormanderStruct_l2_estimate E]
    (d : WeightData_hormander_l2_estimate E) :
    h.l2norm (adjoint_operator_hormander_l2_estimate (E := E) d) + h.curvature d.state ≤
      h.l2norm d.state + d.weight ∧
    h.l2norm (adjoint_operator_hormander_l2_estimate (E := E) d) + h.curvature d.state ≤
      h.l2norm d.state + curvature_form_hormander_l2_estimate (E := E) d := by
  have hBochner :
      h.l2norm (h.adjoint d.state) + h.curvature d.state ≤ h.l2norm d.state + h.rhs d.state :=
    h.bochner_axiom d.state
  have hRhsLift :
      h.l2norm d.state + h.rhs d.state ≤ h.l2norm d.state + d.weight :=
    h.add_le_add_left_nat (h.rhs d.state) d.weight (h.l2norm d.state) d.rhs_le_weight
  have hFirstRaw :
      h.l2norm (h.adjoint d.state) + h.curvature d.state ≤ h.l2norm d.state + d.weight :=
    h.le_trans_nat _ _ _ hBochner hRhsLift
  have hWeightLift : d.weight ≤ h.curvature d.state + d.weight :=
    h.le_add_left_nat (h.curvature d.state) d.weight
  have hSecondLift :
      h.l2norm d.state + d.weight ≤ h.l2norm d.state + (h.curvature d.state + d.weight) :=
    h.add_le_add_left_nat d.weight (h.curvature d.state + d.weight) (h.l2norm d.state) hWeightLift
  have hSecondRaw :
      h.l2norm (h.adjoint d.state) + h.curvature d.state ≤
        h.l2norm d.state + (h.curvature d.state + d.weight) :=
    h.le_trans_nat _ _ _ hFirstRaw hSecondLift
  have hFirst :
      h.l2norm (adjoint_operator_hormander_l2_estimate (E := E) d) + h.curvature d.state ≤
        h.l2norm d.state + d.weight := by
    simpa [adjoint_operator_hormander_l2_estimate] using hFirstRaw
  have hSecond :
      h.l2norm (adjoint_operator_hormander_l2_estimate (E := E) d) + h.curvature d.state ≤
        h.l2norm d.state + curvature_form_hormander_l2_estimate (E := E) d := by
    simpa [adjoint_operator_hormander_l2_estimate, curvature_form_hormander_l2_estimate] using hSecondRaw
  exact And.intro hFirst hSecond

theorem positivity_curvature_step_hormander_l2_estimate
    {E : Type u} [h : HormanderStruct_l2_estimate E]
    (d : WeightData_hormander_l2_estimate E) :
    h.rhs d.state ≤ curvature_form_hormander_l2_estimate (E := E) d ∧
    h.error d.state ≤ curvature_form_hormander_l2_estimate (E := E) d + h.curvature d.state := by
  have hWeightLift : d.weight ≤ h.curvature d.state + d.weight :=
    h.le_add_left_nat (h.curvature d.state) d.weight
  have hRhsCurved : h.rhs d.state ≤ h.curvature d.state + d.weight :=
    h.le_trans_nat _ _ _ d.rhs_le_weight hWeightLift
  have hFirst : h.rhs d.state ≤ curvature_form_hormander_l2_estimate (E := E) d := by
    simpa [curvature_form_hormander_l2_estimate] using hRhsCurved
  have hErrBase : h.error d.state ≤ h.curvature d.state + d.weight :=
    d.error_le_curved
  have hErrLift :
      h.curvature d.state + d.weight ≤ (h.curvature d.state + d.weight) + h.curvature d.state :=
    h.le_add_right_nat (h.curvature d.state + d.weight) (h.curvature d.state)
  have hErrRaw :
      h.error d.state ≤ (h.curvature d.state + d.weight) + h.curvature d.state :=
    h.le_trans_nat _ _ _ hErrBase hErrLift
  have hSecond :
      h.error d.state ≤ curvature_form_hormander_l2_estimate (E := E) d + h.curvature d.state := by
    simpa [curvature_form_hormander_l2_estimate] using hErrRaw
  exact And.intro hFirst hSecond

theorem weighted_cauchy_step_hormander_l2_estimate
    {E : Type u} [h : HormanderStruct_l2_estimate E]
    (d : WeightData_hormander_l2_estimate E) :
    h.rhs d.state ≤ h.l2norm d.state + h.error d.state ∧
    h.rhs d.state ≤ h.l2norm d.state + (curvature_form_hormander_l2_estimate (E := E) d + h.error d.state) := by
  have hBase : h.rhs d.state ≤ h.l2norm d.state + h.error d.state :=
    h.weighted_cauchy_axiom d.state
  have hErrLift :
      h.error d.state ≤ curvature_form_hormander_l2_estimate (E := E) d + h.error d.state :=
    h.le_add_left_nat (curvature_form_hormander_l2_estimate (E := E) d) (h.error d.state)
  have hRightLift :
      h.l2norm d.state + h.error d.state ≤
        h.l2norm d.state + (curvature_form_hormander_l2_estimate (E := E) d + h.error d.state) :=
    h.add_le_add_left_nat
      (h.error d.state)
      (curvature_form_hormander_l2_estimate (E := E) d + h.error d.state)
      (h.l2norm d.state)
      hErrLift
  have hSecond :
      h.rhs d.state ≤ h.l2norm d.state + (curvature_form_hormander_l2_estimate (E := E) d + h.error d.state) :=
    h.le_trans_nat _ _ _ hBase hRightLift
  exact And.intro hBase hSecond

theorem coercivity_chain_hormander_l2_estimate
    {E : Type u} [h : HormanderStruct_l2_estimate E]
    (d : WeightData_hormander_l2_estimate E) :
    h.l2norm (solution_operator_hormander_l2_estimate (E := E) d) ≤ d.weight + h.error d.state ∧
    h.l2norm (solution_operator_hormander_l2_estimate (E := E) d) ≤
      (d.weight + h.error d.state) + h.error (solution_operator_hormander_l2_estimate (E := E) d) := by
  have hCoercive : h.l2norm (h.solver d.state) ≤ h.rhs d.state + h.error d.state :=
    h.coercive_axiom d.state
  have hWeightShift :
      h.rhs d.state + h.error d.state ≤ d.weight + h.error d.state :=
    h.add_le_add_right_nat (h.rhs d.state) d.weight (h.error d.state) d.rhs_le_weight
  have hFirstRaw : h.l2norm (h.solver d.state) ≤ d.weight + h.error d.state :=
    h.le_trans_nat _ _ _ hCoercive hWeightShift
  have hSecondLift :
      d.weight + h.error d.state ≤ (d.weight + h.error d.state) + h.error (h.solver d.state) :=
    h.le_add_right_nat (d.weight + h.error d.state) (h.error (h.solver d.state))
  have hSecondRaw :
      h.l2norm (h.solver d.state) ≤ (d.weight + h.error d.state) + h.error (h.solver d.state) :=
    h.le_trans_nat _ _ _ hFirstRaw hSecondLift
  have hFirst :
      h.l2norm (solution_operator_hormander_l2_estimate (E := E) d) ≤ d.weight + h.error d.state := by
    simpa [solution_operator_hormander_l2_estimate] using hFirstRaw
  have hSecond :
      h.l2norm (solution_operator_hormander_l2_estimate (E := E) d) ≤
        (d.weight + h.error d.state) + h.error (solution_operator_hormander_l2_estimate (E := E) d) := by
    simpa [solution_operator_hormander_l2_estimate] using hSecondRaw
  exact And.intro hFirst hSecond

theorem weak_solution_exists_hormander_l2_estimate
    {E : Type u} [h : HormanderStruct_l2_estimate E]
    (d : WeightData_hormander_l2_estimate E) :
    ∃ u : E, u = solution_operator_hormander_l2_estimate (E := E) d ∧
      h.error u ≤ h.error d.state + h.curvature d.state := by
  have hWeakRaw : h.error (h.solver d.state) ≤ h.error d.state + h.curvature d.state :=
    h.weak_solution_axiom d.state
  have hWeak :
      h.error (solution_operator_hormander_l2_estimate (E := E) d) ≤
        h.error d.state + h.curvature d.state := by
    simpa [solution_operator_hormander_l2_estimate] using hWeakRaw
  refine ⟨solution_operator_hormander_l2_estimate (E := E) d, ?_, ?_⟩
  · rfl
  · exact hWeak

theorem minimizer_characterization_hormander_l2_estimate
    {E : Type u} [h : HormanderStruct_l2_estimate E]
    (d : WeightData_hormander_l2_estimate E) :
    h.l2norm (h.adjoint (solution_operator_hormander_l2_estimate (E := E) d)) ≤
      h.l2norm (solution_operator_hormander_l2_estimate (E := E) d) + h.curvature d.state ∧
    h.l2norm (h.adjoint (solution_operator_hormander_l2_estimate (E := E) d)) ≤
      h.l2norm (solution_operator_hormander_l2_estimate (E := E) d) +
        curvature_form_hormander_l2_estimate (E := E) d := by
  have hMinRaw :
      h.l2norm (h.adjoint (h.solver d.state)) ≤ h.l2norm (h.solver d.state) + h.curvature d.state :=
    h.minimizer_axiom d.state
  have hCurvLift : h.curvature d.state ≤ h.curvature d.state + d.weight :=
    h.le_add_right_nat (h.curvature d.state) d.weight
  have hSecondLift :
      h.l2norm (h.solver d.state) + h.curvature d.state ≤
        h.l2norm (h.solver d.state) + (h.curvature d.state + d.weight) :=
    h.add_le_add_left_nat
      (h.curvature d.state)
      (h.curvature d.state + d.weight)
      (h.l2norm (h.solver d.state))
      hCurvLift
  have hSecondRaw :
      h.l2norm (h.adjoint (h.solver d.state)) ≤
        h.l2norm (h.solver d.state) + (h.curvature d.state + d.weight) :=
    h.le_trans_nat _ _ _ hMinRaw hSecondLift
  have hFirst :
      h.l2norm (h.adjoint (solution_operator_hormander_l2_estimate (E := E) d)) ≤
        h.l2norm (solution_operator_hormander_l2_estimate (E := E) d) + h.curvature d.state := by
    simpa [solution_operator_hormander_l2_estimate] using hMinRaw
  have hSecond :
      h.l2norm (h.adjoint (solution_operator_hormander_l2_estimate (E := E) d)) ≤
        h.l2norm (solution_operator_hormander_l2_estimate (E := E) d) +
          curvature_form_hormander_l2_estimate (E := E) d := by
    simpa [solution_operator_hormander_l2_estimate, curvature_form_hormander_l2_estimate] using hSecondRaw
  exact And.intro hFirst hSecond

theorem l2_estimate_final_hormander_l2_estimate
    {E : Type u} [h : HormanderStruct_l2_estimate E]
    (d : WeightData_hormander_l2_estimate E) :
    ∃ u : E, u = solution_operator_hormander_l2_estimate (E := E) d ∧
      h.l2norm u + h.l2norm (h.adjoint u) ≤
        d.weight + d.weight + h.curvature d.state + h.error d.state := by
  have hFinalRaw :
      h.l2norm (h.solver d.state) + h.l2norm (h.adjoint (h.solver d.state)) ≤
        h.rhs d.state + h.rhs d.state + h.curvature d.state :=
    h.final_axiom d.state
  have hRhsDouble : h.rhs d.state + h.rhs d.state ≤ d.weight + d.weight :=
    h.add_le_add_nat
      (h.rhs d.state) d.weight
      (h.rhs d.state) d.weight
      d.rhs_le_weight d.rhs_le_weight
  have hCurvLift :
      h.rhs d.state + h.rhs d.state + h.curvature d.state ≤
        (d.weight + d.weight) + h.curvature d.state :=
    h.add_le_add_right_nat
      (h.rhs d.state + h.rhs d.state)
      (d.weight + d.weight)
      (h.curvature d.state)
      hRhsDouble
  have hMain :
      h.l2norm (h.solver d.state) + h.l2norm (h.adjoint (h.solver d.state)) ≤
        (d.weight + d.weight) + h.curvature d.state :=
    h.le_trans_nat _ _ _ hFinalRaw hCurvLift
  have hWithError :
      (d.weight + d.weight) + h.curvature d.state ≤
        ((d.weight + d.weight) + h.curvature d.state) + h.error d.state :=
    h.le_add_right_nat ((d.weight + d.weight) + h.curvature d.state) (h.error d.state)
  have hBound :
      h.l2norm (h.solver d.state) + h.l2norm (h.adjoint (h.solver d.state)) ≤
        ((d.weight + d.weight) + h.curvature d.state) + h.error d.state :=
    h.le_trans_nat _ _ _ hMain hWithError
  refine ⟨solution_operator_hormander_l2_estimate (E := E) d, ?_, ?_⟩
  · rfl
  · simpa [solution_operator_hormander_l2_estimate] using hBound
