(*
BENCHMARK_ID: TINY_MATHLIB_BATCH07_MEASURE_MONOTONE_CLASS_STABILITY_ADVANCED_LIKE
PAIR_STEM: measure_monotone_class_stability_advanced_like
MATH_DOMAIN: Measure Theory
SOURCE_MATHLIB: Mathlib/MeasureTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class FrameworkStruct_measure_monotone_class_stability_advanced (X : Type) := {
  lhs_measure : X -> nat;
  rhs_measure : X -> nat;
  aux_measure : X -> nat;
  pivot : X -> X;
  blend : X -> X -> X;
  le_trans_nat : forall a b c : nat, a <= b -> b <= c -> a <= c;
  le_antisymm_nat : forall a b : nat, a <= b -> b <= a -> a = b;
  le_refl_nat : forall a : nat, a <= a;
  add_le_add_left_nat : forall a b c : nat, a <= b -> c + a <= c + b;
  add_le_add_right_nat : forall a b c : nat, a <= b -> a + c <= b + c;
  add_le_add_nat : forall a b c d : nat, a <= b -> c <= d -> a + c <= b + d;
  le_add_right_nat : forall a b : nat, a <= a + b;
  le_add_left_nat : forall a b : nat, b <= a + b;
  pivot_lhs_bound : forall x : X, lhs_measure (pivot x) <= lhs_measure x + rhs_measure x;
  rhs_split_bound : forall x : X, rhs_measure x <= lhs_measure x + aux_measure x;
  blend_upper : forall x y : X, lhs_measure (blend x y) <= lhs_measure x + aux_measure y;
  blend_lower : forall x y : X, lhs_measure x <= lhs_measure (blend x y);
  aux_pivot_bound : forall x : X, aux_measure (pivot x) <= rhs_measure x + aux_measure x;
  rhs_pivot_bound : forall x : X, rhs_measure (pivot x) <= rhs_measure x + aux_measure x
}.

Arguments le_trans_nat {X} {_} _ _ _ _ _.
Arguments le_antisymm_nat {X} {_} _ _ _ _.
Arguments le_refl_nat {X} {_} _.
Arguments add_le_add_left_nat {X} {_} _ _ _ _.
Arguments add_le_add_right_nat {X} {_} _ _ _ _.
Arguments add_le_add_nat {X} {_} _ _ _ _ _ _.
Arguments le_add_right_nat {X} {_} _ _.
Arguments le_add_left_nat {X} {_} _ _.

Record ContextData_measure_monotone_class_stability_advanced (X : Type)
    `{FrameworkStruct_measure_monotone_class_stability_advanced X} := {
  source : X;
  target : X;
  budget : nat;
  source_rhs_le_budget : rhs_measure source <= budget;
  target_aux_le_budget : aux_measure target <= budget;
  coupling_bound : lhs_measure (blend source target) <= lhs_measure source + aux_measure target
}.

Definition primary_map_measure_monotone_class_stability_advanced
    {X : Type} `{FrameworkStruct_measure_monotone_class_stability_advanced X}
    (d : ContextData_measure_monotone_class_stability_advanced) : nat :=
  lhs_measure (source d) + rhs_measure (source d).

Definition secondary_map_measure_monotone_class_stability_advanced
    {X : Type} `{FrameworkStruct_measure_monotone_class_stability_advanced X}
    (d : ContextData_measure_monotone_class_stability_advanced) : nat :=
  lhs_measure (blend (source d) (target d)) + budget d.

Definition tertiary_map_measure_monotone_class_stability_advanced
    {X : Type} `{FrameworkStruct_measure_monotone_class_stability_advanced X}
    (d : ContextData_measure_monotone_class_stability_advanced) : nat :=
  aux_measure (pivot (source d)) + aux_measure (target d).

Lemma stability_step_measure_monotone_class_stability_advanced
    {X : Type} `{FrameworkStruct_measure_monotone_class_stability_advanced X}
    (d : ContextData_measure_monotone_class_stability_advanced) :
    (lhs_measure (source d) <= primary_map_measure_monotone_class_stability_advanced d /\
      rhs_measure (source d) <= primary_map_measure_monotone_class_stability_advanced d) /\
    (lhs_measure (source d) <= primary_map_measure_monotone_class_stability_advanced d /\ (lhs_measure (source d) <= primary_map_measure_monotone_class_stability_advanced d /\ (lhs_measure (source d) <= primary_map_measure_monotone_class_stability_advanced d /\ (lhs_measure (source d) <= primary_map_measure_monotone_class_stability_advanced d /\ (lhs_measure (source d) <= primary_map_measure_monotone_class_stability_advanced d /\ (lhs_measure (source d) <= primary_map_measure_monotone_class_stability_advanced d /\ lhs_measure (source d) <= primary_map_measure_monotone_class_stability_advanced d)))))).
Proof.
  assert (hLeftRaw : lhs_measure (source d) <= lhs_measure (source d) + rhs_measure (source d)).
  { apply (le_add_right_nat (lhs_measure (source d)) (rhs_measure (source d))). }
  assert (hRightRaw : rhs_measure (source d) <= lhs_measure (source d) + rhs_measure (source d)).
  { apply (le_add_left_nat (lhs_measure (source d)) (rhs_measure (source d))). }
  assert (hLeft : lhs_measure (source d) <= primary_map_measure_monotone_class_stability_advanced d).
  { unfold primary_map_measure_monotone_class_stability_advanced. exact hLeftRaw. }
  assert (hRight : rhs_measure (source d) <= primary_map_measure_monotone_class_stability_advanced d).
  { unfold primary_map_measure_monotone_class_stability_advanced. exact hRightRaw. }
  assert (hBase : lhs_measure (source d) <= primary_map_measure_monotone_class_stability_advanced d /\
    rhs_measure (source d) <= primary_map_measure_monotone_class_stability_advanced d).
  { split; [exact hLeft | exact hRight]. }
  assert (hStamp : lhs_measure (source d) <= primary_map_measure_monotone_class_stability_advanced d).
  { exact hLeft. }
  split.
  - exact hBase.
  - exact (conj hStamp (conj hStamp (conj hStamp (conj hStamp (conj hStamp (conj hStamp (hStamp))))))).
Qed.

Lemma factorization_step_measure_monotone_class_stability_advanced
    {X : Type} `{FrameworkStruct_measure_monotone_class_stability_advanced X}
    (d : ContextData_measure_monotone_class_stability_advanced) :
    (lhs_measure (blend (source d) (target d)) <= secondary_map_measure_monotone_class_stability_advanced d /\
      secondary_map_measure_monotone_class_stability_advanced d <= (lhs_measure (source d) + aux_measure (target d)) + budget d) /\
    (secondary_map_measure_monotone_class_stability_advanced d <= (lhs_measure (source d) + aux_measure (target d)) + budget d /\ (secondary_map_measure_monotone_class_stability_advanced d <= (lhs_measure (source d) + aux_measure (target d)) + budget d /\ (secondary_map_measure_monotone_class_stability_advanced d <= (lhs_measure (source d) + aux_measure (target d)) + budget d /\ (secondary_map_measure_monotone_class_stability_advanced d <= (lhs_measure (source d) + aux_measure (target d)) + budget d /\ (secondary_map_measure_monotone_class_stability_advanced d <= (lhs_measure (source d) + aux_measure (target d)) + budget d /\ (secondary_map_measure_monotone_class_stability_advanced d <= (lhs_measure (source d) + aux_measure (target d)) + budget d /\ secondary_map_measure_monotone_class_stability_advanced d <= (lhs_measure (source d) + aux_measure (target d)) + budget d)))))).
Proof.
  assert (hLowerRaw :
    lhs_measure (blend (source d) (target d)) <=
      lhs_measure (blend (source d) (target d)) + budget d).
  { apply (le_add_right_nat (lhs_measure (blend (source d) (target d))) (budget d)). }
  assert (hUpperRaw :
    lhs_measure (blend (source d) (target d)) + budget d <=
      (lhs_measure (source d) + aux_measure (target d)) + budget d).
  {
    apply (add_le_add_right_nat
      (lhs_measure (blend (source d) (target d)))
      (lhs_measure (source d) + aux_measure (target d))
      (budget d)).
    exact (coupling_bound d).
  }
  assert (hLower : lhs_measure (blend (source d) (target d)) <= secondary_map_measure_monotone_class_stability_advanced d).
  { unfold secondary_map_measure_monotone_class_stability_advanced. exact hLowerRaw. }
  assert (hUpper : secondary_map_measure_monotone_class_stability_advanced d <= (lhs_measure (source d) + aux_measure (target d)) + budget d).
  { unfold secondary_map_measure_monotone_class_stability_advanced. exact hUpperRaw. }
  assert (hBase : lhs_measure (blend (source d) (target d)) <= secondary_map_measure_monotone_class_stability_advanced d /\
    secondary_map_measure_monotone_class_stability_advanced d <= (lhs_measure (source d) + aux_measure (target d)) + budget d).
  { split; [exact hLower | exact hUpper]. }
  assert (hStamp : secondary_map_measure_monotone_class_stability_advanced d <= (lhs_measure (source d) + aux_measure (target d)) + budget d).
  { exact hUpper. }
  split.
  - exact hBase.
  - exact (conj hStamp (conj hStamp (conj hStamp (conj hStamp (conj hStamp (conj hStamp (hStamp))))))).
Qed.

Lemma comparison_step_measure_monotone_class_stability_advanced
    {X : Type} `{FrameworkStruct_measure_monotone_class_stability_advanced X}
    (d : ContextData_measure_monotone_class_stability_advanced) :
    (primary_map_measure_monotone_class_stability_advanced d <= lhs_measure (source d) + budget d /\
      lhs_measure (source d) + budget d <= lhs_measure (source d) + (rhs_measure (source d) + budget d)) /\
    (lhs_measure (source d) + budget d <= lhs_measure (source d) + (rhs_measure (source d) + budget d) /\ (lhs_measure (source d) + budget d <= lhs_measure (source d) + (rhs_measure (source d) + budget d) /\ (lhs_measure (source d) + budget d <= lhs_measure (source d) + (rhs_measure (source d) + budget d) /\ (lhs_measure (source d) + budget d <= lhs_measure (source d) + (rhs_measure (source d) + budget d) /\ (lhs_measure (source d) + budget d <= lhs_measure (source d) + (rhs_measure (source d) + budget d) /\ (lhs_measure (source d) + budget d <= lhs_measure (source d) + (rhs_measure (source d) + budget d) /\ lhs_measure (source d) + budget d <= lhs_measure (source d) + (rhs_measure (source d) + budget d))))))).
Proof.
  assert (hFirstRaw :
    lhs_measure (source d) + rhs_measure (source d) <= lhs_measure (source d) + budget d).
  {
    apply (add_le_add_left_nat (rhs_measure (source d)) (budget d) (lhs_measure (source d))).
    exact (source_rhs_le_budget d).
  }
  assert (hFirst : primary_map_measure_monotone_class_stability_advanced d <= lhs_measure (source d) + budget d).
  { unfold primary_map_measure_monotone_class_stability_advanced. exact hFirstRaw. }
  assert (hBudgetLift : budget d <= rhs_measure (source d) + budget d).
  { apply (le_add_left_nat (rhs_measure (source d)) (budget d)). }
  assert (hSecond : lhs_measure (source d) + budget d <= lhs_measure (source d) + (rhs_measure (source d) + budget d)).
  {
    apply (add_le_add_left_nat
      (budget d)
      (rhs_measure (source d) + budget d)
      (lhs_measure (source d))).
    exact hBudgetLift.
  }
  assert (hBase : primary_map_measure_monotone_class_stability_advanced d <= lhs_measure (source d) + budget d /\
    lhs_measure (source d) + budget d <= lhs_measure (source d) + (rhs_measure (source d) + budget d)).
  { split; [exact hFirst | exact hSecond]. }
  assert (hStamp : lhs_measure (source d) + budget d <= lhs_measure (source d) + (rhs_measure (source d) + budget d)).
  { exact hSecond. }
  split.
  - exact hBase.
  - exact (conj hStamp (conj hStamp (conj hStamp (conj hStamp (conj hStamp (conj hStamp (hStamp))))))).
Qed.

Lemma transport_step_measure_monotone_class_stability_advanced
    {X : Type} `{FrameworkStruct_measure_monotone_class_stability_advanced X}
    (d : ContextData_measure_monotone_class_stability_advanced) :
    (tertiary_map_measure_monotone_class_stability_advanced d <=
      (rhs_measure (source d) + aux_measure (source d)) + aux_measure (target d) /\
      aux_measure (target d) <= tertiary_map_measure_monotone_class_stability_advanced d) /\
    (aux_measure (target d) <= tertiary_map_measure_monotone_class_stability_advanced d /\ (aux_measure (target d) <= tertiary_map_measure_monotone_class_stability_advanced d /\ (aux_measure (target d) <= tertiary_map_measure_monotone_class_stability_advanced d /\ (aux_measure (target d) <= tertiary_map_measure_monotone_class_stability_advanced d /\ (aux_measure (target d) <= tertiary_map_measure_monotone_class_stability_advanced d /\ (aux_measure (target d) <= tertiary_map_measure_monotone_class_stability_advanced d /\ aux_measure (target d) <= tertiary_map_measure_monotone_class_stability_advanced d)))))).
Proof.
  assert (hPivotRaw :
    aux_measure (pivot (source d)) <= rhs_measure (source d) + aux_measure (source d)).
  { apply (aux_pivot_bound (source d)). }
  assert (hFirstRaw :
    aux_measure (pivot (source d)) + aux_measure (target d) <=
      (rhs_measure (source d) + aux_measure (source d)) + aux_measure (target d)).
  {
    apply (add_le_add_right_nat
      (aux_measure (pivot (source d)))
      (rhs_measure (source d) + aux_measure (source d))
      (aux_measure (target d))).
    exact hPivotRaw.
  }
  assert (hSecondRaw :
    aux_measure (target d) <= aux_measure (pivot (source d)) + aux_measure (target d)).
  { apply (le_add_left_nat (aux_measure (pivot (source d))) (aux_measure (target d))). }
  assert (hFirst : tertiary_map_measure_monotone_class_stability_advanced d <=
      (rhs_measure (source d) + aux_measure (source d)) + aux_measure (target d)).
  { unfold tertiary_map_measure_monotone_class_stability_advanced. exact hFirstRaw. }
  assert (hSecond : aux_measure (target d) <= tertiary_map_measure_monotone_class_stability_advanced d).
  { unfold tertiary_map_measure_monotone_class_stability_advanced. exact hSecondRaw. }
  assert (hBase : tertiary_map_measure_monotone_class_stability_advanced d <=
      (rhs_measure (source d) + aux_measure (source d)) + aux_measure (target d) /\
      aux_measure (target d) <= tertiary_map_measure_monotone_class_stability_advanced d).
  { split; [exact hFirst | exact hSecond]. }
  assert (hStamp : aux_measure (target d) <= tertiary_map_measure_monotone_class_stability_advanced d).
  { exact hSecond. }
  split.
  - exact hBase.
  - exact (conj hStamp (conj hStamp (conj hStamp (conj hStamp (conj hStamp (conj hStamp (hStamp))))))).
Qed.

Lemma coherence_step_measure_monotone_class_stability_advanced
    {X : Type} `{FrameworkStruct_measure_monotone_class_stability_advanced X}
    (d : ContextData_measure_monotone_class_stability_advanced) :
    ((primary_map_measure_monotone_class_stability_advanced d = secondary_map_measure_monotone_class_stability_advanced d <->
      primary_map_measure_monotone_class_stability_advanced d <= secondary_map_measure_monotone_class_stability_advanced d /\
      secondary_map_measure_monotone_class_stability_advanced d <= primary_map_measure_monotone_class_stability_advanced d) /\
      secondary_map_measure_monotone_class_stability_advanced d <=
        ((lhs_measure (source d) + aux_measure (target d)) + budget d) + rhs_measure (source d)) /\
    (secondary_map_measure_monotone_class_stability_advanced d <= ((lhs_measure (source d) + aux_measure (target d)) + budget d) + rhs_measure (source d) /\ (secondary_map_measure_monotone_class_stability_advanced d <= ((lhs_measure (source d) + aux_measure (target d)) + budget d) + rhs_measure (source d) /\ (secondary_map_measure_monotone_class_stability_advanced d <= ((lhs_measure (source d) + aux_measure (target d)) + budget d) + rhs_measure (source d) /\ (secondary_map_measure_monotone_class_stability_advanced d <= ((lhs_measure (source d) + aux_measure (target d)) + budget d) + rhs_measure (source d) /\ (secondary_map_measure_monotone_class_stability_advanced d <= ((lhs_measure (source d) + aux_measure (target d)) + budget d) + rhs_measure (source d) /\ (secondary_map_measure_monotone_class_stability_advanced d <= ((lhs_measure (source d) + aux_measure (target d)) + budget d) + rhs_measure (source d) /\ secondary_map_measure_monotone_class_stability_advanced d <= ((lhs_measure (source d) + aux_measure (target d)) + budget d) + rhs_measure (source d))))))).
Proof.
  assert (hForward :
    primary_map_measure_monotone_class_stability_advanced d = secondary_map_measure_monotone_class_stability_advanced d ->
    primary_map_measure_monotone_class_stability_advanced d <= secondary_map_measure_monotone_class_stability_advanced d /\
    secondary_map_measure_monotone_class_stability_advanced d <= primary_map_measure_monotone_class_stability_advanced d).
  {
    intro hEq.
    split.
    - rewrite hEq. apply (le_refl_nat (secondary_map_measure_monotone_class_stability_advanced d)).
    - rewrite hEq. apply (le_refl_nat (secondary_map_measure_monotone_class_stability_advanced d)).
  }
  assert (hBackward :
    (primary_map_measure_monotone_class_stability_advanced d <= secondary_map_measure_monotone_class_stability_advanced d /\
     secondary_map_measure_monotone_class_stability_advanced d <= primary_map_measure_monotone_class_stability_advanced d) ->
    primary_map_measure_monotone_class_stability_advanced d = secondary_map_measure_monotone_class_stability_advanced d).
  {
    intro hBoth.
    apply (le_antisymm_nat (primary_map_measure_monotone_class_stability_advanced d) (secondary_map_measure_monotone_class_stability_advanced d)).
    - exact (proj1 hBoth).
    - exact (proj2 hBoth).
  }
  assert (hIff :
    primary_map_measure_monotone_class_stability_advanced d = secondary_map_measure_monotone_class_stability_advanced d <->
    primary_map_measure_monotone_class_stability_advanced d <= secondary_map_measure_monotone_class_stability_advanced d /\
    secondary_map_measure_monotone_class_stability_advanced d <= primary_map_measure_monotone_class_stability_advanced d).
  { split; [apply hForward | apply hBackward]. }
  assert (hFact := factorization_step_measure_monotone_class_stability_advanced d).
  assert (hUpper1 : lhs_measure (blend (source d) (target d)) <= secondary_map_measure_monotone_class_stability_advanced d).
  { exact (proj1 (proj1 hFact)). }
  assert (hUpper2 : secondary_map_measure_monotone_class_stability_advanced d <= (lhs_measure (source d) + aux_measure (target d)) + budget d).
  { exact (proj2 (proj1 hFact)). }
  assert (hUpper3Raw :
    (lhs_measure (source d) + aux_measure (target d)) + budget d <=
      ((lhs_measure (source d) + aux_measure (target d)) + budget d) + rhs_measure (source d)).
  {
    apply (le_add_right_nat ((lhs_measure (source d) + aux_measure (target d)) + budget d) (rhs_measure (source d))).
  }
  assert (hBound : secondary_map_measure_monotone_class_stability_advanced d <=
      ((lhs_measure (source d) + aux_measure (target d)) + budget d) + rhs_measure (source d)).
  { apply (le_trans_nat _ _ _ hUpper2 hUpper3Raw). }
  assert (hBase : (primary_map_measure_monotone_class_stability_advanced d = secondary_map_measure_monotone_class_stability_advanced d <->
      primary_map_measure_monotone_class_stability_advanced d <= secondary_map_measure_monotone_class_stability_advanced d /\
      secondary_map_measure_monotone_class_stability_advanced d <= primary_map_measure_monotone_class_stability_advanced d) /\
      secondary_map_measure_monotone_class_stability_advanced d <=
        ((lhs_measure (source d) + aux_measure (target d)) + budget d) + rhs_measure (source d)).
  { split; [exact hIff | exact hBound]. }
  assert (hStamp : secondary_map_measure_monotone_class_stability_advanced d <= ((lhs_measure (source d) + aux_measure (target d)) + budget d) + rhs_measure (source d)).
  { exact hBound. }
  assert (hWitness : lhs_measure (blend (source d) (target d)) <= secondary_map_measure_monotone_class_stability_advanced d).
  { exact hUpper1. }
  split.
  - exact hBase.
  - exact (conj hStamp (conj hStamp (conj hStamp (conj hStamp (conj hStamp (conj hStamp (hStamp))))))).
Qed.

Lemma iteration_step_measure_monotone_class_stability_advanced
    {X : Type} `{FrameworkStruct_measure_monotone_class_stability_advanced X}
    (d : ContextData_measure_monotone_class_stability_advanced) :
    (exists z : X, z = pivot (source d) /\
      lhs_measure z <= primary_map_measure_monotone_class_stability_advanced d /\
      rhs_measure z <= primary_map_measure_monotone_class_stability_advanced d + aux_measure (source d)) /\
    (rhs_measure (pivot (source d)) <= primary_map_measure_monotone_class_stability_advanced d + aux_measure (source d) /\ (rhs_measure (pivot (source d)) <= primary_map_measure_monotone_class_stability_advanced d + aux_measure (source d) /\ (rhs_measure (pivot (source d)) <= primary_map_measure_monotone_class_stability_advanced d + aux_measure (source d) /\ (rhs_measure (pivot (source d)) <= primary_map_measure_monotone_class_stability_advanced d + aux_measure (source d) /\ (rhs_measure (pivot (source d)) <= primary_map_measure_monotone_class_stability_advanced d + aux_measure (source d) /\ (rhs_measure (pivot (source d)) <= primary_map_measure_monotone_class_stability_advanced d + aux_measure (source d) /\ rhs_measure (pivot (source d)) <= primary_map_measure_monotone_class_stability_advanced d + aux_measure (source d))))))).
Proof.
  assert (hLhsRaw :
    lhs_measure (pivot (source d)) <= lhs_measure (source d) + rhs_measure (source d)).
  { apply (pivot_lhs_bound (source d)). }
  assert (hLhs : lhs_measure (pivot (source d)) <= primary_map_measure_monotone_class_stability_advanced d).
  { unfold primary_map_measure_monotone_class_stability_advanced. exact hLhsRaw. }
  assert (hRhsPivot :
    rhs_measure (pivot (source d)) <= rhs_measure (source d) + aux_measure (source d)).
  { apply (rhs_pivot_bound (source d)). }
  assert (hRhsToPrimary :
    rhs_measure (source d) <= lhs_measure (source d) + rhs_measure (source d)).
  { apply (le_add_left_nat (lhs_measure (source d)) (rhs_measure (source d))). }
  assert (hRhsLift :
    rhs_measure (source d) + aux_measure (source d) <=
      (lhs_measure (source d) + rhs_measure (source d)) + aux_measure (source d)).
  {
    apply (add_le_add_right_nat
      (rhs_measure (source d))
      (lhs_measure (source d) + rhs_measure (source d))
      (aux_measure (source d))).
    exact hRhsToPrimary.
  }
  assert (hRhsFinal :
    rhs_measure (pivot (source d)) <= primary_map_measure_monotone_class_stability_advanced d + aux_measure (source d)).
  {
    apply (le_trans_nat _ _ _ hRhsPivot).
    unfold primary_map_measure_monotone_class_stability_advanced.
    exact hRhsLift.
  }
  assert (hExists : exists z : X, z = pivot (source d) /\
      lhs_measure z <= primary_map_measure_monotone_class_stability_advanced d /\
      rhs_measure z <= primary_map_measure_monotone_class_stability_advanced d + aux_measure (source d)).
  {
    exists (pivot (source d)).
    split.
    - reflexivity.
    - split.
      + exact hLhs.
      + exact hRhsFinal.
  }
  assert (hStamp : rhs_measure (pivot (source d)) <= primary_map_measure_monotone_class_stability_advanced d + aux_measure (source d)).
  { exact hRhsFinal. }
  split.
  - exact hExists.
  - exact (conj hStamp (conj hStamp (conj hStamp (conj hStamp (conj hStamp (conj hStamp (hStamp))))))).
Qed.

Lemma main_result_measure_monotone_class_stability_advanced
    {X : Type} `{FrameworkStruct_measure_monotone_class_stability_advanced X}
    (d : ContextData_measure_monotone_class_stability_advanced) :
    (exists n : nat,
      primary_map_measure_monotone_class_stability_advanced d <= n /\
      secondary_map_measure_monotone_class_stability_advanced d <= n /\
      ((primary_map_measure_monotone_class_stability_advanced d = secondary_map_measure_monotone_class_stability_advanced d <->
          primary_map_measure_monotone_class_stability_advanced d <= secondary_map_measure_monotone_class_stability_advanced d /\
          secondary_map_measure_monotone_class_stability_advanced d <= primary_map_measure_monotone_class_stability_advanced d) /\
        tertiary_map_measure_monotone_class_stability_advanced d <= n + tertiary_map_measure_monotone_class_stability_advanced d)) /\
    (primary_map_measure_monotone_class_stability_advanced d <= primary_map_measure_monotone_class_stability_advanced d + secondary_map_measure_monotone_class_stability_advanced d /\ (primary_map_measure_monotone_class_stability_advanced d <= primary_map_measure_monotone_class_stability_advanced d + secondary_map_measure_monotone_class_stability_advanced d /\ (primary_map_measure_monotone_class_stability_advanced d <= primary_map_measure_monotone_class_stability_advanced d + secondary_map_measure_monotone_class_stability_advanced d /\ (primary_map_measure_monotone_class_stability_advanced d <= primary_map_measure_monotone_class_stability_advanced d + secondary_map_measure_monotone_class_stability_advanced d /\ (primary_map_measure_monotone_class_stability_advanced d <= primary_map_measure_monotone_class_stability_advanced d + secondary_map_measure_monotone_class_stability_advanced d /\ (primary_map_measure_monotone_class_stability_advanced d <= primary_map_measure_monotone_class_stability_advanced d + secondary_map_measure_monotone_class_stability_advanced d /\ primary_map_measure_monotone_class_stability_advanced d <= primary_map_measure_monotone_class_stability_advanced d + secondary_map_measure_monotone_class_stability_advanced d)))))).
Proof.
  set (n := primary_map_measure_monotone_class_stability_advanced d + secondary_map_measure_monotone_class_stability_advanced d).
  assert (hPrimary : primary_map_measure_monotone_class_stability_advanced d <= n).
  {
    unfold n.
    apply (le_add_right_nat (primary_map_measure_monotone_class_stability_advanced d) (secondary_map_measure_monotone_class_stability_advanced d)).
  }
  assert (hSecondary : secondary_map_measure_monotone_class_stability_advanced d <= n).
  {
    unfold n.
    apply (le_add_left_nat (primary_map_measure_monotone_class_stability_advanced d) (secondary_map_measure_monotone_class_stability_advanced d)).
  }
  assert (hForward :
    primary_map_measure_monotone_class_stability_advanced d = secondary_map_measure_monotone_class_stability_advanced d ->
    primary_map_measure_monotone_class_stability_advanced d <= secondary_map_measure_monotone_class_stability_advanced d /\
    secondary_map_measure_monotone_class_stability_advanced d <= primary_map_measure_monotone_class_stability_advanced d).
  {
    intro hEq.
    split.
    - rewrite hEq. apply (le_refl_nat (secondary_map_measure_monotone_class_stability_advanced d)).
    - rewrite hEq. apply (le_refl_nat (secondary_map_measure_monotone_class_stability_advanced d)).
  }
  assert (hBackward :
    (primary_map_measure_monotone_class_stability_advanced d <= secondary_map_measure_monotone_class_stability_advanced d /\
      secondary_map_measure_monotone_class_stability_advanced d <= primary_map_measure_monotone_class_stability_advanced d) ->
    primary_map_measure_monotone_class_stability_advanced d = secondary_map_measure_monotone_class_stability_advanced d).
  {
    intro hBoth.
    apply (le_antisymm_nat (primary_map_measure_monotone_class_stability_advanced d) (secondary_map_measure_monotone_class_stability_advanced d)).
    - exact (proj1 hBoth).
    - exact (proj2 hBoth).
  }
  assert (hIff :
    primary_map_measure_monotone_class_stability_advanced d = secondary_map_measure_monotone_class_stability_advanced d <->
      primary_map_measure_monotone_class_stability_advanced d <= secondary_map_measure_monotone_class_stability_advanced d /\
      secondary_map_measure_monotone_class_stability_advanced d <= primary_map_measure_monotone_class_stability_advanced d).
  { split; [apply hForward | apply hBackward]. }
  assert (hTertiary : tertiary_map_measure_monotone_class_stability_advanced d <= n + tertiary_map_measure_monotone_class_stability_advanced d).
  { apply (le_add_left_nat n (tertiary_map_measure_monotone_class_stability_advanced d)). }
  assert (hExist : exists n : nat,
      primary_map_measure_monotone_class_stability_advanced d <= n /\
      secondary_map_measure_monotone_class_stability_advanced d <= n /\
      ((primary_map_measure_monotone_class_stability_advanced d = secondary_map_measure_monotone_class_stability_advanced d <->
          primary_map_measure_monotone_class_stability_advanced d <= secondary_map_measure_monotone_class_stability_advanced d /\
          secondary_map_measure_monotone_class_stability_advanced d <= primary_map_measure_monotone_class_stability_advanced d) /\
        tertiary_map_measure_monotone_class_stability_advanced d <= n + tertiary_map_measure_monotone_class_stability_advanced d)).
  {
    exists n.
    split.
    - exact hPrimary.
    - split.
      + exact hSecondary.
      + split.
        * exact hIff.
        * exact hTertiary.
  }
  assert (hStamp : primary_map_measure_monotone_class_stability_advanced d <= primary_map_measure_monotone_class_stability_advanced d + secondary_map_measure_monotone_class_stability_advanced d).
  { apply (le_add_right_nat (primary_map_measure_monotone_class_stability_advanced d) (secondary_map_measure_monotone_class_stability_advanced d)). }
  split.
  - exact hExist.
  - exact (conj hStamp (conj hStamp (conj hStamp (conj hStamp (conj hStamp (conj hStamp (hStamp))))))).
Qed.
