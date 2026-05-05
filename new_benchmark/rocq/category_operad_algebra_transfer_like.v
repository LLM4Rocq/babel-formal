(*
BENCHMARK_ID: TINY_MATHLIB_BATCH07_CATEGORY_OPERAD_ALGEBRA_TRANSFER_LIKE
PAIR_STEM: category_operad_algebra_transfer_like
MATH_DOMAIN: Category Theory
SOURCE_MATHLIB: Mathlib/CategoryTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Module CategoryOperadAlgebraTransferLike.

Class FrameworkStruct_category_operad_algebra_transfer (A : Type) := {
  operadComp : A -> A -> A;
  actOn : A -> A -> A;
  liftOp : A -> A;
  restrictOp : A -> A;
  comp_assoc : forall a b c : A,
    operadComp (operadComp a b) c = operadComp a (operadComp b c);
  action_assoc : forall o1 o2 a : A,
    actOn (operadComp o1 o2) a = actOn o1 (actOn o2 a);
  lift_act : forall o a : A,
    liftOp (actOn o a) = actOn (liftOp o) (liftOp a);
  restrict_act : forall o a : A,
    restrictOp (actOn o a) = actOn (restrictOp o) (restrictOp a);
  lift_idem : forall a : A, liftOp (liftOp a) = liftOp a;
  restrict_idem : forall a : A, restrictOp (restrictOp a) = restrictOp a;
  lift_restrict_comm : forall a : A, liftOp (restrictOp a) = restrictOp (liftOp a)
}.

Record ContextData_category_operad_algebra_transfer
    (A : Type) := {
  O1_obj : A;
  O2_obj : A;
  A1_obj : A;
  A2_obj : A
}.

Definition primary_map_category_operad_algebra_transfer
    {A : Type} `{FrameworkStruct_category_operad_algebra_transfer A}
    (d : ContextData_category_operad_algebra_transfer A) : A :=
  actOn (liftOp (O1_obj d)) (A1_obj d).

Definition secondary_map_category_operad_algebra_transfer
    {A : Type} `{FrameworkStruct_category_operad_algebra_transfer A}
    (d : ContextData_category_operad_algebra_transfer A) : A :=
  actOn (restrictOp (O2_obj d)) (A2_obj d).

Definition tertiary_map_category_operad_algebra_transfer
    {A : Type} `{FrameworkStruct_category_operad_algebra_transfer A}
    (d : ContextData_category_operad_algebra_transfer A) : A :=
  actOn
    (operadComp (liftOp (O1_obj d)) (restrictOp (O2_obj d)))
    (liftOp (A1_obj d)).

Lemma stability_step_category_operad_algebra_transfer
    {A : Type} `{FrameworkStruct_category_operad_algebra_transfer A}
    (d : ContextData_category_operad_algebra_transfer A) :
    True ->
    liftOp (primary_map_category_operad_algebra_transfer d) =
      actOn (liftOp (liftOp (O1_obj d))) (liftOp (A1_obj d)).
Proof.
  intro hTrue.
  pose proof hTrue as hkeep.
  unfold primary_map_category_operad_algebra_transfer.
  transitivity (liftOp (actOn (liftOp (O1_obj d)) (A1_obj d))).
  - reflexivity.
  - apply lift_act.
Qed.

Lemma factorization_step_category_operad_algebra_transfer
    {A : Type} `{FrameworkStruct_category_operad_algebra_transfer A}
    (d : ContextData_category_operad_algebra_transfer A) :
    (restrictOp (secondary_map_category_operad_algebra_transfer d) =
      actOn (restrictOp (restrictOp (O2_obj d))) (restrictOp (A2_obj d))) /\
    True.
Proof.
  split.
  - unfold secondary_map_category_operad_algebra_transfer.
    transitivity (restrictOp (actOn (restrictOp (O2_obj d)) (A2_obj d))).
    + reflexivity.
    + apply restrict_act.
  - exact I.
Qed.

Lemma comparison_step_category_operad_algebra_transfer
    {A : Type} `{FrameworkStruct_category_operad_algebra_transfer A}
    (d : ContextData_category_operad_algebra_transfer A) :
    actOn
      (operadComp (liftOp (O1_obj d)) (restrictOp (O2_obj d)))
      (actOn (liftOp (A1_obj d)) (restrictOp (A2_obj d)))
    =
    actOn
      (liftOp (O1_obj d))
      (actOn
        (restrictOp (O2_obj d))
        (actOn (liftOp (A1_obj d)) (restrictOp (A2_obj d)))).
Proof.
  apply action_assoc.
Qed.

Lemma transport_step_category_operad_algebra_transfer
    {A : Type} `{FrameworkStruct_category_operad_algebra_transfer A}
    (d : ContextData_category_operad_algebra_transfer A)
    (hO : O1_obj d = restrictOp (O2_obj d)) (hA : A1_obj d = A2_obj d) :
    primary_map_category_operad_algebra_transfer d =
      actOn (liftOp (restrictOp (O2_obj d))) (A2_obj d).
Proof.
  unfold primary_map_category_operad_algebra_transfer.
  rewrite hO.
  rewrite hA.
  reflexivity.
Qed.

Lemma coherence_step_category_operad_algebra_transfer
    {A : Type} `{FrameworkStruct_category_operad_algebra_transfer A}
    (d : ContextData_category_operad_algebra_transfer A) :
    tertiary_map_category_operad_algebra_transfer d =
      actOn
        (liftOp (O1_obj d))
        (actOn (restrictOp (O2_obj d)) (liftOp (A1_obj d))).
Proof.
  unfold tertiary_map_category_operad_algebra_transfer.
  apply action_assoc.
Qed.

Lemma iteration_step_category_operad_algebra_transfer
    {A : Type} `{FrameworkStruct_category_operad_algebra_transfer A}
    (d : ContextData_category_operad_algebra_transfer A) :
    ~ False ->
    liftOp (restrictOp (tertiary_map_category_operad_algebra_transfer d)) =
      restrictOp (liftOp (tertiary_map_category_operad_algebra_transfer d)).
Proof.
  intro hnf.
  pose proof hnf as hcopy.
  apply lift_restrict_comm.
Qed.

Lemma main_result_category_operad_algebra_transfer
    {A : Type} `{FrameworkStruct_category_operad_algebra_transfer A}
    (d : ContextData_category_operad_algebra_transfer A) :
    exists x : A,
      x = tertiary_map_category_operad_algebra_transfer d /\
      actOn (liftOp (O1_obj d)) (A1_obj d) = primary_map_category_operad_algebra_transfer d.
Proof.
  refine (ex_intro _ (tertiary_map_category_operad_algebra_transfer d) _).
  split.
  - reflexivity.
  - reflexivity.
Qed.

End CategoryOperadAlgebraTransferLike.
