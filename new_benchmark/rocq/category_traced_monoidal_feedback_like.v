(*
BENCHMARK_ID: TINY_MATHLIB_BATCH07_CATEGORY_TRACED_MONOIDAL_FEEDBACK_LIKE
PAIR_STEM: category_traced_monoidal_feedback_like
MATH_DOMAIN: Category Theory
SOURCE_MATHLIB: Mathlib/CategoryTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Module CategoryTracedMonoidalFeedbackLike.

Class FrameworkStruct_category_traced_monoidal_feedback (A : Type) := {
  tensor : A -> A -> A;
  traceOp : A -> A;
  feedbackOp : A -> A;
  tensor_assoc : forall a b c : A, tensor (tensor a b) c = tensor a (tensor b c);
  trace_tensor : forall a b : A,
    traceOp (tensor a b) = tensor (traceOp a) (traceOp b);
  feedback_tensor : forall a b : A,
    feedbackOp (tensor a b) = tensor (feedbackOp a) (feedbackOp b);
  trace_feedback_comm : forall a : A,
    traceOp (feedbackOp a) = feedbackOp (traceOp a);
  trace_idem : forall a : A, traceOp (traceOp a) = traceOp a;
  feedback_idem : forall a : A, feedbackOp (feedbackOp a) = feedbackOp a;
  braid : forall a b : A, tensor a b = tensor b a
}.

Record ContextData_category_traced_monoidal_feedback
    (A : Type) := {
  M_obj : A;
  N_obj : A;
  P_obj : A;
  Q_obj : A
}.

Definition primary_map_category_traced_monoidal_feedback
    {A : Type} `{FrameworkStruct_category_traced_monoidal_feedback A}
    (d : ContextData_category_traced_monoidal_feedback A) : A :=
  tensor (traceOp (M_obj d)) (feedbackOp (N_obj d)).

Definition secondary_map_category_traced_monoidal_feedback
    {A : Type} `{FrameworkStruct_category_traced_monoidal_feedback A}
    (d : ContextData_category_traced_monoidal_feedback A) : A :=
  tensor (feedbackOp (P_obj d)) (traceOp (Q_obj d)).

Definition tertiary_map_category_traced_monoidal_feedback
    {A : Type} `{FrameworkStruct_category_traced_monoidal_feedback A}
    (d : ContextData_category_traced_monoidal_feedback A) : A :=
  feedbackOp
    (tensor
      (primary_map_category_traced_monoidal_feedback d)
      (secondary_map_category_traced_monoidal_feedback d)).

Lemma stability_step_category_traced_monoidal_feedback
    {A : Type} `{FrameworkStruct_category_traced_monoidal_feedback A}
    (d : ContextData_category_traced_monoidal_feedback A) :
    False \/
      traceOp (primary_map_category_traced_monoidal_feedback d) =
        tensor (traceOp (traceOp (M_obj d))) (traceOp (feedbackOp (N_obj d))).
Proof.
  right.
  unfold primary_map_category_traced_monoidal_feedback.
  transitivity (traceOp (tensor (traceOp (M_obj d)) (feedbackOp (N_obj d)))).
  - reflexivity.
  - apply trace_tensor.
Qed.

Lemma factorization_step_category_traced_monoidal_feedback
    {A : Type} `{FrameworkStruct_category_traced_monoidal_feedback A}
    (d : ContextData_category_traced_monoidal_feedback A) :
    (feedbackOp (secondary_map_category_traced_monoidal_feedback d) =
      tensor (feedbackOp (feedbackOp (P_obj d))) (feedbackOp (traceOp (Q_obj d)))) <->
    True.
Proof.
  split.
  - intro hEq.
    exact I.
  - intro hTrue.
    pose proof hTrue as hcopy.
    unfold secondary_map_category_traced_monoidal_feedback.
    transitivity (feedbackOp (tensor (feedbackOp (P_obj d)) (traceOp (Q_obj d)))).
    + reflexivity.
    + apply feedback_tensor.
Qed.

Lemma comparison_step_category_traced_monoidal_feedback
    {A : Type} `{FrameworkStruct_category_traced_monoidal_feedback A}
    (d : ContextData_category_traced_monoidal_feedback A) :
    True /\
    (tensor
      (traceOp (primary_map_category_traced_monoidal_feedback d))
      (feedbackOp (secondary_map_category_traced_monoidal_feedback d))
    =
    tensor
      (tensor (traceOp (traceOp (M_obj d))) (traceOp (feedbackOp (N_obj d))))
      (tensor (feedbackOp (feedbackOp (P_obj d))) (feedbackOp (traceOp (Q_obj d))))).
Proof.
  split.
  - exact I.
  - unfold primary_map_category_traced_monoidal_feedback.
    unfold secondary_map_category_traced_monoidal_feedback.
    rewrite trace_tensor.
    rewrite feedback_tensor.
    reflexivity.
Qed.

Lemma transport_step_category_traced_monoidal_feedback
    {A : Type} `{FrameworkStruct_category_traced_monoidal_feedback A}
    (d : ContextData_category_traced_monoidal_feedback A)
    (hM : M_obj d = P_obj d) (hQ : Q_obj d = N_obj d) :
    (tensor (traceOp (M_obj d)) (feedbackOp (Q_obj d)) =
      tensor (traceOp (P_obj d)) (feedbackOp (N_obj d))) /\
    Q_obj d = Q_obj d.
Proof.
  split.
  - rewrite hM.
    rewrite hQ.
    reflexivity.
  - reflexivity.
Qed.

Lemma coherence_step_category_traced_monoidal_feedback
    {A : Type} `{FrameworkStruct_category_traced_monoidal_feedback A}
    (d : ContextData_category_traced_monoidal_feedback A) :
    True ->
    tertiary_map_category_traced_monoidal_feedback d =
      tensor
        (feedbackOp (primary_map_category_traced_monoidal_feedback d))
        (feedbackOp (secondary_map_category_traced_monoidal_feedback d)).
Proof.
  intro hTrue.
  pose proof hTrue as hcopy.
  unfold tertiary_map_category_traced_monoidal_feedback.
  apply feedback_tensor.
Qed.

Lemma iteration_step_category_traced_monoidal_feedback
    {A : Type} `{FrameworkStruct_category_traced_monoidal_feedback A}
    (d : ContextData_category_traced_monoidal_feedback A) :
    False \/
    traceOp (feedbackOp (tertiary_map_category_traced_monoidal_feedback d)) =
      feedbackOp (traceOp (tertiary_map_category_traced_monoidal_feedback d)).
Proof.
  right.
  apply trace_feedback_comm.
Qed.

Lemma main_result_category_traced_monoidal_feedback
    {A : Type} `{FrameworkStruct_category_traced_monoidal_feedback A}
    (d : ContextData_category_traced_monoidal_feedback A) :
    tensor
      (primary_map_category_traced_monoidal_feedback d)
      (secondary_map_category_traced_monoidal_feedback d)
    =
    tensor
      (secondary_map_category_traced_monoidal_feedback d)
      (primary_map_category_traced_monoidal_feedback d)
    /\ exists t : A, t = tertiary_map_category_traced_monoidal_feedback d.
Proof.
  refine (conj _ _).
  - apply braid.
  - exists (tertiary_map_category_traced_monoidal_feedback d).
    reflexivity.
Qed.

End CategoryTracedMonoidalFeedbackLike.
