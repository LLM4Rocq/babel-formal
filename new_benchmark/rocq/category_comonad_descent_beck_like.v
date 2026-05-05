(*
BENCHMARK_ID: TINY_MATHLIB_BATCH07_CATEGORY_COMONAD_DESCENT_BECK_LIKE
PAIR_STEM: category_comonad_descent_beck_like
MATH_DOMAIN: Category Theory
SOURCE_MATHLIB: Mathlib/CategoryTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Module CategoryComonadDescentBeckLike.

Class FrameworkStruct_category_comonad_descent_beck (A : Type) := {
  merge : A -> A -> A;
  extend : A -> A;
  reduce : A -> A;
  merge_assoc : forall a b c : A, merge (merge a b) c = merge a (merge b c);
  extend_merge : forall a b : A, extend (merge a b) = merge (extend a) (extend b);
  reduce_merge : forall a b : A, reduce (merge a b) = merge (reduce a) (reduce b);
  extend_idem : forall a : A, extend (extend a) = extend a;
  reduce_idem : forall a : A, reduce (reduce a) = reduce a;
  reduce_extend_comm : forall a : A, reduce (extend a) = extend (reduce a);
  beck_swap : forall a b : A, merge (extend a) (reduce b) = merge (reduce b) (extend a)
}.

Record ContextData_category_comonad_descent_beck
    (A : Type) := {
  P_obj : A;
  Q_obj : A;
  R_obj : A;
  S_obj : A
}.

Definition primary_map_category_comonad_descent_beck
    {A : Type} `{FrameworkStruct_category_comonad_descent_beck A}
    (d : ContextData_category_comonad_descent_beck A) : A :=
  merge (extend (P_obj d)) (reduce (Q_obj d)).

Definition secondary_map_category_comonad_descent_beck
    {A : Type} `{FrameworkStruct_category_comonad_descent_beck A}
    (d : ContextData_category_comonad_descent_beck A) : A :=
  merge (extend (R_obj d)) (reduce (S_obj d)).

Definition tertiary_map_category_comonad_descent_beck
    {A : Type} `{FrameworkStruct_category_comonad_descent_beck A}
    (d : ContextData_category_comonad_descent_beck A) : A :=
  extend
    (merge
      (primary_map_category_comonad_descent_beck d)
      (secondary_map_category_comonad_descent_beck d)).

Lemma stability_step_category_comonad_descent_beck
    {A : Type} `{FrameworkStruct_category_comonad_descent_beck A}
    (d : ContextData_category_comonad_descent_beck A) :
    exists y : A,
      y = reduce (primary_map_category_comonad_descent_beck d) /\
      y = merge (reduce (extend (P_obj d))) (reduce (reduce (Q_obj d))).
Proof.
  refine (ex_intro _ (reduce (primary_map_category_comonad_descent_beck d)) _).
  split.
  - reflexivity.
  - unfold primary_map_category_comonad_descent_beck.
    transitivity (reduce (merge (extend (P_obj d)) (reduce (Q_obj d)))).
    + reflexivity.
    + apply reduce_merge.
Qed.

Lemma factorization_step_category_comonad_descent_beck
    {A : Type} `{FrameworkStruct_category_comonad_descent_beck A}
    (d : ContextData_category_comonad_descent_beck A) :
    R_obj d = R_obj d ->
    extend (secondary_map_category_comonad_descent_beck d) =
      merge (extend (extend (R_obj d))) (extend (reduce (S_obj d))).
Proof.
  intro hR.
  pose proof hR as hcopy.
  unfold secondary_map_category_comonad_descent_beck.
  transitivity (extend (merge (extend (R_obj d)) (reduce (S_obj d)))).
  - reflexivity.
  - apply extend_merge.
Qed.

Lemma comparison_step_category_comonad_descent_beck
    {A : Type} `{FrameworkStruct_category_comonad_descent_beck A}
    (d : ContextData_category_comonad_descent_beck A) :
    (merge
      (reduce (primary_map_category_comonad_descent_beck d))
      (extend (secondary_map_category_comonad_descent_beck d))
    =
    merge
      (merge (reduce (extend (P_obj d))) (reduce (reduce (Q_obj d))))
      (merge (extend (extend (R_obj d))) (extend (reduce (S_obj d))))) /\
    True.
Proof.
  assert (hf := factorization_step_category_comonad_descent_beck d eq_refl).
  split.
  - unfold primary_map_category_comonad_descent_beck.
    rewrite reduce_merge.
    rewrite hf.
    reflexivity.
  - exact I.
Qed.

Lemma transport_step_category_comonad_descent_beck
    {A : Type} `{FrameworkStruct_category_comonad_descent_beck A}
    (d : ContextData_category_comonad_descent_beck A)
    (hP : P_obj d = R_obj d) (hQ : Q_obj d = S_obj d) :
    primary_map_category_comonad_descent_beck d =
      secondary_map_category_comonad_descent_beck d.
Proof.
  unfold primary_map_category_comonad_descent_beck.
  unfold secondary_map_category_comonad_descent_beck.
  rewrite hP.
  rewrite hQ.
  reflexivity.
Qed.

Lemma coherence_step_category_comonad_descent_beck
    {A : Type} `{FrameworkStruct_category_comonad_descent_beck A}
    (d : ContextData_category_comonad_descent_beck A) :
    (tertiary_map_category_comonad_descent_beck d =
      merge
        (extend (primary_map_category_comonad_descent_beck d))
        (extend (secondary_map_category_comonad_descent_beck d))) <->
    True.
Proof.
  split.
  - intro hEq.
    exact I.
  - intro hTrue.
    pose proof hTrue as hcopy.
    unfold tertiary_map_category_comonad_descent_beck.
    apply extend_merge.
Qed.

Lemma iteration_step_category_comonad_descent_beck
    {A : Type} `{FrameworkStruct_category_comonad_descent_beck A}
    (d : ContextData_category_comonad_descent_beck A) :
    exists z : A,
      z = reduce (extend (tertiary_map_category_comonad_descent_beck d)) /\
      z = extend (reduce (tertiary_map_category_comonad_descent_beck d)).
Proof.
  refine (ex_intro _ (reduce (extend (tertiary_map_category_comonad_descent_beck d))) _).
  split.
  - reflexivity.
  - apply reduce_extend_comm.
Qed.

Lemma main_result_category_comonad_descent_beck
    {A : Type} `{FrameworkStruct_category_comonad_descent_beck A}
    (d : ContextData_category_comonad_descent_beck A) :
    merge (extend (P_obj d)) (reduce (Q_obj d)) =
      merge (reduce (Q_obj d)) (extend (P_obj d)) /\
    exists t : A, t = tertiary_map_category_comonad_descent_beck d.
Proof.
  split.
  - apply beck_swap.
  - refine (ex_intro _ (tertiary_map_category_comonad_descent_beck d) _).
    reflexivity.
Qed.

End CategoryComonadDescentBeckLike.
