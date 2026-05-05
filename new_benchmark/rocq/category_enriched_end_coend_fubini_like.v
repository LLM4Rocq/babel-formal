(*
BENCHMARK_ID: TINY_MATHLIB_BATCH07_CATEGORY_ENRICHED_END_COEND_FUBINI_LIKE
PAIR_STEM: category_enriched_end_coend_fubini_like
MATH_DOMAIN: Category Theory
SOURCE_MATHLIB: Mathlib/CategoryTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 11
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Module CategoryEnrichedEndCoendFubiniLike.

Class FrameworkStruct_category_enriched_end_coend_fubini (A : Type) := {
  tensor : A -> A -> A;
  endOp : A -> A;
  coendOp : A -> A;
  tensor_assoc : forall a b c : A, tensor (tensor a b) c = tensor a (tensor b c);
  end_tensor : forall a b : A, endOp (tensor a b) = tensor (endOp a) (endOp b);
  coend_tensor : forall a b : A, coendOp (tensor a b) = tensor (coendOp a) (coendOp b);
  end_idem : forall a : A, endOp (endOp a) = endOp a;
  coend_idem : forall a : A, coendOp (coendOp a) = coendOp a;
  end_coend_comm : forall a : A, endOp (coendOp a) = coendOp (endOp a);
  fubini_swap : forall a b : A, tensor (endOp a) (coendOp b) = tensor (coendOp b) (endOp a)
}.

Record ContextData_category_enriched_end_coend_fubini
    (A : Type) := {
  U_obj : A;
  V_obj : A;
  W_obj : A;
  Z_obj : A
}.

Definition primary_map_category_enriched_end_coend_fubini
    {A : Type} `{FrameworkStruct_category_enriched_end_coend_fubini A}
    (d : ContextData_category_enriched_end_coend_fubini A) : A :=
  tensor (endOp (U_obj d)) (coendOp (V_obj d)).

Definition secondary_map_category_enriched_end_coend_fubini
    {A : Type} `{FrameworkStruct_category_enriched_end_coend_fubini A}
    (d : ContextData_category_enriched_end_coend_fubini A) : A :=
  tensor (coendOp (V_obj d)) (endOp (W_obj d)).

Definition tertiary_map_category_enriched_end_coend_fubini
    {A : Type} `{FrameworkStruct_category_enriched_end_coend_fubini A}
    (d : ContextData_category_enriched_end_coend_fubini A) : A :=
  tensor
    (tensor
      (primary_map_category_enriched_end_coend_fubini d)
      (secondary_map_category_enriched_end_coend_fubini d))
    (endOp (Z_obj d)).

Lemma stability_step_category_enriched_end_coend_fubini
    {A : Type} `{FrameworkStruct_category_enriched_end_coend_fubini A}
    (d : ContextData_category_enriched_end_coend_fubini A) :
    U_obj d = U_obj d ->
    (endOp (primary_map_category_enriched_end_coend_fubini d) =
      tensor (endOp (endOp (U_obj d))) (endOp (coendOp (V_obj d)))) /\
    True.
Proof.
  intro hUU.
  pose proof hUU as hkeep.
  split.
  - unfold primary_map_category_enriched_end_coend_fubini.
    transitivity (endOp (tensor (endOp (U_obj d)) (coendOp (V_obj d)))).
    + reflexivity.
    + apply end_tensor.
  - exact I.
Qed.

Lemma factorization_step_category_enriched_end_coend_fubini
    {A : Type} `{FrameworkStruct_category_enriched_end_coend_fubini A}
    (d : ContextData_category_enriched_end_coend_fubini A) :
    (coendOp (secondary_map_category_enriched_end_coend_fubini d) =
      tensor (coendOp (coendOp (V_obj d))) (coendOp (endOp (W_obj d)))) /\
    V_obj d = V_obj d.
Proof.
  split.
  - unfold secondary_map_category_enriched_end_coend_fubini.
    transitivity (coendOp (tensor (coendOp (V_obj d)) (endOp (W_obj d)))).
    + reflexivity.
    + apply coend_tensor.
  - reflexivity.
Qed.

Lemma comparison_step_category_enriched_end_coend_fubini
    {A : Type} `{FrameworkStruct_category_enriched_end_coend_fubini A}
    (d : ContextData_category_enriched_end_coend_fubini A) :
    exists x : A,
      x =
        tensor
          (endOp (primary_map_category_enriched_end_coend_fubini d))
          (coendOp (secondary_map_category_enriched_end_coend_fubini d)) /\
      x =
        tensor
          (tensor (endOp (endOp (U_obj d))) (endOp (coendOp (V_obj d))))
          (tensor (coendOp (coendOp (V_obj d))) (coendOp (endOp (W_obj d)))).
Proof.
  assert (hs := stability_step_category_enriched_end_coend_fubini d eq_refl).
  assert (hf := factorization_step_category_enriched_end_coend_fubini d).
  refine (ex_intro _
    (tensor
      (endOp (primary_map_category_enriched_end_coend_fubini d))
      (coendOp (secondary_map_category_enriched_end_coend_fubini d))) _).
  split.
  - reflexivity.
  - rewrite (proj1 hs).
    rewrite (proj1 hf).
    reflexivity.
Qed.

Lemma transport_step_category_enriched_end_coend_fubini
    {A : Type} `{FrameworkStruct_category_enriched_end_coend_fubini A}
    (d : ContextData_category_enriched_end_coend_fubini A)
    (hU : U_obj d = W_obj d) (hV : V_obj d = Z_obj d) :
    exists p : A,
      p = tensor (endOp (U_obj d)) (coendOp (V_obj d)) /\
      p = tensor (endOp (W_obj d)) (coendOp (Z_obj d)).
Proof.
  refine (ex_intro _ (tensor (endOp (U_obj d)) (coendOp (V_obj d))) _).
  split.
  - reflexivity.
  - rewrite hU.
    rewrite hV.
    reflexivity.
Qed.

Lemma iteration_step_category_enriched_end_coend_fubini
    {A : Type} `{FrameworkStruct_category_enriched_end_coend_fubini A}
    (d : ContextData_category_enriched_end_coend_fubini A) :
    True ->
    endOp (coendOp (primary_map_category_enriched_end_coend_fubini d)) =
      coendOp (endOp (primary_map_category_enriched_end_coend_fubini d)).
Proof.
  intro hTrue.
  pose proof hTrue as hcopy.
  apply end_coend_comm.
Qed.

Lemma main_result_category_enriched_end_coend_fubini
    {A : Type} `{FrameworkStruct_category_enriched_end_coend_fubini A}
    (d : ContextData_category_enriched_end_coend_fubini A) :
    exists t : A,
      t = tertiary_map_category_enriched_end_coend_fubini d /\
      tensor (endOp (U_obj d)) (coendOp (V_obj d)) =
        tensor (coendOp (V_obj d)) (endOp (U_obj d)).
Proof.
  refine (ex_intro _ (tertiary_map_category_enriched_end_coend_fubini d) _).
  split.
  - reflexivity.
  - apply fubini_swap.
Qed.

End CategoryEnrichedEndCoendFubiniLike.
