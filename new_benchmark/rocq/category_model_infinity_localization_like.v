(*
BENCHMARK_ID: TINY_MATHLIB_BATCH07_CATEGORY_MODEL_INFINITY_LOCALIZATION_LIKE
PAIR_STEM: category_model_infinity_localization_like
MATH_DOMAIN: Category Theory
SOURCE_MATHLIB: Mathlib/CategoryTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Module CategoryModelInfinityLocalizationLike.

Class FrameworkStruct_category_model_infinity_localization (A : Type) := {
  objComp : A -> A -> A;
  localize : A -> A;
  witness : A -> A;
  comp_assoc : forall a b c : A,
    objComp (objComp a b) c = objComp a (objComp b c);
  loc_idem : forall a : A, localize (localize a) = localize a;
  loc_comp : forall a b : A,
    localize (objComp a b) = objComp (localize a) (localize b);
  witness_loc : forall a : A, witness (localize a) = localize (witness a);
  witness_comp : forall a b : A,
    witness (objComp a b) = objComp (witness a) (witness b);
  loc_witness : forall a : A, localize (witness a) = witness (localize a)
}.

Record ContextData_category_model_infinity_localization
    (A : Type) := {
  f_obj : A;
  g_obj : A;
  h_obj : A;
  k_obj : A
}.

Definition primary_map_category_model_infinity_localization
    {A : Type} `{FrameworkStruct_category_model_infinity_localization A}
    (d : ContextData_category_model_infinity_localization A) : A :=
  localize (objComp (f_obj d) (g_obj d)).

Definition secondary_map_category_model_infinity_localization
    {A : Type} `{FrameworkStruct_category_model_infinity_localization A}
    (d : ContextData_category_model_infinity_localization A) : A :=
  objComp (localize (f_obj d)) (localize (g_obj d)).

Definition tertiary_map_category_model_infinity_localization
    {A : Type} `{FrameworkStruct_category_model_infinity_localization A}
    (d : ContextData_category_model_infinity_localization A) : A :=
  localize
    (objComp
      (secondary_map_category_model_infinity_localization d)
      (witness (h_obj d))).

Lemma stability_step_category_model_infinity_localization
    {A : Type} `{FrameworkStruct_category_model_infinity_localization A}
    (d : ContextData_category_model_infinity_localization A) :
    primary_map_category_model_infinity_localization d =
      secondary_map_category_model_infinity_localization d.
Proof.
  unfold primary_map_category_model_infinity_localization.
  unfold secondary_map_category_model_infinity_localization.
  rewrite loc_comp.
  reflexivity.
Qed.

Lemma factorization_step_category_model_infinity_localization
    {A : Type} `{FrameworkStruct_category_model_infinity_localization A}
    (d : ContextData_category_model_infinity_localization A) :
    localize (secondary_map_category_model_infinity_localization d) =
      secondary_map_category_model_infinity_localization d.
Proof.
  unfold secondary_map_category_model_infinity_localization.
  rewrite loc_comp.
  rewrite loc_idem.
  rewrite loc_idem.
  reflexivity.
Qed.

Lemma comparison_step_category_model_infinity_localization
    {A : Type} `{FrameworkStruct_category_model_infinity_localization A}
    (d : ContextData_category_model_infinity_localization A) :
    localize
      (objComp
        (primary_map_category_model_infinity_localization d)
        (witness (h_obj d)))
    = tertiary_map_category_model_infinity_localization d.
Proof.
  assert (hstable := stability_step_category_model_infinity_localization d).
  rewrite hstable.
  unfold tertiary_map_category_model_infinity_localization.
  reflexivity.
Qed.

Lemma transport_step_category_model_infinity_localization
    {A : Type} `{FrameworkStruct_category_model_infinity_localization A}
    (d : ContextData_category_model_infinity_localization A)
    (hf : f_obj d = k_obj d) (hg : g_obj d = h_obj d) :
    primary_map_category_model_infinity_localization d =
      localize (objComp (k_obj d) (h_obj d)).
Proof.
  unfold primary_map_category_model_infinity_localization.
  rewrite hf.
  rewrite hg.
  reflexivity.
Qed.

Lemma coherence_step_category_model_infinity_localization
    {A : Type} `{FrameworkStruct_category_model_infinity_localization A}
    (d : ContextData_category_model_infinity_localization A) :
    objComp
      (primary_map_category_model_infinity_localization d)
      (objComp (witness (h_obj d)) (witness (k_obj d)))
    =
    objComp
      (objComp
        (primary_map_category_model_infinity_localization d)
        (witness (h_obj d)))
      (witness (k_obj d)).
Proof.
  symmetry.
  apply comp_assoc.
Qed.

Lemma iteration_step_category_model_infinity_localization
    {A : Type} `{FrameworkStruct_category_model_infinity_localization A}
    (d : ContextData_category_model_infinity_localization A) :
    (witness
      (localize (tertiary_map_category_model_infinity_localization d))
    =
    localize
      (witness (tertiary_map_category_model_infinity_localization d))) /\
    True.
Proof.
  split.
  - apply witness_loc.
  - exact I.
Qed.

Lemma main_result_category_model_infinity_localization
    {A : Type} `{FrameworkStruct_category_model_infinity_localization A}
    (d : ContextData_category_model_infinity_localization A) :
    exists m : A,
      localize m = tertiary_map_category_model_infinity_localization d /\
      m = objComp
        (secondary_map_category_model_infinity_localization d)
        (witness (h_obj d)).
Proof.
  refine (ex_intro _
    (objComp
      (secondary_map_category_model_infinity_localization d)
      (witness (h_obj d))) _).
  split.
  - unfold tertiary_map_category_model_infinity_localization.
    reflexivity.
  - reflexivity.
Qed.

End CategoryModelInfinityLocalizationLike.
