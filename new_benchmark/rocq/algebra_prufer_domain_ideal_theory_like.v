(*
BENCHMARK_ID: TINY_MATHLIB_BATCH06_ALG_PRUFER_DOMAIN_IDEAL_THEORY_LIKE
PAIR_STEM: algebra_prufer_domain_ideal_theory_like
MATH_DOMAIN: Algebra
SOURCE_MATHLIB: Mathlib/Algebra/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class PruferStruct_domain_ideal_theory (R : Type) := {
  Ideal : Type;
  rmul : R -> R -> R;
  le : Ideal -> Ideal -> Prop;
  top : Ideal;
  mul : Ideal -> Ideal -> Ideal;
  inter : Ideal -> Ideal -> Ideal;
  inv : Ideal -> Ideal;
  hull : Ideal -> Ideal;
  localization : Ideal -> Ideal;
  content : R -> Ideal;
  principal : Ideal -> Prop;
  le_refl : forall I : Ideal, le I I;
  le_trans : forall I J K : Ideal, le I J -> le J K -> le I K;
  le_antisymm : forall I J : Ideal, le I J -> le J I -> I = J;
  mul_assoc : forall I J K : Ideal, mul (mul I J) K = mul I (mul J K);
  mul_comm : forall I J : Ideal, mul I J = mul J I;
  mul_top : forall I : Ideal, mul I top = I;
  mul_mono_left : forall I J K : Ideal, le I J -> le (mul K I) (mul K J);
  inter_left : forall I J : Ideal, le (inter I J) I;
  inter_right : forall I J : Ideal, le (inter I J) J;
  inter_glb : forall K I J : Ideal, le K I -> le K J -> le K (inter I J);
  hull_extensive : forall I : Ideal, le I (hull I);
  hull_idem : forall I : Ideal, hull (hull I) = hull I;
  hull_mono : forall I J : Ideal, le I J -> le (hull I) (hull J);
  localization_mono : forall I J : Ideal, le I J -> le (localization I) (localization J);
  localization_hull : forall I : Ideal, le (hull I) (localization I);
  localization_principal : forall I : Ideal, principal (localization I);
  inv_mul_top : forall I : Ideal, le top (mul I (inv I));
  mul_inv_top : forall I : Ideal, le (mul I (inv I)) top;
  content_mul_le : forall x y : R, le (content (rmul x y)) (mul (content x) (content y));
  content_mul_ge : forall x y : R, le (mul (content x) (content y)) (content (rmul x y));
  divisor_closed : forall I J : Ideal, le I J -> le J (hull I)
}.

Record IdealData_prufer_domain_ideal_theory (R : Type)
    {P : PruferStruct_domain_ideal_theory R} := {
  base_ideal : Ideal;
  aux_ideal : Ideal;
  coeff_left : R;
  coeff_right : R
}.

Definition invertible_hull_prufer_domain_ideal_theory {R : Type}
    {P : PruferStruct_domain_ideal_theory R}
    (d : IdealData_prufer_domain_ideal_theory) : Ideal :=
  hull (base_ideal d).

Definition localization_ideal_prufer_domain_ideal_theory {R : Type}
    {P : PruferStruct_domain_ideal_theory R}
    (d : IdealData_prufer_domain_ideal_theory) : Ideal :=
  localization (base_ideal d).

Definition content_ideal_prufer_domain_ideal_theory {R : Type}
    {P : PruferStruct_domain_ideal_theory R}
    (d : IdealData_prufer_domain_ideal_theory) : Ideal :=
  content (coeff_left d).

Lemma localization_principal_step_prufer_domain_ideal_theory {R : Type}
    {P : PruferStruct_domain_ideal_theory R}
    (d : IdealData_prufer_domain_ideal_theory) :
    principal (localization_ideal_prufer_domain_ideal_theory d) /\
      le (invertible_hull_prufer_domain_ideal_theory d)
        (localization_ideal_prufer_domain_ideal_theory d).
Proof.
  assert (h_principal : principal (localization (base_ideal d))).
  { apply localization_principal. }
  assert (h_hull_to_loc : le (hull (base_ideal d)) (localization (base_ideal d))).
  { apply localization_hull. }
  assert (h_principal' : principal (localization_ideal_prufer_domain_ideal_theory d)).
  { exact h_principal. }
  assert (h_hull_to_loc' :
    le (invertible_hull_prufer_domain_ideal_theory d)
      (localization_ideal_prufer_domain_ideal_theory d)).
  { exact h_hull_to_loc. }
  split.
  - exact h_principal'.
  - exact h_hull_to_loc'.
Qed.

Lemma invertible_times_inverse_prufer_domain_ideal_theory {R : Type}
    {P : PruferStruct_domain_ideal_theory R}
    (d : IdealData_prufer_domain_ideal_theory) :
    mul (invertible_hull_prufer_domain_ideal_theory d)
      (inv (invertible_hull_prufer_domain_ideal_theory d)) = top.
Proof.
  set (I := invertible_hull_prufer_domain_ideal_theory d).
  assert (h_left : le top (mul I (inv I))).
  { apply inv_mul_top. }
  assert (h_right : le (mul I (inv I)) top).
  { apply mul_inv_top. }
  assert (h_top_stable : mul top top = top).
  { apply mul_top. }
  assert (h_top_rewrite : top = mul top top).
  { symmetry. exact h_top_stable. }
  assert (h_reflexive : le top top).
  { apply le_refl. }
  assert (h_eq : mul I (inv I) = top).
  { apply le_antisymm; assumption. }
  exact h_eq.
Qed.

Lemma finite_intersection_stable_prufer_domain_ideal_theory {R : Type}
    {P : PruferStruct_domain_ideal_theory R}
    (d : IdealData_prufer_domain_ideal_theory) :
    le (inter (invertible_hull_prufer_domain_ideal_theory d)
      (content_ideal_prufer_domain_ideal_theory d))
      (invertible_hull_prufer_domain_ideal_theory d) /\
    le (inter (invertible_hull_prufer_domain_ideal_theory d)
      (content_ideal_prufer_domain_ideal_theory d))
      (content_ideal_prufer_domain_ideal_theory d) /\
    le (inter (invertible_hull_prufer_domain_ideal_theory d)
      (content_ideal_prufer_domain_ideal_theory d))
      (hull (invertible_hull_prufer_domain_ideal_theory d)).
Proof.
  assert (h_left :
    le (inter (invertible_hull_prufer_domain_ideal_theory d)
      (content_ideal_prufer_domain_ideal_theory d))
      (invertible_hull_prufer_domain_ideal_theory d)).
  {
    apply inter_left.
  }
  assert (h_right :
    le (inter (invertible_hull_prufer_domain_ideal_theory d)
      (content_ideal_prufer_domain_ideal_theory d))
      (content_ideal_prufer_domain_ideal_theory d)).
  {
    apply inter_right.
  }
  assert (h_hull :
    le (invertible_hull_prufer_domain_ideal_theory d)
      (hull (invertible_hull_prufer_domain_ideal_theory d))).
  {
    apply hull_extensive.
  }
  assert (h_left_to_hull :
    le (inter (invertible_hull_prufer_domain_ideal_theory d)
      (content_ideal_prufer_domain_ideal_theory d))
      (hull (invertible_hull_prufer_domain_ideal_theory d))).
  {
    apply (le_trans _ (invertible_hull_prufer_domain_ideal_theory d)).
    - exact h_left.
    - exact h_hull.
  }
  split.
  - exact h_left.
  - split.
    + exact h_right.
    + exact h_left_to_hull.
Qed.

Lemma content_multiplicative_step_prufer_domain_ideal_theory {R : Type}
    {P : PruferStruct_domain_ideal_theory R}
    (d : IdealData_prufer_domain_ideal_theory) :
    content (rmul (coeff_left d) (coeff_right d)) =
      mul (content_ideal_prufer_domain_ideal_theory d) (content (coeff_right d)).
Proof.
  assert (h_le_raw :
    le (content (rmul (coeff_left d) (coeff_right d)))
      (mul (content (coeff_left d)) (content (coeff_right d)))).
  {
    apply content_mul_le.
  }
  assert (h_ge_raw :
    le (mul (content (coeff_left d)) (content (coeff_right d)))
      (content (rmul (coeff_left d) (coeff_right d)))).
  {
    apply content_mul_ge.
  }
  assert (h_le :
    le (content (rmul (coeff_left d) (coeff_right d)))
      (mul (content_ideal_prufer_domain_ideal_theory d) (content (coeff_right d)))).
  {
    unfold content_ideal_prufer_domain_ideal_theory.
    exact h_le_raw.
  }
  assert (h_ge :
    le (mul (content_ideal_prufer_domain_ideal_theory d) (content (coeff_right d)))
      (content (rmul (coeff_left d) (coeff_right d)))).
  {
    unfold content_ideal_prufer_domain_ideal_theory.
    exact h_ge_raw.
  }
  apply le_antisymm.
  - exact h_le.
  - exact h_ge.
Qed.

Lemma divisor_closure_step_prufer_domain_ideal_theory {R : Type}
    {P : PruferStruct_domain_ideal_theory R}
    (d : IdealData_prufer_domain_ideal_theory) :
    le (localization_ideal_prufer_domain_ideal_theory d)
      (invertible_hull_prufer_domain_ideal_theory d).
Proof.
  assert (h_local_step := localization_principal_step_prufer_domain_ideal_theory d).
  destruct h_local_step as [h_principal h_hull_to_loc].
  assert (h_divisor_raw :
    le (localization_ideal_prufer_domain_ideal_theory d)
      (hull (invertible_hull_prufer_domain_ideal_theory d))).
  {
    apply divisor_closed.
    exact h_hull_to_loc.
  }
  assert (h_hull_idem :
    hull (invertible_hull_prufer_domain_ideal_theory d) =
      invertible_hull_prufer_domain_ideal_theory d).
  {
    unfold invertible_hull_prufer_domain_ideal_theory.
    apply hull_idem.
  }
  rewrite h_hull_idem in h_divisor_raw.
  exact h_divisor_raw.
Qed.

Lemma invertible_factorization_prufer_domain_ideal_theory {R : Type}
    {P : PruferStruct_domain_ideal_theory R}
    (d : IdealData_prufer_domain_ideal_theory) :
    le (mul (localization_ideal_prufer_domain_ideal_theory d)
      (inv (invertible_hull_prufer_domain_ideal_theory d))) top.
Proof.
  set (I := invertible_hull_prufer_domain_ideal_theory d).
  assert (h_div : le (localization_ideal_prufer_domain_ideal_theory d) I).
  { apply divisor_closure_step_prufer_domain_ideal_theory. }
  assert (h_mono_raw :
    le (mul (inv I) (localization_ideal_prufer_domain_ideal_theory d))
      (mul (inv I) I)).
  {
    apply mul_mono_left.
    exact h_div.
  }
  assert (h_mono :
    le (mul (localization_ideal_prufer_domain_ideal_theory d) (inv I))
      (mul I (inv I))).
  {
    rewrite (mul_comm (localization_ideal_prufer_domain_ideal_theory d) (inv I)).
    rewrite (mul_comm I (inv I)).
    exact h_mono_raw.
  }
  assert (h_top : le (mul I (inv I)) top).
  {
    apply mul_inv_top.
  }
  assert (h_chain :
    le (mul (localization_ideal_prufer_domain_ideal_theory d) (inv I)) top).
  {
    apply (le_trans _ (mul I (inv I))).
    - exact h_mono.
    - exact h_top.
  }
  exact h_chain.
Qed.

Lemma ideal_classification_final_prufer_domain_ideal_theory {R : Type}
    {P : PruferStruct_domain_ideal_theory R}
    (d : IdealData_prufer_domain_ideal_theory) :
    principal (localization_ideal_prufer_domain_ideal_theory d) /\
    localization_ideal_prufer_domain_ideal_theory d =
      invertible_hull_prufer_domain_ideal_theory d /\
    le (mul (localization_ideal_prufer_domain_ideal_theory d)
      (inv (invertible_hull_prufer_domain_ideal_theory d))) top /\
    mul (invertible_hull_prufer_domain_ideal_theory d)
      (inv (invertible_hull_prufer_domain_ideal_theory d)) = top.
Proof.
  assert (h_local := localization_principal_step_prufer_domain_ideal_theory d).
  destruct h_local as [h_principal h_backward].
  assert (h_forward :
    le (localization_ideal_prufer_domain_ideal_theory d)
      (invertible_hull_prufer_domain_ideal_theory d)).
  {
    apply divisor_closure_step_prufer_domain_ideal_theory.
  }
  assert (h_eq :
    localization_ideal_prufer_domain_ideal_theory d =
      invertible_hull_prufer_domain_ideal_theory d).
  {
    apply le_antisymm.
    - exact h_forward.
    - exact h_backward.
  }
  assert (h_factor :
    le (mul (localization_ideal_prufer_domain_ideal_theory d)
      (inv (invertible_hull_prufer_domain_ideal_theory d))) top).
  {
    apply invertible_factorization_prufer_domain_ideal_theory.
  }
  assert (h_unit :
    mul (invertible_hull_prufer_domain_ideal_theory d)
      (inv (invertible_hull_prufer_domain_ideal_theory d)) = top).
  {
    apply invertible_times_inverse_prufer_domain_ideal_theory.
  }
  split.
  - exact h_principal.
  - split.
    + exact h_eq.
    + split.
      * exact h_factor.
      * exact h_unit.
Qed.
