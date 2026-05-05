(*
BENCHMARK_ID: TINY_MATHLIB_BATCH03_ALG_GROUP_CANCEL
PAIR_STEM: algebra_group_cancel
MATH_DOMAIN: Algebra
SOURCE_MATHLIB: Mathlib/Algebra/Group/Basic
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Implicit Arguments.

Class GroupLike (A : Type) := {
  one : A;
  mul : A -> A -> A;
  inv : A -> A;
  mul_assoc : forall a b c : A, mul (mul a b) c = mul a (mul b c);
  one_mul : forall a : A, mul one a = a;
  mul_one : forall a : A, mul a one = a;
  inv_mul_axiom : forall a : A, mul (inv a) a = one
}.

Infix "*" := mul (at level 40, left associativity).
Notation "1" := one.

Lemma mul_left_cancel {A : Type} `{GroupLike A}
    (a b c : A) (h : a * b = a * c) : b = c.
Proof.
  transitivity (1 * b).
  - symmetry.
    apply one_mul.
  - rewrite <- (inv_mul_axiom a).
    rewrite (mul_assoc (inv a) a b).
    rewrite h.
    rewrite <- (mul_assoc (inv a) a c).
    rewrite (inv_mul_axiom a).
    apply one_mul.
Qed.

Lemma mul_right_cancel {A : Type} `{GroupLike A}
    (a b c : A) (h : b * a = c * a) : b = c.
Proof.
  assert (h_aux : inv a * (a * inv a) = inv a * 1).
  {
    rewrite <- (mul_assoc (inv a) a (inv a)).
    rewrite (inv_mul_axiom a).
    rewrite (one_mul (inv a)).
    symmetry.
    apply mul_one.
  }
  assert (h_inv : a * inv a = 1).
  {
    exact (mul_left_cancel (inv a) (a * inv a) 1 h_aux).
  }
  transitivity (b * 1).
  - symmetry.
    apply mul_one.
  - rewrite <- h_inv.
    rewrite <- (mul_assoc b a (inv a)).
    rewrite h.
    rewrite (mul_assoc c a (inv a)).
    rewrite h_inv.
    apply mul_one.
Qed.

Lemma inv_inv {A : Type} `{GroupLike A} (a : A) : inv (inv a) = a.
Proof.
  assert (h_inv : a * inv a = 1).
  {
    assert (h_cancel : inv a * (a * inv a) = inv a * 1).
    {
      rewrite <- (mul_assoc (inv a) a (inv a)).
      rewrite (inv_mul_axiom a).
      rewrite (one_mul (inv a)).
      symmetry.
      apply mul_one.
    }
    exact (mul_left_cancel (inv a) (a * inv a) 1 h_cancel).
  }
  assert (h_aux : inv (inv a) * inv a = a * inv a).
  {
    rewrite (inv_mul_axiom (inv a)).
    symmetry.
    exact h_inv.
  }
  exact (mul_right_cancel (inv a) (inv (inv a)) a h_aux).
Qed.

Lemma inv_mul {A : Type} `{GroupLike A} (a b : A) : inv a * (a * b) = b.
Proof.
  rewrite <- (mul_assoc (inv a) a b).
  rewrite (inv_mul_axiom a).
  apply one_mul.
Qed.

Lemma mul_inv_cancel {A : Type} `{GroupLike A} (a : A) : a * inv a = 1.
Proof.
  assert (h_aux : inv a * (a * inv a) = inv a * 1).
  {
    rewrite (inv_mul a (inv a)).
    symmetry.
    apply mul_one.
  }
  exact (mul_left_cancel (inv a) (a * inv a) 1 h_aux).
Qed.

Lemma inv_mul_cancel {A : Type} `{GroupLike A} (a : A) : inv a * a = 1.
Proof.
  assert (h_one : a = a * 1).
  {
    symmetry.
    apply mul_one.
  }
  assert (h_congr : inv a * a = inv a * (a * 1)).
  {
    exact (f_equal (fun t : A => inv a * t) h_one).
  }
  rewrite h_congr.
  exact (inv_mul a 1).
Qed.

Lemma eq_inv_of_mul_eq_one {A : Type} `{GroupLike A}
    (a b : A) (h : a * b = 1) : a = inv b.
Proof.
  assert (h_aux : a * b = inv b * b).
  {
    rewrite h.
    symmetry.
    apply inv_mul_cancel.
  }
  exact (mul_right_cancel b a (inv b) h_aux).
Qed.

Lemma eq_inv_of_mul_eq_one_left {A : Type} `{GroupLike A}
    (a b : A) (h : b * a = 1) : a = inv b.
Proof.
  assert (h_swap : b = inv a).
  {
    apply (eq_inv_of_mul_eq_one b a).
    exact h.
  }
  rewrite h_swap.
  symmetry.
  apply inv_inv.
Qed.

Lemma inv_eq_of_mul_eq_one {A : Type} `{GroupLike A}
    (a b : A) (h : a * b = 1) : inv a = b.
Proof.
  assert (h_aux : a * inv a = a * b).
  {
    rewrite mul_inv_cancel.
    symmetry.
    exact h.
  }
  exact (mul_left_cancel a (inv a) b h_aux).
Qed.

Lemma inv_eq_of_eq_inv {A : Type} `{GroupLike A}
    (a b : A) (h : a = inv b) : inv a = b.
Proof.
  assert (h_mul : a * b = 1).
  {
    rewrite h.
    apply inv_mul_cancel.
  }
  apply (inv_eq_of_mul_eq_one a b).
  exact h_mul.
Qed.

Lemma mul_eq_one_of_eq_inv {A : Type} `{GroupLike A}
    (a b : A) (h : a = inv b) : a * b = 1.
Proof.
  rewrite h.
  apply inv_mul_cancel.
Qed.
