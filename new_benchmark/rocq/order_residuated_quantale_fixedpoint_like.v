(*
BENCHMARK_ID: TINY_MATHLIB_BATCH04_ORDER_RESIDUATED_QUANTALE_FIXEDPOINT_LIKE
PAIR_STEM: order_residuated_quantale_fixedpoint_like
MATH_DOMAIN: Order Theory
SOURCE_MATHLIB: Mathlib/Order/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 17
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class QuantaleLike (Q : Type) := {
  leq : Q -> Q -> Prop;
  leq_refl : forall a : Q, leq a a;
  leq_trans : forall {a b c : Q}, leq a b -> leq b c -> leq a c;
  mul : Q -> Q -> Q;
  lres : Q -> Q -> Q;
  rres : Q -> Q -> Q;
  resid_left : forall a b c : Q, leq (mul a b) c <-> leq b (lres a c);
  resid_right : forall a b c : Q, leq (mul a b) c <-> leq a (rres b c)
}.

Infix "<=" := leq (at level 70).
Infix "*q" := mul (at level 40, left associativity).

Definition leftRes {Q : Type} `{QuantaleLike Q} (a c : Q) : Q :=
  lres a c.

Definition rightRes {Q : Type} `{QuantaleLike Q} (b c : Q) : Q :=
  rres b c.

Definition monotone {Q : Type} `{QuantaleLike Q} (f : Q -> Q) : Prop :=
  forall x y : Q, x <= y -> f x <= f y.

Definition closureOp {Q : Type} `{QuantaleLike Q} (a x : Q) : Q :=
  leftRes a (a *q x).

Definition interiorOp {Q : Type} `{QuantaleLike Q} (a x : Q) : Q :=
  (rightRes a x) *q a.

Lemma residuation_left {Q : Type} `{QuantaleLike Q} (a b c : Q) :
    a *q b <= c <-> b <= leftRes a c.
Proof.
  assert (hRaw : a *q b <= c <-> b <= lres a c).
  { exact (resid_left a b c). }
  assert (hDef : leftRes a c = lres a c).
  { reflexivity. }
  split.
  - intro hMul.
    assert (hTo : b <= lres a c).
    { exact (proj1 hRaw hMul). }
    rewrite hDef.
    exact hTo.
  - intro hRes.
    assert (hBack : b <= lres a c).
    {
      rewrite hDef in hRes.
      exact hRes.
    }
    exact (proj2 hRaw hBack).
Qed.

Lemma residuation_right {Q : Type} `{QuantaleLike Q} (a b c : Q) :
    a *q b <= c <-> a <= rightRes b c.
Proof.
  assert (hRaw : a *q b <= c <-> a <= rres b c).
  { exact (resid_right a b c). }
  assert (hDef : rightRes b c = rres b c).
  { reflexivity. }
  split.
  - intro hMul.
    assert (hTo : a <= rres b c).
    { exact (proj1 hRaw hMul). }
    rewrite hDef.
    exact hTo.
  - intro hRes.
    assert (hBack : a <= rres b c).
    {
      rewrite hDef in hRes.
      exact hRes.
    }
    exact (proj2 hRaw hBack).
Qed.

Lemma closure_extensive {Q : Type} `{QuantaleLike Q} (a x : Q) :
    x <= closureOp a x.
Proof.
  assert (hDiag : a *q x <= a *q x).
  { apply leq_refl. }
  assert (hRes : x <= leftRes a (a *q x)).
  { exact (proj1 (residuation_left a x (a *q x)) hDiag). }
  unfold closureOp.
  exact hRes.
Qed.

Lemma interior_reductive {Q : Type} `{QuantaleLike Q} (a x : Q) :
    interiorOp a x <= x.
Proof.
  assert (hRefl : rightRes a x <= rightRes a x).
  { apply leq_refl. }
  assert (hRaw : (rightRes a x) *q a <= x).
  { exact (proj2 (residuation_right (rightRes a x) a x) hRefl). }
  unfold interiorOp.
  exact hRaw.
Qed.

Lemma fixedpoint_transfer_left {Q : Type} `{QuantaleLike Q}
    (a : Q) (f : Q -> Q) (hf : monotone f)
    (hcl_mono : forall u v : Q, u <= v -> closureOp a u <= closureOp a v)
    (x : Q) (hfix : f x <= x) :
    closureOp a (f x) <= closureOp a x.
Proof.
  assert (hSelf : x <= x).
  { apply leq_refl. }
  assert (hMonoSelf : f x <= f x).
  { apply hf; exact hSelf. }
  assert (hAnchor : closureOp a (f x) <= closureOp a (f x)).
  { apply hcl_mono; exact hMonoSelf. }
  assert (hDirect : closureOp a (f x) <= closureOp a x).
  { apply hcl_mono; exact hfix. }
  exact (leq_trans hAnchor hDirect).
Qed.

Lemma fixedpoint_transfer_right {Q : Type} `{QuantaleLike Q}
    (a : Q) (f : Q -> Q) (hf : monotone f)
    (hint_mono : forall u v : Q, u <= v -> interiorOp a u <= interiorOp a v)
    (x : Q) (hfix : x <= f x) :
    interiorOp a x <= interiorOp a (f (f x)).
Proof.
  assert (hFirst : interiorOp a x <= interiorOp a (f x)).
  { apply hint_mono; exact hfix. }
  assert (hNext : f x <= f (f x)).
  { apply hf; exact hfix. }
  assert (hSecond : interiorOp a (f x) <= interiorOp a (f (f x))).
  { apply hint_mono; exact hNext. }
  exact (leq_trans hFirst hSecond).
Qed.

Lemma closure_mul_upper {Q : Type} `{QuantaleLike Q} (a x : Q) :
    a *q closureOp a x <= a *q x.
Proof.
  assert (hResSelf : closureOp a x <= leftRes a (a *q x)).
  {
    unfold closureOp.
    apply leq_refl.
  }
  exact (proj2 (residuation_left a (closureOp a x) (a *q x)) hResSelf).
Qed.

Lemma closure_le_of_mul_le {Q : Type} `{QuantaleLike Q} (a x y : Q)
    (hMul : a *q y <= a *q x) :
    y <= closureOp a x.
Proof.
  assert (hRes : y <= leftRes a (a *q x)).
  { exact (proj1 (residuation_left a y (a *q x)) hMul). }
  unfold closureOp.
  exact hRes.
Qed.

Lemma interior_le_of_rightRes_le {Q : Type} `{QuantaleLike Q} (a x y : Q)
    (hRes : rightRes a x <= rightRes a y) :
    interiorOp a x <= y.
Proof.
  assert (hMul : (rightRes a x) *q a <= y).
  { exact (proj2 (residuation_right (rightRes a x) a y) hRes). }
  unfold interiorOp.
  exact hMul.
Qed.

Lemma fixedpoint_transfer_left_twice {Q : Type} `{QuantaleLike Q}
    (a : Q) (f : Q -> Q) (hf : monotone f)
    (hcl_mono : forall u v : Q, u <= v -> closureOp a u <= closureOp a v)
    (x : Q) (hfix : f x <= x) :
    closureOp a (f (f x)) <= closureOp a x.
Proof.
  assert (hStep1 : f (f x) <= f x).
  { apply hf; exact hfix. }
  assert (hLift1 : closureOp a (f (f x)) <= closureOp a (f x)).
  { apply hcl_mono; exact hStep1. }
  assert (hLift2 : closureOp a (f x) <= closureOp a x).
  { apply hcl_mono; exact hfix. }
  assert (hSelf : closureOp a x <= closureOp a x).
  { apply leq_refl. }
  assert (hTail : closureOp a (f x) <= closureOp a x).
  { exact (leq_trans hLift2 hSelf). }
  exact (leq_trans hLift1 hTail).
Qed.

Lemma fixedpoint_transfer_right_triple {Q : Type} `{QuantaleLike Q}
    (a : Q) (f : Q -> Q) (hf : monotone f)
    (hint_mono : forall u v : Q, u <= v -> interiorOp a u <= interiorOp a v)
    (x : Q) (hfix : x <= f x) :
    interiorOp a x <= interiorOp a (f (f (f x))).
Proof.
  assert (hFirst : interiorOp a x <= interiorOp a (f x)).
  { apply hint_mono; exact hfix. }
  assert (hSecondArg : f x <= f (f x)).
  { apply hf; exact hfix. }
  assert (hSecond : interiorOp a (f x) <= interiorOp a (f (f x))).
  { apply hint_mono; exact hSecondArg. }
  assert (hThirdArg : f (f x) <= f (f (f x))).
  { apply hf; exact hSecondArg. }
  assert (hThird : interiorOp a (f (f x)) <= interiorOp a (f (f (f x)))).
  { apply hint_mono; exact hThirdArg. }
  assert (hFirstSecond : interiorOp a x <= interiorOp a (f (f x))).
  { exact (leq_trans hFirst hSecond). }
  exact (leq_trans hFirstSecond hThird).
Qed.
