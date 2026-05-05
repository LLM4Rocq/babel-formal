(*
BENCHMARK_ID: TINY_MATHLIB_BATCH05_ORDER_QUANTIFIER_ALGEBRA_SEMANTICS_LIKE
PAIR_STEM: order_quantifier_algebra_semantics_like
MATH_DOMAIN: Order Theory
SOURCE_MATHLIB: Mathlib/Order/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class OrderStruct_quantifier_algebra_semantics (A : Type) := {
  le : A -> A -> Prop;
  le_refl : forall x : A, le x x;
  le_trans : forall {x y z : A}, le x y -> le y z -> le x z;
  lower : A -> A;
  upper : A -> A;
  lower_mono_axiom : forall {x y : A}, le x y -> le (lower x) (lower y);
  upper_mono_axiom : forall {x y : A}, le x y -> le (upper x) (upper y);
  lower_extensive_axiom : forall x : A, le x (lower x);
  upper_reductive_axiom : forall x : A, le (upper x) x;
  lower_idem_axiom : forall x : A, lower (lower x) = lower x;
  upper_idem_axiom : forall x : A, upper (upper x) = upper x;
  adjoint_axiom : forall x y : A, le (lower x) y <-> le x (upper y)
}.

Arguments le {A} {_} _ _.
Arguments lower {A} {_} _.
Arguments upper {A} {_} _.

Infix "⊑" := le (at level 70).

Definition LowerOp_quantifier_algebra_semantics
    {A : Type} `{OrderStruct_quantifier_algebra_semantics A} (x : A) : A :=
  lower x.

Definition UpperOp_quantifier_algebra_semantics
    {A : Type} `{OrderStruct_quantifier_algebra_semantics A} (x : A) : A :=
  upper x.

Definition FixedSet_quantifier_algebra_semantics
    {A : Type} `{OrderStruct_quantifier_algebra_semantics A} (x : A) : Prop :=
  LowerOp_quantifier_algebra_semantics x = x /\
  UpperOp_quantifier_algebra_semantics x = x.

Definition IterStep_quantifier_algebra_semantics
    {A : Type} `{OrderStruct_quantifier_algebra_semantics A} (x : A) : A :=
  LowerOp_quantifier_algebra_semantics (LowerOp_quantifier_algebra_semantics x).

Lemma lower_mono_quantifier_algebra_semantics
    {A : Type} `{OrderStruct_quantifier_algebra_semantics A}
    {x y : A} (hxy : x ⊑ y) :
    LowerOp_quantifier_algebra_semantics x ⊑
      LowerOp_quantifier_algebra_semantics y.
Proof.
  assert (hbase : lower x ⊑ lower y).
  { exact (lower_mono_axiom hxy). }
  exact hbase.
Qed.

Lemma upper_mono_quantifier_algebra_semantics
    {A : Type} `{OrderStruct_quantifier_algebra_semantics A}
    {x y : A} (hxy : x ⊑ y) :
    UpperOp_quantifier_algebra_semantics x ⊑
      UpperOp_quantifier_algebra_semantics y.
Proof.
  assert (hbase : upper x ⊑ upper y).
  { exact (upper_mono_axiom hxy). }
  exact hbase.
Qed.

Lemma lower_extensive_quantifier_algebra_semantics
    {A : Type} `{OrderStruct_quantifier_algebra_semantics A}
    (x : A) :
    x ⊑ LowerOp_quantifier_algebra_semantics x.
Proof.
  assert (hraw : x ⊑ lower x).
  { exact (lower_extensive_axiom x). }
  exact hraw.
Qed.

Lemma upper_reductive_quantifier_algebra_semantics
    {A : Type} `{OrderStruct_quantifier_algebra_semantics A}
    (x : A) :
    UpperOp_quantifier_algebra_semantics x ⊑ x.
Proof.
  assert (hraw : upper x ⊑ x).
  { exact (upper_reductive_axiom x). }
  exact hraw.
Qed.

Lemma iteration_stable_quantifier_algebra_semantics
    {A : Type} `{OrderStruct_quantifier_algebra_semantics A}
    (x : A) :
    IterStep_quantifier_algebra_semantics (IterStep_quantifier_algebra_semantics x) =
      IterStep_quantifier_algebra_semantics x /\
    LowerOp_quantifier_algebra_semantics (IterStep_quantifier_algebra_semantics x) =
      IterStep_quantifier_algebra_semantics x.
Proof.
  assert (hIdemOuter :
      LowerOp_quantifier_algebra_semantics
          (LowerOp_quantifier_algebra_semantics
            (LowerOp_quantifier_algebra_semantics
              (LowerOp_quantifier_algebra_semantics x))) =
        LowerOp_quantifier_algebra_semantics
          (LowerOp_quantifier_algebra_semantics x)).
  {
    unfold LowerOp_quantifier_algebra_semantics.
    rewrite (lower_idem_axiom (lower (lower x))).
    rewrite (lower_idem_axiom (lower x)).
    reflexivity.
  }
  assert (hIdemInner :
      LowerOp_quantifier_algebra_semantics
        (IterStep_quantifier_algebra_semantics x) =
      IterStep_quantifier_algebra_semantics x).
  {
    unfold IterStep_quantifier_algebra_semantics, LowerOp_quantifier_algebra_semantics.
    rewrite (lower_idem_axiom (lower x)).
    reflexivity.
  }
  split.
  - exact hIdemOuter.
  - exact hIdemInner.
Qed.

Lemma fixedpoint_characterization_quantifier_algebra_semantics
    {A : Type} `{OrderStruct_quantifier_algebra_semantics A}
    {x : A} (hfix : FixedSet_quantifier_algebra_semantics x) :
    IterStep_quantifier_algebra_semantics x = x /\
    UpperOp_quantifier_algebra_semantics x = x.
Proof.
  destruct hfix as [hlower hupper].
  assert (hIter : IterStep_quantifier_algebra_semantics x =
      LowerOp_quantifier_algebra_semantics (LowerOp_quantifier_algebra_semantics x)).
  { reflexivity. }
  assert (hMain : IterStep_quantifier_algebra_semantics x = x).
  {
    rewrite hIter.
    unfold LowerOp_quantifier_algebra_semantics.
    rewrite (lower_idem_axiom x).
    exact hlower.
  }
  split.
  - exact hMain.
  - exact hupper.
Qed.

Lemma duality_bridge_quantifier_algebra_semantics
    {A : Type} `{OrderStruct_quantifier_algebra_semantics A}
    (x y : A) :
    (LowerOp_quantifier_algebra_semantics x ⊑ y) <->
      (x ⊑ UpperOp_quantifier_algebra_semantics y).
Proof.
  assert (hAdj : le (lower x) y <-> le x (upper y)).
  { exact (adjoint_axiom x y). }
  split.
  - intro hxy.
    assert (hNorm : le (lower x) y).
    { exact hxy. }
    assert (hOut : le x (upper y)).
    { apply (proj1 hAdj). exact hNorm. }
    exact hOut.
  - intro hxy.
    assert (hNorm : le x (upper y)).
    { exact hxy. }
    assert (hOut : le (lower x) y).
    { apply (proj2 hAdj). exact hNorm. }
    exact hOut.
Qed.
