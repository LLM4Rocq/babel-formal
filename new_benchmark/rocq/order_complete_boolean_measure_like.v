(*
BENCHMARK_ID: TINY_MATHLIB_BATCH05_ORDER_COMPLETE_BOOLEAN_MEASURE_LIKE
PAIR_STEM: order_complete_boolean_measure_like
MATH_DOMAIN: Order Theory
SOURCE_MATHLIB: Mathlib/Order/CompleteBooleanAlgebra
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class OrderStruct_complete_boolean_measure (A : Type) := {
  le : A -> A -> Prop;
  lower : A -> A;
  upper : A -> A;
  measure : A -> nat;
  le_refl : forall x : A, le x x;
  le_trans : forall {x y z : A}, le x y -> le y z -> le x z;
  lower_mono_axiom : forall {x y : A}, le x y -> le (lower x) (lower y);
  upper_mono_axiom : forall {x y : A}, le x y -> le (upper x) (upper y);
  lower_extensive_axiom : forall x : A, le x (lower x);
  upper_reductive_axiom : forall x : A, le (upper x) x;
  absorb_axiom : forall x : A, lower (upper x) = upper x;
  fixed_axiom : forall x : A, lower x = x <-> upper x = x;
  duality_axiom : forall x : A, measure (lower x) + measure (upper x) = measure x + measure x
}.

Definition LowerOp_complete_boolean_measure {A : Type}
    {S : OrderStruct_complete_boolean_measure A} (x : A) : A :=
  lower x.

Definition UpperOp_complete_boolean_measure {A : Type}
    {S : OrderStruct_complete_boolean_measure A} (x : A) : A :=
  upper x.

Definition FixedSet_complete_boolean_measure {A : Type}
    {S : OrderStruct_complete_boolean_measure A} (x : A) : Prop :=
  LowerOp_complete_boolean_measure x = x /\ UpperOp_complete_boolean_measure x = x.

Definition IterStep_complete_boolean_measure {A : Type}
    {S : OrderStruct_complete_boolean_measure A} (x : A) : A :=
  LowerOp_complete_boolean_measure (UpperOp_complete_boolean_measure x).

Lemma lower_mono_complete_boolean_measure {A : Type}
    {S : OrderStruct_complete_boolean_measure A}
    {x y : A} (hxy : le x y) :
    le (LowerOp_complete_boolean_measure x) (LowerOp_complete_boolean_measure y).
Proof.
  assert (hLower : le (lower x) (lower y)).
  { apply lower_mono_axiom. exact hxy. }
  assert (hx : LowerOp_complete_boolean_measure x = lower x).
  { reflexivity. }
  assert (hy : LowerOp_complete_boolean_measure y = lower y).
  { reflexivity. }
  rewrite hx.
  rewrite hy.
  exact hLower.
Qed.

Lemma upper_mono_complete_boolean_measure {A : Type}
    {S : OrderStruct_complete_boolean_measure A}
    {x y : A} (hxy : le x y) :
    le (UpperOp_complete_boolean_measure x) (UpperOp_complete_boolean_measure y).
Proof.
  assert (hUpper : le (upper x) (upper y)).
  { apply upper_mono_axiom. exact hxy. }
  assert (hx : UpperOp_complete_boolean_measure x = upper x).
  { reflexivity. }
  assert (hy : UpperOp_complete_boolean_measure y = upper y).
  { reflexivity. }
  rewrite hx.
  rewrite hy.
  exact hUpper.
Qed.

Lemma lower_extensive_complete_boolean_measure {A : Type}
    {S : OrderStruct_complete_boolean_measure A}
    (x : A) :
    le x (LowerOp_complete_boolean_measure x).
Proof.
  assert (hExt : le x (lower x)).
  { apply lower_extensive_axiom. }
  assert (hLower : LowerOp_complete_boolean_measure x = lower x).
  { reflexivity. }
  rewrite hLower.
  exact hExt.
Qed.

Lemma upper_reductive_complete_boolean_measure {A : Type}
    {S : OrderStruct_complete_boolean_measure A}
    (x : A) :
    le (UpperOp_complete_boolean_measure x) x.
Proof.
  assert (hRed : le (upper x) x).
  { apply upper_reductive_axiom. }
  assert (hUpper : UpperOp_complete_boolean_measure x = upper x).
  { reflexivity. }
  rewrite hUpper.
  exact hRed.
Qed.

Lemma iteration_stable_complete_boolean_measure {A : Type}
    {S : OrderStruct_complete_boolean_measure A}
    (x : A) :
    IterStep_complete_boolean_measure x = UpperOp_complete_boolean_measure x.
Proof.
  assert (hAbs : lower (upper x) = upper x).
  { apply absorb_axiom. }
  assert (hIter : IterStep_complete_boolean_measure x = lower (upper x)).
  { reflexivity. }
  assert (hUpper : UpperOp_complete_boolean_measure x = upper x).
  { reflexivity. }
  rewrite hIter.
  rewrite hAbs.
  rewrite hUpper.
  reflexivity.
Qed.

Lemma fixedpoint_characterization_complete_boolean_measure {A : Type}
    {S : OrderStruct_complete_boolean_measure A}
    (x : A) :
    FixedSet_complete_boolean_measure x <->
      (LowerOp_complete_boolean_measure x = x /\ IterStep_complete_boolean_measure x = x).
Proof.
  split.
  - intro hFixed.
    destruct hFixed as [hLowerEq hUpperEq].
    assert (hIterUpper : IterStep_complete_boolean_measure x = UpperOp_complete_boolean_measure x).
    { apply iteration_stable_complete_boolean_measure. }
    assert (hIterEq : IterStep_complete_boolean_measure x = x).
    {
      rewrite hIterUpper.
      exact hUpperEq.
    }
    split.
    + exact hLowerEq.
    + exact hIterEq.
  - intro hData.
    destruct hData as [hLowerEq hIterEq].
    assert (hIterUpper : IterStep_complete_boolean_measure x = UpperOp_complete_boolean_measure x).
    { apply iteration_stable_complete_boolean_measure. }
    assert (hUpperEq : UpperOp_complete_boolean_measure x = x).
    {
      rewrite <- hIterUpper.
      exact hIterEq.
    }
    split.
    + exact hLowerEq.
    + exact hUpperEq.
Qed.

Lemma duality_bridge_complete_boolean_measure {A : Type}
    {S : OrderStruct_complete_boolean_measure A}
    (x : A)
    (hFixed : FixedSet_complete_boolean_measure x) :
    measure (LowerOp_complete_boolean_measure x)
      = measure (UpperOp_complete_boolean_measure x).
Proof.
  destruct hFixed as [hLowerEq hUpperEq].
  assert (hDual : measure (lower x) + measure (upper x) = measure x + measure x).
  { apply duality_axiom. }
  assert (hLowerRaw : lower x = x).
  { exact hLowerEq. }
  assert (hUpperRaw : upper x = x).
  { exact hUpperEq. }
  assert (hLowerVal : measure (LowerOp_complete_boolean_measure x) = measure x).
  { rewrite hLowerEq. reflexivity. }
  assert (hUpperVal : measure (UpperOp_complete_boolean_measure x) = measure x).
  { rewrite hUpperEq. reflexivity. }
  assert (hCheck : measure x + measure x = measure x + measure x).
  {
    rewrite <- hDual.
    rewrite hLowerRaw.
    rewrite hUpperRaw.
    reflexivity.
  }
  rewrite hLowerVal.
  rewrite hUpperVal.
  reflexivity.
Qed.
