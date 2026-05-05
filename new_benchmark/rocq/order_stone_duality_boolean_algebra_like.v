(*
BENCHMARK_ID: TINY_MATHLIB_BATCH05_ORDER_STONE_DUALITY_BOOLEAN_ALGEBRA_LIKE
PAIR_STEM: order_stone_duality_boolean_algebra_like
MATH_DOMAIN: Order Theory
SOURCE_MATHLIB: Mathlib/Order/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class OrderStruct_stone_duality_boolean (A : Type) := {
  le : A -> A -> Prop;
  le_refl : forall x : A, le x x;
  le_trans : forall {x y z : A}, le x y -> le y z -> le x z;
  compl : A -> A;
  lower : A -> A;
  upper : A -> A;
  lower_mono_axiom : forall {x y : A}, le x y -> le (lower x) (lower y);
  upper_mono_axiom : forall {x y : A}, le x y -> le (upper x) (upper y);
  lower_extensive_axiom : forall x : A, le x (lower x);
  upper_reductive_axiom : forall x : A, le (upper x) x;
  stone_bridge_axiom : forall x y : A, le (lower x) y <-> le x (upper y);
  deMorgan_axiom : forall x : A, lower (compl x) = compl (upper x)
}.

Infix "<=b" := le (at level 70).

Definition LowerOp_stone_duality_boolean {A : Type}
    {OA : OrderStruct_stone_duality_boolean A} (x : A) : A :=
  lower x.

Definition UpperOp_stone_duality_boolean {A : Type}
    {OA : OrderStruct_stone_duality_boolean A} (x : A) : A :=
  upper x.

Definition FixedSet_stone_duality_boolean {A : Type}
    {OA : OrderStruct_stone_duality_boolean A} (x : A) : Prop :=
  LowerOp_stone_duality_boolean x = x /\
    UpperOp_stone_duality_boolean x = x.

Definition IterStep_stone_duality_boolean {A : Type}
    {OA : OrderStruct_stone_duality_boolean A} (x : A) : A :=
  LowerOp_stone_duality_boolean (UpperOp_stone_duality_boolean x).

Lemma lower_mono_stone_duality_boolean {A : Type}
    {OA : OrderStruct_stone_duality_boolean A}
    {x y : A} (hxy : x <=b y) :
    LowerOp_stone_duality_boolean x <=b LowerOp_stone_duality_boolean y.
Proof.
  assert (hRaw : lower x <=b lower y).
  { exact (lower_mono_axiom hxy). }
  unfold LowerOp_stone_duality_boolean.
  exact hRaw.
Qed.

Lemma upper_mono_stone_duality_boolean {A : Type}
    {OA : OrderStruct_stone_duality_boolean A}
    {x y : A} (hxy : x <=b y) :
    UpperOp_stone_duality_boolean x <=b UpperOp_stone_duality_boolean y.
Proof.
  assert (hRaw : upper x <=b upper y).
  { exact (upper_mono_axiom hxy). }
  unfold UpperOp_stone_duality_boolean.
  exact hRaw.
Qed.

Lemma lower_extensive_stone_duality_boolean {A : Type}
    {OA : OrderStruct_stone_duality_boolean A}
    (x : A) :
    x <=b LowerOp_stone_duality_boolean x.
Proof.
  assert (hRaw : x <=b lower x).
  { exact (lower_extensive_axiom x). }
  unfold LowerOp_stone_duality_boolean.
  exact hRaw.
Qed.

Lemma upper_reductive_stone_duality_boolean {A : Type}
    {OA : OrderStruct_stone_duality_boolean A}
    (x : A) :
    UpperOp_stone_duality_boolean x <=b x.
Proof.
  assert (hRaw : upper x <=b x).
  { exact (upper_reductive_axiom x). }
  unfold UpperOp_stone_duality_boolean.
  exact hRaw.
Qed.

Lemma iteration_stable_stone_duality_boolean {A : Type}
    {OA : OrderStruct_stone_duality_boolean A}
    (hLowerIdem :
      forall x : A,
        LowerOp_stone_duality_boolean (LowerOp_stone_duality_boolean x) =
          LowerOp_stone_duality_boolean x)
    (hUpperIdem :
      forall x : A,
        UpperOp_stone_duality_boolean (UpperOp_stone_duality_boolean x) =
          UpperOp_stone_duality_boolean x)
    (hComm :
      forall x : A,
        UpperOp_stone_duality_boolean (LowerOp_stone_duality_boolean x) =
          LowerOp_stone_duality_boolean (UpperOp_stone_duality_boolean x))
    (x : A) :
    IterStep_stone_duality_boolean (IterStep_stone_duality_boolean x) =
      IterStep_stone_duality_boolean x.
Proof.
  assert (hCommX :
      UpperOp_stone_duality_boolean
          (LowerOp_stone_duality_boolean (UpperOp_stone_duality_boolean x)) =
        LowerOp_stone_duality_boolean
          (UpperOp_stone_duality_boolean (UpperOp_stone_duality_boolean x))).
  { exact (hComm (UpperOp_stone_duality_boolean x)). }
  assert (hLift :
      LowerOp_stone_duality_boolean
          (UpperOp_stone_duality_boolean
            (LowerOp_stone_duality_boolean (UpperOp_stone_duality_boolean x))) =
        LowerOp_stone_duality_boolean
          (LowerOp_stone_duality_boolean
            (UpperOp_stone_duality_boolean (UpperOp_stone_duality_boolean x)))).
  { rewrite hCommX. reflexivity. }
  assert (hCollapseLower :
      LowerOp_stone_duality_boolean
          (LowerOp_stone_duality_boolean
            (UpperOp_stone_duality_boolean (UpperOp_stone_duality_boolean x))) =
        LowerOp_stone_duality_boolean
          (UpperOp_stone_duality_boolean (UpperOp_stone_duality_boolean x))).
  { exact (hLowerIdem (UpperOp_stone_duality_boolean (UpperOp_stone_duality_boolean x))). }
  assert (hCollapseUpper :
      UpperOp_stone_duality_boolean (UpperOp_stone_duality_boolean x) =
        UpperOp_stone_duality_boolean x).
  { exact (hUpperIdem x). }
  unfold IterStep_stone_duality_boolean.
  rewrite hLift.
  rewrite hCollapseLower.
  rewrite hCollapseUpper.
  reflexivity.
Qed.

Lemma fixedpoint_characterization_stone_duality_boolean {A : Type}
    {OA : OrderStruct_stone_duality_boolean A}
    (x : A) (hFix : FixedSet_stone_duality_boolean x) :
    IterStep_stone_duality_boolean x = x.
Proof.
  destruct hFix as [hLower hUpper].
  unfold IterStep_stone_duality_boolean.
  rewrite hUpper.
  exact hLower.
Qed.

Lemma duality_bridge_stone_duality_boolean {A : Type}
    {OA : OrderStruct_stone_duality_boolean A}
    (x y : A) :
    LowerOp_stone_duality_boolean x <=b y <->
      x <=b UpperOp_stone_duality_boolean y.
Proof.
  assert (hRaw : lower x <=b y <-> x <=b upper y).
  { exact (stone_bridge_axiom x y). }
  split.
  - intro hxy.
    assert (hLeft : lower x <=b y).
    { unfold LowerOp_stone_duality_boolean in hxy. exact hxy. }
    assert (hTo : x <=b upper y).
    { exact (proj1 hRaw hLeft). }
    unfold UpperOp_stone_duality_boolean.
    exact hTo.
  - intro hxy.
    assert (hRight : x <=b upper y).
    { unfold UpperOp_stone_duality_boolean in hxy. exact hxy. }
    assert (hTo : lower x <=b y).
    { exact (proj2 hRaw hRight). }
    unfold LowerOp_stone_duality_boolean.
    exact hTo.
Qed.
