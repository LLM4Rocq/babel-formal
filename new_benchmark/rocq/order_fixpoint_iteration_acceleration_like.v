(*
BENCHMARK_ID: TINY_MATHLIB_BATCH05_ORDER_FIXPOINT_ITERATION_ACCELERATION_LIKE
PAIR_STEM: order_fixpoint_iteration_acceleration_like
MATH_DOMAIN: Order Theory
SOURCE_MATHLIB: Mathlib/Order/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class OrderStruct_fixpoint_iteration_acceleration (A : Type) := {
  le : A -> A -> Prop;
  le_refl : forall x : A, le x x;
  le_trans : forall {x y z : A}, le x y -> le y z -> le x z;
  lower : A -> A;
  upper : A -> A;
  lower_mono_axiom : forall {x y : A}, le x y -> le (lower x) (lower y);
  upper_mono_axiom : forall {x y : A}, le x y -> le (upper x) (upper y);
  lower_extensive_axiom : forall x : A, le x (lower x);
  upper_reductive_axiom : forall x : A, le (upper x) x;
  bridge_axiom : forall x : A, le (lower (upper x)) (upper (lower x))
}.

Definition LowerOp_fixpoint_iteration_acceleration {A : Type}
    `{OrderStruct_fixpoint_iteration_acceleration A} : A -> A :=
  lower.

Definition UpperOp_fixpoint_iteration_acceleration {A : Type}
    `{OrderStruct_fixpoint_iteration_acceleration A} : A -> A :=
  upper.

Definition FixedSet_fixpoint_iteration_acceleration {A : Type}
    `{OrderStruct_fixpoint_iteration_acceleration A} (x : A) : Prop :=
  LowerOp_fixpoint_iteration_acceleration x = x /\
    UpperOp_fixpoint_iteration_acceleration x = x.

Definition IterStep_fixpoint_iteration_acceleration {A : Type}
    `{OrderStruct_fixpoint_iteration_acceleration A} (x : A) : A :=
  LowerOp_fixpoint_iteration_acceleration (UpperOp_fixpoint_iteration_acceleration x).

Lemma lower_mono_fixpoint_iteration_acceleration {A : Type}
    `{OrderStruct_fixpoint_iteration_acceleration A}
    {x y : A}
    (hxy : le x y) :
    le (LowerOp_fixpoint_iteration_acceleration x)
      (LowerOp_fixpoint_iteration_acceleration y).
Proof.
  assert (hraw : le (lower x) (lower y)).
  { apply lower_mono_axiom. exact hxy. }
  exact hraw.
Qed.

Lemma upper_mono_fixpoint_iteration_acceleration {A : Type}
    `{OrderStruct_fixpoint_iteration_acceleration A}
    {x y : A}
    (hxy : le x y) :
    le (UpperOp_fixpoint_iteration_acceleration x)
      (UpperOp_fixpoint_iteration_acceleration y).
Proof.
  assert (hraw : le (upper x) (upper y)).
  { apply upper_mono_axiom. exact hxy. }
  exact hraw.
Qed.

Lemma lower_extensive_fixpoint_iteration_acceleration {A : Type}
    `{OrderStruct_fixpoint_iteration_acceleration A}
    (x : A) :
    le x (LowerOp_fixpoint_iteration_acceleration x) /\
    le (LowerOp_fixpoint_iteration_acceleration x)
      (LowerOp_fixpoint_iteration_acceleration x).
Proof.
  assert (hbase : le x (lower x)).
  { apply lower_extensive_axiom. }
  assert (hself : le (LowerOp_fixpoint_iteration_acceleration x)
      (LowerOp_fixpoint_iteration_acceleration x)).
  { apply le_refl. }
  split.
  - exact hbase.
  - exact hself.
Qed.

Lemma upper_reductive_fixpoint_iteration_acceleration {A : Type}
    `{OrderStruct_fixpoint_iteration_acceleration A}
    (x : A) :
    le (UpperOp_fixpoint_iteration_acceleration x) x.
Proof.
  assert (hbase : le (upper x) x).
  { apply upper_reductive_axiom. }
  exact hbase.
Qed.

Lemma iteration_stable_fixpoint_iteration_acceleration {A : Type}
    `{OrderStruct_fixpoint_iteration_acceleration A}
    (x : A) :
    le (IterStep_fixpoint_iteration_acceleration x)
      (UpperOp_fixpoint_iteration_acceleration
        (LowerOp_fixpoint_iteration_acceleration x)) /\
    le (IterStep_fixpoint_iteration_acceleration x)
      (IterStep_fixpoint_iteration_acceleration x).
Proof.
  assert (hbridge : le (lower (upper x)) (upper (lower x))).
  { apply bridge_axiom. }
  assert (hstable : le (IterStep_fixpoint_iteration_acceleration x)
      (UpperOp_fixpoint_iteration_acceleration
        (LowerOp_fixpoint_iteration_acceleration x))).
  { exact hbridge. }
  assert (hself : le (IterStep_fixpoint_iteration_acceleration x)
      (IterStep_fixpoint_iteration_acceleration x)).
  { apply le_refl. }
  split.
  - exact hstable.
  - exact hself.
Qed.

Lemma fixedpoint_characterization_fixpoint_iteration_acceleration {A : Type}
    `{OrderStruct_fixpoint_iteration_acceleration A}
    {x : A}
    (hfix : FixedSet_fixpoint_iteration_acceleration x) :
    IterStep_fixpoint_iteration_acceleration x = x /\
    le (IterStep_fixpoint_iteration_acceleration x)
      (IterStep_fixpoint_iteration_acceleration x).
Proof.
  destruct hfix as [hlow hupp].
  assert (hEq : IterStep_fixpoint_iteration_acceleration x = x).
  {
    unfold IterStep_fixpoint_iteration_acceleration.
    rewrite hupp.
    exact hlow.
  }
  assert (hSelf : le (IterStep_fixpoint_iteration_acceleration x)
      (IterStep_fixpoint_iteration_acceleration x)).
  { apply le_refl. }
  split.
  - exact hEq.
  - exact hSelf.
Qed.

Lemma duality_bridge_fixpoint_iteration_acceleration {A : Type}
    `{OrderStruct_fixpoint_iteration_acceleration A}
    (x : A) :
    exists y : A,
      y = IterStep_fixpoint_iteration_acceleration x /\
      le y
        (UpperOp_fixpoint_iteration_acceleration
          (LowerOp_fixpoint_iteration_acceleration x)).
Proof.
  exists (IterStep_fixpoint_iteration_acceleration x).
  split.
  - reflexivity.
  - pose proof (iteration_stable_fixpoint_iteration_acceleration x) as hPair.
    exact (proj1 hPair).
Qed.
