(*
BENCHMARK_ID: TINY_MATHLIB_BATCH06_TOPOLOGY_ANR_HOMOTOPY_EXTENSION_LIKE
PAIR_STEM: topology_anr_homotopy_extension_like
MATH_DOMAIN: Topology
SOURCE_MATHLIB: Mathlib/Topology/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class ANRStruct_homotopy_extension (A X Y : Type) := {
  cofibration : A -> X;
  extend : (A -> Y) -> X -> Y;
  extend_on_cofibration :
    forall (f : A -> Y) (a : A),
      extend f (cofibration a) = f a;
  extend_natural :
    forall (f : A -> Y) (u : X -> X),
      (forall a : A, u (cofibration a) = cofibration a) ->
      forall x : X,
        extend f (u x) = extend f x;
  homotopy_fill : (A -> nat -> Y) -> (X -> Y) -> X -> nat -> Y;
  homotopy_boundary :
    forall (F : A -> nat -> Y) (g : X -> Y) (a : A) (n : nat),
      homotopy_fill F g (cofibration a) n = F a n;
  homotopy_start :
    forall (F : A -> nat -> Y) (g : X -> Y) (x : X),
      homotopy_fill F g x 0 = g x;
  glue_fill :
    forall (H1 H2 : X -> nat -> Y),
      (forall x : X, H1 x 0 = H2 x 0) ->
      (forall x : X, H1 x 1 = H2 x 1) ->
      exists H : X -> nat -> Y,
        (forall x : X, H x 0 = H1 x 0) /\
        (forall x : X, H x 1 = H2 x 1);
  endpoint_stable :
    forall (H : X -> nat -> Y),
      (forall a : A, H (cofibration a) 0 = H (cofibration a) 1) ->
      forall a : A,
        forall n : nat,
          H (cofibration a) n = H (cofibration a) 0
}.

Record CylData_anr_homotopy_extension (X Y : Type) := {
  left : X -> Y;
  right : X -> Y;
  bridge : X -> nat -> Y;
  bridge_left : forall x : X, bridge x 0 = left x;
  bridge_right : forall x : X, bridge x 1 = right x
}.

Definition cofibration_map_anr_homotopy_extension
    {A X Y : Type}
    `{ANRStruct_homotopy_extension A X Y} :
    A -> X :=
  cofibration.

Definition extension_operator_anr_homotopy_extension
    {A X Y : Type}
    `{ANRStruct_homotopy_extension A X Y}
    (f : A -> Y) :
    X -> Y :=
  extend f.

Definition terminal_homotopy_anr_homotopy_extension
    {A X Y : Type}
    `{ANRStruct_homotopy_extension A X Y}
    (F : A -> nat -> Y)
    (g : X -> Y) :
    X -> Y :=
  fun x : X => homotopy_fill F g x 1.

Lemma cofibration_lift_exists_anr_homotopy_extension
    {A X Y : Type}
    `{ANRStruct_homotopy_extension A X Y}
    (f : A -> Y) :
    exists g : X -> Y,
      (forall a : A,
        g (cofibration_map_anr_homotopy_extension a) = f a) /\
      (exists Hcyl : X -> nat -> Y,
        (forall x : X, Hcyl x 0 = g x) /\
        (forall a : A,
          Hcyl (cofibration_map_anr_homotopy_extension a) 0 = f a)).
Proof.
  exists (extension_operator_anr_homotopy_extension f).
  split.
  - intro a.
    assert (hAgree : extend f (cofibration a) = f a).
    { exact (extend_on_cofibration f a). }
    exact hAgree.
  - exists (homotopy_fill (fun a : A => fun _n : nat => f a) (extend f)).
    split.
    + intro x.
      exact (homotopy_start (fun a : A => fun _n : nat => f a) (extend f) x).
    + intro a.
      assert (hBoundary0 :
        homotopy_fill (fun a0 : A => fun _n : nat => f a0) (extend f) (cofibration a) 0 =
        (fun a0 : A => fun _n : nat => f a0) a 0).
      { exact (homotopy_boundary (fun a0 : A => fun _n : nat => f a0) (extend f) a 0). }
      transitivity ((fun a0 : A => fun _n : nat => f a0) a 0).
      * exact hBoundary0.
      * reflexivity.
Qed.

Lemma homotopy_extension_step_anr_homotopy_extension
    {A X Y : Type}
    `{ANRStruct_homotopy_extension A X Y}
    (F : A -> nat -> Y)
    (g : X -> Y) :
    (forall a : A,
      terminal_homotopy_anr_homotopy_extension F g
        (cofibration_map_anr_homotopy_extension a) = F a 1) /\
    (forall a : A,
      homotopy_fill F g (cofibration_map_anr_homotopy_extension a) 0 = F a 0) /\
    (forall x : X, homotopy_fill F g x 0 = g x).
Proof.
  split.
  - intro a.
    assert (hBoundary : homotopy_fill F g (cofibration a) 1 = F a 1).
    { exact (homotopy_boundary F g a 1). }
    exact hBoundary.
  - split.
    + intro a.
      assert (hBoundary0 : homotopy_fill F g (cofibration a) 0 = F a 0).
      { exact (homotopy_boundary F g a 0). }
      exact hBoundary0.
    + intro x.
      assert (hStart : homotopy_fill F g x 0 = g x).
      { exact (homotopy_start F g x). }
      exact hStart.
Qed.

Lemma relative_retraction_anr_homotopy_extension
    {A X Y : Type}
    `{ANRStruct_homotopy_extension A X Y}
    (f : A -> Y)
    (u : X -> X)
    (hu : forall a : A,
      u (cofibration_map_anr_homotopy_extension a) =
        cofibration_map_anr_homotopy_extension a) :
    (forall a : A,
      extension_operator_anr_homotopy_extension f
        (u (cofibration_map_anr_homotopy_extension a)) = f a) /\
    (forall x : X,
      extension_operator_anr_homotopy_extension f (u x) =
        extension_operator_anr_homotopy_extension f x).
Proof.
  assert (hNat : forall x : X, extend f (u x) = extend f x).
  { apply (extend_natural f u hu). }
  split.
  - intro a.
    assert (hStep1 : extend f (u (cofibration a)) = extend f (cofibration a)).
    { exact (hNat (cofibration a)). }
    assert (hStep2 : extend f (cofibration a) = f a).
    { exact (extend_on_cofibration f a). }
    transitivity (extend f (cofibration a)).
    + exact hStep1.
    + exact hStep2.
  - intro x.
    exact (hNat x).
Qed.

Lemma glued_homotopy_continuous_anr_homotopy_extension
    {A X Y : Type}
    `{ANRStruct_homotopy_extension A X Y}
    (c : CylData_anr_homotopy_extension X Y)
    (hEq : forall x : X, left c x = right c x) :
    exists H : X -> nat -> Y,
      (forall x : X, H x 0 = left c x) /\
      (forall x : X, H x 1 = right c x) /\
      (forall x : X, H x 0 = H x 1).
Proof.
  assert (hSame0 : forall x : X, bridge c x 0 = bridge c x 0).
  {
    intro x.
    reflexivity.
  }
  assert (hSame1 : forall x : X, bridge c x 1 = bridge c x 1).
  {
    intro x.
    reflexivity.
  }
  destruct (glue_fill (bridge c) (bridge c) hSame0 hSame1) as [Hglue [hH0 hH1]].
  exists Hglue.
  split.
  - intro x.
    transitivity (bridge c x 0).
    + exact (hH0 x).
    + exact (bridge_left c x).
  - split.
    + intro x.
      transitivity (bridge c x 1).
      * exact (hH1 x).
      * exact (bridge_right c x).
    + intro x.
      transitivity (bridge c x 0).
      * exact (hH0 x).
      * transitivity (left c x).
        { exact (bridge_left c x). }
        transitivity (right c x).
        { exact (hEq x). }
        transitivity (bridge c x 1).
        { symmetry. exact (bridge_right c x). }
        { symmetry. exact (hH1 x). }
Qed.

Lemma endpoint_control_anr_homotopy_extension
    {A X Y : Type}
    `{ANRStruct_homotopy_extension A X Y}
    (HH : X -> nat -> Y)
    (hRel : forall a : A,
      HH (cofibration_map_anr_homotopy_extension a) 0 =
        HH (cofibration_map_anr_homotopy_extension a) 1) :
    (forall a : A,
      HH (cofibration_map_anr_homotopy_extension a) 1 =
        HH (cofibration_map_anr_homotopy_extension a) 0) /\
    (forall a : A,
      HH (cofibration_map_anr_homotopy_extension a) 2 =
        HH (cofibration_map_anr_homotopy_extension a) 1).
Proof.
  assert (hStable : forall a : A, forall n : nat, HH (cofibration a) n = HH (cofibration a) 0).
  { apply (endpoint_stable HH hRel). }
  split.
  - intro a.
    exact (hStable a 1).
  - intro a.
    assert (hTwo : HH (cofibration a) 2 = HH (cofibration a) 0).
    { exact (hStable a 2). }
    assert (hOne : HH (cofibration a) 1 = HH (cofibration a) 0).
    { exact (hStable a 1). }
    transitivity (HH (cofibration a) 0).
    + exact hTwo.
    + symmetry. exact hOne.
Qed.

Lemma extension_naturality_anr_homotopy_extension
    {A X Y : Type}
    `{ANRStruct_homotopy_extension A X Y}
    (f : A -> Y)
    (u v : X -> X)
    (hu : forall a : A,
      u (cofibration_map_anr_homotopy_extension a) =
        cofibration_map_anr_homotopy_extension a)
    (hv : forall a : A,
      v (cofibration_map_anr_homotopy_extension a) =
        cofibration_map_anr_homotopy_extension a) :
    (forall x : X,
      extension_operator_anr_homotopy_extension f (u (v x)) =
        extension_operator_anr_homotopy_extension f x) /\
    (forall x : X,
      extension_operator_anr_homotopy_extension f (v (u x)) =
        extension_operator_anr_homotopy_extension f x).
Proof.
  assert (hNatU : forall x : X, extend f (u x) = extend f x).
  { apply (extend_natural f u hu). }
  assert (hNatV : forall x : X, extend f (v x) = extend f x).
  { apply (extend_natural f v hv). }
  split.
  - intro x.
    transitivity (extend f (v x)).
    + exact (hNatU (v x)).
    + exact (hNatV x).
  - intro x.
    transitivity (extend f (u x)).
    + exact (hNatV (u x)).
    + exact (hNatU x).
Qed.

Lemma homotopy_extension_universal_anr_homotopy_extension
    {A X Y : Type}
    `{ANRStruct_homotopy_extension A X Y}
    (F : A -> nat -> Y)
    (g0 : X -> Y)
    (h0 : forall a : A,
      g0 (cofibration_map_anr_homotopy_extension a) = F a 0) :
    exists H : X -> nat -> Y,
      (forall x : X, H x 0 = g0 x) /\
      (forall a : A,
        H (cofibration_map_anr_homotopy_extension a) 1 = F a 1) /\
      (forall a : A,
        H (cofibration_map_anr_homotopy_extension a) 0 =
          g0 (cofibration_map_anr_homotopy_extension a)).
Proof.
  exists (homotopy_fill F g0).
  split.
  - intro x.
    exact (homotopy_start F g0 x).
  - split.
    + intro a.
      exact (homotopy_boundary F g0 a 1).
    + intro a.
      assert (hBdry0 : homotopy_fill F g0 (cofibration a) 0 = F a 0).
      { exact (homotopy_boundary F g0 a 0). }
      transitivity (F a 0).
      * exact hBdry0.
      * symmetry. exact (h0 a).
Qed.
