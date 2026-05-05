(*
BENCHMARK_ID: TINY_MATHLIB_BATCH04_TOPOLOGY_COMPACT_OPEN_DUALITY
PAIR_STEM: topology_compact_open_duality_like
MATH_DOMAIN: Topology
SOURCE_MATHLIB: Mathlib/Topology/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class TopologicalSpaceLike (X : Type) := {
  IsOpen : (X -> Prop) -> Prop;
  open_univ : IsOpen (fun _ : X => True);
  open_inter :
    forall U V : X -> Prop,
      IsOpen U -> IsOpen V -> IsOpen (fun x : X => U x /\ V x);
  open_ext :
    forall U V : X -> Prop,
      (forall x : X, U x <-> V x) -> IsOpen U -> IsOpen V
}.

Definition CompactLike {X : Type} `{TopologicalSpaceLike X} (K : X -> Prop) : Prop :=
  forall C : (X -> Prop) -> Prop,
    (forall U : X -> Prop, C U -> IsOpen U) ->
    (forall x : X, K x -> exists U : X -> Prop, C U /\ U x) ->
    exists U : X -> Prop, C U /\ forall x : X, K x -> U x.

Definition OpenLike {X : Type} `{TopologicalSpaceLike X} (U : X -> Prop) : Prop :=
  IsOpen U.

Definition CompactOpenLike {X : Type} `{TopologicalSpaceLike X}
    (K U : X -> Prop) : Prop :=
  CompactLike K /\ OpenLike U.

Definition EvaluationLike {X Y : Type}
    (ev : (X -> Y) -> X -> Y) : Prop :=
  forall f : X -> Y, forall x : X, ev f x = f x.

Definition ExponentialLike {X Y Z : Type}
    (tr : (X -> Y -> Z) -> Y -> X -> Z) : Prop :=
  forall g : X -> Y -> Z, forall y : Y, forall x : X, tr g y x = g x y.

Lemma compact_open_mono_left {X : Type} `{TopologicalSpaceLike X}
    (K1 K2 U : X -> Prop)
    (hsub : forall x : X, K1 x -> K2 x)
    (hext :
      forall C : (X -> Prop) -> Prop,
        (forall Uo : X -> Prop, C Uo -> OpenLike Uo) ->
        (forall x : X, K1 x -> exists Uo : X -> Prop, C Uo /\ Uo x) ->
        (forall x : X, K2 x -> exists Uo : X -> Prop, C Uo /\ Uo x))
    (hco : CompactOpenLike K2 U) :
    CompactOpenLike K1 U.
Proof.
  destruct hco as [hK2 hOpenU].
  assert (hK1 : CompactLike K1).
  {
    intros C hOpen hCover1.
    assert (hCover2 : forall x : X, K2 x -> exists Uo : X -> Prop, C Uo /\ Uo x).
    {
      apply hext.
      - exact hOpen.
      - exact hCover1.
    }
    destruct (hK2 C hOpen hCover2) as [Uo [hCUo hKUo]].
    exists Uo.
    split.
    - exact hCUo.
    - intros x hxK1.
      assert (hxK2 : K2 x).
      {
        apply hsub.
        exact hxK1.
      }
      exact (hKUo x hxK2).
  }
  assert (hsub_copy : forall x : X, K1 x -> K2 x).
  {
    exact hsub.
  }
  split.
  - exact hK1.
  - exact hOpenU.
Qed.

Lemma compact_open_mono_right {X : Type} `{TopologicalSpaceLike X}
    (K U1 U2 : X -> Prop)
    (hopen_mono :
      forall V W : X -> Prop,
        OpenLike V ->
        (forall x : X, V x -> W x) ->
        OpenLike W)
    (hsub : forall x : X, U1 x -> U2 x)
    (hco : CompactOpenLike K U1) :
    CompactOpenLike K U2.
Proof.
  destruct hco as [hK hOpen1].
  assert (hOpen2 : OpenLike U2).
  {
    apply (hopen_mono U1 U2).
    - exact hOpen1.
    - exact hsub.
  }
  split.
  - exact hK.
  - exact hOpen2.
Qed.

Lemma evaluation_continuous_like {X Y : Type}
    `{TopologicalSpaceLike Y} `{TopologicalSpaceLike (X -> Y)}
    (ev : (X -> Y) -> X -> Y)
    (hev : EvaluationLike ev)
    (x : X)
    (V : Y -> Prop)
    (hopenV : OpenLike V)
    (hcont : OpenLike (fun f : X -> Y => V (ev f x))) :
    OpenLike (fun f : X -> Y => V (f x)).
Proof.
  assert (hEq : forall f : X -> Y,
      (fun g : X -> Y => V (ev g x)) f <-> (fun g : X -> Y => V (g x)) f).
  {
    intro f.
    split; intro hf.
    - rewrite (hev f x) in hf.
      exact hf.
    - rewrite (hev f x).
      exact hf.
  }
  assert (hOpenEval : IsOpen (fun f : X -> Y => V (ev f x))).
  {
    exact hcont.
  }
  assert (hopenV_copy : OpenLike V).
  {
    exact hopenV.
  }
  exact (open_ext _ _ hEq hOpenEval).
Qed.

Lemma transpose_continuous_like {X Y Z : Type}
    `{TopologicalSpaceLike X} `{TopologicalSpaceLike Z}
    (tr : (X -> Y -> Z) -> Y -> X -> Z)
    (htr : ExponentialLike tr)
    (g : X -> Y -> Z)
    (y : Y)
    (W : Z -> Prop)
    (hopenW : OpenLike W)
    (hcont : OpenLike (fun x : X => W (tr g y x))) :
    OpenLike (fun x : X => W (g x y)).
Proof.
  assert (hEq : forall x : X,
      (fun t : X => W (tr g y t)) x <-> (fun t : X => W (g t y)) x).
  {
    intro x0.
    split; intro hx.
    - rewrite (htr g y x0) in hx.
      exact hx.
    - rewrite (htr g y x0).
      exact hx.
  }
  assert (hOpenTr : IsOpen (fun x0 : X => W (tr g y x0))).
  {
    exact hcont.
  }
  assert (hopenW_copy : OpenLike W).
  {
    exact hopenW.
  }
  exact (open_ext _ _ hEq hOpenTr).
Qed.

Lemma compact_open_universal_like {X : Type} `{TopologicalSpaceLike X}
    (hleft :
      forall K1 K2 U : X -> Prop,
        (forall x : X, K1 x -> K2 x) ->
        (forall C : (X -> Prop) -> Prop,
          (forall Uo : X -> Prop, C Uo -> OpenLike Uo) ->
          (forall x : X, K1 x -> exists Uo : X -> Prop, C Uo /\ Uo x) ->
          (forall x : X, K2 x -> exists Uo : X -> Prop, C Uo /\ Uo x)) ->
        CompactOpenLike K2 U ->
        CompactOpenLike K1 U)
    (hright :
      forall K U1 U2 : X -> Prop,
        (forall V W : X -> Prop, OpenLike V -> (forall x : X, V x -> W x) -> OpenLike W) ->
        (forall x : X, U1 x -> U2 x) ->
        CompactOpenLike K U1 ->
        CompactOpenLike K U2)
    (K1 K2 U1 U2 : X -> Prop)
    (hsubK : forall x : X, K1 x -> K2 x)
    (hExt :
      forall C : (X -> Prop) -> Prop,
        (forall Uo : X -> Prop, C Uo -> OpenLike Uo) ->
        (forall x : X, K1 x -> exists Uo : X -> Prop, C Uo /\ Uo x) ->
        (forall x : X, K2 x -> exists Uo : X -> Prop, C Uo /\ Uo x))
    (hopen_mono :
      forall V W : X -> Prop,
        OpenLike V ->
        (forall x : X, V x -> W x) ->
        OpenLike W)
    (hsubU : forall x : X, U1 x -> U2 x)
    (hco : CompactOpenLike K2 U1) :
    CompactOpenLike K1 U2.
Proof.
  assert (hleft_step : CompactOpenLike K1 U1).
  {
    apply (hleft K1 K2 U1).
    - exact hsubK.
    - exact hExt.
    - exact hco.
  }
  assert (hright_step : CompactOpenLike K1 U2).
  {
    apply (hright K1 U1 U2).
    - exact hopen_mono.
    - exact hsubU.
    - exact hleft_step.
  }
  exact hright_step.
Qed.

Lemma alexander_subbase_like {X : Type} `{TopologicalSpaceLike X}
    (K : X -> Prop)
    (Sub : (X -> Prop) -> Prop)
    (hSubOpen : forall U : X -> Prop, Sub U -> OpenLike U)
    (hSubCriterion :
      forall C : (X -> Prop) -> Prop,
        (forall U : X -> Prop, C U -> Sub U) ->
        (forall x : X, K x -> exists U : X -> Prop, C U /\ U x) ->
        exists U : X -> Prop, C U /\ forall x : X, K x -> U x)
    (hRefine :
      forall C : (X -> Prop) -> Prop,
        (forall U : X -> Prop, C U -> OpenLike U) ->
        (forall x : X, K x -> exists U : X -> Prop, C U /\ U x) ->
        exists C' : (X -> Prop) -> Prop,
          (forall U : X -> Prop, C' U -> Sub U) /\
          (forall x : X, K x -> exists U : X -> Prop, C' U /\ U x) /\
          (forall U : X -> Prop, C' U -> C U)) :
    CompactLike K.
Proof.
  intros C hOpen hCover.
  destruct (hRefine C hOpen hCover) as [C' [hCSub [hCCover hCToC]]].
  assert (hCOpen : forall U : X -> Prop, C' U -> OpenLike U).
  {
    intros U hU.
    apply hSubOpen.
    apply hCSub.
    exact hU.
  }
  assert (hCOpen_copy : forall U : X -> Prop, C' U -> OpenLike U).
  {
    exact hCOpen.
  }
  destruct (hSubCriterion C' hCSub hCCover) as [U [hU' hUK]].
  exists U.
  split.
  - apply hCToC.
    exact hU'.
  - exact hUK.
Qed.
