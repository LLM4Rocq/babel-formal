(*
BENCHMARK_ID: TINY_MATHLIB_BATCH04_TOPOLOGY_BAIRE_CATEGORY_ADVANCED
PAIR_STEM: topology_baire_category_advanced_like
MATH_DOMAIN: Topology / Analysis
SOURCE_MATHLIB: Mathlib/Topology/Baire/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class TopologicalSpaceLike (A : Type) := {
  IsOpen : (A -> Prop) -> Prop;
  open_univ : IsOpen (fun _ => True);
  open_inter :
    forall s t : A -> Prop, IsOpen s -> IsOpen t -> IsOpen (fun x => s x /\ t x);
  inhabited_space : exists x : A, True
}.

Definition DenseLike {A : Type} `{TopologicalSpaceLike A} (s : A -> Prop) : Prop :=
  forall U : A -> Prop,
    IsOpen U ->
      (exists x : A, U x) ->
        exists x : A, s x /\ U x.

Definition OpenLike {A : Type} `{TopologicalSpaceLike A} (s : A -> Prop) : Prop :=
  IsOpen s.

Definition MeagreLike {A : Type} `{TopologicalSpaceLike A} (s : A -> Prop) : Prop :=
  exists F : nat -> A -> Prop,
    (forall n : nat, DenseLike (fun x => ~ F n x)) /\
      (forall x : A, s x -> exists n : nat, F n x).

Definition ResidualLike {A : Type} `{TopologicalSpaceLike A} (s : A -> Prop) : Prop :=
  exists t : A -> Prop,
    MeagreLike t /\
      (forall x : A, s x <-> ~ t x).

Definition GdeltaLike {A : Type} `{TopologicalSpaceLike A} (s : A -> Prop) : Prop :=
  exists U : nat -> A -> Prop,
    (forall n : nat, OpenLike (U n)) /\
      (forall x : A, s x <-> forall n : nat, U n x).

Lemma residual_dense {A : Type} `{TopologicalSpaceLike A}
    (R : A -> Prop)
    (hR : ResidualLike R)
    (hDenseComp : forall t : A -> Prop, MeagreLike t -> DenseLike (fun x => ~ t x)) :
    DenseLike R.
Proof.
  intros U hU hNonempty.
  destruct hR as [t [htMeagre htChar]].
  assert (hDenseNotT : DenseLike (fun x => ~ t x)).
  {
    apply hDenseComp.
    exact htMeagre.
  }
  assert (hMeet : exists x : A, (~ t x) /\ U x).
  {
    apply hDenseNotT.
    - exact hU.
    - exact hNonempty.
  }
  destruct hMeet as [x [hNotTx hUx]].
  assert (hRx : R x).
  {
    apply (proj2 (htChar x)).
    exact hNotTx.
  }
  exists x.
  split.
  - exact hRx.
  - exact hUx.
Qed.

Lemma meagre_union {A : Type} `{TopologicalSpaceLike A}
    (A0 B0 : A -> Prop)
    (hA : MeagreLike A0)
    (hB : MeagreLike B0)
    (hUnionClosure :
      forall s t : A -> Prop,
        MeagreLike s -> MeagreLike t -> MeagreLike (fun x => s x \/ t x)) :
    MeagreLike (fun x => A0 x \/ B0 x).
Proof.
  assert (hAB : MeagreLike (fun x => A0 x \/ B0 x)).
  {
    apply hUnionClosure.
    - exact hA.
    - exact hB.
  }
  assert (hKeep : MeagreLike (fun x => A0 x \/ B0 x)).
  {
    exact hAB.
  }
  exact hKeep.
Qed.

Lemma baire_intersection_dense {A : Type} `{TopologicalSpaceLike A}
    (U : nat -> A -> Prop)
    (hOpen : forall n : nat, OpenLike (U n))
    (hDenseEach : forall n : nat, DenseLike (U n))
    (hBaire :
      forall V : A -> Prop,
        OpenLike V ->
          (exists x : A, V x) ->
            exists x : A, (forall n : nat, U n x) /\ V x) :
    DenseLike (fun x => forall n : nat, U n x).
Proof.
  intros V hV hNonempty.
  assert (hMeetAll : exists x : A, (forall n : nat, U n x) /\ V x).
  {
    apply hBaire.
    - exact hV.
    - exact hNonempty.
  }
  destruct hMeetAll as [x [hxAll hxV]].
  assert (hKeepDense : forall n : nat, DenseLike (U n)).
  {
    exact hDenseEach.
  }
  assert (hKeepOpen : forall n : nat, OpenLike (U n)).
  {
    exact hOpen.
  }
  exists x.
  split.
  - exact hxAll.
  - exact hxV.
Qed.

Lemma generic_point_exists_like {A : Type} `{TopologicalSpaceLike A}
    (R : A -> Prop)
    (hDenseR : DenseLike R) :
    exists x : A, R x.
Proof.
  assert (hUnivOpen : OpenLike (fun _ : A => True)).
  {
    exact open_univ.
  }
  destruct inhabited_space as [x0 hx0].
  assert (hNonemptyUniv : exists x : A, (fun _ : A => True) x).
  {
    exists x0.
    trivial.
  }
  assert (hMeet : exists x : A, R x /\ (fun _ : A => True) x).
  {
    apply hDenseR.
    - exact hUnivOpen.
    - exact hNonemptyUniv.
  }
  destruct hMeet as [x [hRx hTrue]].
  assert (htriv : True).
  {
    exact hTrue.
  }
  exists x.
  exact hRx.
Qed.

Lemma open_mapping_baire_step {A : Type} `{TopologicalSpaceLike A}
    (f : A -> A) (A0 B0 : A -> Prop)
    (hAOpen : OpenLike A0)
    (hADense : DenseLike A0)
    (hImageOpen :
      forall s : A -> Prop,
        OpenLike s -> OpenLike (fun y => exists x : A, s x /\ y = f x))
    (hImageDense :
      forall s : A -> Prop,
        DenseLike s -> DenseLike (fun y => exists x : A, s x /\ y = f x))
    (hAB : forall x : A, A0 x -> B0 (f x)) :
    DenseLike B0.
Proof.
  intros U hU hNonempty.
  assert (hImgOpenA : OpenLike (fun y => exists x : A, A0 x /\ y = f x)).
  {
    apply hImageOpen.
    exact hAOpen.
  }
  assert (hImgDenseA : DenseLike (fun y : A => exists x : A, A0 x /\ y = f x)).
  {
    apply hImageDense.
    exact hADense.
  }
  assert (hMeet : exists y : A, (exists x : A, A0 x /\ y = f x) /\ U y).
  {
    apply hImgDenseA.
    - exact hU.
    - exact hNonempty.
  }
  destruct hMeet as [y [hyImg hyU]].
  destruct hyImg as [x [hAx hyEq]].
  assert (hBx : B0 (f x)).
  {
    apply hAB.
    exact hAx.
  }
  assert (hkeep : OpenLike (fun y => exists x : A, A0 x /\ y = f x)).
  {
    exact hImgOpenA.
  }
  exists (f x).
  split.
  - exact hBx.
  - rewrite <- hyEq.
    exact hyU.
Qed.

Lemma baire_category_transfer {A : Type} `{TopologicalSpaceLike A}
    (f : A -> A) (R : A -> Prop)
    (hResidual : ResidualLike R)
    (hPreResidual :
      forall s : A -> Prop,
        ResidualLike s -> ResidualLike (fun x => s (f x)))
    (hDenseOfResidual :
      forall s : A -> Prop,
        ResidualLike s -> DenseLike s) :
    DenseLike (fun x => R (f x)).
Proof.
  assert (hPre : ResidualLike (fun x => R (f x))).
  {
    apply hPreResidual.
    exact hResidual.
  }
  assert (hDensePre : DenseLike (fun x => R (f x))).
  {
    apply hDenseOfResidual.
    exact hPre.
  }
  assert (hKeep : DenseLike (fun x => R (f x))).
  {
    exact hDensePre.
  }
  exact hKeep.
Qed.
