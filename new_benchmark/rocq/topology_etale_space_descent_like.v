(*
BENCHMARK_ID: TINY_MATHLIB_BATCH05_TOPOLOGY_ETALE_SPACE_DESCENT_LIKE
PAIR_STEM: topology_etale_space_descent_like
MATH_DOMAIN: Topology
SOURCE_MATHLIB: Mathlib/Topology/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class TopStruct_etale_space_descent (X S : Type) := {
  IsOpen : (X -> Prop) -> Prop;
  fullCover : nat -> X -> Prop;
  full_cover_spec : forall x : X, exists i : nat, fullCover i x;
  Refines : (nat -> X -> Prop) -> (nat -> X -> Prop) -> Prop;
  refines_refl : forall U : nat -> X -> Prop, Refines U U;
  refines_comp :
    forall U V W : nat -> X -> Prop,
      Refines U V -> Refines V W -> Refines U W;
  localSection : (nat -> X -> Prop) -> (nat -> S) -> Prop;
  compatible : (nat -> X -> Prop) -> (nat -> S) -> Prop;
  restrict : S -> nat -> S;
  glue :
    forall U : nat -> X -> Prop,
      forall sigma : nat -> S,
        localSection U sigma ->
        compatible U sigma ->
          exists s : S, forall i : nat, restrict s i = sigma i;
  section_ext :
    forall s t : S,
      (forall i : nat, restrict s i = restrict t i) ->
        s = t;
  compact_transfer_axiom :
    forall U : nat -> X -> Prop,
      Refines U fullCover ->
      (forall sigma : nat -> S, localSection U sigma -> compatible U sigma) ->
      (forall s : S, localSection fullCover (restrict s)) ->
        True
}.

Definition OpenFamily_etale_space_descent (X : Type) : Type :=
  nat -> X -> Prop.

Definition CoverRefine_etale_space_descent
    {X S : Type} `{TopStruct_etale_space_descent X S}
    (U V : OpenFamily_etale_space_descent X) : Prop :=
  Refines U V.

Definition GlueData_etale_space_descent (S : Type) : Type :=
  nat -> S.

Definition SectionMap_etale_space_descent
    {X S : Type} `{TopStruct_etale_space_descent X S}
    (s : S) : GlueData_etale_space_descent S :=
  restrict s.

Lemma refine_trans_etale_space_descent
    {X S : Type} `{TopStruct_etale_space_descent X S}
    (U V W : OpenFamily_etale_space_descent X)
    (hUV : CoverRefine_etale_space_descent U V)
    (hVW : CoverRefine_etale_space_descent V W)
    (x : X) :
    exists i : nat, fullCover i x /\ CoverRefine_etale_space_descent U W.
Proof.
  assert (hUW : Refines U W).
  { apply (refines_comp U V W); assumption. }
  destruct (full_cover_spec x) as [i hi].
  exists i.
  split.
  - exact hi.
  - exact hUW.
Qed.

Lemma local_to_global_etale_space_descent
    {X S : Type} `{TopStruct_etale_space_descent X S}
    (s t : S)
    (hEq : s = t) :
    forall i : nat,
      SectionMap_etale_space_descent s i =
        SectionMap_etale_space_descent t i /\
      restrict s = restrict t.
Proof.
  assert (hMap : restrict s = restrict t).
  { now rewrite hEq. }
  intro i.
  assert (hPt :
      SectionMap_etale_space_descent s i =
        SectionMap_etale_space_descent t i).
  { exact (f_equal (fun f : nat -> S => f i) hMap). }
  split.
  - exact hPt.
  - exact hMap.
Qed.

Lemma global_to_local_etale_space_descent
    {X S : Type} `{TopStruct_etale_space_descent X S}
    (s t : S)
    (hEq :
      forall i : nat,
        SectionMap_etale_space_descent s i =
          SectionMap_etale_space_descent t i) :
    s = t /\
      (forall i : nat,
        SectionMap_etale_space_descent s i =
          SectionMap_etale_space_descent t i) /\
      restrict s = restrict t.
Proof.
  assert (hPointwise : forall i : nat, restrict s i = restrict t i).
  {
    intro i.
    exact (hEq i).
  }
  assert (hST : s = t).
  { apply (section_ext s t). exact hPointwise. }
  assert (hMap : restrict s = restrict t).
  { now rewrite hST. }
  split.
  - exact hST.
  - split.
    + exact hEq.
    + exact hMap.
Qed.

Lemma glue_exists_etale_space_descent
    {X S : Type} `{TopStruct_etale_space_descent X S}
    (U : OpenFamily_etale_space_descent X)
    (sigma : GlueData_etale_space_descent S)
    (hLoc : localSection U sigma)
    (hCmp : compatible U sigma) :
    exists s : S,
      (forall i : nat, SectionMap_etale_space_descent s i = sigma i) /\
      localSection U sigma /\ compatible U sigma.
Proof.
  destruct (glue U sigma hLoc hCmp) as [s hs].
  exists s.
  split.
  - exact hs.
  - split.
    + exact hLoc.
    + exact hCmp.
Qed.

Lemma glue_unique_etale_space_descent
    {X S : Type} `{TopStruct_etale_space_descent X S}
    (sigma : GlueData_etale_space_descent S)
    (s t : S)
    (hs : forall i : nat, SectionMap_etale_space_descent s i = sigma i)
    (ht : forall i : nat, SectionMap_etale_space_descent t i = sigma i) :
    s = t /\
      (forall i : nat,
        SectionMap_etale_space_descent s i =
          SectionMap_etale_space_descent t i) /\
      restrict s = restrict t.
Proof.
  assert (hPointwise :
      forall i : nat,
        SectionMap_etale_space_descent s i =
          SectionMap_etale_space_descent t i).
  {
    intro i.
    transitivity (sigma i).
    - apply hs.
    - symmetry. apply ht.
  }
  assert (hST : s = t).
  { apply (section_ext s t). exact hPointwise. }
  assert (hMap : restrict s = restrict t).
  { now rewrite hST. }
  split.
  - exact hST.
  - split.
    + exact hPointwise.
    + exact hMap.
Qed.

Lemma descent_equiv_etale_space_descent
    {X S : Type} `{TopStruct_etale_space_descent X S}
    (U : OpenFamily_etale_space_descent X) :
    (forall sigma : GlueData_etale_space_descent S,
      localSection U sigma ->
      compatible U sigma ->
      exists s : S, forall i : nat, SectionMap_etale_space_descent s i = sigma i) <->
    (forall sigma : GlueData_etale_space_descent S,
      localSection U sigma ->
      compatible U sigma ->
      exists s : S,
        (forall i : nat, SectionMap_etale_space_descent s i = sigma i) /\
        localSection U sigma /\ compatible U sigma).
Proof.
  split.
  - intros hForward sigma hLoc hCmp.
    destruct (hForward sigma hLoc hCmp) as [s hs].
    exists s.
    split.
    + exact hs.
    + split.
      * exact hLoc.
      * exact hCmp.
  - intros hBackward sigma hLoc hCmp.
    destruct (hBackward sigma hLoc hCmp) as [s [hs _]].
    exists s.
    exact hs.
Qed.

Lemma compactness_transfer_etale_space_descent
    {X S : Type} `{TopStruct_etale_space_descent X S}
    (U : OpenFamily_etale_space_descent X)
    (hRef : CoverRefine_etale_space_descent U fullCover)
    (hComp : forall sigma : GlueData_etale_space_descent S, localSection U sigma -> compatible U sigma)
    (hBase : forall s : S, localSection fullCover (SectionMap_etale_space_descent s))
    (x : X) :
    exists i : nat, fullCover i x /\ True /\ CoverRefine_etale_space_descent U fullCover.
Proof.
  assert (hTransfer : True).
  { apply (compact_transfer_axiom U hRef hComp hBase). }
  destruct (full_cover_spec x) as [i hi].
  exists i.
  split.
  - exact hi.
  - split.
    + exact hTransfer.
    + exact hRef.
Qed.
