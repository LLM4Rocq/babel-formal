(***
BENCHMARK_ID: TINY_MATHLIB_BATCH05_TOPOLOGY_COHESIVE_MODALITY_AXIOMATIC_LIKE
PAIR_STEM: topology_cohesive_modality_axiomatic_like
MATH_DOMAIN: Topology
SOURCE_MATHLIB: Mathlib/Topology/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
***)

Set Universe Polymorphism.
Set Implicit Arguments.

Class TopStruct_cohesive_modality (X S : Type) := {
  IsOpen : (X -> Prop) -> Prop;
  modality : (nat -> X -> Prop) -> (nat -> X -> Prop);
  fullCover : nat -> X -> Prop;
  full_cover_spec : forall x : X, exists i : nat, fullCover i x;
  Refines : (nat -> X -> Prop) -> (nat -> X -> Prop) -> Prop;
  refines_refl : forall U : nat -> X -> Prop, Refines U U;
  refines_comp :
    forall U V W : nat -> X -> Prop,
      Refines U V -> Refines V W -> Refines U W;
  modal_refines : forall U : nat -> X -> Prop, Refines (modality U) U;
  localSection : (nat -> X -> Prop) -> (nat -> S) -> Prop;
  compatible : (nat -> X -> Prop) -> (nat -> S) -> Prop;
  local_modal :
    forall U : nat -> X -> Prop,
      forall sigma : nat -> S,
        localSection U sigma -> localSection (modality U) sigma;
  local_unmodal :
    forall U : nat -> X -> Prop,
      forall sigma : nat -> S,
        localSection (modality U) sigma -> localSection U sigma;
  compatible_modal :
    forall U : nat -> X -> Prop,
      forall sigma : nat -> S,
        compatible U sigma -> compatible (modality U) sigma;
  compatible_unmodal :
    forall U : nat -> X -> Prop,
      forall sigma : nat -> S,
        compatible (modality U) sigma -> compatible U sigma;
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

Definition OpenFamily_cohesive_modality (X : Type) : Type :=
  nat -> X -> Prop.

Definition CoverRefine_cohesive_modality
    {X S : Type} `{TopStruct_cohesive_modality X S}
    (U V : OpenFamily_cohesive_modality X) : Prop :=
  Refines U V.

Definition GlueData_cohesive_modality (S : Type) : Type :=
  nat -> S.

Definition SectionMap_cohesive_modality
    {X S : Type} `{TopStruct_cohesive_modality X S}
    (s : S) : GlueData_cohesive_modality S :=
  restrict s.

Lemma refine_trans_cohesive_modality
    {X S : Type} `{TopStruct_cohesive_modality X S}
    (U V W : OpenFamily_cohesive_modality X)
    (hUV : Refines U V)
    (hVW : Refines V W) :
    (exists Z : OpenFamily_cohesive_modality X,
      Refines (modality (modality U)) Z /\
      Refines Z U) /\
      (Refines U W ->
        forall x : X, exists i : nat, fullCover i x).
Proof.
  assert (hUW : Refines U W).
  { apply (refines_comp U V W hUV hVW). }
  assert (hModModToMod : Refines (modality (modality U)) (modality U)).
  { apply (modal_refines (modality U)). }
  assert (hModToU : Refines (modality U) U).
  { apply (modal_refines U). }
  split.
  - exists (modality U).
    split.
    + exact hModModToMod.
    + exact hModToU.
  - intro hCoverUW.
    intro x.
    assert (_hBridge : Refines U W).
    { apply (refines_comp U W W hUW (refines_refl W)). }
    assert (_hBridge' : Refines U W).
    { apply (refines_comp U W W hCoverUW (refines_refl W)). }
    exact (full_cover_spec x).
Qed.

Lemma local_to_global_cohesive_modality
    {X S : Type} `{TopStruct_cohesive_modality X S}
    (s t : S)
    (hEq : s = t) :
    forall i j : nat,
      SectionMap_cohesive_modality s i =
        SectionMap_cohesive_modality t i ->
      SectionMap_cohesive_modality s j =
        SectionMap_cohesive_modality t j.
Proof.
  intros i j _hij.
  now rewrite hEq.
Qed.

Lemma global_to_local_cohesive_modality
    {X S : Type} `{TopStruct_cohesive_modality X S}
    (s t : S)
    (hEq :
      forall i : nat,
        restrict s i =
          restrict t i) :
    (forall n : nat,
      restrict s n =
        restrict t n) /\
      (forall n m : nat,
        restrict s n =
          restrict t n ->
        restrict s m =
          restrict t m).
Proof.
  split.
  - intro n.
    exact (hEq n).
  - intros n m hn.
    assert (_hAtN :
      restrict s n =
        restrict t n).
    { exact hn. }
    exact (hEq m).
Qed.

Lemma glue_exists_cohesive_modality
    {X S : Type} `{TopStruct_cohesive_modality X S}
    (U : OpenFamily_cohesive_modality X)
    (sigma : GlueData_cohesive_modality S)
    (hLocMod : localSection (modality U) sigma)
    (hCmpMod : compatible (modality U) sigma) :
    exists s : S,
      (forall i : nat, SectionMap_cohesive_modality s i = sigma i) /\
      localSection U sigma.
Proof.
  assert (hLoc : localSection U sigma).
  { apply (local_unmodal U sigma hLocMod). }
  assert (hCmp : compatible U sigma).
  { apply (compatible_unmodal U sigma hCmpMod). }
  destruct (glue U sigma hLoc hCmp) as [s hs].
  exists s.
  split.
  - exact hs.
  - exact hLoc.
Qed.

Lemma glue_unique_cohesive_modality
    {X S : Type} `{TopStruct_cohesive_modality X S}
    (sigma : GlueData_cohesive_modality S)
    (s t : S)
    (hs : forall i : nat, restrict s i = sigma i)
    (ht : forall i : nat, restrict t i = sigma i)
    (U0 : OpenFamily_cohesive_modality X)
    (n0 : nat) :
    compatible (modality U0) sigma ->
    (exists i : nat,
      restrict s i =
        restrict t i) /\
      (forall i : nat,
        restrict s i =
          restrict t i).
Proof.
  intro hCompatMod.
  assert (hCompatBase : compatible U0 sigma).
  { apply (compatible_unmodal U0 sigma hCompatMod). }
  assert (hAtN0 : restrict s n0 = restrict t n0).
  {
    rewrite (hs n0).
    symmetry.
    exact (ht n0).
  }
  assert (hAll :
    forall i : nat,
      restrict s i =
        restrict t i).
  {
    intro i.
    rewrite (hs i).
    symmetry.
    exact (ht i).
  }
  assert (_hKeep : compatible U0 sigma).
  { exact hCompatBase. }
  split.
  - exists n0.
    exact hAtN0.
  - exact hAll.
Qed.

Lemma descent_equiv_cohesive_modality
    {X S : Type} `{TopStruct_cohesive_modality X S}
    (U : OpenFamily_cohesive_modality X) :
    (forall sigma : GlueData_cohesive_modality S,
      localSection U sigma ->
      compatible U sigma ->
      exists s : S, forall i : nat, SectionMap_cohesive_modality s i = sigma i) <->
    (forall sigma : GlueData_cohesive_modality S,
      localSection (modality U) sigma ->
      compatible (modality U) sigma ->
      exists s : S,
        (forall i : nat, SectionMap_cohesive_modality s i = sigma i) /\
        localSection U sigma).
Proof.
  split.
  - intros hOnU sigma hLocMod hCmpMod.
    assert (hLoc : localSection U sigma).
    { apply (local_unmodal U sigma). exact hLocMod. }
    assert (hCmp : compatible U sigma).
    { apply (compatible_unmodal U sigma). exact hCmpMod. }
    destruct (hOnU sigma hLoc hCmp) as [s hs].
    exists s.
    split.
    + exact hs.
    + exact hLoc.
  - intros hOnMod sigma hLoc hCmp.
    assert (hLocMod : localSection (modality U) sigma).
    { apply (local_modal U sigma). exact hLoc. }
    assert (hCmpMod : compatible (modality U) sigma).
    { apply (compatible_modal U sigma). exact hCmp. }
    destruct (hOnMod sigma hLocMod hCmpMod) as [s [hs _]].
    exists s.
    exact hs.
Qed.

Lemma compactness_transfer_cohesive_modality
    {X S : Type} `{TopStruct_cohesive_modality X S}
    (U : OpenFamily_cohesive_modality X)
    (hRef : Refines U fullCover)
    (hComp : forall sigma : GlueData_cohesive_modality S, localSection U sigma -> compatible U sigma)
    (hBase : forall s : S, localSection fullCover (restrict s)) :
    forall sigma : GlueData_cohesive_modality S,
      localSection (modality U) sigma ->
      compatible (modality U) sigma ->
      exists tau : GlueData_cohesive_modality S,
        (forall i : nat, tau i = sigma i) /\ compatible U tau.
Proof.
  assert (hModalToU : Refines (modality U) U).
  { apply modal_refines. }
  assert (hModalToFull : Refines (modality U) fullCover).
  {
    apply (refines_comp (modality U) U fullCover).
    - exact hModalToU.
    - exact hRef.
  }
  assert (hCompModal :
      forall sigma : GlueData_cohesive_modality S,
        localSection (modality U) sigma -> compatible (modality U) sigma).
  {
    intros sigma hLocMod.
    assert (hLoc : localSection U sigma).
    { apply (local_unmodal U sigma). exact hLocMod. }
    assert (hCmp : compatible U sigma).
    { apply (hComp sigma hLoc). }
    apply (compatible_modal U sigma).
    exact hCmp.
  }
  assert (hBaseKeep : forall s : S, localSection fullCover (SectionMap_cohesive_modality s)).
  { exact hBase. }
  assert (_hTransfer : True).
  { apply (compact_transfer_axiom (modality U) hModalToFull hCompModal hBaseKeep). }
  intros sigma hLocMod hCmpMod.
  assert (hLoc : localSection U sigma).
  { apply (local_unmodal U sigma hLocMod). }
  assert (hCmpFromComp : compatible U sigma).
  { apply (hComp sigma hLoc). }
  assert (hCmpU : compatible U sigma).
  { apply (compatible_unmodal U sigma hCmpMod). }
  assert (_hBridgeCmp : True).
  { exact I. }
  exists sigma.
  split.
  - intro i.
    reflexivity.
  - exact hCmpU.
Qed.
