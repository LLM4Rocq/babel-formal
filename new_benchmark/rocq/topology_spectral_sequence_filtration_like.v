(*
BENCHMARK_ID: TINY_MATHLIB_BATCH05_TOPOLOGY_SPECTRAL_SEQUENCE_FILTRATION_LIKE
PAIR_STEM: topology_spectral_sequence_filtration_like
MATH_DOMAIN: Topology
SOURCE_MATHLIB: Mathlib/Topology/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class TopStruct_spectral_sequence_filtration (X E : Type) := {
  IsOpen : (X -> Prop) -> Prop;
  filtrationCover : nat -> X -> Prop;
  Refines : (nat -> X -> Prop) -> (nat -> X -> Prop) -> Prop;
  refines_refl : forall U : nat -> X -> Prop, Refines U U;
  refines_trans :
    forall U V W : nat -> X -> Prop,
      Refines U V -> Refines V W -> Refines U W;
  pageRestrict : E -> nat -> E;
  gluable : (nat -> X -> Prop) -> (nat -> E) -> Prop;
  compatible : (nat -> X -> Prop) -> (nat -> E) -> Prop;
  glue :
    forall U : nat -> X -> Prop,
      forall sigma : nat -> E,
        gluable U sigma ->
        compatible U sigma ->
          exists s : E, forall n : nat, pageRestrict s n = sigma n;
  page_coherence :
    forall s : E,
      forall m n : nat,
        n <= m ->
          pageRestrict (pageRestrict s m) n = pageRestrict s n;
  page_ext :
    forall s t : E,
      (forall n : nat, pageRestrict s n = pageRestrict t n) ->
        s = t;
  compact_transfer_axiom :
    forall U : nat -> X -> Prop,
      Refines U filtrationCover ->
      (forall sigma : nat -> E, gluable U sigma -> compatible U sigma) ->
      (forall s : E, gluable filtrationCover (pageRestrict s)) ->
        True
}.

Definition OpenFamily_spectral_sequence_filtration (X : Type) : Type :=
  nat -> X -> Prop.

Definition CoverRefine_spectral_sequence_filtration
    {X E : Type} `{TopStruct_spectral_sequence_filtration X E}
    (U V : OpenFamily_spectral_sequence_filtration X) : Prop :=
  Refines U V.

Definition GlueData_spectral_sequence_filtration (E : Type) : Type :=
  nat -> E.

Definition SectionMap_spectral_sequence_filtration
    {X E : Type} `{TopStruct_spectral_sequence_filtration X E}
    (s : E) : GlueData_spectral_sequence_filtration E :=
  pageRestrict s.

Lemma refine_trans_spectral_sequence_filtration
    {X E : Type} `{TopStruct_spectral_sequence_filtration X E}
    (U V W : OpenFamily_spectral_sequence_filtration X)
    (hUV : CoverRefine_spectral_sequence_filtration U V)
    (hVW : CoverRefine_spectral_sequence_filtration V W) :
    forall s : E,
      forall m n : nat,
        n <= m ->
          CoverRefine_spectral_sequence_filtration U W /\
          pageRestrict
            (SectionMap_spectral_sequence_filtration s m) n =
          SectionMap_spectral_sequence_filtration s n.
Proof.
  assert (hUW : Refines U W).
  { apply (refines_trans U V W); assumption. }
  intros s m n hnm.
  assert (hCoh :
      pageRestrict
        (SectionMap_spectral_sequence_filtration s m) n =
      SectionMap_spectral_sequence_filtration s n).
  {
    unfold SectionMap_spectral_sequence_filtration.
    apply page_coherence.
    exact hnm.
  }
  split.
  - exact hUW.
  - exact hCoh.
Qed.

Lemma local_to_global_spectral_sequence_filtration
    {X E : Type} `{TopStruct_spectral_sequence_filtration X E}
    (s t : E)
    (hEq : s = t) :
    forall m n : nat,
      n <= m ->
        SectionMap_spectral_sequence_filtration s n =
          SectionMap_spectral_sequence_filtration t n /\
        pageRestrict
          (SectionMap_spectral_sequence_filtration s m) n =
          SectionMap_spectral_sequence_filtration t n.
Proof.
  intros m n hnm.
  assert (hPoint :
      SectionMap_spectral_sequence_filtration s n =
        SectionMap_spectral_sequence_filtration t n).
  { now rewrite hEq. }
  assert (hCoh :
      pageRestrict
        (SectionMap_spectral_sequence_filtration s m) n =
        SectionMap_spectral_sequence_filtration s n).
  {
    unfold SectionMap_spectral_sequence_filtration.
    apply page_coherence.
    exact hnm.
  }
  split.
  - exact hPoint.
  - rewrite hCoh. exact hPoint.
Qed.

Lemma global_to_local_spectral_sequence_filtration
    {X E : Type} `{TopStruct_spectral_sequence_filtration X E}
    (s t : E)
    (hEq :
      forall n : nat,
        SectionMap_spectral_sequence_filtration s n =
          SectionMap_spectral_sequence_filtration t n) :
    (forall m n : nat,
      n <= m ->
        pageRestrict
          (SectionMap_spectral_sequence_filtration s m) n =
          SectionMap_spectral_sequence_filtration t n) ->
    s = t.
Proof.
  intro hCoh.
  assert (hPointwise : forall n : nat, pageRestrict s n = pageRestrict t n).
  {
    intro n.
    exact (hEq n).
  }
  assert (hKeep : forall m n : nat,
      n <= m ->
        pageRestrict
          (SectionMap_spectral_sequence_filtration s m) n =
          SectionMap_spectral_sequence_filtration t n).
  { exact hCoh. }
  apply (page_ext s t).
  exact hPointwise.
Qed.

Lemma glue_exists_spectral_sequence_filtration
    {X E : Type} `{TopStruct_spectral_sequence_filtration X E}
    (U : OpenFamily_spectral_sequence_filtration X)
    (sigma : GlueData_spectral_sequence_filtration E)
    (hLoc : gluable U sigma)
    (hCmp : compatible U sigma) :
    exists s : E,
      (forall n : nat, SectionMap_spectral_sequence_filtration s n = sigma n) /\
      (compatible U sigma ->
        forall m n : nat,
          n <= m ->
            pageRestrict
              (SectionMap_spectral_sequence_filtration s m) n = sigma n).
Proof.
  destruct (glue U sigma hLoc hCmp) as [s hs].
  exists s.
  split.
  - exact hs.
  - intros hCompat m n hnm.
    assert (hKeepCompat : compatible U sigma).
    { exact hCompat. }
    transitivity (SectionMap_spectral_sequence_filtration s n).
    + unfold SectionMap_spectral_sequence_filtration.
      apply page_coherence.
      exact hnm.
    + apply hs.
Qed.

Lemma glue_unique_spectral_sequence_filtration
    {X E : Type} `{TopStruct_spectral_sequence_filtration X E}
    (sigma : GlueData_spectral_sequence_filtration E)
    (s t : E)
    (hs : forall n : nat, SectionMap_spectral_sequence_filtration s n = sigma n)
    (ht : forall n : nat, SectionMap_spectral_sequence_filtration t n = sigma n) :
    exists p : s = t,
      forall m n : nat,
        n <= m ->
          pageRestrict
            (SectionMap_spectral_sequence_filtration s m) n =
          SectionMap_spectral_sequence_filtration t n.
Proof.
  assert (hPointwise :
      forall n : nat,
        SectionMap_spectral_sequence_filtration s n =
          SectionMap_spectral_sequence_filtration t n).
  {
    intro n.
    transitivity (sigma n).
    - apply hs.
    - symmetry. apply ht.
  }
  assert (hST : s = t).
  { apply (page_ext s t). exact hPointwise. }
  exists hST.
  intros m n hnm.
  transitivity (SectionMap_spectral_sequence_filtration s n).
  - unfold SectionMap_spectral_sequence_filtration.
    apply page_coherence.
    exact hnm.
  - apply hPointwise.
Qed.

Lemma descent_equiv_spectral_sequence_filtration
    {X E : Type} `{TopStruct_spectral_sequence_filtration X E}
    (U : OpenFamily_spectral_sequence_filtration X) :
    (forall sigma : GlueData_spectral_sequence_filtration E,
      gluable U sigma ->
      compatible U sigma ->
      exists s : E, forall n : nat, SectionMap_spectral_sequence_filtration s n = sigma n) <->
    (forall sigma : GlueData_spectral_sequence_filtration E,
      gluable U sigma ->
      compatible U sigma ->
      exists s : E,
        (gluable U sigma -> compatible U sigma) /\
        (forall n : nat, SectionMap_spectral_sequence_filtration s n = sigma n)).
Proof.
  split.
  - intros hForward sigma hLoc hCmp.
    destruct (hForward sigma hLoc hCmp) as [s hs].
    exists s.
    split.
    + intro hLocAgain.
      assert (hKeepLoc : gluable U sigma).
      { exact hLocAgain. }
      exact hCmp.
    + exact hs.
  - intros hBackward sigma hLoc hCmp.
    destruct (hBackward sigma hLoc hCmp) as [s [_ hs]].
    exists s.
    exact hs.
Qed.

Lemma compactness_transfer_spectral_sequence_filtration
    {X E : Type} `{TopStruct_spectral_sequence_filtration X E}
    (U : OpenFamily_spectral_sequence_filtration X)
    (hRef : CoverRefine_spectral_sequence_filtration U filtrationCover)
    (hLocCmp : forall sigma : GlueData_spectral_sequence_filtration E, gluable U sigma -> compatible U sigma)
    (hBase : forall s : E, gluable filtrationCover (SectionMap_spectral_sequence_filtration s)) :
    (CoverRefine_spectral_sequence_filtration U filtrationCover -> True) /\
      (forall s : E,
        forall m n : nat,
          n <= m ->
            pageRestrict
              (SectionMap_spectral_sequence_filtration s m) n =
            SectionMap_spectral_sequence_filtration s n).
Proof.
  assert (hTransfer : True).
  { apply (compact_transfer_axiom U hRef hLocCmp hBase). }
  split.
  - intro hRefAgain.
    assert (hKeepRef : CoverRefine_spectral_sequence_filtration U filtrationCover).
    { exact hRefAgain. }
    exact hTransfer.
  - intros s m n hnm.
    unfold SectionMap_spectral_sequence_filtration.
    apply page_coherence.
    exact hnm.
Qed.
