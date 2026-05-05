(*
BENCHMARK_ID: TINY_MATHLIB_BATCH05_TOPOLOGY_SHAPE_THEORY_PROFINITE_LIKE
PAIR_STEM: topology_shape_theory_profinite_like
MATH_DOMAIN: Topology
SOURCE_MATHLIB: Mathlib/Topology/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class TopStruct_shape_theory_profinite (X S : Type) := {
  IsClopen : (X -> Prop) -> Prop;
  profiniteCover : nat -> X -> Prop;
  cover_spec : forall x : X, exists i : nat, profiniteCover i x;
  Refines : (nat -> X -> Prop) -> (nat -> X -> Prop) -> Prop;
  refines_refl : forall U : nat -> X -> Prop, Refines U U;
  refines_comp :
    forall U V W : nat -> X -> Prop,
      Refines U V -> Refines V W -> Refines U W;
  localSection : (nat -> X -> Prop) -> (nat -> S) -> Prop;
  coherent : (nat -> X -> Prop) -> (nat -> S) -> Prop;
  restrict : S -> nat -> S;
  glue :
    forall U : nat -> X -> Prop,
      forall sigma : nat -> S,
        localSection U sigma ->
        coherent U sigma ->
          exists s : S, forall i : nat, restrict s i = sigma i;
  section_ext :
    forall s t : S,
      (forall i : nat, restrict s i = restrict t i) ->
        s = t;
  compact_transfer_axiom :
    forall U : nat -> X -> Prop,
      Refines U profiniteCover ->
      (forall sigma : nat -> S, localSection U sigma -> coherent U sigma) ->
      (forall s : S, localSection profiniteCover (restrict s)) ->
      forall sigma : nat -> S,
        localSection U sigma ->
        coherent U sigma ->
          exists s : S, forall i : nat, restrict s i = sigma i
}.

Definition OpenFamily_shape_theory_profinite (X : Type) : Type :=
  nat -> X -> Prop.

Definition CoverRefine_shape_theory_profinite
    {X S : Type} `{TopStruct_shape_theory_profinite X S}
    (U V : OpenFamily_shape_theory_profinite X) : Prop :=
  Refines U V.

Definition GlueData_shape_theory_profinite (S : Type) : Type :=
  nat -> S.

Definition SectionMap_shape_theory_profinite
    {X S : Type} `{TopStruct_shape_theory_profinite X S}
    (s : S) : GlueData_shape_theory_profinite S :=
  restrict s.

Lemma refine_trans_shape_theory_profinite
    {X S : Type} `{TopStruct_shape_theory_profinite X S}
    (U V W : OpenFamily_shape_theory_profinite X)
    (hUV : CoverRefine_shape_theory_profinite U V)
    (hVW : CoverRefine_shape_theory_profinite V W) :
    CoverRefine_shape_theory_profinite U W /\
      (forall x : X, exists i : nat, profiniteCover i x).
Proof.
  assert (hStepUV : Refines U V).
  { exact hUV. }
  assert (hStepVW : Refines V W).
  { exact hVW. }
  assert (hComposed : Refines U W).
  { apply (refines_comp U V W); assumption. }
  split.
  - exact hComposed.
  - intro x.
    apply cover_spec.
Qed.

Lemma local_to_global_shape_theory_profinite
    {X S : Type} `{TopStruct_shape_theory_profinite X S}
    (s t : S)
    (hEq : s = t) :
    (forall i : nat,
      SectionMap_shape_theory_profinite s i =
      SectionMap_shape_theory_profinite t i) /\
      restrict s = restrict t.
Proof.
  assert (hMapEq : restrict s = restrict t).
  { now rewrite hEq. }
  split.
  - intro i.
    exact (f_equal (fun f : nat -> S => f i) hMapEq).
  - exact hMapEq.
Qed.

Lemma global_to_local_shape_theory_profinite
    {X S : Type} `{TopStruct_shape_theory_profinite X S}
    (s t : S)
    (hEq :
      forall i : nat,
        SectionMap_shape_theory_profinite s i =
          SectionMap_shape_theory_profinite t i) :
    s = t /\ restrict s = restrict t.
Proof.
  assert (hPointwise : forall i : nat, restrict s i = restrict t i).
  {
    intro i.
    exact (hEq i).
  }
  assert (hCollapsed : s = t).
  { apply (section_ext s t). exact hPointwise. }
  assert (hRestrict : restrict s = restrict t).
  { now rewrite hCollapsed. }
  split.
  - exact hCollapsed.
  - exact hRestrict.
Qed.

Lemma glue_exists_shape_theory_profinite
    {X S : Type} `{TopStruct_shape_theory_profinite X S}
    (U : OpenFamily_shape_theory_profinite X)
    (sigma : GlueData_shape_theory_profinite S)
    (hLoc : localSection U sigma)
    (hCoh : coherent U sigma) :
    exists s : S,
      (forall i : nat,
        SectionMap_shape_theory_profinite s i = sigma i) /\
      forall t : S,
        (forall i : nat,
          SectionMap_shape_theory_profinite t i = sigma i) ->
          t = s.
Proof.
  destruct (glue U sigma hLoc hCoh) as [s hs].
  assert (hPointwise : forall i : nat, SectionMap_shape_theory_profinite s i = sigma i).
  {
    intro i.
    exact (hs i).
  }
  assert (hUnique :
      forall t : S,
        (forall i : nat,
          SectionMap_shape_theory_profinite t i = sigma i) ->
          t = s).
  {
    intros t ht.
    assert (hPair :
      forall i : nat,
        SectionMap_shape_theory_profinite t i =
          SectionMap_shape_theory_profinite s i).
    {
      intro i.
      transitivity (sigma i).
      - exact (ht i).
      - symmetry. exact (hs i).
    }
    apply (section_ext t s).
    exact hPair.
  }
  exists s.
  split.
  - exact hPointwise.
  - exact hUnique.
Qed.

Lemma glue_unique_shape_theory_profinite
    {X S : Type} `{TopStruct_shape_theory_profinite X S}
    (sigma : GlueData_shape_theory_profinite S)
    (s t : S)
    (hs : forall i : nat, SectionMap_shape_theory_profinite s i = sigma i)
    (ht : forall i : nat, SectionMap_shape_theory_profinite t i = sigma i) :
    s = t /\ restrict s = restrict t.
Proof.
  assert (hPointwise :
      forall i : nat,
        SectionMap_shape_theory_profinite s i =
        SectionMap_shape_theory_profinite t i).
  {
    intro i.
    pose proof (hs i) as hs'.
    pose proof (ht i) as ht'.
    rewrite hs'.
    rewrite ht'.
    reflexivity.
  }
  assert (hEq : s = t).
  { apply (section_ext s t). exact hPointwise. }
  assert (hRestrict : restrict s = restrict t).
  { now rewrite hEq. }
  split.
  - exact hEq.
  - exact hRestrict.
Qed.

Lemma descent_equiv_shape_theory_profinite
    {X S : Type} `{TopStruct_shape_theory_profinite X S}
    (U : OpenFamily_shape_theory_profinite X) :
    (forall sigma : GlueData_shape_theory_profinite S,
      localSection U sigma ->
      coherent U sigma ->
      exists s : S,
        (forall i : nat,
          SectionMap_shape_theory_profinite s i = sigma i) /\
        forall t : S,
          (forall i : nat,
            SectionMap_shape_theory_profinite t i = sigma i) ->
            t = s) <->
    (forall sigma : GlueData_shape_theory_profinite S,
      localSection U sigma ->
      coherent U sigma ->
      exists s : S,
        (forall i : nat,
          SectionMap_shape_theory_profinite s i = sigma i) /\
        forall t : S,
          (forall i : nat,
            SectionMap_shape_theory_profinite t i = sigma i) ->
            restrict t = restrict s).
Proof.
  split.
  - intros hExist sigma hLoc hCoh.
    destruct (hExist sigma hLoc hCoh) as [s [hs huniq]].
    exists s.
    split.
    + exact hs.
    + intros t ht.
      now rewrite (huniq t ht).
  - intros hStrong sigma hLoc hCoh.
    destruct (hStrong sigma hLoc hCoh) as [s [hs huniqR]].
    exists s.
    split.
    + exact hs.
    + intros t ht.
      assert (hEqRestrict : restrict t = restrict s).
      { exact (huniqR t ht). }
      assert (hEqPoint :
        forall i : nat,
          SectionMap_shape_theory_profinite t i =
            SectionMap_shape_theory_profinite s i).
      {
        intro i.
        exact (f_equal (fun f : nat -> S => f i) hEqRestrict).
      }
      apply (section_ext t s).
      exact hEqPoint.
Qed.

Lemma compactness_transfer_shape_theory_profinite
    {X S : Type} `{TopStruct_shape_theory_profinite X S}
    (U : OpenFamily_shape_theory_profinite X)
    (hRef : CoverRefine_shape_theory_profinite U profiniteCover)
    (hComp : forall sigma : GlueData_shape_theory_profinite S, localSection U sigma -> coherent U sigma)
    (hBase :
      forall s : S,
        localSection profiniteCover (SectionMap_shape_theory_profinite s)) :
    forall sigma : GlueData_shape_theory_profinite S,
      localSection U sigma ->
      coherent U sigma ->
      exists s : S,
        (forall i : nat,
          SectionMap_shape_theory_profinite s i = sigma i) /\
        forall t : S,
          (forall i : nat,
            SectionMap_shape_theory_profinite t i = sigma i) ->
            restrict t = restrict s.
Proof.
  intros sigma hLoc hCoh.
  assert (hExist :
      exists s : S,
        forall i : nat,
          SectionMap_shape_theory_profinite s i = sigma i).
  { apply (compact_transfer_axiom U hRef hComp hBase sigma hLoc hCoh). }
  destruct hExist as [s hs].
  exists s.
  split.
  - exact hs.
  - intros t ht.
    assert (hUnique : t = s).
    {
      apply (section_ext t s).
      intro i.
      transitivity (sigma i).
      - exact (ht i).
      - symmetry. exact (hs i).
    }
    now rewrite hUnique.
Qed.
