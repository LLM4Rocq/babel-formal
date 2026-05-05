(*
BENCHMARK_ID: TINY_MATHLIB_BATCH05_TOPOLOGY_HOMOTOPY_LIMIT_GLUING_LIKE
PAIR_STEM: topology_homotopy_limit_gluing_like
MATH_DOMAIN: Topology
SOURCE_MATHLIB: Mathlib/Topology/Sheaves
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.

Class TopStruct_homotopy_limit_gluing (X : Type) := {
  Cover : Type;
  GlobalSection : Type;
  LocalSection : Cover -> Type;
  Refines : Cover -> Cover -> Prop;
  restrict : forall {U V : Cover}, Refines V U -> LocalSection U -> LocalSection V;
  refine_refl : forall U : Cover, Refines U U;
  refine_trans_axiom :
    forall {U V W : Cover}, Refines U V -> Refines V W -> Refines U W;
  local_from_global : forall (U : Cover), GlobalSection -> LocalSection U;
  global_from_local : forall (U : Cover), LocalSection U -> GlobalSection;
  section_roundtrip :
    forall (U : Cover) (s : LocalSection U),
      local_from_global U (global_from_local U s) = s;
  global_roundtrip :
    forall (U : Cover) (g : GlobalSection),
      global_from_local U (local_from_global U g) = g;
  glue_axiom :
    forall (U : Cover) (s : LocalSection U),
      exists g : GlobalSection, local_from_global U g = s;
  glue_unique_axiom :
    forall (U : Cover) (s : LocalSection U) (g1 g2 : GlobalSection),
      local_from_global U g1 = s ->
      local_from_global U g2 = s ->
      g1 = g2;
  compact : GlobalSection -> Prop;
  compact_transfer_axiom :
    forall (U : Cover) (s : LocalSection U), compact (global_from_local U s)
}.

Definition OpenFamily_homotopy_limit_gluing
    (X : Type) `{TopStruct_homotopy_limit_gluing X} : Type :=
  Cover.

Definition CoverRefine_homotopy_limit_gluing
    {X : Type} `{TopStruct_homotopy_limit_gluing X}
    (U V : OpenFamily_homotopy_limit_gluing X) : Prop :=
  Refines U V.

Definition GlueData_homotopy_limit_gluing
    (X : Type) `{TopStruct_homotopy_limit_gluing X} : Type :=
  { U : OpenFamily_homotopy_limit_gluing X & LocalSection U }.

Definition SectionMap_homotopy_limit_gluing
    {X : Type} `{TopStruct_homotopy_limit_gluing X}
    (U : OpenFamily_homotopy_limit_gluing X)
    (g : GlobalSection) : LocalSection U :=
  local_from_global U g.

Lemma refine_trans_homotopy_limit_gluing
    {X : Type} `{TopStruct_homotopy_limit_gluing X}
    {U V W : OpenFamily_homotopy_limit_gluing X}
    (hUV : CoverRefine_homotopy_limit_gluing U V)
    (hVW : CoverRefine_homotopy_limit_gluing V W) :
    CoverRefine_homotopy_limit_gluing U W.
Proof.
  assert (hStep1 : Refines U V).
  { exact hUV. }
  assert (hStep2 : Refines V W).
  { exact hVW. }
  assert (hStep3 : Refines U W).
  { exact (refine_trans_axiom hStep1 hStep2). }
  exact hStep3.
Qed.

Lemma local_to_global_homotopy_limit_gluing
    {X : Type} `{TopStruct_homotopy_limit_gluing X}
    (U : OpenFamily_homotopy_limit_gluing X)
    (s : LocalSection U) :
    exists g : GlobalSection, SectionMap_homotopy_limit_gluing U g = s.
Proof.
  assert (hExist : exists g : GlobalSection, local_from_global U g = s).
  { apply glue_axiom. }
  destruct hExist as [g hg].
  assert (hRewrite : SectionMap_homotopy_limit_gluing U g = s).
  { exact hg. }
  exact (ex_intro _ g hRewrite).
Qed.

Lemma global_to_local_homotopy_limit_gluing
    {X : Type} `{TopStruct_homotopy_limit_gluing X}
    (U : OpenFamily_homotopy_limit_gluing X)
    (g : GlobalSection) :
    global_from_local U (SectionMap_homotopy_limit_gluing U g) = g.
Proof.
  assert (hRaw : global_from_local U (local_from_global U g) = g).
  { apply global_roundtrip. }
  assert (hAsWritten : global_from_local U (SectionMap_homotopy_limit_gluing U g) = g).
  { exact hRaw. }
  exact hAsWritten.
Qed.

Lemma glue_exists_homotopy_limit_gluing
    {X : Type} `{TopStruct_homotopy_limit_gluing X}
    (d : GlueData_homotopy_limit_gluing X) :
    exists g : GlobalSection, SectionMap_homotopy_limit_gluing (projT1 d) g = projT2 d.
Proof.
  destruct d as [U s].
  assert (hLocal : exists g : GlobalSection, SectionMap_homotopy_limit_gluing U g = s).
  { apply local_to_global_homotopy_limit_gluing. }
  destruct hLocal as [g hg].
  exact (ex_intro _ g hg).
Qed.

Lemma glue_unique_homotopy_limit_gluing
    {X : Type} `{TopStruct_homotopy_limit_gluing X}
    (U : OpenFamily_homotopy_limit_gluing X)
    (s : LocalSection U)
    (g1 g2 : GlobalSection)
    (hg1 : SectionMap_homotopy_limit_gluing U g1 = s)
    (hg2 : SectionMap_homotopy_limit_gluing U g2 = s) :
    g1 = g2.
Proof.
  assert (h1 : local_from_global U g1 = s).
  { exact hg1. }
  assert (h2 : local_from_global U g2 = s).
  { exact hg2. }
  assert (hCore : g1 = g2).
  { apply (glue_unique_axiom U s g1 g2); assumption. }
  exact hCore.
Qed.

Lemma descent_equiv_homotopy_limit_gluing
    {X : Type} `{TopStruct_homotopy_limit_gluing X}
    (U : OpenFamily_homotopy_limit_gluing X)
    (s : LocalSection U) :
    (exists g : GlobalSection, SectionMap_homotopy_limit_gluing U g = s) /\
    (forall g1 g2 : GlobalSection,
      SectionMap_homotopy_limit_gluing U g1 = s ->
      SectionMap_homotopy_limit_gluing U g2 = s ->
      g1 = g2).
Proof.
  split.
  - apply local_to_global_homotopy_limit_gluing.
  - intros g1 g2 hg1 hg2.
    apply (glue_unique_homotopy_limit_gluing U s g1 g2); assumption.
Qed.

Lemma compactness_transfer_homotopy_limit_gluing
    {X : Type} `{TopStruct_homotopy_limit_gluing X}
    (U : OpenFamily_homotopy_limit_gluing X)
    (s : LocalSection U)
    (g : GlobalSection)
    (hg : SectionMap_homotopy_limit_gluing U g = s) :
    compact g.
Proof.
  assert (hRound : SectionMap_homotopy_limit_gluing U (global_from_local U s) = s).
  { apply section_roundtrip. }
  assert (hUnique : g = global_from_local U s).
  { apply (glue_unique_axiom U s g (global_from_local U s)); assumption. }
  assert (hCompactBase : compact (global_from_local U s)).
  { apply compact_transfer_axiom. }
  rewrite hUnique.
  exact hCompactBase.
Qed.
