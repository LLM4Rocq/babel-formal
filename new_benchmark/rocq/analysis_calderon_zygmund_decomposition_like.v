(*
BENCHMARK_ID: TINY_MATHLIB_BATCH05_ANALYSIS_CALDERON_ZYGMUND_DECOMPOSITION_LIKE
PAIR_STEM: analysis_calderon_zygmund_decomposition_like
MATH_DOMAIN: Analysis
SOURCE_MATHLIB: Mathlib/Analysis/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 17
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class AnalysisStruct_calderon_zygmund_decomposition (H : Type) := {
  dominates : H -> H -> Prop;
  dominates_refl : forall x : H, dominates x x;
  dominates_trans : forall x y z : H, dominates x y -> dominates y z -> dominates x z;
  normCtrl : H -> Prop;
  localChart : H -> H;
  transform : H -> H;
  generator : H -> H;
  local_estimate_axiom : forall x : H, dominates (localChart x) x;
  patching_axiom : forall x y : H, dominates x y -> dominates (transform x) (transform y);
  isometry_axiom : forall x : H, dominates (transform x) x /\ dominates x (transform x);
  decomposition_axiom : forall x : H, dominates (generator x) (localChart x);
  symbol_comp_axiom : forall x : H, transform (generator x) = generator (transform x);
  semigroup_axiom : forall x : H, dominates (generator (generator x)) (generator x);
  regularity_axiom :
    forall x : H,
      dominates (generator x) x ->
      normCtrl x ->
        normCtrl (transform x)
}.

Definition NormCtrl_calderon_zygmund_decomposition
    {H : Type} `{AnalysisStruct_calderon_zygmund_decomposition H} :
    H -> Prop :=
  normCtrl.

Definition LocalChart_calderon_zygmund_decomposition
    {H : Type} `{AnalysisStruct_calderon_zygmund_decomposition H} :
    H -> H :=
  localChart.

Definition Transform_calderon_zygmund_decomposition
    {H : Type} `{AnalysisStruct_calderon_zygmund_decomposition H} :
    H -> H :=
  transform.

Definition Generator_calderon_zygmund_decomposition
    {H : Type} `{AnalysisStruct_calderon_zygmund_decomposition H} :
    H -> H :=
  generator.

Lemma local_estimate_calderon_zygmund_decomposition
    {H : Type} `{AnalysisStruct_calderon_zygmund_decomposition H}
    (x : H) :
    dominates
      (LocalChart_calderon_zygmund_decomposition x)
      x /\
    dominates
      (Transform_calderon_zygmund_decomposition
        (LocalChart_calderon_zygmund_decomposition x))
      (Transform_calderon_zygmund_decomposition x).
Proof.
  assert (hLocal : dominates (localChart x) x).
  { apply (local_estimate_axiom x). }
  assert (hTransformed : dominates (transform (localChart x)) (transform x)).
  { apply (patching_axiom (localChart x) x hLocal). }
  split.
  - exact hLocal.
  - exact hTransformed.
Qed.

Lemma patching_estimate_calderon_zygmund_decomposition
    {H : Type} `{AnalysisStruct_calderon_zygmund_decomposition H}
    (x : H) :
    dominates
      (Transform_calderon_zygmund_decomposition
        (LocalChart_calderon_zygmund_decomposition x))
      (Transform_calderon_zygmund_decomposition x) /\
    dominates
      (Transform_calderon_zygmund_decomposition x)
      (Transform_calderon_zygmund_decomposition x).
Proof.
  pose proof (local_estimate_calderon_zygmund_decomposition x) as hBase.
  assert (hForward :
      dominates
        (Transform_calderon_zygmund_decomposition
          (LocalChart_calderon_zygmund_decomposition x))
        (Transform_calderon_zygmund_decomposition x)).
  { exact (proj2 hBase). }
  assert (hRefl : dominates (transform x) (transform x)).
  { apply dominates_refl. }
  split.
  - exact hForward.
  - exact hRefl.
Qed.

Lemma transform_isometry_calderon_zygmund_decomposition
    {H : Type} `{AnalysisStruct_calderon_zygmund_decomposition H}
    (x : H) :
    dominates (Transform_calderon_zygmund_decomposition x) x /\
    dominates x (Transform_calderon_zygmund_decomposition x) /\
    dominates (Transform_calderon_zygmund_decomposition x)
      (Transform_calderon_zygmund_decomposition x).
Proof.
  assert (hIso : dominates (transform x) x /\ dominates x (transform x)).
  { apply (isometry_axiom x). }
  assert (hDiag : dominates (transform x) (transform x)).
  { apply (dominates_trans (transform x) x (transform x) (proj1 hIso) (proj2 hIso)). }
  split.
  - exact (proj1 hIso).
  - split.
    + exact (proj2 hIso).
    + exact hDiag.
Qed.

Lemma decomposition_bound_calderon_zygmund_decomposition
    {H : Type} `{AnalysisStruct_calderon_zygmund_decomposition H}
    (x : H) :
    dominates
      (Generator_calderon_zygmund_decomposition x)
      x /\
    dominates
      (Generator_calderon_zygmund_decomposition x)
      (LocalChart_calderon_zygmund_decomposition x).
Proof.
  assert (hToChart : dominates (generator x) (localChart x)).
  { apply (decomposition_axiom x). }
  assert (hChartToX : dominates (localChart x) x).
  { apply (local_estimate_axiom x). }
  assert (hToX : dominates (generator x) x).
  { apply (dominates_trans (generator x) (localChart x) x hToChart hChartToX). }
  split.
  - exact hToX.
  - exact hToChart.
Qed.

Lemma symbol_composition_calderon_zygmund_decomposition
    {H : Type} `{AnalysisStruct_calderon_zygmund_decomposition H}
    (x : H) :
    Transform_calderon_zygmund_decomposition
      (Generator_calderon_zygmund_decomposition x) =
    Generator_calderon_zygmund_decomposition
      (Transform_calderon_zygmund_decomposition x) /\
    dominates
      (Generator_calderon_zygmund_decomposition x)
      (LocalChart_calderon_zygmund_decomposition x).
Proof.
  assert (hEq : transform (generator x) = generator (transform x)).
  { apply (symbol_comp_axiom x). }
  assert (hDom : dominates (generator x) (localChart x)).
  { apply (decomposition_axiom x). }
  split.
  - exact hEq.
  - exact hDom.
Qed.

Lemma semigroup_generation_calderon_zygmund_decomposition
    {H : Type} `{AnalysisStruct_calderon_zygmund_decomposition H}
    (x : H) :
    dominates
      (Generator_calderon_zygmund_decomposition
        (Generator_calderon_zygmund_decomposition x))
      x /\
    dominates
      (Generator_calderon_zygmund_decomposition
        (Generator_calderon_zygmund_decomposition x))
      (Generator_calderon_zygmund_decomposition x).
Proof.
  assert (hStep1 : dominates (generator (generator x)) (generator x)).
  { apply (semigroup_axiom x). }
  assert (hStep2 : dominates (generator x) x).
  { exact (proj1 (decomposition_bound_calderon_zygmund_decomposition x)). }
  assert (hStep3 : dominates (generator (generator x)) x).
  { apply (dominates_trans (generator (generator x)) (generator x) x hStep1 hStep2). }
  split.
  - exact hStep3.
  - exact hStep1.
Qed.

Lemma regularity_upgrade_calderon_zygmund_decomposition
    {H : Type} `{AnalysisStruct_calderon_zygmund_decomposition H}
    (x : H)
    (hNorm : NormCtrl_calderon_zygmund_decomposition x) :
    NormCtrl_calderon_zygmund_decomposition
      (Transform_calderon_zygmund_decomposition x) /\
    dominates
      (Transform_calderon_zygmund_decomposition
        (Generator_calderon_zygmund_decomposition x))
      (Transform_calderon_zygmund_decomposition x).
Proof.
  assert (hGen : dominates (generator x) x).
  { exact (proj1 (decomposition_bound_calderon_zygmund_decomposition x)). }
  assert (hReg : normCtrl (transform x)).
  { apply (regularity_axiom x hGen hNorm). }
  assert (hPatch : dominates (transform (generator x)) (transform x)).
  { apply (patching_axiom (generator x) x hGen). }
  split.
  - exact hReg.
  - exact hPatch.
Qed.

Lemma transformed_semigroup_localchart_bound_calderon_zygmund_decomposition
    {H : Type} `{AnalysisStruct_calderon_zygmund_decomposition H}
    (x : H) :
    dominates
      (Transform_calderon_zygmund_decomposition
        (Generator_calderon_zygmund_decomposition
          (Generator_calderon_zygmund_decomposition x)))
      (Transform_calderon_zygmund_decomposition
        (LocalChart_calderon_zygmund_decomposition x)) /\
    dominates
      (Transform_calderon_zygmund_decomposition
        (Generator_calderon_zygmund_decomposition
          (Generator_calderon_zygmund_decomposition x)))
      (Transform_calderon_zygmund_decomposition x).
Proof.
  assert (hSemi : dominates (generator (generator x)) (generator x)).
  { apply (semigroup_axiom x). }
  assert (hGenChart : dominates (generator x) (localChart x)).
  { apply (decomposition_axiom x). }
  assert (hGen2Chart : dominates (generator (generator x)) (localChart x)).
  { apply (dominates_trans (generator (generator x)) (generator x) (localChart x) hSemi hGenChart). }
  assert (hChartX : dominates (localChart x) x).
  { apply (local_estimate_axiom x). }
  assert (hGen2X : dominates (generator (generator x)) x).
  { apply (dominates_trans (generator (generator x)) (localChart x) x hGen2Chart hChartX). }
  assert (hPatchChart :
      dominates (transform (generator (generator x))) (transform (localChart x))).
  { apply (patching_axiom (generator (generator x)) (localChart x) hGen2Chart). }
  assert (hPatchX :
      dominates (transform (generator (generator x))) (transform x)).
  { apply (patching_axiom (generator (generator x)) x hGen2X). }
  split.
  - exact hPatchChart.
  - exact hPatchX.
Qed.

Lemma regularity_double_transform_calderon_zygmund_decomposition
    {H : Type} `{AnalysisStruct_calderon_zygmund_decomposition H}
    (x : H)
    (hNorm : NormCtrl_calderon_zygmund_decomposition x) :
    NormCtrl_calderon_zygmund_decomposition
      (Transform_calderon_zygmund_decomposition
        (Transform_calderon_zygmund_decomposition x)) /\
    dominates
      (Transform_calderon_zygmund_decomposition
        (Transform_calderon_zygmund_decomposition
          (Generator_calderon_zygmund_decomposition x)))
      (Transform_calderon_zygmund_decomposition
        (Transform_calderon_zygmund_decomposition x)).
Proof.
  pose proof (regularity_upgrade_calderon_zygmund_decomposition x hNorm) as hFirstReg.
  assert (hNormTx : normCtrl (transform x)).
  { exact (proj1 hFirstReg). }
  assert (hGenTx : dominates (generator (transform x)) (transform x)).
  { exact (proj1 (decomposition_bound_calderon_zygmund_decomposition (transform x))). }
  assert (hNormTTx : normCtrl (transform (transform x))).
  { apply (regularity_axiom (transform x) hGenTx hNormTx). }
  assert (hDomTgTx : dominates (transform (generator x)) (transform x)).
  { exact (proj2 hFirstReg). }
  assert (hPatch :
      dominates (transform (transform (generator x))) (transform (transform x))).
  { apply (patching_axiom (transform (generator x)) (transform x) hDomTgTx). }
  split.
  - exact hNormTTx.
  - exact hPatch.
Qed.

Lemma symbol_semigroup_transport_calderon_zygmund_decomposition
    {H : Type} `{AnalysisStruct_calderon_zygmund_decomposition H}
    (x : H) :
    Transform_calderon_zygmund_decomposition
      (Generator_calderon_zygmund_decomposition
        (Generator_calderon_zygmund_decomposition x)) =
    Generator_calderon_zygmund_decomposition
      (Generator_calderon_zygmund_decomposition
        (Transform_calderon_zygmund_decomposition x)) /\
    dominates
      (Generator_calderon_zygmund_decomposition
        (Generator_calderon_zygmund_decomposition
          (Transform_calderon_zygmund_decomposition x)))
      (Transform_calderon_zygmund_decomposition x).
Proof.
  assert (hEq1 :
      transform (generator (generator x)) =
      generator (transform (generator x))).
  { apply (symbol_comp_axiom (generator x)). }
  assert (hEq2 : transform (generator x) = generator (transform x)).
  { apply (symbol_comp_axiom x). }
  assert (hEqFinal :
      transform (generator (generator x)) =
      generator (generator (transform x))).
  {
    rewrite hEq2 in hEq1.
    exact hEq1.
  }
  assert (hDom : dominates (generator (generator (transform x))) (transform x)).
  { exact (proj1 (semigroup_generation_calderon_zygmund_decomposition (transform x))). }
  split.
  - exact hEqFinal.
  - exact hDom.
Qed.

Lemma transform_generator_chart_chain_calderon_zygmund_decomposition
    {H : Type} `{AnalysisStruct_calderon_zygmund_decomposition H}
    (x : H) :
    dominates
      (Transform_calderon_zygmund_decomposition
        (Generator_calderon_zygmund_decomposition x))
      (LocalChart_calderon_zygmund_decomposition x) /\
    dominates
      (Transform_calderon_zygmund_decomposition
        (Generator_calderon_zygmund_decomposition x))
      x.
Proof.
  pose proof (transform_isometry_calderon_zygmund_decomposition (generator x)) as hIsoGen.
  assert (hToGen : dominates (transform (generator x)) (generator x)).
  { exact (proj1 hIsoGen). }
  assert (hGenChart : dominates (generator x) (localChart x)).
  { exact (proj2 (decomposition_bound_calderon_zygmund_decomposition x)). }
  assert (hToChart : dominates (transform (generator x)) (localChart x)).
  { apply (dominates_trans (transform (generator x)) (generator x) (localChart x) hToGen hGenChart). }
  assert (hChartX : dominates (localChart x) x).
  { apply (local_estimate_axiom x). }
  assert (hToX : dominates (transform (generator x)) x).
  { apply (dominates_trans (transform (generator x)) (localChart x) x hToChart hChartX). }
  split.
  - exact hToChart.
  - exact hToX.
Qed.

Lemma generator_transform_chart_bound_calderon_zygmund_decomposition
    {H : Type} `{AnalysisStruct_calderon_zygmund_decomposition H}
    (x : H) :
    dominates
      (Generator_calderon_zygmund_decomposition
        (Transform_calderon_zygmund_decomposition x))
      (Transform_calderon_zygmund_decomposition
        (LocalChart_calderon_zygmund_decomposition x)) /\
    dominates
      (Generator_calderon_zygmund_decomposition
        (Transform_calderon_zygmund_decomposition x))
      (Transform_calderon_zygmund_decomposition x).
Proof.
  assert (hEq : transform (generator x) = generator (transform x)).
  { apply (symbol_comp_axiom x). }
  assert (hDom : dominates (generator x) (localChart x)).
  { apply (decomposition_axiom x). }
  assert (hPatch : dominates (transform (generator x)) (transform (localChart x))).
  { apply (patching_axiom (generator x) (localChart x) hDom). }
  assert (hFirst : dominates (generator (transform x)) (transform (localChart x))).
  {
    rewrite <- hEq.
    exact hPatch.
  }
  assert (hSecond : dominates (generator (transform x)) (transform x)).
  { exact (proj1 (decomposition_bound_calderon_zygmund_decomposition (transform x))). }
  split.
  - exact hFirst.
  - exact hSecond.
Qed.
