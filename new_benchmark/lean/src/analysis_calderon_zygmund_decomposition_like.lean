/-
BENCHMARK_ID: TINY_MATHLIB_BATCH05_ANALYSIS_CALDERON_ZYGMUND_DECOMPOSITION_LIKE
PAIR_STEM: analysis_calderon_zygmund_decomposition_like
MATH_DOMAIN: Analysis
SOURCE_MATHLIB: Mathlib/Analysis/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 17
-/

universe u

class AnalysisStruct_calderon_zygmund_decomposition (H : Type u) where
  dominates : H → H → Prop
  dominates_refl : ∀ x : H, dominates x x
  dominates_trans : ∀ x y z : H, dominates x y → dominates y z → dominates x z
  normCtrl : H → Prop
  localChart : H → H
  transform : H → H
  generator : H → H
  local_estimate_axiom : ∀ x : H, dominates (localChart x) x
  patching_axiom : ∀ x y : H, dominates x y → dominates (transform x) (transform y)
  isometry_axiom : ∀ x : H, dominates (transform x) x ∧ dominates x (transform x)
  decomposition_axiom : ∀ x : H, dominates (generator x) (localChart x)
  symbol_comp_axiom : ∀ x : H, transform (generator x) = generator (transform x)
  semigroup_axiom : ∀ x : H, dominates (generator (generator x)) (generator x)
  regularity_axiom :
    ∀ x : H,
      dominates (generator x) x →
      normCtrl x →
        normCtrl (transform x)

def NormCtrl_calderon_zygmund_decomposition
    {H : Type u} [h : AnalysisStruct_calderon_zygmund_decomposition H] :
    H → Prop :=
  h.normCtrl

def LocalChart_calderon_zygmund_decomposition
    {H : Type u} [h : AnalysisStruct_calderon_zygmund_decomposition H] :
    H → H :=
  h.localChart

def Transform_calderon_zygmund_decomposition
    {H : Type u} [h : AnalysisStruct_calderon_zygmund_decomposition H] :
    H → H :=
  h.transform

def Generator_calderon_zygmund_decomposition
    {H : Type u} [h : AnalysisStruct_calderon_zygmund_decomposition H] :
    H → H :=
  h.generator

theorem local_estimate_calderon_zygmund_decomposition
    {H : Type u} [h : AnalysisStruct_calderon_zygmund_decomposition H]
    (x : H) :
    h.dominates
      (LocalChart_calderon_zygmund_decomposition (H := H) x)
      x ∧
    h.dominates
      (Transform_calderon_zygmund_decomposition (H := H)
        (LocalChart_calderon_zygmund_decomposition (H := H) x))
      (Transform_calderon_zygmund_decomposition (H := H) x) := by
  have hLocal : h.dominates (h.localChart x) x := h.local_estimate_axiom x
  have hTransformed : h.dominates (h.transform (h.localChart x)) (h.transform x) :=
    h.patching_axiom (h.localChart x) x hLocal
  exact And.intro hLocal hTransformed

theorem patching_estimate_calderon_zygmund_decomposition
    {H : Type u} [h : AnalysisStruct_calderon_zygmund_decomposition H]
    (x : H) :
    h.dominates
      (Transform_calderon_zygmund_decomposition (H := H)
        (LocalChart_calderon_zygmund_decomposition (H := H) x))
      (Transform_calderon_zygmund_decomposition (H := H) x) ∧
    h.dominates
      (Transform_calderon_zygmund_decomposition (H := H) x)
      (Transform_calderon_zygmund_decomposition (H := H) x) := by
  have hBase := local_estimate_calderon_zygmund_decomposition (H := H) x
  have hForward :
      h.dominates
        (Transform_calderon_zygmund_decomposition (H := H)
          (LocalChart_calderon_zygmund_decomposition (H := H) x))
        (Transform_calderon_zygmund_decomposition (H := H) x) := hBase.right
  have hRefl : h.dominates (h.transform x) (h.transform x) := h.dominates_refl (h.transform x)
  exact And.intro hForward hRefl

theorem transform_isometry_calderon_zygmund_decomposition
    {H : Type u} [h : AnalysisStruct_calderon_zygmund_decomposition H]
    (x : H) :
    h.dominates (Transform_calderon_zygmund_decomposition (H := H) x) x ∧
    h.dominates x (Transform_calderon_zygmund_decomposition (H := H) x) ∧
    h.dominates (Transform_calderon_zygmund_decomposition (H := H) x)
      (Transform_calderon_zygmund_decomposition (H := H) x) := by
  have hIso : h.dominates (h.transform x) x ∧ h.dominates x (h.transform x) :=
    h.isometry_axiom x
  have hDiag : h.dominates (h.transform x) (h.transform x) :=
    h.dominates_trans (h.transform x) x (h.transform x) hIso.left hIso.right
  exact And.intro hIso.left (And.intro hIso.right hDiag)

theorem decomposition_bound_calderon_zygmund_decomposition
    {H : Type u} [h : AnalysisStruct_calderon_zygmund_decomposition H]
    (x : H) :
    h.dominates
      (Generator_calderon_zygmund_decomposition (H := H) x)
      x ∧
    h.dominates
      (Generator_calderon_zygmund_decomposition (H := H) x)
      (LocalChart_calderon_zygmund_decomposition (H := H) x) := by
  have hToChart : h.dominates (h.generator x) (h.localChart x) :=
    h.decomposition_axiom x
  have hChartToX : h.dominates (h.localChart x) x := h.local_estimate_axiom x
  have hToX : h.dominates (h.generator x) x :=
    h.dominates_trans (h.generator x) (h.localChart x) x hToChart hChartToX
  exact And.intro hToX hToChart

theorem symbol_composition_calderon_zygmund_decomposition
    {H : Type u} [h : AnalysisStruct_calderon_zygmund_decomposition H]
    (x : H) :
    Transform_calderon_zygmund_decomposition (H := H)
      (Generator_calderon_zygmund_decomposition (H := H) x) =
    Generator_calderon_zygmund_decomposition (H := H)
      (Transform_calderon_zygmund_decomposition (H := H) x) ∧
    h.dominates
      (Generator_calderon_zygmund_decomposition (H := H) x)
      (LocalChart_calderon_zygmund_decomposition (H := H) x) := by
  have hEq : h.transform (h.generator x) = h.generator (h.transform x) :=
    h.symbol_comp_axiom x
  have hDom : h.dominates (h.generator x) (h.localChart x) :=
    h.decomposition_axiom x
  exact And.intro hEq hDom

theorem semigroup_generation_calderon_zygmund_decomposition
    {H : Type u} [h : AnalysisStruct_calderon_zygmund_decomposition H]
    (x : H) :
    h.dominates
      (Generator_calderon_zygmund_decomposition (H := H)
        (Generator_calderon_zygmund_decomposition (H := H) x))
      x ∧
    h.dominates
      (Generator_calderon_zygmund_decomposition (H := H)
        (Generator_calderon_zygmund_decomposition (H := H) x))
      (Generator_calderon_zygmund_decomposition (H := H) x) := by
  have hStep1 : h.dominates (h.generator (h.generator x)) (h.generator x) :=
    h.semigroup_axiom x
  have hStep2 : h.dominates (h.generator x) x :=
    (decomposition_bound_calderon_zygmund_decomposition (H := H) x).left
  have hStep3 : h.dominates (h.generator (h.generator x)) x :=
    h.dominates_trans (h.generator (h.generator x)) (h.generator x) x hStep1 hStep2
  exact And.intro hStep3 hStep1

theorem regularity_upgrade_calderon_zygmund_decomposition
    {H : Type u} [h : AnalysisStruct_calderon_zygmund_decomposition H]
    (x : H)
    (hNorm : NormCtrl_calderon_zygmund_decomposition (H := H) x) :
    NormCtrl_calderon_zygmund_decomposition (H := H)
      (Transform_calderon_zygmund_decomposition (H := H) x) ∧
    h.dominates
      (Transform_calderon_zygmund_decomposition (H := H)
        (Generator_calderon_zygmund_decomposition (H := H) x))
      (Transform_calderon_zygmund_decomposition (H := H) x) := by
  have hGen : h.dominates (h.generator x) x :=
    (decomposition_bound_calderon_zygmund_decomposition (H := H) x).left
  have hReg : h.normCtrl (h.transform x) :=
    h.regularity_axiom x hGen hNorm
  have hPatch : h.dominates (h.transform (h.generator x)) (h.transform x) :=
    h.patching_axiom (h.generator x) x hGen
  exact And.intro hReg hPatch

theorem transformed_semigroup_localchart_bound_calderon_zygmund_decomposition
    {H : Type u} [h : AnalysisStruct_calderon_zygmund_decomposition H]
    (x : H) :
    h.dominates
      (Transform_calderon_zygmund_decomposition (H := H)
        (Generator_calderon_zygmund_decomposition (H := H)
          (Generator_calderon_zygmund_decomposition (H := H) x)))
      (Transform_calderon_zygmund_decomposition (H := H)
        (LocalChart_calderon_zygmund_decomposition (H := H) x)) ∧
    h.dominates
      (Transform_calderon_zygmund_decomposition (H := H)
        (Generator_calderon_zygmund_decomposition (H := H)
          (Generator_calderon_zygmund_decomposition (H := H) x)))
      (Transform_calderon_zygmund_decomposition (H := H) x) := by
  have hSemi : h.dominates (h.generator (h.generator x)) (h.generator x) :=
    h.semigroup_axiom x
  have hGenChart : h.dominates (h.generator x) (h.localChart x) :=
    h.decomposition_axiom x
  have hGen2Chart : h.dominates (h.generator (h.generator x)) (h.localChart x) :=
    h.dominates_trans (h.generator (h.generator x)) (h.generator x) (h.localChart x) hSemi hGenChart
  have hChartX : h.dominates (h.localChart x) x :=
    h.local_estimate_axiom x
  have hGen2X : h.dominates (h.generator (h.generator x)) x :=
    h.dominates_trans (h.generator (h.generator x)) (h.localChart x) x hGen2Chart hChartX
  have hPatchChart :
      h.dominates
        (h.transform (h.generator (h.generator x)))
        (h.transform (h.localChart x)) :=
    h.patching_axiom (h.generator (h.generator x)) (h.localChart x) hGen2Chart
  have hPatchX :
      h.dominates
        (h.transform (h.generator (h.generator x)))
        (h.transform x) :=
    h.patching_axiom (h.generator (h.generator x)) x hGen2X
  exact And.intro hPatchChart hPatchX

theorem regularity_double_transform_calderon_zygmund_decomposition
    {H : Type u} [h : AnalysisStruct_calderon_zygmund_decomposition H]
    (x : H)
    (hNorm : NormCtrl_calderon_zygmund_decomposition (H := H) x) :
    NormCtrl_calderon_zygmund_decomposition (H := H)
      (Transform_calderon_zygmund_decomposition (H := H)
        (Transform_calderon_zygmund_decomposition (H := H) x)) ∧
    h.dominates
      (Transform_calderon_zygmund_decomposition (H := H)
        (Transform_calderon_zygmund_decomposition (H := H)
          (Generator_calderon_zygmund_decomposition (H := H) x)))
      (Transform_calderon_zygmund_decomposition (H := H)
        (Transform_calderon_zygmund_decomposition (H := H) x)) := by
  have hFirstReg :=
    regularity_upgrade_calderon_zygmund_decomposition (H := H) x hNorm
  have hNormTx : h.normCtrl (h.transform x) :=
    hFirstReg.left
  have hGenTx : h.dominates (h.generator (h.transform x)) (h.transform x) :=
    (decomposition_bound_calderon_zygmund_decomposition (H := H) (h.transform x)).left
  have hNormTTx : h.normCtrl (h.transform (h.transform x)) :=
    h.regularity_axiom (h.transform x) hGenTx hNormTx
  have hDomTgTx : h.dominates (h.transform (h.generator x)) (h.transform x) :=
    hFirstReg.right
  have hPatch :
      h.dominates
        (h.transform (h.transform (h.generator x)))
        (h.transform (h.transform x)) :=
    h.patching_axiom (h.transform (h.generator x)) (h.transform x) hDomTgTx
  exact And.intro hNormTTx hPatch

theorem symbol_semigroup_transport_calderon_zygmund_decomposition
    {H : Type u} [h : AnalysisStruct_calderon_zygmund_decomposition H]
    (x : H) :
    Transform_calderon_zygmund_decomposition (H := H)
      (Generator_calderon_zygmund_decomposition (H := H)
        (Generator_calderon_zygmund_decomposition (H := H) x)) =
    Generator_calderon_zygmund_decomposition (H := H)
      (Generator_calderon_zygmund_decomposition (H := H)
        (Transform_calderon_zygmund_decomposition (H := H) x)) ∧
    h.dominates
      (Generator_calderon_zygmund_decomposition (H := H)
        (Generator_calderon_zygmund_decomposition (H := H)
          (Transform_calderon_zygmund_decomposition (H := H) x)))
      (Transform_calderon_zygmund_decomposition (H := H) x) := by
  have hEq1 :
      h.transform (h.generator (h.generator x)) =
      h.generator (h.transform (h.generator x)) :=
    h.symbol_comp_axiom (h.generator x)
  have hEq2 : h.transform (h.generator x) = h.generator (h.transform x) :=
    h.symbol_comp_axiom x
  have hEqFinal :
      h.transform (h.generator (h.generator x)) =
      h.generator (h.generator (h.transform x)) := by
    rw [hEq2] at hEq1
    exact hEq1
  have hDom :
      h.dominates
        (h.generator (h.generator (h.transform x)))
        (h.transform x) :=
    (semigroup_generation_calderon_zygmund_decomposition (H := H) (h.transform x)).left
  exact And.intro hEqFinal hDom

theorem transform_generator_chart_chain_calderon_zygmund_decomposition
    {H : Type u} [h : AnalysisStruct_calderon_zygmund_decomposition H]
    (x : H) :
    h.dominates
      (Transform_calderon_zygmund_decomposition (H := H)
        (Generator_calderon_zygmund_decomposition (H := H) x))
      (LocalChart_calderon_zygmund_decomposition (H := H) x) ∧
    h.dominates
      (Transform_calderon_zygmund_decomposition (H := H)
        (Generator_calderon_zygmund_decomposition (H := H) x))
      x := by
  have hIsoGen :=
    transform_isometry_calderon_zygmund_decomposition
      (H := H) (h.generator x)
  have hToGen : h.dominates (h.transform (h.generator x)) (h.generator x) :=
    hIsoGen.left
  have hGenChart : h.dominates (h.generator x) (h.localChart x) :=
    (decomposition_bound_calderon_zygmund_decomposition (H := H) x).right
  have hToChart : h.dominates (h.transform (h.generator x)) (h.localChart x) :=
    h.dominates_trans (h.transform (h.generator x)) (h.generator x) (h.localChart x) hToGen hGenChart
  have hChartX : h.dominates (h.localChart x) x :=
    h.local_estimate_axiom x
  have hToX : h.dominates (h.transform (h.generator x)) x :=
    h.dominates_trans (h.transform (h.generator x)) (h.localChart x) x hToChart hChartX
  exact And.intro hToChart hToX

theorem generator_transform_chart_bound_calderon_zygmund_decomposition
    {H : Type u} [h : AnalysisStruct_calderon_zygmund_decomposition H]
    (x : H) :
    h.dominates
      (Generator_calderon_zygmund_decomposition (H := H)
        (Transform_calderon_zygmund_decomposition (H := H) x))
      (Transform_calderon_zygmund_decomposition (H := H)
        (LocalChart_calderon_zygmund_decomposition (H := H) x)) ∧
    h.dominates
      (Generator_calderon_zygmund_decomposition (H := H)
        (Transform_calderon_zygmund_decomposition (H := H) x))
      (Transform_calderon_zygmund_decomposition (H := H) x) := by
  have hEq : h.transform (h.generator x) = h.generator (h.transform x) :=
    h.symbol_comp_axiom x
  have hDom : h.dominates (h.generator x) (h.localChart x) :=
    h.decomposition_axiom x
  have hPatch :
      h.dominates
        (h.transform (h.generator x))
        (h.transform (h.localChart x)) :=
    h.patching_axiom (h.generator x) (h.localChart x) hDom
  have hFirst : h.dominates (h.generator (h.transform x)) (h.transform (h.localChart x)) := by
    rw [← hEq]
    exact hPatch
  have hSecond : h.dominates (h.generator (h.transform x)) (h.transform x) :=
    (decomposition_bound_calderon_zygmund_decomposition (H := H) (h.transform x)).left
  exact And.intro hFirst hSecond
