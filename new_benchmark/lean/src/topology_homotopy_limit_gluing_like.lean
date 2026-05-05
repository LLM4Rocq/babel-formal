/-
BENCHMARK_ID: TINY_MATHLIB_BATCH05_TOPOLOGY_HOMOTOPY_LIMIT_GLUING_LIKE
PAIR_STEM: topology_homotopy_limit_gluing_like
MATH_DOMAIN: Topology
SOURCE_MATHLIB: Mathlib/Topology/Sheaves
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class TopStruct_homotopy_limit_gluing (X : Type u) where
  Cover : Type u
  GlobalSection : Type u
  LocalSection : Cover → Type u
  Refines : Cover → Cover → Prop
  restrict : {U V : Cover} → Refines V U → LocalSection U → LocalSection V
  refine_refl : ∀ U : Cover, Refines U U
  refine_trans_axiom :
    ∀ {U V W : Cover}, Refines U V → Refines V W → Refines U W
  local_from_global : (U : Cover) → GlobalSection → LocalSection U
  global_from_local : (U : Cover) → LocalSection U → GlobalSection
  section_roundtrip :
    ∀ (U : Cover) (s : LocalSection U),
      local_from_global U (global_from_local U s) = s
  global_roundtrip :
    ∀ (U : Cover) (g : GlobalSection),
      global_from_local U (local_from_global U g) = g
  glue_axiom :
    ∀ (U : Cover) (s : LocalSection U),
      ∃ g : GlobalSection, local_from_global U g = s
  glue_unique_axiom :
    ∀ (U : Cover) (s : LocalSection U) (g₁ g₂ : GlobalSection),
      local_from_global U g₁ = s →
      local_from_global U g₂ = s →
      g₁ = g₂
  compact : GlobalSection → Prop
  compact_transfer_axiom :
    ∀ (U : Cover) (s : LocalSection U), compact (global_from_local U s)

def OpenFamily_homotopy_limit_gluing
    (X : Type u) [h : TopStruct_homotopy_limit_gluing X] : Type u :=
  h.Cover

def CoverRefine_homotopy_limit_gluing
    {X : Type u} [h : TopStruct_homotopy_limit_gluing X]
    (U V : OpenFamily_homotopy_limit_gluing X) : Prop :=
  h.Refines U V

def GlueData_homotopy_limit_gluing
    (X : Type u) [h : TopStruct_homotopy_limit_gluing X] : Type u :=
  Sigma (fun U : OpenFamily_homotopy_limit_gluing X => h.LocalSection U)

def SectionMap_homotopy_limit_gluing
    {X : Type u} [h : TopStruct_homotopy_limit_gluing X]
    (U : OpenFamily_homotopy_limit_gluing X)
    (g : h.GlobalSection) : h.LocalSection U :=
  h.local_from_global U g

theorem refine_trans_homotopy_limit_gluing
    {X : Type u} [h : TopStruct_homotopy_limit_gluing X]
    {U V W : OpenFamily_homotopy_limit_gluing X}
    (hUV : CoverRefine_homotopy_limit_gluing U V)
    (hVW : CoverRefine_homotopy_limit_gluing V W) :
    CoverRefine_homotopy_limit_gluing U W := by
  have hStep₁ : h.Refines U V := hUV
  have hStep₂ : h.Refines V W := hVW
  have hStep₃ : h.Refines U W := h.refine_trans_axiom hStep₁ hStep₂
  exact hStep₃

theorem local_to_global_homotopy_limit_gluing
    {X : Type u} [h : TopStruct_homotopy_limit_gluing X]
    (U : OpenFamily_homotopy_limit_gluing X)
    (s : h.LocalSection U) :
    ∃ g : h.GlobalSection, SectionMap_homotopy_limit_gluing U g = s := by
  have hExist : ∃ g : h.GlobalSection, h.local_from_global U g = s :=
    h.glue_axiom U s
  rcases hExist with ⟨g, hg⟩
  have hRewrite : SectionMap_homotopy_limit_gluing U g = s := by
    exact hg
  exact ⟨g, hRewrite⟩

theorem global_to_local_homotopy_limit_gluing
    {X : Type u} [h : TopStruct_homotopy_limit_gluing X]
    (U : OpenFamily_homotopy_limit_gluing X)
    (g : h.GlobalSection) :
    h.global_from_local U (SectionMap_homotopy_limit_gluing U g) = g := by
  have hRaw : h.global_from_local U (h.local_from_global U g) = g :=
    h.global_roundtrip U g
  have hAsWritten : h.global_from_local U (SectionMap_homotopy_limit_gluing U g) = g := by
    exact hRaw
  exact hAsWritten

theorem glue_exists_homotopy_limit_gluing
    {X : Type u} [h : TopStruct_homotopy_limit_gluing X]
    (d : GlueData_homotopy_limit_gluing X) :
    ∃ g : h.GlobalSection, SectionMap_homotopy_limit_gluing d.1 g = d.2 := by
  rcases d with ⟨U, s⟩
  have hLocal : ∃ g : h.GlobalSection, SectionMap_homotopy_limit_gluing U g = s :=
    local_to_global_homotopy_limit_gluing U s
  rcases hLocal with ⟨g, hg⟩
  exact ⟨g, hg⟩

theorem glue_unique_homotopy_limit_gluing
    {X : Type u} [h : TopStruct_homotopy_limit_gluing X]
    (U : OpenFamily_homotopy_limit_gluing X)
    (s : h.LocalSection U)
    (g₁ g₂ : h.GlobalSection)
    (hg₁ : SectionMap_homotopy_limit_gluing U g₁ = s)
    (hg₂ : SectionMap_homotopy_limit_gluing U g₂ = s) :
    g₁ = g₂ := by
  have h₁ : h.local_from_global U g₁ = s := hg₁
  have h₂ : h.local_from_global U g₂ = s := hg₂
  have hCore : g₁ = g₂ := h.glue_unique_axiom U s g₁ g₂ h₁ h₂
  exact hCore

theorem descent_equiv_homotopy_limit_gluing
    {X : Type u} [h : TopStruct_homotopy_limit_gluing X]
    (U : OpenFamily_homotopy_limit_gluing X)
    (s : h.LocalSection U) :
    (∃ g : h.GlobalSection, SectionMap_homotopy_limit_gluing U g = s) ∧
    (∀ g₁ g₂ : h.GlobalSection,
      SectionMap_homotopy_limit_gluing U g₁ = s →
      SectionMap_homotopy_limit_gluing U g₂ = s →
      g₁ = g₂) := by
  constructor
  · exact local_to_global_homotopy_limit_gluing U s
  · intro g₁ g₂ hg₁ hg₂
    exact glue_unique_homotopy_limit_gluing U s g₁ g₂ hg₁ hg₂

theorem compactness_transfer_homotopy_limit_gluing
    {X : Type u} [h : TopStruct_homotopy_limit_gluing X]
    (U : OpenFamily_homotopy_limit_gluing X)
    (s : h.LocalSection U)
    (g : h.GlobalSection)
    (hg : SectionMap_homotopy_limit_gluing U g = s) :
    h.compact g := by
  have hRound : SectionMap_homotopy_limit_gluing U (h.global_from_local U s) = s :=
    h.section_roundtrip U s
  have hUnique : g = h.global_from_local U s :=
    h.glue_unique_axiom U s g (h.global_from_local U s) hg hRound
  have hCompactBase : h.compact (h.global_from_local U s) :=
    h.compact_transfer_axiom U s
  have hCompactLift : h.compact g := by
    rw [hUnique]
    exact hCompactBase
  exact hCompactLift
