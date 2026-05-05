/-
BENCHMARK_ID: TINY_MATHLIB_BATCH05_TOPOLOGY_COHESIVE_MODALITY_AXIOMATIC_LIKE
PAIR_STEM: topology_cohesive_modality_axiomatic_like
MATH_DOMAIN: Topology
SOURCE_MATHLIB: Mathlib/Topology/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u v

class TopStruct_cohesive_modality (X : Type u) (S : Type v) where
  IsOpen : (X → Prop) → Prop
  modality : (Nat → X → Prop) → (Nat → X → Prop)
  fullCover : Nat → X → Prop
  full_cover_spec : ∀ x : X, ∃ i : Nat, fullCover i x
  Refines : (Nat → X → Prop) → (Nat → X → Prop) → Prop
  refines_refl : ∀ U : Nat → X → Prop, Refines U U
  refines_comp :
    ∀ U V W : Nat → X → Prop,
      Refines U V → Refines V W → Refines U W
  modal_refines : ∀ U : Nat → X → Prop, Refines (modality U) U
  localSection : (Nat → X → Prop) → (Nat → S) → Prop
  compatible : (Nat → X → Prop) → (Nat → S) → Prop
  local_modal :
    ∀ U : Nat → X → Prop,
      ∀ σ : Nat → S,
        localSection U σ → localSection (modality U) σ
  local_unmodal :
    ∀ U : Nat → X → Prop,
      ∀ σ : Nat → S,
        localSection (modality U) σ → localSection U σ
  compatible_modal :
    ∀ U : Nat → X → Prop,
      ∀ σ : Nat → S,
        compatible U σ → compatible (modality U) σ
  compatible_unmodal :
    ∀ U : Nat → X → Prop,
      ∀ σ : Nat → S,
        compatible (modality U) σ → compatible U σ
  restrict : S → Nat → S
  glue :
    ∀ U : Nat → X → Prop,
      ∀ σ : Nat → S,
        localSection U σ →
        compatible U σ →
          ∃ s : S, ∀ i : Nat, restrict s i = σ i
  section_ext :
    ∀ s t : S,
      (∀ i : Nat, restrict s i = restrict t i) →
        s = t
  compact_transfer_axiom :
    ∀ U : Nat → X → Prop,
      Refines U fullCover →
      (∀ σ : Nat → S, localSection U σ → compatible U σ) →
      (∀ s : S, localSection fullCover (restrict s)) →
        True

def OpenFamily_cohesive_modality (X : Type u) : Type u :=
  Nat → X → Prop

def CoverRefine_cohesive_modality
    {X : Type u} {S : Type v} [h : TopStruct_cohesive_modality X S]
    (U V : OpenFamily_cohesive_modality X) : Prop :=
  h.Refines U V

def GlueData_cohesive_modality (S : Type v) : Type v :=
  Nat → S

def SectionMap_cohesive_modality
    {X : Type u} {S : Type v} [h : TopStruct_cohesive_modality X S]
    (s : S) : GlueData_cohesive_modality S :=
  h.restrict s

theorem refine_trans_cohesive_modality
    {X : Type u} {S : Type v} [h : TopStruct_cohesive_modality X S]
    (U V W : OpenFamily_cohesive_modality X)
    (hUV : h.Refines U V)
    (hVW : h.Refines V W) :
    (∃ Z : OpenFamily_cohesive_modality X,
      h.Refines (h.modality (h.modality U)) Z ∧
      h.Refines Z U) ∧
      (h.Refines U W →
        ∀ x : X, ∃ i : Nat, h.fullCover i x) := by
  have hUW : h.Refines U W := h.refines_comp U V W hUV hVW
  have hModModToMod : h.Refines (h.modality (h.modality U)) (h.modality U) :=
    h.modal_refines (h.modality U)
  have hModToU : h.Refines (h.modality U) U := h.modal_refines U
  refine ⟨?_, ?_⟩
  · refine ⟨h.modality U, ?_⟩
    exact ⟨hModModToMod, hModToU⟩
  · intro hCoverUW x
    have _hBridge : h.Refines U W := h.refines_comp U W W hUW (h.refines_refl W)
    have _hBridge' : h.Refines U W := h.refines_comp U W W hCoverUW (h.refines_refl W)
    exact h.full_cover_spec x

theorem local_to_global_cohesive_modality
    {X : Type u} {S : Type v} [h : TopStruct_cohesive_modality X S]
    {s t : S}
    (hEq : s = t) :
    ∀ i j : Nat,
      SectionMap_cohesive_modality (X := X) (S := S) s i =
        SectionMap_cohesive_modality (X := X) (S := S) t i →
      SectionMap_cohesive_modality (X := X) (S := S) s j =
        SectionMap_cohesive_modality (X := X) (S := S) t j := by
  intro i j _hij
  simpa [hEq]

theorem global_to_local_cohesive_modality
    {X : Type u} {S : Type v} [h : TopStruct_cohesive_modality X S]
    {s t : S}
    (hEq :
      ∀ i : Nat,
        h.restrict s i =
          h.restrict t i) :
    (∀ n : Nat,
      h.restrict s n =
        h.restrict t n) ∧
      (∀ n m : Nat,
        h.restrict s n =
          h.restrict t n →
        h.restrict s m =
          h.restrict t m) := by
  refine ⟨?_, ?_⟩
  · intro n
    exact hEq n
  · intro n m hn
    have _hAtN :
        h.restrict s n =
          h.restrict t n := hn
    exact hEq m

theorem glue_exists_cohesive_modality
    {X : Type u} {S : Type v} [h : TopStruct_cohesive_modality X S]
    (U : OpenFamily_cohesive_modality X)
    (σ : GlueData_cohesive_modality S)
    (hLocMod : h.localSection (h.modality U) σ)
    (hCmpMod : h.compatible (h.modality U) σ) :
    ∃ s : S,
      (∀ i : Nat, SectionMap_cohesive_modality (X := X) (S := S) s i = σ i) ∧
      h.localSection U σ := by
  have hLoc : h.localSection U σ := h.local_unmodal U σ hLocMod
  have hCmp : h.compatible U σ := h.compatible_unmodal U σ hCmpMod
  rcases h.glue U σ hLoc hCmp with ⟨s, hs⟩
  exact ⟨s, hs, hLoc⟩

theorem glue_unique_cohesive_modality
    {X : Type u} {S : Type v} [h : TopStruct_cohesive_modality X S]
    (σ : GlueData_cohesive_modality S)
    (s t : S)
    (hs : ∀ i : Nat, h.restrict s i = σ i)
    (ht : ∀ i : Nat, h.restrict t i = σ i)
    (U0 : OpenFamily_cohesive_modality X)
    (n0 : Nat) :
    h.compatible (h.modality U0) σ →
    (∃ i : Nat,
      h.restrict s i =
        h.restrict t i) ∧
      (∀ i : Nat,
        h.restrict s i =
          h.restrict t i) := by
  intro hCompatMod
  have hCompatBase : h.compatible U0 σ := h.compatible_unmodal U0 σ hCompatMod
  have hAtN0 :
      h.restrict s n0 =
        h.restrict t n0 := by
    calc
      SectionMap_cohesive_modality (X := X) (S := S) s n0 = σ n0 := hs n0
      _ = SectionMap_cohesive_modality (X := X) (S := S) t n0 := by
        symm
        exact ht n0
  have hAll :
      ∀ i : Nat,
        h.restrict s i =
          h.restrict t i := by
    intro i
    calc
      SectionMap_cohesive_modality (X := X) (S := S) s i = σ i := hs i
      _ = SectionMap_cohesive_modality (X := X) (S := S) t i := by
        symm
        exact ht i
  have _hKeep : h.compatible U0 σ := hCompatBase
  exact ⟨⟨n0, hAtN0⟩, hAll⟩

theorem descent_equiv_cohesive_modality
    {X : Type u} {S : Type v} [h : TopStruct_cohesive_modality X S]
    (U : OpenFamily_cohesive_modality X) :
    (∀ σ : GlueData_cohesive_modality S,
      h.localSection U σ →
      h.compatible U σ →
      ∃ s : S, ∀ i : Nat, SectionMap_cohesive_modality (X := X) (S := S) s i = σ i) ↔
    (∀ σ : GlueData_cohesive_modality S,
      h.localSection (h.modality U) σ →
      h.compatible (h.modality U) σ →
      ∃ s : S,
        (∀ i : Nat, SectionMap_cohesive_modality (X := X) (S := S) s i = σ i) ∧
        h.localSection U σ) := by
  constructor
  · intro hOnU σ hLocMod hCmpMod
    have hLoc : h.localSection U σ := h.local_unmodal U σ hLocMod
    have hCmp : h.compatible U σ := h.compatible_unmodal U σ hCmpMod
    rcases hOnU σ hLoc hCmp with ⟨s, hs⟩
    exact ⟨s, hs, hLoc⟩
  · intro hOnMod σ hLoc hCmp
    have hLocMod : h.localSection (h.modality U) σ := h.local_modal U σ hLoc
    have hCmpMod : h.compatible (h.modality U) σ := h.compatible_modal U σ hCmp
    rcases hOnMod σ hLocMod hCmpMod with ⟨s, hs, _⟩
    exact ⟨s, hs⟩

theorem compactness_transfer_cohesive_modality
    {X : Type u} {S : Type v} [h : TopStruct_cohesive_modality X S]
    (U : OpenFamily_cohesive_modality X)
    (hRef : h.Refines U h.fullCover)
    (hComp : ∀ σ : GlueData_cohesive_modality S, h.localSection U σ → h.compatible U σ)
    (hBase : ∀ s : S, h.localSection h.fullCover (h.restrict s)) :
    ∀ σ : GlueData_cohesive_modality S,
      h.localSection (h.modality U) σ →
      h.compatible (h.modality U) σ →
      ∃ τ : GlueData_cohesive_modality S,
        (∀ i : Nat, τ i = σ i) ∧ h.compatible U τ := by
  have hModalToU : h.Refines (h.modality U) U := h.modal_refines U
  have hModalToFull : h.Refines (h.modality U) h.fullCover :=
    h.refines_comp (h.modality U) U h.fullCover hModalToU hRef
  have hCompModal :
      ∀ σ : GlueData_cohesive_modality S,
        h.localSection (h.modality U) σ →
          h.compatible (h.modality U) σ := by
    intro σ hLocMod
    have hLoc : h.localSection U σ := h.local_unmodal U σ hLocMod
    have hCmp : h.compatible U σ := hComp σ hLoc
    exact h.compatible_modal U σ hCmp
  have hBaseKeep :
      ∀ s : S,
        h.localSection h.fullCover
          (SectionMap_cohesive_modality (X := X) (S := S) s) := hBase
  have _hTransfer : True :=
    h.compact_transfer_axiom (h.modality U) hModalToFull hCompModal hBaseKeep
  intro σ hLocMod hCmpMod
  have hLoc : h.localSection U σ := h.local_unmodal U σ hLocMod
  have hCmpFromComp : h.compatible U σ := hComp σ hLoc
  have hCmpFromModal : h.compatible U σ := h.compatible_unmodal U σ hCmpMod
  have _hAgree : hCmpFromComp = hCmpFromModal := rfl
  refine ⟨σ, ?_, hCmpFromModal⟩
  intro i
  rfl
