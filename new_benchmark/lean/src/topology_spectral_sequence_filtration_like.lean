/-
BENCHMARK_ID: TINY_MATHLIB_BATCH05_TOPOLOGY_SPECTRAL_SEQUENCE_FILTRATION_LIKE
PAIR_STEM: topology_spectral_sequence_filtration_like
MATH_DOMAIN: Topology
SOURCE_MATHLIB: Mathlib/Topology/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u v

class TopStruct_spectral_sequence_filtration (X : Type u) (E : Type v) where
  IsOpen : (X → Prop) → Prop
  filtrationCover : Nat → X → Prop
  Refines : (Nat → X → Prop) → (Nat → X → Prop) → Prop
  refines_refl : ∀ U : Nat → X → Prop, Refines U U
  refines_trans :
    ∀ U V W : Nat → X → Prop,
      Refines U V → Refines V W → Refines U W
  pageRestrict : E → Nat → E
  gluable : (Nat → X → Prop) → (Nat → E) → Prop
  compatible : (Nat → X → Prop) → (Nat → E) → Prop
  glue :
    ∀ U : Nat → X → Prop,
      ∀ σ : Nat → E,
        gluable U σ →
        compatible U σ →
          ∃ s : E, ∀ n : Nat, pageRestrict s n = σ n
  page_coherence :
    ∀ s : E,
      ∀ m n : Nat,
        n ≤ m →
          pageRestrict (pageRestrict s m) n = pageRestrict s n
  page_ext :
    ∀ s t : E,
      (∀ n : Nat, pageRestrict s n = pageRestrict t n) →
        s = t
  compact_transfer_axiom :
    ∀ U : Nat → X → Prop,
      Refines U filtrationCover →
      (∀ σ : Nat → E, gluable U σ → compatible U σ) →
      (∀ s : E, gluable filtrationCover (pageRestrict s)) →
        True

def OpenFamily_spectral_sequence_filtration (X : Type u) : Type u :=
  Nat → X → Prop

def CoverRefine_spectral_sequence_filtration
    {X : Type u} {E : Type v} [h : TopStruct_spectral_sequence_filtration X E]
    (U V : OpenFamily_spectral_sequence_filtration X) : Prop :=
  h.Refines U V

def GlueData_spectral_sequence_filtration (E : Type v) : Type v :=
  Nat → E

def SectionMap_spectral_sequence_filtration
    {X : Type u} {E : Type v} [h : TopStruct_spectral_sequence_filtration X E]
    (s : E) : GlueData_spectral_sequence_filtration E :=
  h.pageRestrict s

theorem refine_trans_spectral_sequence_filtration
    {X : Type u} {E : Type v} [h : TopStruct_spectral_sequence_filtration X E]
    (U V W : OpenFamily_spectral_sequence_filtration X)
    (hUV : @CoverRefine_spectral_sequence_filtration X E h U V)
    (hVW : @CoverRefine_spectral_sequence_filtration X E h V W) :
    ∀ s : E,
      ∀ m n : Nat,
        n ≤ m →
          @CoverRefine_spectral_sequence_filtration X E h U W ∧
          h.pageRestrict
            (@SectionMap_spectral_sequence_filtration X E h s m) n =
          @SectionMap_spectral_sequence_filtration X E h s n := by
  have hUW : h.Refines U W := h.refines_trans U V W hUV hVW
  intro s m n hnm
  have hCoh :
      h.pageRestrict
        (@SectionMap_spectral_sequence_filtration X E h s m) n =
      @SectionMap_spectral_sequence_filtration X E h s n := by
    simpa [SectionMap_spectral_sequence_filtration] using h.page_coherence s m n hnm
  exact ⟨hUW, hCoh⟩

theorem local_to_global_spectral_sequence_filtration
    {X : Type u} {E : Type v} [h : TopStruct_spectral_sequence_filtration X E]
    {s t : E}
    (hEq : s = t) :
    ∀ m n : Nat,
      n ≤ m →
        @SectionMap_spectral_sequence_filtration X E h s n =
          @SectionMap_spectral_sequence_filtration X E h t n ∧
        h.pageRestrict
          (@SectionMap_spectral_sequence_filtration X E h s m) n =
          @SectionMap_spectral_sequence_filtration X E h t n := by
  intro m n hnm
  have hPoint :
      @SectionMap_spectral_sequence_filtration X E h s n =
        @SectionMap_spectral_sequence_filtration X E h t n := by
    cases hEq
    rfl
  have hCoh :
      h.pageRestrict
        (@SectionMap_spectral_sequence_filtration X E h s m) n =
        @SectionMap_spectral_sequence_filtration X E h s n := by
    simpa [SectionMap_spectral_sequence_filtration] using h.page_coherence s m n hnm
  exact ⟨hPoint, hCoh.trans hPoint⟩

theorem global_to_local_spectral_sequence_filtration
    {X : Type u} {E : Type v} [h : TopStruct_spectral_sequence_filtration X E]
    {s t : E}
    (hEq :
      ∀ n : Nat,
        @SectionMap_spectral_sequence_filtration X E h s n =
          @SectionMap_spectral_sequence_filtration X E h t n) :
    (∀ m n : Nat,
      n ≤ m →
        h.pageRestrict
          (@SectionMap_spectral_sequence_filtration X E h s m) n =
          @SectionMap_spectral_sequence_filtration X E h t n) →
    s = t := by
  intro _
  have hPointwise : ∀ n : Nat, h.pageRestrict s n = h.pageRestrict t n := by
    intro n
    exact hEq n
  exact h.page_ext s t hPointwise

theorem glue_exists_spectral_sequence_filtration
    {X : Type u} {E : Type v} [h : TopStruct_spectral_sequence_filtration X E]
    (U : OpenFamily_spectral_sequence_filtration X)
    (σ : GlueData_spectral_sequence_filtration E)
    (hLoc : h.gluable U σ)
    (hCmp : h.compatible U σ) :
    ∃ s : E,
      (∀ n : Nat, @SectionMap_spectral_sequence_filtration X E h s n = σ n) ∧
      (h.compatible U σ →
        ∀ m n : Nat,
          n ≤ m →
            h.pageRestrict
              (@SectionMap_spectral_sequence_filtration X E h s m) n = σ n) := by
  rcases h.glue U σ hLoc hCmp with ⟨s, hs⟩
  refine ⟨s, hs, ?_⟩
  intro _ m n hnm
  calc
    h.pageRestrict
        (@SectionMap_spectral_sequence_filtration X E h s m) n
        = @SectionMap_spectral_sequence_filtration X E h s n := by
          simpa [SectionMap_spectral_sequence_filtration] using h.page_coherence s m n hnm
    _ = σ n := hs n

theorem glue_unique_spectral_sequence_filtration
    {X : Type u} {E : Type v} [h : TopStruct_spectral_sequence_filtration X E]
    (σ : GlueData_spectral_sequence_filtration E)
    (s t : E)
    (hs : ∀ n : Nat, @SectionMap_spectral_sequence_filtration X E h s n = σ n)
    (ht : ∀ n : Nat, @SectionMap_spectral_sequence_filtration X E h t n = σ n) :
    ∃ p : s = t,
      ∀ m n : Nat,
        n ≤ m →
          h.pageRestrict
            (@SectionMap_spectral_sequence_filtration X E h s m) n =
          @SectionMap_spectral_sequence_filtration X E h t n := by
  have hPointwise :
      ∀ n : Nat,
        @SectionMap_spectral_sequence_filtration X E h s n =
          @SectionMap_spectral_sequence_filtration X E h t n := by
    intro n
    calc
      @SectionMap_spectral_sequence_filtration X E h s n = σ n := hs n
      _ = @SectionMap_spectral_sequence_filtration X E h t n := by
        symm
        exact ht n
  have hST : s = t := h.page_ext s t hPointwise
  refine ⟨hST, ?_⟩
  intro m n hnm
  calc
    h.pageRestrict
        (@SectionMap_spectral_sequence_filtration X E h s m) n
        = @SectionMap_spectral_sequence_filtration X E h s n := by
          simpa [SectionMap_spectral_sequence_filtration] using h.page_coherence s m n hnm
    _ = @SectionMap_spectral_sequence_filtration X E h t n := hPointwise n

theorem descent_equiv_spectral_sequence_filtration
    {X : Type u} {E : Type v} [h : TopStruct_spectral_sequence_filtration X E]
    (U : OpenFamily_spectral_sequence_filtration X) :
    (∀ σ : GlueData_spectral_sequence_filtration E,
      h.gluable U σ →
      h.compatible U σ →
      ∃ s : E, ∀ n : Nat, @SectionMap_spectral_sequence_filtration X E h s n = σ n) ↔
    (∀ σ : GlueData_spectral_sequence_filtration E,
      h.gluable U σ →
      h.compatible U σ →
      ∃ s : E,
        (h.gluable U σ → h.compatible U σ) ∧
        (∀ n : Nat, @SectionMap_spectral_sequence_filtration X E h s n = σ n)) := by
  constructor
  · intro hForward σ hLoc hCmp
    rcases hForward σ hLoc hCmp with ⟨s, hs⟩
    refine ⟨s, ?_, hs⟩
    intro _
    exact hCmp
  · intro hBackward σ hLoc hCmp
    rcases hBackward σ hLoc hCmp with ⟨s, _, hs⟩
    exact ⟨s, hs⟩

theorem compactness_transfer_spectral_sequence_filtration
    {X : Type u} {E : Type v} [h : TopStruct_spectral_sequence_filtration X E]
    (U : OpenFamily_spectral_sequence_filtration X)
    (hRef : @CoverRefine_spectral_sequence_filtration X E h U h.filtrationCover)
    (hLocCmp : ∀ σ : GlueData_spectral_sequence_filtration E, h.gluable U σ → h.compatible U σ)
    (hBase : ∀ s : E, h.gluable h.filtrationCover (@SectionMap_spectral_sequence_filtration X E h s)) :
    (@CoverRefine_spectral_sequence_filtration X E h U h.filtrationCover → True) ∧
      (∀ s : E,
        ∀ m n : Nat,
          n ≤ m →
            h.pageRestrict
              (@SectionMap_spectral_sequence_filtration X E h s m) n =
            @SectionMap_spectral_sequence_filtration X E h s n) := by
  have hTransfer : True := h.compact_transfer_axiom U hRef hLocCmp hBase
  refine ⟨?_, ?_⟩
  · intro _
    exact hTransfer
  · intro s m n hnm
    simpa [SectionMap_spectral_sequence_filtration] using h.page_coherence s m n hnm
