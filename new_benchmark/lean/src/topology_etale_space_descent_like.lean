/-
BENCHMARK_ID: TINY_MATHLIB_BATCH05_TOPOLOGY_ETALE_SPACE_DESCENT_LIKE
PAIR_STEM: topology_etale_space_descent_like
MATH_DOMAIN: Topology
SOURCE_MATHLIB: Mathlib/Topology/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u v

class TopStruct_etale_space_descent (X : Type u) (S : Type v) where
  IsOpen : (X → Prop) → Prop
  fullCover : Nat → X → Prop
  full_cover_spec : ∀ x : X, ∃ i : Nat, fullCover i x
  Refines : (Nat → X → Prop) → (Nat → X → Prop) → Prop
  refines_refl : ∀ U : Nat → X → Prop, Refines U U
  refines_comp :
    ∀ U V W : Nat → X → Prop,
      Refines U V → Refines V W → Refines U W
  localSection : (Nat → X → Prop) → (Nat → S) → Prop
  compatible : (Nat → X → Prop) → (Nat → S) → Prop
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

def OpenFamily_etale_space_descent (X : Type u) : Type u :=
  Nat → X → Prop

def CoverRefine_etale_space_descent
    {X : Type u} {S : Type v} [h : TopStruct_etale_space_descent X S]
    (U V : OpenFamily_etale_space_descent X) : Prop :=
  h.Refines U V

def GlueData_etale_space_descent (S : Type v) : Type v :=
  Nat → S

def SectionMap_etale_space_descent
    {X : Type u} {S : Type v} [h : TopStruct_etale_space_descent X S]
    (s : S) : GlueData_etale_space_descent S :=
  h.restrict s

theorem refine_trans_etale_space_descent
    {X : Type u} {S : Type v} [h : TopStruct_etale_space_descent X S]
    (U V W : OpenFamily_etale_space_descent X)
    (hUV : @CoverRefine_etale_space_descent X S h U V)
    (hVW : @CoverRefine_etale_space_descent X S h V W)
    (x : X) :
    ∃ i : Nat, h.fullCover i x ∧ @CoverRefine_etale_space_descent X S h U W := by
  have hUW : h.Refines U W := h.refines_comp U V W hUV hVW
  rcases h.full_cover_spec x with ⟨i, hi⟩
  exact ⟨i, hi, hUW⟩

theorem local_to_global_etale_space_descent
    {X : Type u} {S : Type v} [h : TopStruct_etale_space_descent X S]
    {s t : S}
    (hEq : s = t) :
    ∀ i : Nat,
      @SectionMap_etale_space_descent X S h s i =
        @SectionMap_etale_space_descent X S h t i ∧
      h.restrict s = h.restrict t := by
  have hMap : h.restrict s = h.restrict t := congrArg h.restrict hEq
  intro i
  have hPt :
      @SectionMap_etale_space_descent X S h s i =
        @SectionMap_etale_space_descent X S h t i :=
    congrArg (fun f : Nat → S => f i) hMap
  exact ⟨hPt, hMap⟩

theorem global_to_local_etale_space_descent
    {X : Type u} {S : Type v} [h : TopStruct_etale_space_descent X S]
    {s t : S}
    (hEq :
      ∀ i : Nat,
        @SectionMap_etale_space_descent X S h s i =
          @SectionMap_etale_space_descent X S h t i) :
    s = t ∧
      (∀ i : Nat,
        @SectionMap_etale_space_descent X S h s i =
          @SectionMap_etale_space_descent X S h t i) ∧
      h.restrict s = h.restrict t := by
  have hPointwise : ∀ i : Nat, h.restrict s i = h.restrict t i := by
    intro i
    exact hEq i
  have hST : s = t := h.section_ext s t hPointwise
  have hMap : h.restrict s = h.restrict t := congrArg h.restrict hST
  exact ⟨hST, hEq, hMap⟩

theorem glue_exists_etale_space_descent
    {X : Type u} {S : Type v} [h : TopStruct_etale_space_descent X S]
    (U : OpenFamily_etale_space_descent X)
    (σ : GlueData_etale_space_descent S)
    (hLoc : h.localSection U σ)
    (hCmp : h.compatible U σ) :
    ∃ s : S,
      (∀ i : Nat, @SectionMap_etale_space_descent X S h s i = σ i) ∧
      h.localSection U σ ∧ h.compatible U σ := by
  rcases h.glue U σ hLoc hCmp with ⟨s, hs⟩
  exact ⟨s, hs, hLoc, hCmp⟩

theorem glue_unique_etale_space_descent
    {X : Type u} {S : Type v} [h : TopStruct_etale_space_descent X S]
    (σ : GlueData_etale_space_descent S)
    (s t : S)
    (hs : ∀ i : Nat, @SectionMap_etale_space_descent X S h s i = σ i)
    (ht : ∀ i : Nat, @SectionMap_etale_space_descent X S h t i = σ i) :
    s = t ∧
      (∀ i : Nat,
        @SectionMap_etale_space_descent X S h s i =
          @SectionMap_etale_space_descent X S h t i) ∧
      h.restrict s = h.restrict t := by
  have hPointwise :
      ∀ i : Nat,
        @SectionMap_etale_space_descent X S h s i =
          @SectionMap_etale_space_descent X S h t i := by
    intro i
    calc
      @SectionMap_etale_space_descent X S h s i = σ i := hs i
      _ = @SectionMap_etale_space_descent X S h t i := by
        symm
        exact ht i
  have hST : s = t := h.section_ext s t hPointwise
  have hMap : h.restrict s = h.restrict t := congrArg h.restrict hST
  exact ⟨hST, hPointwise, hMap⟩

theorem descent_equiv_etale_space_descent
    {X : Type u} {S : Type v} [h : TopStruct_etale_space_descent X S]
    (U : OpenFamily_etale_space_descent X) :
    (∀ σ : GlueData_etale_space_descent S,
      h.localSection U σ →
      h.compatible U σ →
      ∃ s : S, ∀ i : Nat, @SectionMap_etale_space_descent X S h s i = σ i) ↔
    (∀ σ : GlueData_etale_space_descent S,
      h.localSection U σ →
      h.compatible U σ →
      ∃ s : S,
        (∀ i : Nat, @SectionMap_etale_space_descent X S h s i = σ i) ∧
        h.localSection U σ ∧ h.compatible U σ) := by
  constructor
  · intro hForward σ hLoc hCmp
    rcases hForward σ hLoc hCmp with ⟨s, hs⟩
    exact ⟨s, hs, hLoc, hCmp⟩
  · intro hBackward σ hLoc hCmp
    rcases hBackward σ hLoc hCmp with ⟨s, hs, _, _⟩
    exact ⟨s, hs⟩

theorem compactness_transfer_etale_space_descent
    {X : Type u} {S : Type v} [h : TopStruct_etale_space_descent X S]
    (U : OpenFamily_etale_space_descent X)
    (hRef : @CoverRefine_etale_space_descent X S h U h.fullCover)
    (hComp : ∀ σ : GlueData_etale_space_descent S, h.localSection U σ → h.compatible U σ)
    (hBase : ∀ s : S, h.localSection h.fullCover (@SectionMap_etale_space_descent X S h s))
    (x : X) :
    ∃ i : Nat, h.fullCover i x ∧ True ∧ @CoverRefine_etale_space_descent X S h U h.fullCover := by
  have hTransfer : True := h.compact_transfer_axiom U hRef hComp hBase
  rcases h.full_cover_spec x with ⟨i, hi⟩
  exact ⟨i, hi, hTransfer, hRef⟩
