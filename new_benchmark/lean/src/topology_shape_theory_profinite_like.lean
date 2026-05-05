/-
BENCHMARK_ID: TINY_MATHLIB_BATCH05_TOPOLOGY_SHAPE_THEORY_PROFINITE_LIKE
PAIR_STEM: topology_shape_theory_profinite_like
MATH_DOMAIN: Topology
SOURCE_MATHLIB: Mathlib/Topology/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u v

class TopStruct_shape_theory_profinite (X : Type u) (S : Type v) where
  IsClopen : (X → Prop) → Prop
  profiniteCover : Nat → X → Prop
  cover_spec : ∀ x : X, ∃ i : Nat, profiniteCover i x
  Refines : (Nat → X → Prop) → (Nat → X → Prop) → Prop
  refines_refl : ∀ U : Nat → X → Prop, Refines U U
  refines_comp :
    ∀ U V W : Nat → X → Prop,
      Refines U V → Refines V W → Refines U W
  localSection : (Nat → X → Prop) → (Nat → S) → Prop
  coherent : (Nat → X → Prop) → (Nat → S) → Prop
  restrict : S → Nat → S
  glue :
    ∀ U : Nat → X → Prop,
      ∀ σ : Nat → S,
        localSection U σ →
        coherent U σ →
          ∃ s : S, ∀ i : Nat, restrict s i = σ i
  section_ext :
    ∀ s t : S,
      (∀ i : Nat, restrict s i = restrict t i) →
        s = t
  compact_transfer_axiom :
    ∀ U : Nat → X → Prop,
      Refines U profiniteCover →
      (∀ σ : Nat → S, localSection U σ → coherent U σ) →
      (∀ s : S, localSection profiniteCover (restrict s)) →
      ∀ σ : Nat → S,
        localSection U σ →
        coherent U σ →
          ∃ s : S, ∀ i : Nat, restrict s i = σ i

def OpenFamily_shape_theory_profinite (X : Type u) : Type u :=
  Nat → X → Prop

def CoverRefine_shape_theory_profinite
    {X : Type u} {S : Type v} [h : TopStruct_shape_theory_profinite X S]
    (U V : OpenFamily_shape_theory_profinite X) : Prop :=
  h.Refines U V

def GlueData_shape_theory_profinite (S : Type v) : Type v :=
  Nat → S

def SectionMap_shape_theory_profinite
    {X : Type u} {S : Type v} [h : TopStruct_shape_theory_profinite X S]
    (s : S) : GlueData_shape_theory_profinite S :=
  h.restrict s

theorem refine_trans_shape_theory_profinite
    {X : Type u} {S : Type v} [h : TopStruct_shape_theory_profinite X S]
    (U V W : OpenFamily_shape_theory_profinite X)
    (hUV : CoverRefine_shape_theory_profinite (X := X) (S := S) U V)
    (hVW : CoverRefine_shape_theory_profinite (X := X) (S := S) V W) :
    CoverRefine_shape_theory_profinite (X := X) (S := S) U W ∧
      (∀ x : X, ∃ i : Nat, h.profiniteCover i x) := by
  have hStepUV : h.Refines U V := hUV
  have hStepVW : h.Refines V W := hVW
  have hComposed : h.Refines U W := h.refines_comp U V W hStepUV hStepVW
  refine ⟨hComposed, ?_⟩
  intro x
  exact h.cover_spec x

theorem local_to_global_shape_theory_profinite
    {X : Type u} {S : Type v} [h : TopStruct_shape_theory_profinite X S]
    {s t : S}
    (hEq : s = t) :
    (∀ i : Nat,
      SectionMap_shape_theory_profinite (X := X) (S := S) s i =
        SectionMap_shape_theory_profinite (X := X) (S := S) t i) ∧
      h.restrict s = h.restrict t := by
  have hMapEq : h.restrict s = h.restrict t := by
    exact congrArg h.restrict hEq
  refine ⟨?_, hMapEq⟩
  intro i
  exact congrArg (fun f : Nat → S => f i) hMapEq

theorem global_to_local_shape_theory_profinite
    {X : Type u} {S : Type v} [h : TopStruct_shape_theory_profinite X S]
    {s t : S}
    (hEq :
      ∀ i : Nat,
        SectionMap_shape_theory_profinite (X := X) (S := S) s i =
          SectionMap_shape_theory_profinite (X := X) (S := S) t i) :
    s = t ∧ h.restrict s = h.restrict t := by
  have hPointwise :
      ∀ i : Nat,
        h.restrict s i = h.restrict t i := by
    intro i
    exact hEq i
  have hCollapsed : s = t := h.section_ext s t hPointwise
  have hRestrict : h.restrict s = h.restrict t := congrArg h.restrict hCollapsed
  exact ⟨hCollapsed, hRestrict⟩

theorem glue_exists_shape_theory_profinite
    {X : Type u} {S : Type v} [h : TopStruct_shape_theory_profinite X S]
    (U : OpenFamily_shape_theory_profinite X)
    (σ : GlueData_shape_theory_profinite S)
    (hLoc : h.localSection U σ)
    (hCoh : h.coherent U σ) :
    ∃ s : S,
      (∀ i : Nat,
        SectionMap_shape_theory_profinite (X := X) (S := S) s i = σ i) ∧
      ∀ t : S,
        (∀ i : Nat,
          SectionMap_shape_theory_profinite (X := X) (S := S) t i = σ i) →
          t = s := by
  rcases h.glue U σ hLoc hCoh with ⟨s, hs⟩
  have hPointwise :
      ∀ i : Nat,
        SectionMap_shape_theory_profinite (X := X) (S := S) s i = σ i := by
    intro i
    exact hs i
  have hUnique :
      ∀ t : S,
        (∀ i : Nat,
          SectionMap_shape_theory_profinite (X := X) (S := S) t i = σ i) →
          t = s := by
    intro t ht
    have hPair :
        ∀ i : Nat,
          SectionMap_shape_theory_profinite (X := X) (S := S) t i =
            SectionMap_shape_theory_profinite (X := X) (S := S) s i := by
      intro i
      calc
        SectionMap_shape_theory_profinite (X := X) (S := S) t i = σ i := ht i
        _ = SectionMap_shape_theory_profinite (X := X) (S := S) s i := by
          symm
          exact hs i
    exact h.section_ext t s hPair
  exact ⟨s, hPointwise, hUnique⟩

theorem glue_unique_shape_theory_profinite
    {X : Type u} {S : Type v} [h : TopStruct_shape_theory_profinite X S]
    (σ : GlueData_shape_theory_profinite S)
    (s t : S)
    (hs : ∀ i : Nat, SectionMap_shape_theory_profinite (X := X) (S := S) s i = σ i)
    (ht : ∀ i : Nat, SectionMap_shape_theory_profinite (X := X) (S := S) t i = σ i) :
    s = t ∧ h.restrict s = h.restrict t := by
  have hPointwise :
      ∀ i : Nat,
        SectionMap_shape_theory_profinite (X := X) (S := S) s i =
        SectionMap_shape_theory_profinite (X := X) (S := S) t i := by
    intro i
    have hs' : SectionMap_shape_theory_profinite (X := X) (S := S) s i = σ i := hs i
    have ht' : SectionMap_shape_theory_profinite (X := X) (S := S) t i = σ i := ht i
    calc
      SectionMap_shape_theory_profinite (X := X) (S := S) s i = σ i := hs'
      _ = SectionMap_shape_theory_profinite (X := X) (S := S) t i := by
        symm
        exact ht'
  have hEq : s = t := h.section_ext s t hPointwise
  have hRestrict : h.restrict s = h.restrict t := congrArg h.restrict hEq
  exact ⟨hEq, hRestrict⟩

theorem descent_equiv_shape_theory_profinite
    {X : Type u} {S : Type v} [h : TopStruct_shape_theory_profinite X S]
    (U : OpenFamily_shape_theory_profinite X) :
    (∀ σ : GlueData_shape_theory_profinite S,
      h.localSection U σ →
      h.coherent U σ →
      ∃ s : S,
        (∀ i : Nat,
          SectionMap_shape_theory_profinite (X := X) (S := S) s i = σ i) ∧
        ∀ t : S,
          (∀ i : Nat,
            SectionMap_shape_theory_profinite (X := X) (S := S) t i = σ i) →
            t = s) ↔
    (∀ σ : GlueData_shape_theory_profinite S,
      h.localSection U σ →
      h.coherent U σ →
      ∃ s : S,
        (∀ i : Nat,
          SectionMap_shape_theory_profinite (X := X) (S := S) s i = σ i) ∧
        ∀ t : S,
          (∀ i : Nat,
            SectionMap_shape_theory_profinite (X := X) (S := S) t i = σ i) →
            h.restrict t = h.restrict s) := by
  constructor
  · intro hExist σ hLoc hCoh
    rcases hExist σ hLoc hCoh with ⟨s, hs, huniq⟩
    refine ⟨s, hs, ?_⟩
    intro t ht
    exact congrArg h.restrict (huniq t ht)
  · intro hStrong σ hLoc hCoh
    rcases hStrong σ hLoc hCoh with ⟨s, hs, huniqR⟩
    refine ⟨s, hs, ?_⟩
    intro t ht
    have hEqRestrict : h.restrict t = h.restrict s := huniqR t ht
    have hEqPoint :
        ∀ i : Nat,
          SectionMap_shape_theory_profinite (X := X) (S := S) t i =
            SectionMap_shape_theory_profinite (X := X) (S := S) s i := by
      intro i
      exact congrArg (fun f : Nat → S => f i) hEqRestrict
    exact h.section_ext t s hEqPoint

theorem compactness_transfer_shape_theory_profinite
    {X : Type u} {S : Type v} [h : TopStruct_shape_theory_profinite X S]
    (U : OpenFamily_shape_theory_profinite X)
    (hRef : CoverRefine_shape_theory_profinite (X := X) (S := S) U h.profiniteCover)
    (hComp : ∀ σ : GlueData_shape_theory_profinite S, h.localSection U σ → h.coherent U σ)
    (hBase :
      ∀ s : S,
        h.localSection h.profiniteCover
          (SectionMap_shape_theory_profinite (X := X) (S := S) s)) :
    ∀ σ : GlueData_shape_theory_profinite S,
      h.localSection U σ →
      h.coherent U σ →
      ∃ s : S,
        (∀ i : Nat,
          SectionMap_shape_theory_profinite (X := X) (S := S) s i = σ i) ∧
        ∀ t : S,
          (∀ i : Nat,
            SectionMap_shape_theory_profinite (X := X) (S := S) t i = σ i) →
            h.restrict t = h.restrict s := by
  intro σ hLoc hCoh
  have hExist :
      ∃ s : S,
        ∀ i : Nat,
          SectionMap_shape_theory_profinite (X := X) (S := S) s i = σ i :=
    h.compact_transfer_axiom U hRef hComp hBase σ hLoc hCoh
  rcases hExist with ⟨s, hs⟩
  refine ⟨s, hs, ?_⟩
  intro t ht
  have hUnique : t = s :=
    h.section_ext t s (by
      intro i
      calc
        SectionMap_shape_theory_profinite (X := X) (S := S) t i = σ i := ht i
        _ = SectionMap_shape_theory_profinite (X := X) (S := S) s i := by
          symm
          exact hs i)
  exact congrArg h.restrict hUnique
