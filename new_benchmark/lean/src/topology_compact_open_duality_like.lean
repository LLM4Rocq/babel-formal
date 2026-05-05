/-
BENCHMARK_ID: TINY_MATHLIB_BATCH04_TOPOLOGY_COMPACT_OPEN_DUALITY
PAIR_STEM: topology_compact_open_duality_like
MATH_DOMAIN: Topology
SOURCE_MATHLIB: Mathlib/Topology/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u v w

class TopologicalSpaceLike (X : Type u) where
  IsOpen : (X → Prop) → Prop
  open_univ : IsOpen (fun _ : X => True)
  open_inter :
    ∀ U V : X → Prop,
      IsOpen U → IsOpen V → IsOpen (fun x : X => U x ∧ V x)
  open_ext :
    ∀ U V : X → Prop,
      (∀ x : X, U x ↔ V x) → IsOpen U → IsOpen V

def CompactLike {X : Type u} [TopologicalSpaceLike X] (K : X → Prop) : Prop :=
  ∀ C : (X → Prop) → Prop,
    (∀ U : X → Prop, C U → TopologicalSpaceLike.IsOpen U) →
    (∀ x : X, K x → ∃ U : X → Prop, C U ∧ U x) →
    ∃ U : X → Prop, C U ∧ ∀ x : X, K x → U x

def OpenLike {X : Type u} [TopologicalSpaceLike X] (U : X → Prop) : Prop :=
  TopologicalSpaceLike.IsOpen U

def CompactOpenLike {X : Type u} [TopologicalSpaceLike X]
    (K U : X → Prop) : Prop :=
  CompactLike K ∧ OpenLike U

def EvaluationLike {X : Type u} {Y : Type v}
    (ev : (X → Y) → X → Y) : Prop :=
  ∀ f : X → Y, ∀ x : X, ev f x = f x

def ExponentialLike {X : Type u} {Y : Type v} {Z : Type w}
    (tr : (X → Y → Z) → Y → X → Z) : Prop :=
  ∀ g : X → Y → Z, ∀ y : Y, ∀ x : X, tr g y x = g x y

theorem compact_open_mono_left {X : Type u} [TopologicalSpaceLike X]
    {K₁ K₂ U : X → Prop}
    (hsub : ∀ x : X, K₁ x → K₂ x)
    (hext :
      ∀ C : (X → Prop) → Prop,
        (∀ Uo : X → Prop, C Uo → OpenLike Uo) →
        (∀ x : X, K₁ x → ∃ Uo : X → Prop, C Uo ∧ Uo x) →
        (∀ x : X, K₂ x → ∃ Uo : X → Prop, C Uo ∧ Uo x))
    (hco : CompactOpenLike K₂ U) :
    CompactOpenLike K₁ U := by
  have hK₂ : CompactLike K₂ := hco.1
  have hOpenU : OpenLike U := hco.2
  have hK₁ : CompactLike K₁ := by
    intro C hOpen hCover₁
    have hCover₂ : ∀ x : X, K₂ x → ∃ Uo : X → Prop, C Uo ∧ Uo x :=
      hext C hOpen hCover₁
    rcases hK₂ C hOpen hCover₂ with ⟨Uo, hCUo, hKUo⟩
    refine ⟨Uo, hCUo, ?_⟩
    intro x hxK₁
    have hxK₂ : K₂ x := hsub x hxK₁
    exact hKUo x hxK₂
  exact And.intro hK₁ hOpenU

theorem compact_open_mono_right {X : Type u} [TopologicalSpaceLike X]
    {K U₁ U₂ : X → Prop}
    (hopen_mono :
      ∀ {V W : X → Prop},
        OpenLike V →
        (∀ x : X, V x → W x) →
        OpenLike W)
    (hsub : ∀ x : X, U₁ x → U₂ x)
    (hco : CompactOpenLike K U₁) :
    CompactOpenLike K U₂ := by
  have hK : CompactLike K := hco.1
  have hOpen₁ : OpenLike U₁ := hco.2
  have hOpen₂ : OpenLike U₂ := hopen_mono hOpen₁ hsub
  exact And.intro hK hOpen₂

theorem evaluation_continuous_like {X : Type u} {Y : Type v}
    [TopologicalSpaceLike Y] [TopologicalSpaceLike (X → Y)]
    (ev : (X → Y) → X → Y)
    (hev : EvaluationLike ev)
    (x : X)
    (V : Y → Prop)
    (hopenV : OpenLike V)
    (hcont : OpenLike (fun f : X → Y => V (ev f x))) :
    OpenLike (fun f : X → Y => V (f x)) := by
  have hEq :
      ∀ f : X → Y,
        (fun g : X → Y => V (ev g x)) f ↔ (fun g : X → Y => V (g x)) f := by
    intro f
    constructor
    · intro hf
      simpa [hev f x] using hf
    · intro hf
      simpa [hev f x] using hf
  have hOpenEval : TopologicalSpaceLike.IsOpen (fun f : X → Y => V (ev f x)) := hcont
  have _ : OpenLike V := hopenV
  exact TopologicalSpaceLike.open_ext _ _ hEq hOpenEval

theorem transpose_continuous_like {X : Type u} {Y : Type v} {Z : Type w}
    [TopologicalSpaceLike X] [TopologicalSpaceLike Z]
    (tr : (X → Y → Z) → Y → X → Z)
    (htr : ExponentialLike tr)
    (g : X → Y → Z)
    (y : Y)
    (W : Z → Prop)
    (hopenW : OpenLike W)
    (hcont : OpenLike (fun x : X => W (tr g y x))) :
    OpenLike (fun x : X => W (g x y)) := by
  have hEq :
      ∀ x : X,
        (fun t : X => W (tr g y t)) x ↔ (fun t : X => W (g t y)) x := by
    intro x
    constructor
    · intro hx
      simpa [htr g y x] using hx
    · intro hx
      simpa [htr g y x] using hx
  have hOpenTr : TopologicalSpaceLike.IsOpen (fun x : X => W (tr g y x)) := hcont
  have _ : OpenLike W := hopenW
  exact TopologicalSpaceLike.open_ext _ _ hEq hOpenTr

theorem compact_open_universal_like {X : Type u} [TopologicalSpaceLike X]
    (hleft :
      ∀ {K₁ K₂ U : X → Prop},
        (∀ x : X, K₁ x → K₂ x) →
        (∀ C : (X → Prop) → Prop,
          (∀ Uo : X → Prop, C Uo → OpenLike Uo) →
          (∀ x : X, K₁ x → ∃ Uo : X → Prop, C Uo ∧ Uo x) →
          (∀ x : X, K₂ x → ∃ Uo : X → Prop, C Uo ∧ Uo x)) →
        CompactOpenLike K₂ U →
        CompactOpenLike K₁ U)
    (hright :
      ∀ {K U₁ U₂ : X → Prop},
        (∀ {V W : X → Prop}, OpenLike V → (∀ x : X, V x → W x) → OpenLike W) →
        (∀ x : X, U₁ x → U₂ x) →
        CompactOpenLike K U₁ →
        CompactOpenLike K U₂)
    {K₁ K₂ U₁ U₂ : X → Prop}
    (hsubK : ∀ x : X, K₁ x → K₂ x)
    (hExt :
      ∀ C : (X → Prop) → Prop,
        (∀ Uo : X → Prop, C Uo → OpenLike Uo) →
        (∀ x : X, K₁ x → ∃ Uo : X → Prop, C Uo ∧ Uo x) →
        (∀ x : X, K₂ x → ∃ Uo : X → Prop, C Uo ∧ Uo x))
    (hopen_mono :
      ∀ {V W : X → Prop},
        OpenLike V →
        (∀ x : X, V x → W x) →
        OpenLike W)
    (hsubU : ∀ x : X, U₁ x → U₂ x)
    (hco : CompactOpenLike K₂ U₁) :
    CompactOpenLike K₁ U₂ := by
  have hleft_step : CompactOpenLike K₁ U₁ := hleft hsubK hExt hco
  have hright_step : CompactOpenLike K₁ U₂ := hright hopen_mono hsubU hleft_step
  exact hright_step

theorem alexander_subbase_like {X : Type u} [TopologicalSpaceLike X]
    (K : X → Prop)
    (Sub : (X → Prop) → Prop)
    (hSubOpen : ∀ U : X → Prop, Sub U → OpenLike U)
    (hSubCriterion :
      ∀ C : (X → Prop) → Prop,
        (∀ U : X → Prop, C U → Sub U) →
        (∀ x : X, K x → ∃ U : X → Prop, C U ∧ U x) →
        ∃ U : X → Prop, C U ∧ ∀ x : X, K x → U x)
    (hRefine :
      ∀ C : (X → Prop) → Prop,
        (∀ U : X → Prop, C U → OpenLike U) →
        (∀ x : X, K x → ∃ U : X → Prop, C U ∧ U x) →
        ∃ C' : (X → Prop) → Prop,
          (∀ U : X → Prop, C' U → Sub U) ∧
          (∀ x : X, K x → ∃ U : X → Prop, C' U ∧ U x) ∧
          (∀ U : X → Prop, C' U → C U)) :
    CompactLike K := by
  intro C hOpen hCover
  rcases hRefine C hOpen hCover with ⟨C', hCSub, hCCover, hCToC⟩
  have hCOpen : ∀ U : X → Prop, C' U → OpenLike U := by
    intro U hU
    exact hSubOpen U (hCSub U hU)
  have _ : ∀ U : X → Prop, C' U → OpenLike U := hCOpen
  rcases hSubCriterion C' hCSub hCCover with ⟨U, hU', hUK⟩
  refine ⟨U, ?_⟩
  refine And.intro (hCToC U hU') ?_
  intro x hx
  exact hUK x hx
