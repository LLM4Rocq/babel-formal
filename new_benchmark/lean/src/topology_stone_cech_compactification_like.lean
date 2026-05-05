/-
BENCHMARK_ID: TINY_MATHLIB_BATCH06_TOPOLOGY_STONE_CECH_COMPACTIFICATION_LIKE
PAIR_STEM: topology_stone_cech_compactification_like
MATH_DOMAIN: Topology
SOURCE_MATHLIB: Mathlib/Topology/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u v w

class CompactificationStruct_stone_cech (X : Type u) (βX : Type v) (Y : Type w) where
  embed : X → βX
  dense_image : Prop
  compact_space : Prop
  extend : (X → Y) → βX → Y
  extend_on_embed :
    ∀ (f : X → Y) (x : X),
      extend f (embed x) = f x
  extend_unique :
    ∀ (f : X → Y) (g h : βX → Y),
      (∀ x : X, g (embed x) = f x) →
      (∀ x : X, h (embed x) = f x) →
      g = h
  extend_comp :
    ∀ (f : X → Y) (k : Y → Y) (b : βX),
      extend (fun x : X => k (f x)) b = k (extend f b)
  core : (βX → Prop) → Prop
  core_from_dense :
    ∀ p : βX → Prop,
      (∀ x : X, p (embed x)) →
      core p
  core_mono :
    ∀ p q : βX → Prop,
      core p →
      (∀ b : βX, p b → q b) →
      core q

structure ExtensionData_stone_cech_compactification
    (X : Type u) (βX : Type v) (Y : Type w)
    [h : CompactificationStruct_stone_cech X βX Y] where
  base_map : X → Y
  extended_map : βX → Y
  agree_on_dense :
    ∀ x : X,
      extended_map (h.embed x) = base_map x

def dense_embedding_stone_cech_compactification
    {X : Type u} {βX : Type v} {Y : Type w}
    [h : CompactificationStruct_stone_cech X βX Y] :
    X → βX :=
  h.embed

def extension_map_stone_cech_compactification
    {X : Type u} {βX : Type v} {Y : Type w}
    [h : CompactificationStruct_stone_cech X βX Y]
    (f : X → Y) :
    βX → Y :=
  h.extend f

def compact_core_stone_cech_compactification
    {X : Type u} {βX : Type v} {Y : Type w}
    [h : CompactificationStruct_stone_cech X βX Y]
    (p : βX → Prop) :
    Prop :=
  h.core p

theorem extension_exists_stone_cech_compactification
    {X : Type u} {βX : Type v} {Y : Type w}
    [h : CompactificationStruct_stone_cech X βX Y]
    (f : X → Y) :
    ∃ g : βX → Y,
      (∀ x : X,
        g (dense_embedding_stone_cech_compactification (X := X) (βX := βX) (Y := Y) x) = f x) ∧
      (∀ b : βX,
        g b = extension_map_stone_cech_compactification (X := X) (βX := βX) (Y := Y) f b) := by
  refine ⟨extension_map_stone_cech_compactification (X := X) (βX := βX) (Y := Y) f, ?_, ?_⟩
  · intro x
    have hAgree : h.extend f (h.embed x) = f x := h.extend_on_embed f x
    exact hAgree
  · intro b
    have hDef :
        extension_map_stone_cech_compactification (X := X) (βX := βX) (Y := Y) f b = h.extend f b :=
      rfl
    have hSymm : h.extend f b =
        extension_map_stone_cech_compactification (X := X) (βX := βX) (Y := Y) f b :=
      Eq.symm hDef
    exact hSymm

theorem extension_unique_stone_cech_compactification
    {X : Type u} {βX : Type v} {Y : Type w}
    [h : CompactificationStruct_stone_cech X βX Y]
    (f : X → Y)
    (g1 g2 : βX → Y)
    (hg1 : ∀ x : X,
      g1 (dense_embedding_stone_cech_compactification (X := X) (βX := βX) (Y := Y) x) = f x)
    (hg2 : ∀ x : X,
      g2 (dense_embedding_stone_cech_compactification (X := X) (βX := βX) (Y := Y) x) = f x) :
    g1 = g2 ∧
      ∀ b : βX, g1 b = g2 b := by
  have hEq : g1 = g2 := h.extend_unique f g1 g2 hg1 hg2
  have hPointwise : ∀ b : βX, g1 b = g2 b := by
    intro b
    exact congrArg (fun g : βX → Y => g b) hEq
  exact ⟨hEq, hPointwise⟩

theorem extension_respects_comp_stone_cech_compactification
    {X : Type u} {βX : Type v} {Y : Type w}
    [h : CompactificationStruct_stone_cech X βX Y]
    (f : X → Y)
    (k : Y → Y) :
    (∀ x : X,
      extension_map_stone_cech_compactification (X := X) (βX := βX) (Y := Y)
        (fun t : X => k (f t))
        (dense_embedding_stone_cech_compactification (X := X) (βX := βX) (Y := Y) x) = k (f x)) ∧
    (∀ b : βX,
      extension_map_stone_cech_compactification (X := X) (βX := βX) (Y := Y)
        (fun t : X => k (f t)) b =
      k (extension_map_stone_cech_compactification (X := X) (βX := βX) (Y := Y) f b)) := by
  refine ⟨?_, ?_⟩
  · intro x
    have hStep : h.extend (fun t : X => k (f t)) (h.embed x) = k (f x) := by
      calc
        h.extend (fun t : X => k (f t)) (h.embed x)
            = (fun t : X => k (f t)) x := h.extend_on_embed (fun t : X => k (f t)) x
        _ = k (f x) := rfl
    exact hStep
  · intro b
    have hComp : h.extend (fun t : X => k (f t)) b = k (h.extend f b) := h.extend_comp f k b
    exact hComp

theorem dense_image_universal_stone_cech_compactification
    {X : Type u} {βX : Type v} {Y : Type w}
    [h : CompactificationStruct_stone_cech X βX Y]
    (p : βX → Prop)
    (hp : ∀ x : X,
      p (dense_embedding_stone_cech_compactification (X := X) (βX := βX) (Y := Y) x)) :
    compact_core_stone_cech_compactification (X := X) (βX := βX) (Y := Y) p ∧
      ∀ q : βX → Prop,
        (∀ b : βX, p b → q b) →
        compact_core_stone_cech_compactification (X := X) (βX := βX) (Y := Y) q := by
  have hCoreP : h.core p := h.core_from_dense p hp
  refine ⟨hCoreP, ?_⟩
  intro q hpq
  have hCoreQ : h.core q := h.core_mono p q hCoreP hpq
  exact hCoreQ

theorem compact_core_minimal_stone_cech_compactification
    {X : Type u} {βX : Type v} {Y : Type w}
    [h : CompactificationStruct_stone_cech X βX Y]
    (p q r : βX → Prop)
    (hp : compact_core_stone_cech_compactification (X := X) (βX := βX) (Y := Y) p)
    (hpq : ∀ b : βX, p b → q b)
    (hqr : ∀ b : βX, q b → r b) :
    compact_core_stone_cech_compactification (X := X) (βX := βX) (Y := Y) r ∧
      compact_core_stone_cech_compactification (X := X) (βX := βX) (Y := Y) q := by
  have hCoreQ : compact_core_stone_cech_compactification (X := X) (βX := βX) (Y := Y) q :=
    h.core_mono p q hp hpq
  have hCoreR : compact_core_stone_cech_compactification (X := X) (βX := βX) (Y := Y) r :=
    h.core_mono q r hCoreQ hqr
  exact ⟨hCoreR, hCoreQ⟩

theorem factorization_through_core_stone_cech_compactification
    {X : Type u} {βX : Type v} {Y : Type w}
    [h : CompactificationStruct_stone_cech X βX Y]
    (data : ExtensionData_stone_cech_compactification X βX Y)
    (p : βX → Prop)
    (hpDense : ∀ x : X,
      p (dense_embedding_stone_cech_compactification (X := X) (βX := βX) (Y := Y) x))
    (hconst : ∀ b1 b2 : βX, p b1 → p b2 → data.extended_map b1 = data.extended_map b2)
    (hnonempty : ∃ b0 : βX, p b0) :
    ∃ y0 : Y,
      (∀ b : βX, p b → data.extended_map b = y0) ∧
      compact_core_stone_cech_compactification (X := X) (βX := βX) (Y := Y) p := by
  have hCoreP : compact_core_stone_cech_compactification (X := X) (βX := βX) (Y := Y) p :=
    h.core_from_dense p hpDense
  rcases hnonempty with ⟨b0, hb0⟩
  refine ⟨data.extended_map b0, ?_, hCoreP⟩
  intro b hb
  exact hconst b b0 hb hb0

theorem universal_property_stone_cech_compactification
    {X : Type u} {βX : Type v} {Y : Type w}
    [h : CompactificationStruct_stone_cech X βX Y]
    (f : X → Y) :
    (∃ g : βX → Y,
      ∀ x : X,
        g (dense_embedding_stone_cech_compactification (X := X) (βX := βX) (Y := Y) x) = f x) ∧
    (∀ g1 g2 : βX → Y,
      (∀ x : X,
        g1 (dense_embedding_stone_cech_compactification (X := X) (βX := βX) (Y := Y) x) = f x) →
      (∀ x : X,
        g2 (dense_embedding_stone_cech_compactification (X := X) (βX := βX) (Y := Y) x) = f x) →
      ∀ b : βX, g1 b = g2 b) := by
  have hExistStrong :
      ∃ g : βX → Y,
        (∀ x : X,
          g (dense_embedding_stone_cech_compactification (X := X) (βX := βX) (Y := Y) x) = f x) ∧
        (∀ b : βX,
          g b = extension_map_stone_cech_compactification (X := X) (βX := βX) (Y := Y) f b) :=
    extension_exists_stone_cech_compactification (X := X) (βX := βX) (Y := Y) f
  have hExist :
      ∃ g : βX → Y,
        ∀ x : X,
          g (dense_embedding_stone_cech_compactification (X := X) (βX := βX) (Y := Y) x) = f x := by
    rcases hExistStrong with ⟨g, hg, _⟩
    exact ⟨g, hg⟩
  refine ⟨hExist, ?_⟩
  intro g1 g2 hg1 hg2 b
  have hEq : g1 = g2 := h.extend_unique f g1 g2 hg1 hg2
  exact congrArg (fun g : βX → Y => g b) hEq
