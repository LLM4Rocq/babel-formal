/-
BENCHMARK_ID: TINY_MATHLIB_BATCH04_TOPOLOGY_UNIFORM_COMPLETION
PAIR_STEM: topology_uniform_space_completion_like
MATH_DOMAIN: Topology / Uniform Spaces
SOURCE_MATHLIB: Mathlib/Topology/UniformSpace/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u v w

class UniformSpaceLike (α : Type u) where
  entourage : (α → α → Prop) → Prop
  entourage_refl : ∀ {V : α → α → Prop}, entourage V → ∀ x : α, V x x
  entourage_symm :
    ∀ {V : α → α → Prop}, entourage V →
      ∃ W : α → α → Prop, entourage W ∧ (∀ x y : α, W x y → V y x)
  entourage_comp :
    ∀ {V : α → α → Prop}, entourage V →
      ∃ W : α → α → Prop, entourage W ∧
        (∀ x y z : α, W x y → W y z → V x z)
  entourage_mono :
    ∀ {V W : α → α → Prop}, entourage V →
      (∀ x y : α, W x y → V x y) → entourage W

def CauchyLike {α : Type u} [UniformSpaceLike α] (F : (α → Prop) → Prop) : Prop :=
  ∀ V : α → α → Prop,
    UniformSpaceLike.entourage V →
      ∃ s : α → Prop, F s ∧ ∀ x y : α, s x → s y → V x y

def CompleteLike {α : Type u} [UniformSpaceLike α] : Prop :=
  ∀ F : (α → Prop) → Prop,
    CauchyLike F →
      ∃ x : α,
        ∀ V : α → α → Prop,
          UniformSpaceLike.entourage V →
            ∃ s : α → Prop, F s ∧ ∀ y : α, s y → V y x

def DenseLike {α : Type u} {β : Type v} [UniformSpaceLike β] (f : α → β) : Prop :=
  ∀ y : β,
    ∀ V : β → β → Prop,
      UniformSpaceLike.entourage V →
        ∃ x : α, V (f x) y

structure CompletionLike (α : Type u) [UniformSpaceLike α] where
  β : Type v
  uniformβ : UniformSpaceLike β
  emb : α → β
  emb_dense : @DenseLike α β uniformβ emb
  emb_uniform :
    ∀ V : β → β → Prop,
      @UniformSpaceLike.entourage β uniformβ V →
        ∃ W : α → α → Prop,
          UniformSpaceLike.entourage W ∧
            (∀ x y : α, W x y → V (emb x) (emb y))
  completeβ : @CompleteLike β uniformβ

attribute [instance] CompletionLike.uniformβ

def LiftLike {α : Type u} [UniformSpaceLike α]
    (c : CompletionLike α)
    {γ : Type w} [UniformSpaceLike γ]
    (f : α → γ) : Prop :=
  ∃ g : c.β → γ,
    (∀ x : α, g (c.emb x) = f x) ∧
      (∀ V : γ → γ → Prop,
        UniformSpaceLike.entourage V →
          ∃ W : c.β → c.β → Prop,
            UniformSpaceLike.entourage W ∧
              (∀ x y : c.β, W x y → V (g x) (g y)))

theorem completion_map_dense {α : Type u} [UniformSpaceLike α]
    (c : CompletionLike α) :
    DenseLike c.emb := by
  intro y V hV
  have hDense : DenseLike c.emb := c.emb_dense
  have hWitness : ∃ x : α, V (c.emb x) y := hDense y V hV
  rcases hWitness with ⟨x, hx⟩
  refine ⟨x, ?_⟩
  exact hx

theorem completion_map_uniform {α : Type u} [UniformSpaceLike α]
    (c : CompletionLike α) :
    ∀ V : c.β → c.β → Prop,
      UniformSpaceLike.entourage V →
        ∃ W : α → α → Prop,
          UniformSpaceLike.entourage W ∧
            (∀ x y : α, W x y → V (c.emb x) (c.emb y)) := by
  intro V hV
  have hRaw :
      ∃ W : α → α → Prop,
        UniformSpaceLike.entourage W ∧
          (∀ x y : α, W x y → V (c.emb x) (c.emb y)) :=
    c.emb_uniform V hV
  rcases hRaw with ⟨W, hW, hWmap⟩
  refine ⟨W, hW, ?_⟩
  intro x y hxy
  exact hWmap x y hxy

theorem completion_extension_exists {α : Type u} [UniformSpaceLike α]
    (c : CompletionLike α)
    {γ : Type w} [UniformSpaceLike γ]
    (f : α → γ)
    (hLift : LiftLike c f) :
    ∃ g : c.β → γ, ∀ x : α, g (c.emb x) = f x := by
  rcases hLift with ⟨g, hg, hunif⟩
  have hgraph : ∀ x : α, g (c.emb x) = f x := by
    intro x
    exact hg x
  have _ : ∀ V : γ → γ → Prop,
      UniformSpaceLike.entourage V →
        ∃ W : c.β → c.β → Prop,
          UniformSpaceLike.entourage W ∧
            (∀ x y : c.β, W x y → V (g x) (g y)) := hunif
  refine ⟨g, ?_⟩
  intro x
  exact hgraph x

theorem completion_extension_unique {α : Type u} [UniformSpaceLike α]
    (c : CompletionLike α)
    {γ : Type w} [UniformSpaceLike γ]
    (f : α → γ)
    (hDenseExt :
      ∀ g₁ g₂ : c.β → γ,
        (∀ x : α, g₁ (c.emb x) = g₂ (c.emb x)) →
        g₁ = g₂)
    (hLift : LiftLike c f)
    (g₁ g₂ : c.β → γ)
    (hg₁ : ∀ x : α, g₁ (c.emb x) = f x)
    (hg₂ : ∀ x : α, g₂ (c.emb x) = f x) :
    g₁ = g₂ := by
  rcases hLift with ⟨g, hg, hunif⟩
  have hEq₁ : g₁ = g := by
    apply hDenseExt g₁ g
    intro x
    calc
      g₁ (c.emb x) = f x := hg₁ x
      _ = g (c.emb x) := by
        symm
        exact hg x
  have hEq₂ : g₂ = g := by
    apply hDenseExt g₂ g
    intro x
    calc
      g₂ (c.emb x) = f x := hg₂ x
      _ = g (c.emb x) := by
        symm
        exact hg x
  have _ : ∀ V : γ → γ → Prop,
      UniformSpaceLike.entourage V →
        ∃ W : c.β → c.β → Prop,
          UniformSpaceLike.entourage W ∧
            (∀ x y : c.β, W x y → V (g x) (g y)) := hunif
  calc
    g₁ = g := hEq₁
    _ = g₂ := by
      symm
      exact hEq₂

theorem complete_of_completion {α : Type u} [UniformSpaceLike α]
    (c : CompletionLike α) :
    CompleteLike (α := c.β) := by
  have hcomp : CompleteLike (α := c.β) := c.completeβ
  have hkeep : CompleteLike (α := c.β) := hcomp
  exact hkeep

theorem completion_idempotent_like {α : Type u} [UniformSpaceLike α]
    (c : CompletionLike α)
    (c2 : CompletionLike c.β)
    (hLift : LiftLike c2 (fun x : c.β => x))
    (hRetrUniq :
      ∀ r₁ r₂ : c2.β → c.β,
        (∀ x : c.β, r₁ (c2.emb x) = x) →
        (∀ x : c.β, r₂ (c2.emb x) = x) →
        r₁ = r₂) :
    ∃ r : c2.β → c.β,
      (∀ x : c.β, r (c2.emb x) = x) ∧
      (∀ s : c2.β → c.β, (∀ x : c.β, s (c2.emb x) = x) → s = r) := by
  rcases hLift with ⟨r, hr, hunif⟩
  have hr_id : ∀ x : c.β, r (c2.emb x) = x := by
    intro x
    have hraw : r (c2.emb x) = (fun y : c.β => y) x := hr x
    simpa using hraw
  refine ⟨r, hr_id, ?_⟩
  intro s hs
  have hsr : s = r := by
    apply hRetrUniq s r
    · exact hs
    · exact hr_id
  have _ : ∀ V : c.β → c.β → Prop,
      UniformSpaceLike.entourage V →
        ∃ W : c2.β → c2.β → Prop,
          UniformSpaceLike.entourage W ∧
            (∀ x y : c2.β, W x y → V (r x) (r y)) := hunif
  exact hsr
