/-
BENCHMARK_ID: TINY_MATHLIB_BATCH03_ALG_LOCALIZATION_UNIVERSAL_LIKE
PAIR_STEM: algebra_localization_universal_like
MATH_DOMAIN: Commutative Algebra
SOURCE_MATHLIB: Mathlib/RingTheory/Localization/Basic
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u v

class CommMonoidLike (α : Type u) where
  one : α
  mul : α → α → α
  mul_assoc : ∀ a b c : α, mul (mul a b) c = mul a (mul b c)
  one_mul : ∀ a : α, mul one a = a
  mul_one : ∀ a : α, mul a one = a
  mul_comm : ∀ a b : α, mul a b = mul b a

infixl:70 " * " => CommMonoidLike.mul

instance {α : Type u} [CommMonoidLike α] : OfNat α (nat_lit 1) where
  ofNat := CommMonoidLike.one

def IsSubmonoidLike {α : Type u} [CommMonoidLike α] (S : α → Prop) : Prop :=
  S 1 ∧ ∀ a b : α, S a → S b → S (a * b)

structure LocalizationLike (α : Type u) [CommMonoidLike α] (S : α → Prop) where
  L : Type v
  oneL : L
  mulL : L → L → L
  of : α → L
  one_mul : ∀ x : L, mulL oneL x = x
  mul_one : ∀ x : L, mulL x oneL = x
  mul_assoc : ∀ x y z : L, mulL (mulL x y) z = mulL x (mulL y z)
  of_one : of 1 = oneL
  of_mul : ∀ a b : α, of (a * b) = mulL (of a) (of b)
  denom_unit :
    ∀ s : α, S s → ∃ invs : L, mulL (of s) invs = oneL ∧ mulL invs (of s) = oneL
  desc :
    ∀ {β : Type v},
      (oneβ : β) →
      (mulβ : β → β → β) →
      (f : α → β) →
      f 1 = oneβ →
      (∀ a b : α, f (a * b) = mulβ (f a) (f b)) →
      (∀ s : α, S s → ∃ i : β, mulβ (f s) i = oneβ ∧ mulβ i (f s) = oneβ) →
      L → β
  desc_one :
    ∀ {β : Type v} (oneβ : β) (mulβ : β → β → β) (f : α → β)
      (h1 : f 1 = oneβ)
      (hmul : ∀ a b : α, f (a * b) = mulβ (f a) (f b))
      (hS : ∀ s : α, S s → ∃ i : β, mulβ (f s) i = oneβ ∧ mulβ i (f s) = oneβ),
      desc oneβ mulβ f h1 hmul hS oneL = oneβ
  desc_mul :
    ∀ {β : Type v} (oneβ : β) (mulβ : β → β → β) (f : α → β)
      (h1 : f 1 = oneβ)
      (hmul : ∀ a b : α, f (a * b) = mulβ (f a) (f b))
      (hS : ∀ s : α, S s → ∃ i : β, mulβ (f s) i = oneβ ∧ mulβ i (f s) = oneβ)
      (x y : L),
      desc oneβ mulβ f h1 hmul hS (mulL x y) =
        mulβ (desc oneβ mulβ f h1 hmul hS x) (desc oneβ mulβ f h1 hmul hS y)
  desc_of :
    ∀ {β : Type v} (oneβ : β) (mulβ : β → β → β) (f : α → β)
      (h1 : f 1 = oneβ)
      (hmul : ∀ a b : α, f (a * b) = mulβ (f a) (f b))
      (hS : ∀ s : α, S s → ∃ i : β, mulβ (f s) i = oneβ ∧ mulβ i (f s) = oneβ)
      (a : α),
      desc oneβ mulβ f h1 hmul hS (of a) = f a
  desc_unique :
    ∀ {β : Type v} (oneβ : β) (mulβ : β → β → β) (f : α → β)
      (h1 : f 1 = oneβ)
      (hmul : ∀ a b : α, f (a * b) = mulβ (f a) (f b))
      (hS : ∀ s : α, S s → ∃ i : β, mulβ (f s) i = oneβ ∧ mulβ i (f s) = oneβ)
      (g : L → β),
      g oneL = oneβ →
      (∀ x y : L, g (mulL x y) = mulβ (g x) (g y)) →
      (∀ a : α, g (of a) = f a) →
      g = desc oneβ mulβ f h1 hmul hS

def ofMap {α : Type u} [CommMonoidLike α] {S : α → Prop}
    (Loc : LocalizationLike α S) : α → Loc.L :=
  Loc.of

def lift {α : Type u} [CommMonoidLike α] {S : α → Prop}
    (Loc : LocalizationLike α S)
    {β : Type v}
    (oneβ : β)
    (mulβ : β → β → β)
    (f : α → β)
    (h1 : f 1 = oneβ)
    (hmul : ∀ a b : α, f (a * b) = mulβ (f a) (f b))
    (hS : ∀ s : α, S s → ∃ i : β, mulβ (f s) i = oneβ ∧ mulβ i (f s) = oneβ) :
    Loc.L → β :=
  Loc.desc oneβ mulβ f h1 hmul hS

def IsUnitLike {β : Type _} (oneβ : β) (mulβ : β → β → β) (x : β) : Prop :=
  ∃ y : β, mulβ x y = oneβ ∧ mulβ y x = oneβ

theorem of_mem_units {α : Type u} [CommMonoidLike α] {S : α → Prop}
    (Loc : LocalizationLike α S) (s : α) (hs : S s) :
    IsUnitLike Loc.oneL Loc.mulL ((ofMap Loc) s) := by
  rcases Loc.denom_unit s hs with ⟨u, hul, hur⟩
  refine ⟨u, ?_, ?_⟩
  · simpa [ofMap] using hul
  · simpa [ofMap] using hur

theorem lift_comp_of {α : Type u} [CommMonoidLike α] {S : α → Prop}
    (Loc : LocalizationLike α S)
    {β : Type v}
    (oneβ : β)
    (mulβ : β → β → β)
    (f : α → β)
    (h1 : f 1 = oneβ)
    (hmul : ∀ a b : α, f (a * b) = mulβ (f a) (f b))
    (hS : ∀ s : α, S s → ∃ i : β, mulβ (f s) i = oneβ ∧ mulβ i (f s) = oneβ) :
    ∀ a : α, lift Loc oneβ mulβ f h1 hmul hS ((ofMap Loc) a) = f a := by
  intro a
  have hdesc := Loc.desc_of oneβ mulβ f h1 hmul hS a
  simpa [lift, ofMap] using hdesc

theorem lift_unique {α : Type u} [CommMonoidLike α] {S : α → Prop}
    (Loc : LocalizationLike α S)
    {β : Type v}
    (oneβ : β)
    (mulβ : β → β → β)
    (f : α → β)
    (h1 : f 1 = oneβ)
    (hmul : ∀ a b : α, f (a * b) = mulβ (f a) (f b))
    (hS : ∀ s : α, S s → ∃ i : β, mulβ (f s) i = oneβ ∧ mulβ i (f s) = oneβ)
    (g : Loc.L → β)
    (hg_one : g Loc.oneL = oneβ)
    (hg_mul : ∀ x y : Loc.L, g (Loc.mulL x y) = mulβ (g x) (g y))
    (hg_of : ∀ a : α, g ((ofMap Loc) a) = f a) :
    g = lift Loc oneβ mulβ f h1 hmul hS := by
  have huniq := Loc.desc_unique oneβ mulβ f h1 hmul hS g hg_one hg_mul (by
    intro a
    simpa [ofMap] using hg_of a)
  simpa [lift] using huniq

theorem localization_induction {α : Type u} [CommMonoidLike α] {S : α → Prop}
    (Loc : LocalizationLike α S)
    (P : Loc.L → Prop)
    (h_one : P Loc.oneL)
    (h_mul : ∀ x y : Loc.L, P x → P y → P (Loc.mulL x y))
    (h_of : ∀ a : α, P ((ofMap Loc) a))
    (h_cancel : ∀ x : Loc.L, ∀ s : α, S s → P (Loc.mulL x ((ofMap Loc) s)) → P x)
    (x : Loc.L)
    (hx : ∃ a s : α, S s ∧ Loc.mulL x ((ofMap Loc) s) = (ofMap Loc) a) :
    P x := by
  rcases hx with ⟨a, s, hs, hrepr⟩
  have hPa : P ((ofMap Loc) a) := h_of a
  have hPone : P Loc.oneL := h_one
  have hPa_mul_one : P (Loc.mulL ((ofMap Loc) a) Loc.oneL) := h_mul ((ofMap Loc) a) Loc.oneL hPa hPone
  have hPa' : P ((ofMap Loc) a) := by
    simpa [Loc.mul_one] using hPa_mul_one
  have hPxs : P (Loc.mulL x ((ofMap Loc) s)) := by
    rw [hrepr]
    exact hPa'
  exact h_cancel x s hs hPxs

theorem eq_of_cross_multiply {α : Type u} [CommMonoidLike α] {S : α → Prop}
    (Loc : LocalizationLike α S)
    (x y : Loc.L) (s : α) (hs : S s)
    (hxy : Loc.mulL x ((ofMap Loc) s) = Loc.mulL y ((ofMap Loc) s)) :
    x = y := by
  rcases Loc.denom_unit s hs with ⟨u, hsu, hus⟩
  have hmul : Loc.mulL (Loc.mulL x ((ofMap Loc) s)) u =
      Loc.mulL (Loc.mulL y ((ofMap Loc) s)) u := by
    rw [hxy]
  have hx_assoc : Loc.mulL x (Loc.mulL ((ofMap Loc) s) u) =
      Loc.mulL y (Loc.mulL ((ofMap Loc) s) u) := by
    calc
      Loc.mulL x (Loc.mulL ((ofMap Loc) s) u)
          = Loc.mulL (Loc.mulL x ((ofMap Loc) s)) u := by
            symm
            exact Loc.mul_assoc x ((ofMap Loc) s) u
      _ = Loc.mulL (Loc.mulL y ((ofMap Loc) s)) u := hmul
      _ = Loc.mulL y (Loc.mulL ((ofMap Loc) s) u) := by
            exact Loc.mul_assoc y ((ofMap Loc) s) u
  have hreduce : Loc.mulL x Loc.oneL = Loc.mulL y Loc.oneL := by
    have hsu' : Loc.mulL ((ofMap Loc) s) u = Loc.oneL := by
      simpa [ofMap] using hsu
    rw [hsu'] at hx_assoc
    exact hx_assoc
  calc
    x = Loc.mulL x Loc.oneL := by
      symm
      exact Loc.mul_one x
    _ = Loc.mulL y Loc.oneL := hreduce
    _ = y := by
      exact Loc.mul_one y

theorem localization_universal {α : Type u} [CommMonoidLike α] {S : α → Prop}
    (Loc : LocalizationLike.{u, v} α S)
    {β : Type v}
    (oneβ : β)
    (mulβ : β → β → β)
    (f : α → β)
    (h1 : f 1 = oneβ)
    (hmul : ∀ a b : α, f (a * b) = mulβ (f a) (f b))
    (hS : ∀ s : α, S s → ∃ i : β, mulβ (f s) i = oneβ ∧ mulβ i (f s) = oneβ) :
    ∃ g : Loc.L → β,
      g Loc.oneL = oneβ ∧
      (∀ x y : Loc.L, g (Loc.mulL x y) = mulβ (g x) (g y)) ∧
      (∀ a : α, g ((ofMap Loc) a) = f a) ∧
      (∀ g' : Loc.L → β,
        g' Loc.oneL = oneβ →
        (∀ x y : Loc.L, g' (Loc.mulL x y) = mulβ (g' x) (g' y)) →
        (∀ a : α, g' ((ofMap Loc) a) = f a) →
        g' = g) := by
  refine ⟨Loc.desc (β := β) oneβ mulβ f h1 hmul hS, ?_⟩
  constructor
  · exact Loc.desc_one (β := β) oneβ mulβ f h1 hmul hS
  constructor
  · intro x y
    exact Loc.desc_mul (β := β) oneβ mulβ f h1 hmul hS x y
  constructor
  · intro a
    exact Loc.desc_of (β := β) oneβ mulβ f h1 hmul hS a
  · intro g' hg'_one hg'_mul hg'_of
    exact Loc.desc_unique (β := β) oneβ mulβ f h1 hmul hS g' hg'_one hg'_mul (by
      intro a
      simpa [ofMap] using hg'_of a)
