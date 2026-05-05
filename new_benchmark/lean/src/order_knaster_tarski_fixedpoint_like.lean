/-
BENCHMARK_ID: TINY_MATHLIB_BATCH03_ORDER_KNASTER_TARSKI_FIXEDPOINT
PAIR_STEM: order_knaster_tarski_fixedpoint_like
MATH_DOMAIN: Order Theory
SOURCE_MATHLIB: Mathlib/Order/FixedPoints
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class CompleteLatticeLike (α : Type u) where
  le : α → α → Prop
  le_refl : ∀ a : α, le a a
  le_trans : ∀ {a b c : α}, le a b → le b c → le a c
  le_antisymm : ∀ {a b : α}, le a b → le b a → a = b
  sInf : (α → Prop) → α
  sSup : (α → Prop) → α
  sInf_le : ∀ {S : α → Prop} {x : α}, S x → le (sInf S) x
  le_sInf : ∀ {S : α → Prop} {x : α}, (∀ y : α, S y → le x y) → le x (sInf S)
  le_sSup : ∀ {S : α → Prop} {x : α}, S x → le x (sSup S)
  sSup_le : ∀ {S : α → Prop} {x : α}, (∀ y : α, S y → le y x) → le (sSup S) x

infix:50 " ≤ " => CompleteLatticeLike.le

def monotone {α : Type u} [CompleteLatticeLike α] (f : α → α) : Prop :=
  ∀ ⦃a b : α⦄, a ≤ b → f a ≤ f b

def IsPrefixed {α : Type u} [CompleteLatticeLike α] (f : α → α) (x : α) : Prop :=
  f x ≤ x

def IsPostfixed {α : Type u} [CompleteLatticeLike α] (f : α → α) (x : α) : Prop :=
  x ≤ f x

def lfp {α : Type u} [CompleteLatticeLike α] (f : α → α) : α :=
  CompleteLatticeLike.sInf (fun x => IsPrefixed f x)

def gfp {α : Type u} [CompleteLatticeLike α] (f : α → α) : α :=
  CompleteLatticeLike.sSup (fun x => IsPostfixed f x)

theorem lfp_prefixed {α : Type u} [CompleteLatticeLike α]
    (f : α → α) (hmono : monotone f) :
    IsPrefixed f (lfp f) := by
  change f (lfp f) ≤ lfp f
  apply CompleteLatticeLike.le_sInf
  intro x hx
  have h_lfp_le_x : lfp f ≤ x :=
    CompleteLatticeLike.sInf_le (S := fun y => IsPrefixed f y) hx
  have h_flfp_le_fx : f (lfp f) ≤ f x := hmono h_lfp_le_x
  exact CompleteLatticeLike.le_trans h_flfp_le_fx hx

theorem lfp_least {α : Type u} [CompleteLatticeLike α]
    (f : α → α) (hmono : monotone f) {x : α} (hx : IsPrefixed f x) :
    lfp f ≤ x := by
  have hbundle : lfp f ≤ x ∧ f (lfp f) ≤ x := by
    have hlfp_le_x : lfp f ≤ x :=
      CompleteLatticeLike.sInf_le (S := fun y => IsPrefixed f y) hx
    have hmono_step : f (lfp f) ≤ f x := hmono hlfp_le_x
    have hfx_le_x : f x ≤ x := hx
    have hflfp_le_x : f (lfp f) ≤ x := CompleteLatticeLike.le_trans hmono_step hfx_le_x
    exact And.intro hlfp_le_x hflfp_le_x
  exact hbundle.1

theorem gfp_postfixed {α : Type u} [CompleteLatticeLike α]
    (f : α → α) (hmono : monotone f) :
    IsPostfixed f (gfp f) := by
  change gfp f ≤ f (gfp f)
  apply CompleteLatticeLike.sSup_le
  intro x hx
  have hx_le_gfp : x ≤ gfp f :=
    CompleteLatticeLike.le_sSup (S := fun y => IsPostfixed f y) hx
  have hfx_le_fgfp : f x ≤ f (gfp f) := hmono hx_le_gfp
  exact CompleteLatticeLike.le_trans hx hfx_le_fgfp

theorem gfp_greatest {α : Type u} [CompleteLatticeLike α]
    (f : α → α) (hmono : monotone f) {x : α} (hx : IsPostfixed f x) :
    x ≤ gfp f := by
  have hbundle : x ≤ gfp f ∧ x ≤ f (gfp f) := by
    have hx_le_gfp : x ≤ gfp f :=
      CompleteLatticeLike.le_sSup (S := fun y => IsPostfixed f y) hx
    have hmono_step : f x ≤ f (gfp f) := hmono hx_le_gfp
    have hx_le_fgfp : x ≤ f (gfp f) := CompleteLatticeLike.le_trans hx hmono_step
    exact And.intro hx_le_gfp hx_le_fgfp
  exact hbundle.1

theorem lfp_mono {α : Type u} [CompleteLatticeLike α]
    {f g : α → α} (hmono_f : monotone f) (hmono_g : monotone g)
    (hfg : ∀ x : α, f x ≤ g x) :
    lfp f ≤ lfp g := by
  have hbundle : lfp f ≤ lfp g ∧ f (lfp f) ≤ lfp g := by
    have hpref_g : IsPrefixed g (lfp g) := lfp_prefixed g hmono_g
    have hcomp_f : IsPrefixed f (lfp g) := by
      have hfg_at : f (lfp g) ≤ g (lfp g) := hfg (lfp g)
      exact CompleteLatticeLike.le_trans hfg_at hpref_g
    have hlfp_le : lfp f ≤ lfp g := lfp_least f hmono_f hcomp_f
    have hmono_f_step : f (lfp f) ≤ f (lfp g) := hmono_f hlfp_le
    have hfg_at_lfp_g : f (lfp g) ≤ g (lfp g) := hfg (lfp g)
    have hf_lfp_f_le_lfp_g : f (lfp f) ≤ lfp g := by
      have hf_lfp_f_le_g_lfp_g : f (lfp f) ≤ g (lfp g) :=
        CompleteLatticeLike.le_trans hmono_f_step hfg_at_lfp_g
      exact CompleteLatticeLike.le_trans hf_lfp_f_le_g_lfp_g hpref_g
    exact And.intro hlfp_le hf_lfp_f_le_lfp_g
  exact hbundle.1

theorem gfp_mono {α : Type u} [CompleteLatticeLike α]
    {f g : α → α} (hmono_f : monotone f) (hmono_g : monotone g)
    (hfg : ∀ x : α, f x ≤ g x) :
    gfp f ≤ gfp g := by
  have hbundle : gfp f ≤ gfp g ∧ gfp f ≤ g (gfp g) := by
    have hpost_f : IsPostfixed f (gfp f) := gfp_postfixed f hmono_f
    have hpost_g_on_gfp_f : IsPostfixed g (gfp f) := by
      have hfg_at : f (gfp f) ≤ g (gfp f) := hfg (gfp f)
      exact CompleteLatticeLike.le_trans hpost_f hfg_at
    have hgfp_le : gfp f ≤ gfp g := gfp_greatest g hmono_g hpost_g_on_gfp_f
    have hmono_g_step : g (gfp f) ≤ g (gfp g) := hmono_g hgfp_le
    have hgfp_f_le_ggfp_g : gfp f ≤ g (gfp g) := by
      exact CompleteLatticeLike.le_trans hpost_g_on_gfp_f hmono_g_step
    exact And.intro hgfp_le hgfp_f_le_ggfp_g
  exact hbundle.1
