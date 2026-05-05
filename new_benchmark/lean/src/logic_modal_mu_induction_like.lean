/-
BENCHMARK_ID: TINY_MATHLIB_BATCH03_LOGIC_MODAL_MU_INDUCTION
PAIR_STEM: logic_modal_mu_induction_like
MATH_DOMAIN: Logic
SOURCE_MATHLIB: Mathlib/Order/FixedPoints
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class CompleteBooleanAlgebraLike (α : Type u) where
  le : α → α → Prop
  compl : α → α
  box : α → α
  diamond : α → α
  mu : (α → α) → α
  nu : (α → α) → α
  le_refl : ∀ a : α, le a a
  le_trans : ∀ {a b c : α}, le a b → le b c → le a c
  compl_antitone : ∀ {a b : α}, le a b → le (compl b) (compl a)
  compl_involutive : ∀ a : α, compl (compl a) = a
  box_mono : ∀ {a b : α}, le a b → le (box a) (box b)
  diamond_mono : ∀ {a b : α}, le a b → le (diamond a) (diamond b)
  modal_dual : ∀ a : α, le (diamond a) (compl (box (compl a)))
  mu_unfold_axiom :
    ∀ f : α → α,
      (∀ ⦃a b : α⦄, le a b → le (f a) (f b)) →
      le (f (mu f)) (mu f)
  mu_least_axiom :
    ∀ f : α → α,
      (∀ ⦃a b : α⦄, le a b → le (f a) (f b)) →
      ∀ x : α, le (f x) x → le (mu f) x
  nu_unfold_axiom :
    ∀ f : α → α,
      (∀ ⦃a b : α⦄, le a b → le (f a) (f b)) →
      le (nu f) (f (nu f))
  nu_greatest_axiom :
    ∀ f : α → α,
      (∀ ⦃a b : α⦄, le a b → le (f a) (f b)) →
      ∀ x : α, le x (f x) → le x (nu f)
  mu_nu_dual_axiom :
    ∀ f : α → α,
      (∀ ⦃a b : α⦄, le a b → le (f a) (f b)) →
      le (mu f) (compl (nu (fun x => compl (f (compl x)))))

infix:50 " ≤ " => CompleteBooleanAlgebraLike.le

def monotone {α : Type u} [CompleteBooleanAlgebraLike α] (f : α → α) : Prop :=
  ∀ ⦃a b : α⦄, a ≤ b → f a ≤ f b

def boxLike {α : Type u} [CompleteBooleanAlgebraLike α] (x : α) : α :=
  CompleteBooleanAlgebraLike.box x

def diamondLike {α : Type u} [CompleteBooleanAlgebraLike α] (x : α) : α :=
  CompleteBooleanAlgebraLike.diamond x

def muLike {α : Type u} [CompleteBooleanAlgebraLike α] (f : α → α) : α :=
  CompleteBooleanAlgebraLike.mu f

def nuLike {α : Type u} [CompleteBooleanAlgebraLike α] (f : α → α) : α :=
  CompleteBooleanAlgebraLike.nu f

theorem mu_unfold {α : Type u} [CompleteBooleanAlgebraLike α]
    (f : α → α) (hmono : monotone f) :
    f (muLike f) ≤ muLike f := by
  have hmono_explicit :
      ∀ ⦃a b : α⦄,
        CompleteBooleanAlgebraLike.le a b → CompleteBooleanAlgebraLike.le (f a) (f b) := by
    intro a b hab
    exact hmono hab
  have hcore : CompleteBooleanAlgebraLike.le (f (CompleteBooleanAlgebraLike.mu f))
      (CompleteBooleanAlgebraLike.mu f) :=
    CompleteBooleanAlgebraLike.mu_unfold_axiom (f := f) hmono_explicit
  exact hcore

theorem mu_induction {α : Type u} [CompleteBooleanAlgebraLike α]
    (f : α → α) (hmono : monotone f) {x : α} (hx : f x ≤ x) :
    muLike f ≤ x := by
  have hmono_explicit :
      ∀ ⦃a b : α⦄,
        CompleteBooleanAlgebraLike.le a b → CompleteBooleanAlgebraLike.le (f a) (f b) := by
    intro a b hab
    exact hmono hab
  have hleast : CompleteBooleanAlgebraLike.le (CompleteBooleanAlgebraLike.mu f) x :=
    CompleteBooleanAlgebraLike.mu_least_axiom (f := f) hmono_explicit x hx
  have hself : muLike f ≤ muLike f :=
    CompleteBooleanAlgebraLike.le_refl (muLike f)
  have hchain : muLike f ≤ x :=
    CompleteBooleanAlgebraLike.le_trans hself hleast
  exact hchain

theorem nu_unfold {α : Type u} [CompleteBooleanAlgebraLike α]
    (f : α → α) (hmono : monotone f) :
    nuLike f ≤ f (nuLike f) := by
  have hmono_explicit :
      ∀ ⦃a b : α⦄,
        CompleteBooleanAlgebraLike.le a b → CompleteBooleanAlgebraLike.le (f a) (f b) := by
    intro a b hab
    exact hmono hab
  have hcore : CompleteBooleanAlgebraLike.le (CompleteBooleanAlgebraLike.nu f)
      (f (CompleteBooleanAlgebraLike.nu f)) :=
    CompleteBooleanAlgebraLike.nu_unfold_axiom (f := f) hmono_explicit
  exact hcore

theorem nu_coinduction {α : Type u} [CompleteBooleanAlgebraLike α]
    (f : α → α) (hmono : monotone f) {x : α} (hx : x ≤ f x) :
    x ≤ nuLike f := by
  have hmono_explicit :
      ∀ ⦃a b : α⦄,
        CompleteBooleanAlgebraLike.le a b → CompleteBooleanAlgebraLike.le (f a) (f b) := by
    intro a b hab
    exact hmono hab
  have hgreatest : CompleteBooleanAlgebraLike.le x (CompleteBooleanAlgebraLike.nu f) :=
    CompleteBooleanAlgebraLike.nu_greatest_axiom (f := f) hmono_explicit x hx
  have hself : x ≤ x := CompleteBooleanAlgebraLike.le_refl x
  have hchain : x ≤ nuLike f := CompleteBooleanAlgebraLike.le_trans hself hgreatest
  exact hchain

theorem bekic_split_like {α : Type u} [CompleteBooleanAlgebraLike α]
    (f g : α → α)
    (hmono_f : monotone f) (hmono_g : monotone g)
    {x y : α} (hx : f x ≤ x) (hy : y ≤ g y) :
    muLike f ≤ x ∧ y ≤ nuLike g := by
  have hmu : muLike f ≤ x := mu_induction f hmono_f hx
  have hnu : y ≤ nuLike g := nu_coinduction g hmono_g hy
  have hleft : muLike f ≤ x := hmu
  have hright : y ≤ nuLike g := hnu
  exact And.intro hleft hright

theorem modal_mu_duality {α : Type u} [CompleteBooleanAlgebraLike α]
    (f : α → α) (hmono : monotone f) :
    muLike f ≤
      CompleteBooleanAlgebraLike.compl
        (nuLike (fun x => CompleteBooleanAlgebraLike.compl (f (CompleteBooleanAlgebraLike.compl x)))) := by
  let dualOp : α → α := fun x => CompleteBooleanAlgebraLike.compl (f (CompleteBooleanAlgebraLike.compl x))
  have hdual_mono : monotone dualOp := by
    intro a b hab
    have hcbca : CompleteBooleanAlgebraLike.compl b ≤ CompleteBooleanAlgebraLike.compl a :=
      CompleteBooleanAlgebraLike.compl_antitone hab
    have hfbfa : f (CompleteBooleanAlgebraLike.compl b) ≤ f (CompleteBooleanAlgebraLike.compl a) :=
      hmono hcbca
    exact CompleteBooleanAlgebraLike.compl_antitone hfbfa
  have hmono_explicit :
      ∀ ⦃a b : α⦄,
        CompleteBooleanAlgebraLike.le a b → CompleteBooleanAlgebraLike.le (f a) (f b) := by
    intro a b hab
    exact hmono hab
  have hcore :
      CompleteBooleanAlgebraLike.le (CompleteBooleanAlgebraLike.mu f)
        (CompleteBooleanAlgebraLike.compl
          (CompleteBooleanAlgebraLike.nu (fun x => CompleteBooleanAlgebraLike.compl (f (CompleteBooleanAlgebraLike.compl x))))) :=
    CompleteBooleanAlgebraLike.mu_nu_dual_axiom (f := f) hmono_explicit
  have _ : monotone dualOp := hdual_mono
  exact hcore
