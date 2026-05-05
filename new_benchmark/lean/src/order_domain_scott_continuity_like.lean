/-
BENCHMARK_ID: TINY_MATHLIB_BATCH04_ORDER_DOMAIN_SCOTT_CONTINUITY
PAIR_STEM: order_domain_scott_continuity_like
MATH_DOMAIN: Domain Theory
SOURCE_MATHLIB: Mathlib/Order/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class DcpoLike (α : Type u) where
  le : α → α → Prop
  le_refl : ∀ x : α, le x x
  le_trans : ∀ {x y z : α}, le x y → le y z → le x z
  sup : (α → Prop) → α
  le_sup : ∀ {S : α → Prop} {x : α}, S x → le x (sup S)
  sup_le : ∀ {S : α → Prop} {x : α}, (∀ y : α, S y → le y x) → le (sup S) x

infix:50 " <<= " => DcpoLike.le

def DirectedLike {α : Type u} [DcpoLike α] (S : α → Prop) : Prop :=
  ∀ x y : α, S x → S y → ∃ z : α, S z ∧ x <<= z ∧ y <<= z

def SupLike {α : Type u} [DcpoLike α] (S : α → Prop) (x : α) : Prop :=
  x = DcpoLike.sup S

def ScottContinuousLike {α : Type u} [DcpoLike α] (f : α → α) : Prop :=
  (∀ {x y : α}, x <<= y → f x <<= f y) ∧
  (∀ S : α → Prop,
    DirectedLike S →
    f (DcpoLike.sup S) <<= DcpoLike.sup (fun y : α => ∃ x : α, S x ∧ y = f x))

def WayBelowLike {α : Type u} [DcpoLike α] (x y : α) : Prop :=
  ∀ S : α → Prop,
    DirectedLike S →
    y <<= DcpoLike.sup S →
    ∃ z : α, S z ∧ x <<= z

def AlgebraicLike {α : Type u} [DcpoLike α] : Prop :=
  ∀ y : α,
    ∃ S : α → Prop,
      DirectedLike S ∧
      y <<= DcpoLike.sup S ∧
      (∀ z : α, S z → WayBelowLike z y)

theorem scott_mono {α : Type u} [DcpoLike α]
    (f : α → α) (hsc : ScottContinuousLike f) :
    ∀ {x y : α}, x <<= y → f x <<= f y := by
  intro x y hxy
  have hmono : ∀ {a b : α}, a <<= b → f a <<= f b := hsc.1
  have hstep : f x <<= f y := hmono hxy
  exact hstep

theorem scott_preserves_sup {α : Type u} [DcpoLike α]
    (f : α → α) (hsc : ScottContinuousLike f)
    (S : α → Prop) (hdir : DirectedLike S) :
    f (DcpoLike.sup S) <<= DcpoLike.sup (fun y : α => ∃ x : α, S x ∧ y = f x) := by
  have hsup :
      ∀ T : α → Prop,
        DirectedLike T →
        f (DcpoLike.sup T) <<= DcpoLike.sup (fun y : α => ∃ x : α, T x ∧ y = f x) :=
    hsc.2
  have hresult := hsup S hdir
  exact hresult

theorem waybelow_interpolation {α : Type u} [DcpoLike α]
    {x y : α} (hxy : WayBelowLike x y)
    (S : α → Prop) (hdir : DirectedLike S)
    (hyS : y <<= DcpoLike.sup S) :
    ∃ z : α, S z ∧ x <<= z := by
  have hwitness := hxy S hdir hyS
  rcases hwitness with ⟨z, hzS, hxz⟩
  have hpack : S z ∧ x <<= z := And.intro hzS hxz
  exact ⟨z, hpack.1, hpack.2⟩

theorem compact_basis_expand {α : Type u} [DcpoLike α]
    (hAlg : AlgebraicLike (α := α)) (y : α) :
    ∃ S : α → Prop,
      DirectedLike S ∧
      y <<= DcpoLike.sup S ∧
      (∀ z : α, S z → WayBelowLike z y) := by
  have hy := hAlg y
  rcases hy with ⟨S, hdir, hsup, hwb⟩
  have hbundle : DirectedLike S ∧ y <<= DcpoLike.sup S := And.intro hdir hsup
  refine ⟨S, ?_⟩
  exact And.intro hbundle.1 (And.intro hbundle.2 hwb)

theorem fixedpoint_chain_limit {α : Type u} [DcpoLike α]
    (f : α → α) (hsc : ScottContinuousLike f)
    (S : α → Prop) (hdir : DirectedLike S)
    (hstep : ∀ x : α, S x → x <<= f x) :
    DcpoLike.sup S <<= f (DcpoLike.sup S) := by
  have _ : DirectedLike S := hdir
  have hmono : ∀ {x y : α}, x <<= y → f x <<= f y := scott_mono (f := f) hsc
  have hupper : ∀ x : α, S x → x <<= f (DcpoLike.sup S) := by
    intro x hx
    have hxfx : x <<= f x := hstep x hx
    have hxsup : x <<= DcpoLike.sup S := DcpoLike.le_sup hx
    have hfxsup : f x <<= f (DcpoLike.sup S) := hmono hxsup
    exact DcpoLike.le_trans hxfx hfxsup
  have hsup := DcpoLike.sup_le (S := S) (x := f (DcpoLike.sup S)) hupper
  exact hsup

theorem least_fixedpoint_scott {α : Type u} [DcpoLike α]
    (f : α → α) (hsc : ScottContinuousLike f)
    (S : α → Prop) (hdir : DirectedLike S)
    (hpref : ∀ x : α, S x → f x <<= x) :
    f (DcpoLike.sup S) <<= DcpoLike.sup S := by
  have hpres :
      f (DcpoLike.sup S) <<= DcpoLike.sup (fun y : α => ∃ x : α, S x ∧ y = f x) :=
    scott_preserves_sup (f := f) hsc S hdir
  have himage_le :
      DcpoLike.sup (fun y : α => ∃ x : α, S x ∧ y = f x) <<= DcpoLike.sup S := by
    apply DcpoLike.sup_le
    intro y hy
    rcases hy with ⟨x, hxS, hyEq⟩
    have hfx_le_x : f x <<= x := hpref x hxS
    have hx_le_sup : x <<= DcpoLike.sup S := DcpoLike.le_sup hxS
    have hfx_le_sup : f x <<= DcpoLike.sup S := DcpoLike.le_trans hfx_le_x hx_le_sup
    simpa [hyEq] using hfx_le_sup
  exact DcpoLike.le_trans hpres himage_le
