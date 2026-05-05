/-
BENCHMARK_ID: TINY_MATHLIB_BATCH04_COMBINATORICS_POLYNOMIAL_METHOD_INCIDENCE_LIKE
PAIR_STEM: combinatorics_polynomial_method_incidence_like
MATH_DOMAIN: Combinatorics / Algebra
SOURCE_MATHLIB: Mathlib/Combinatorics
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u v

class FiniteFieldLike (F : Type u) where
  zero : F
  one : F
  add : F → F → F
  mul : F → F → F
  card : Nat
  card_pos : 0 < card
  add_assoc : ∀ x y z : F, add (add x y) z = add x (add y z)
  add_zero : ∀ x : F, add x zero = x
  zero_add : ∀ x : F, add zero x = x
  mul_one : ∀ x : F, mul x one = x
  one_mul : ∀ x : F, mul one x = x

infixl:65 " +ₓ " => FiniteFieldLike.add
infixl:70 " *ₓ " => FiniteFieldLike.mul

def PolynomialLike {F : Type u} [FiniteFieldLike F] (P : Type v) : Prop :=
  ∃ eval : P → F → F, ∃ deg : P → Nat, True

def VanishingSetLike {F : Type u} [FiniteFieldLike F] {P : Type v}
    (eval : P → F → F) (p : P) (X : F → Prop) : Prop :=
  ∀ x : F, X x → eval p x = FiniteFieldLike.zero

def DegreeLike {F : Type u} [FiniteFieldLike F] {P : Type v}
    (deg : P → Nat) (p : P) (d : Nat) : Prop :=
  deg p = d

def MultiplicityLike {F : Type u} [FiniteFieldLike F] {P : Type v}
    (eval : P → F → F) (p : P) (x : F) (m : Nat) : Prop :=
  ∀ k : Nat, k < m → eval p x = FiniteFieldLike.zero

def IncidenceSetLike {F : Type u} [FiniteFieldLike F] (Pts Lines : Type v)
    (inc : Pts → Lines → Nat) : Prop :=
  ∀ p : Pts, ∀ l : Lines, inc p l ≤ 1

theorem interpolation_bound_like {F : Type u} [FiniteFieldLike F] {P : Type v}
    (hpoly : PolynomialLike (F := F) P)
    (deg : P → Nat) (p : P) (d n : Nat)
    (hdeg : DegreeLike (F := F) deg p d)
    (hpoints : n ≤ d + 1)
    (hdegree_card : d + 1 ≤ FiniteFieldLike.card (F := F))
    (htrans : ∀ a b c : Nat, a ≤ b → b ≤ c → a ≤ c) :
    n ≤ FiniteFieldLike.card (F := F) := by
  rcases hpoly with ⟨eval0, deg0, htriv⟩
  have hstep1 : n ≤ d + 1 := hpoints
  have hstep2 : d + 1 ≤ FiniteFieldLike.card (F := F) := hdegree_card
  have hbound : n ≤ FiniteFieldLike.card (F := F) := htrans _ _ _ hstep1 hstep2
  have _ : deg p = d := hdeg
  have _ : True := htriv
  have _ : P → Nat := deg0
  have _ : P → F → F := eval0
  exact hbound

theorem vanishing_multiplicity_step {F : Type u} [FiniteFieldLike F] {P : Type v}
    (eval : P → F → F) (p : P) (x : F) (m n : Nat)
    (hmult : MultiplicityLike eval p x m)
    (hstep : ∀ k : Nat, k < n → k < m) :
    MultiplicityLike eval p x n := by
  intro k hk
  have hkm : k < m := hstep k hk
  have hz : eval p x = FiniteFieldLike.zero := hmult k hkm
  exact hz

theorem polynomial_partition_step {F : Type u} [FiniteFieldLike F] {Pts Lines : Type v}
    (inc : Pts → Lines → Nat) (cut : Pts → Nat) (cross : Lines → Nat)
    (c d : Nat)
    (hpartition : ∀ p : Pts, ∀ l : Lines, inc p l ≤ cut p + cross l)
    (hcompress : ∀ p : Pts, ∀ l : Lines, cut p + cross l ≤ c + d)
    (htrans : ∀ a b e : Nat, a ≤ b → b ≤ e → a ≤ e) :
    ∀ p : Pts, ∀ l : Lines, inc p l ≤ c + d := by
  intro p l
  have hlocal : inc p l ≤ cut p + cross l := hpartition p l
  have hfold : cut p + cross l ≤ c + d := hcompress p l
  have hfinal : inc p l ≤ c + d := htrans _ _ _ hlocal hfold
  exact hfinal

theorem incidence_bound_core {F : Type u} [FiniteFieldLike F]
    (pts lines I d e : Nat)
    (hinc : I ≤ pts * lines)
    (hgeom : pts * lines ≤ d * e)
    (htrans : ∀ a b c : Nat, a ≤ b → b ≤ c → a ≤ c) :
    I ≤ d * e := by
  have hstep : I ≤ pts * lines := hinc
  have hmul : pts * lines ≤ d * e := hgeom
  have hfinal : I ≤ d * e := htrans _ _ _ hstep hmul
  exact hfinal

theorem sum_product_interface_like {F : Type u} [FiniteFieldLike F]
    (a b c d e : Nat)
    (hsplit : a ≤ b + c)
    (htransfer : b + c ≤ d + e)
    (hbudget : d + e ≤ FiniteFieldLike.card (F := F))
    (htrans : ∀ x y z : Nat, x ≤ y → y ≤ z → x ≤ z) :
    a ≤ FiniteFieldLike.card (F := F) := by
  have h₁ : a ≤ b + c := hsplit
  have h₂ : b + c ≤ d + e := htransfer
  have h₃ : d + e ≤ FiniteFieldLike.card (F := F) := hbudget
  have h₄ : a ≤ d + e := htrans _ _ _ h₁ h₂
  have h₅ : a ≤ FiniteFieldLike.card (F := F) := htrans _ _ _ h₄ h₃
  exact h₅

theorem polynomial_method_incidence_theorem_like {F : Type u} [FiniteFieldLike F]
    {Pts Lines : Type v}
    (inc : Pts → Lines → Nat) (cut : Pts → Nat) (cross : Lines → Nat)
    (pts lines I c d : Nat)
    (hpartition : ∀ p : Pts, ∀ l : Lines, inc p l ≤ cut p + cross l)
    (hcompress : ∀ p : Pts, ∀ l : Lines, cut p + cross l ≤ c + d)
    (hcount : I ≤ pts * lines)
    (hgeom : pts * lines ≤ c * d)
    (hlift : c * d ≤ c * d + (c + d))
    (hbudget : c * d + (c + d) ≤ FiniteFieldLike.card (F := F))
    (htrans : ∀ x y z : Nat, x ≤ y → y ≤ z → x ≤ z) :
    I ≤ FiniteFieldLike.card (F := F) := by
  have hlocal : ∀ p : Pts, ∀ l : Lines, inc p l ≤ c + d :=
    polynomial_partition_step (F := F) inc cut cross c d hpartition hcompress htrans
  have hcore : I ≤ c * d := incidence_bound_core (F := F) pts lines I c d hcount hgeom htrans
  have hstage : I ≤ c * d + (c + d) := htrans _ _ _ hcore hlift
  have hfinal : I ≤ FiniteFieldLike.card (F := F) := htrans _ _ _ hstage hbudget
  have _ : ∀ p : Pts, ∀ l : Lines, inc p l ≤ c + d := hlocal
  exact hfinal
