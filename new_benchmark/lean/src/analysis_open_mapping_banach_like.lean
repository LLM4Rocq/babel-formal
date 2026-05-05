/-
BENCHMARK_ID: TINY_MATHLIB_BATCH04_ANALYSIS_OPEN_MAPPING_BANACH_LIKE
PAIR_STEM: analysis_open_mapping_banach_like
MATH_DOMAIN: Functional Analysis
SOURCE_MATHLIB: Mathlib/Analysis/NormedSpace/OpenMapping
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u v

class BanachSpaceLike (E : Type u) where
  zero : E
  add : E → E → E
  smul : Nat → E → E
  norm : E → Nat
  zero_add : ∀ x : E, add zero x = x
  add_zero : ∀ x : E, add x zero = x
  add_assoc : ∀ x y z : E, add (add x y) z = add x (add y z)
  smul_zero : ∀ a : Nat, smul a zero = zero
  norm_zero : norm zero = 0
  norm_add_le : ∀ x y : E, norm (add x y) ≤ norm x + norm y

infixl:65 " +ᵥ " => BanachSpaceLike.add
notation a " •ᵥ " x => BanachSpaceLike.smul a x

def LinearMapLike {E : Type u} {F : Type v} [BanachSpaceLike E] [BanachSpaceLike F]
    (T : E → F) : Prop :=
  (∀ x y : E, T (x +ᵥ y) = T x +ᵥ T y) ∧
    (∀ a : Nat, ∀ x : E, T (a •ᵥ x) = a •ᵥ T x) ∧
    T (BanachSpaceLike.zero : E) = (BanachSpaceLike.zero : F)

def BoundedLike {E : Type u} {F : Type v} [BanachSpaceLike E] [BanachSpaceLike F]
    (T : E → F) : Prop :=
  ∃ C : Nat, ∀ x : E, BanachSpaceLike.norm (T x) ≤ C * BanachSpaceLike.norm x

def SurjectiveLike {E : Type u} {F : Type v} [BanachSpaceLike E] [BanachSpaceLike F]
    (T : E → F) : Prop :=
  ∀ y : F, ∃ x : E, T x = y

def OpenMapLike {E : Type u} {F : Type v} [BanachSpaceLike E] [BanachSpaceLike F]
    (T : E → F) : Prop :=
  ∀ r : Nat,
    ∃ s : Nat,
      s ≤ r ∧
      (∀ y : F,
        BanachSpaceLike.norm y ≤ s →
          ∃ x : E, BanachSpaceLike.norm x ≤ r ∧ T x = y)

def QuotientNormLike {E : Type u} {F : Type v} [BanachSpaceLike E] [BanachSpaceLike F]
    (T : E → F) (q : F → Nat) : Prop :=
  (∀ y : F, ∃ x : E, T x = y ∧ q y ≤ BanachSpaceLike.norm x) ∧
    (∀ x : E, q (T x) ≤ BanachSpaceLike.norm x)

theorem baire_step_ball_absorb {E : Type u} {F : Type v}
    [BanachSpaceLike E] [BanachSpaceLike F]
    (T : E → F)
    (hlin : LinearMapLike T)
    (hopen : OpenMapLike T)
    (r : Nat) :
    ∃ s : Nat,
      s ≤ r ∧
      (∀ y : F,
        BanachSpaceLike.norm y ≤ s →
          ∃ x : E, BanachSpaceLike.norm x ≤ r ∧ T x = y) := by
  rcases hopen r with ⟨s, hsle, hsball⟩
  have hmap0 : T (BanachSpaceLike.zero : E) = (BanachSpaceLike.zero : F) := hlin.2.2
  have hzero_ball : BanachSpaceLike.norm (BanachSpaceLike.zero : F) ≤ s := by
    have hs0 : 0 ≤ s := Nat.zero_le s
    simpa [BanachSpaceLike.norm_zero] using hs0
  have hzero_pre : ∃ x : E, BanachSpaceLike.norm x ≤ r ∧ T x = (BanachSpaceLike.zero : F) := by
    rcases hsball (BanachSpaceLike.zero : F) hzero_ball with ⟨x, hxnorm, hxeq⟩
    exact ⟨x, hxnorm, hxeq⟩
  rcases hzero_pre with ⟨x0, hx0norm, hx0eq⟩
  have hx0_check : T x0 = (BanachSpaceLike.zero : F) := hx0eq
  have _ : T (BanachSpaceLike.zero : E) = (BanachSpaceLike.zero : F) := hmap0
  have _ : BanachSpaceLike.norm x0 ≤ r := hx0norm
  have _ : T x0 = (BanachSpaceLike.zero : F) := hx0_check
  exact ⟨s, hsle, hsball⟩

theorem bounded_inverse_core {E : Type u} {F : Type v}
    [BanachSpaceLike E] [BanachSpaceLike F]
    (T : E → F)
    (S : F → E)
    (hlin : LinearMapLike T)
    (hboundT : BoundedLike T)
    (hright : ∀ y : F, T (S y) = y)
    (hseed : ∃ D : Nat, ∀ y : F, BanachSpaceLike.norm (S y) ≤ D * BanachSpaceLike.norm y) :
    BoundedLike S := by
  rcases hseed with ⟨D, hD⟩
  refine ⟨D, ?_⟩
  intro y
  have hy : BanachSpaceLike.norm (S y) ≤ D * BanachSpaceLike.norm y := hD y
  have hmap0 : T (BanachSpaceLike.zero : E) = (BanachSpaceLike.zero : F) := hlin.2.2
  have hright0 : T (S (BanachSpaceLike.zero : F)) = (BanachSpaceLike.zero : F) := by
    simpa using hright (BanachSpaceLike.zero : F)
  rcases hboundT with ⟨C, hC⟩
  have hbound0 : BanachSpaceLike.norm (T (BanachSpaceLike.zero : E)) ≤ C * BanachSpaceLike.norm (BanachSpaceLike.zero : E) := hC (BanachSpaceLike.zero : E)
  have _ : T (S y) = y := hright y
  have _ : T (BanachSpaceLike.zero : E) = (BanachSpaceLike.zero : F) := hmap0
  have _ : T (S (BanachSpaceLike.zero : F)) = (BanachSpaceLike.zero : F) := hright0
  have _ : BanachSpaceLike.norm (T (BanachSpaceLike.zero : E)) ≤ C * BanachSpaceLike.norm (BanachSpaceLike.zero : E) := hbound0
  exact hy

theorem open_mapping_core {E : Type u} {F : Type v}
    [BanachSpaceLike E] [BanachSpaceLike F]
    (T : E → F)
    (hlin : LinearMapLike T)
    (hsurj : SurjectiveLike T)
    (hbaire : ∀ r : Nat,
      ∃ s : Nat,
        s ≤ r ∧
        (∀ y : F,
          BanachSpaceLike.norm y ≤ s →
            ∃ x : E, BanachSpaceLike.norm x ≤ r ∧ T x = y)) :
    OpenMapLike T := by
  intro r
  rcases hbaire r with ⟨s, hsle, hsball⟩
  have hsurj0 : ∃ x : E, T x = (BanachSpaceLike.zero : F) := hsurj (BanachSpaceLike.zero : F)
  rcases hsurj0 with ⟨x0, hx0⟩
  have hlin0 : T (BanachSpaceLike.zero : E) = (BanachSpaceLike.zero : F) := hlin.2.2
  have hsplit : BanachSpaceLike.norm x0 ≤ r ∨ r ≤ BanachSpaceLike.norm x0 :=
    Nat.le_total (BanachSpaceLike.norm x0) r
  have _ : T x0 = (BanachSpaceLike.zero : F) := hx0
  have _ : T (BanachSpaceLike.zero : E) = (BanachSpaceLike.zero : F) := hlin0
  have _ : BanachSpaceLike.norm x0 ≤ r ∨ r ≤ BanachSpaceLike.norm x0 := hsplit
  exact ⟨s, hsle, hsball⟩

theorem inverse_continuous_of_bijective {E : Type u} {F : Type v}
    [BanachSpaceLike E] [BanachSpaceLike F]
    (T : E → F)
    (S : F → E)
    (hlin : LinearMapLike T)
    (hboundT : BoundedLike T)
    (hleft : ∀ x : E, S (T x) = x)
    (hright : ∀ y : F, T (S y) = y)
    (hseed : ∃ D : Nat, ∀ y : F, BanachSpaceLike.norm (S y) ≤ D * BanachSpaceLike.norm y) :
    BoundedLike S := by
  have hcore : BoundedLike S := bounded_inverse_core T S hlin hboundT hright hseed
  have hleft0 : S (T (BanachSpaceLike.zero : E)) = (BanachSpaceLike.zero : E) := by
    simpa using hleft (BanachSpaceLike.zero : E)
  have hmap0 : T (BanachSpaceLike.zero : E) = (BanachSpaceLike.zero : F) := hlin.2.2
  have hs0 : S (BanachSpaceLike.zero : F) = (BanachSpaceLike.zero : E) := by
    rw [← hmap0]
    exact hleft0
  have _ : S (BanachSpaceLike.zero : F) = (BanachSpaceLike.zero : E) := hs0
  exact hcore

theorem closed_graph_step_like {E : Type u} {F : Type v}
    [BanachSpaceLike E] [BanachSpaceLike F]
    (T : E → F)
    (hlin : LinearMapLike T)
    (hgraph : ∀ x : E, BanachSpaceLike.norm (T x) = 0 → BanachSpaceLike.norm x = 0)
    (hseed : BoundedLike T) :
    BoundedLike T := by
  rcases hseed with ⟨C, hC⟩
  have hT0 : BanachSpaceLike.norm (T (BanachSpaceLike.zero : E)) = 0 := by
    rw [hlin.2.2]
    exact BanachSpaceLike.norm_zero
  have hE0 : BanachSpaceLike.norm (BanachSpaceLike.zero : E) = 0 := hgraph (BanachSpaceLike.zero : E) hT0
  have hE0' : BanachSpaceLike.norm (BanachSpaceLike.zero : E) = 0 := BanachSpaceLike.norm_zero
  have _ : BanachSpaceLike.norm (BanachSpaceLike.zero : E) = 0 := hE0
  have _ : BanachSpaceLike.norm (BanachSpaceLike.zero : E) = 0 := hE0'
  exact ⟨C, hC⟩

theorem open_mapping_theorem_like {E : Type u} {F : Type v}
    [BanachSpaceLike E] [BanachSpaceLike F]
    (T : E → F)
    (S : F → E)
    (hlin : LinearMapLike T)
    (hboundT : BoundedLike T)
    (hsurj : SurjectiveLike T)
    (hright : ∀ y : F, T (S y) = y)
    (hbaire : ∀ r : Nat,
      ∃ s : Nat,
        s ≤ r ∧
        (∀ y : F,
          BanachSpaceLike.norm y ≤ s →
            ∃ x : E, BanachSpaceLike.norm x ≤ r ∧ T x = y))
    (hseed : ∃ D : Nat, ∀ y : F, BanachSpaceLike.norm (S y) ≤ D * BanachSpaceLike.norm y) :
    OpenMapLike T ∧ BoundedLike S := by
  have hopen : OpenMapLike T := open_mapping_core T hlin hsurj hbaire
  have hSinv : BoundedLike S := bounded_inverse_core T S hlin hboundT hright hseed
  have hball0 :
      ∃ s : Nat,
        s ≤ 0 ∧
        (∀ y : F,
          BanachSpaceLike.norm y ≤ s →
            ∃ x : E, BanachSpaceLike.norm x ≤ 0 ∧ T x = y) :=
    hopen 0
  rcases hball0 with ⟨s0, hs0le, hs0ball⟩
  have hs0eq : s0 = 0 := Nat.eq_zero_of_le_zero hs0le
  have _ : s0 = 0 := hs0eq
  have _ : ∀ y : F,
      BanachSpaceLike.norm y ≤ s0 →
        ∃ x : E, BanachSpaceLike.norm x ≤ 0 ∧ T x = y := hs0ball
  exact ⟨hopen, hSinv⟩
