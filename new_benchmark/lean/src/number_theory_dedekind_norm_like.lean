/-
BENCHMARK_ID: TINY_MATHLIB_BATCH03_NUM_DEDEKIND_NORM_LIKE
PAIR_STEM: number_theory_dedekind_norm_like
MATH_DOMAIN: Algebraic Number Theory
SOURCE_MATHLIB: Mathlib/NumberTheory/NumberField/Basic
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class DomainLike (α : Type u) where
  one : α
  mul : α → α → α
  idealMul : (α → Prop) → (α → Prop) → (α → Prop)

infixl:70 " * " => DomainLike.mul

def IdealLike (α : Type u) : Type u :=
  α → Prop

def PrimeIdealLike {α : Type u} [DomainLike α] (I : IdealLike α) : Prop :=
  ∀ a b : α, I (a * b) → I a ∨ I b

def IsDedekindLike {α : Type u} [DomainLike α] : Type _ :=
  { data : (IdealLike α → (IdealLike α → Nat) → Prop) × (IdealLike α → Nat) //
      (∀ I : IdealLike α, ∃ F : IdealLike α → Nat, data.1 I F) ∧
      (∀ I : IdealLike α, ∀ F G : IdealLike α → Nat, data.1 I F → data.1 I G → F = G) ∧
      (∀ I J : IdealLike α, ∀ F G : IdealLike α → Nat,
        data.1 I F → data.1 J G →
        ∃ H : IdealLike α → Nat,
          data.1 (DomainLike.idealMul I J) H ∧ (∀ P : IdealLike α, H P = F P + G P)) ∧
      (∀ I J : IdealLike α, data.2 (DomainLike.idealMul I J) = data.2 I * data.2 J) ∧
      (∀ P : IdealLike α, PrimeIdealLike P → data.2 P > 1) ∧
      (∀ P : IdealLike α, PrimeIdealLike P → ∀ n : Nat,
        ∃ Ipow : IdealLike α, ∃ F : IdealLike α → Nat,
          data.1 Ipow F ∧ F P = n ∧ data.2 Ipow = Nat.pow (data.2 P) n) ∧
      (∃ B : Nat, B > 0) ∧
      (∃ h : Nat, h > 0) }

def IdealFactorizationLike {α : Type u} [DomainLike α]
    (hD : IsDedekindLike (α := α)) (I : IdealLike α) (F : IdealLike α → Nat) : Prop :=
  hD.1.1 I F

def IdealNormLike {α : Type u} [DomainLike α]
    (hD : IsDedekindLike (α := α)) (I : IdealLike α) : Nat :=
  hD.1.2 I

theorem factorization_exists {α : Type u} [DomainLike α]
    (hD : IsDedekindLike (α := α)) (I : IdealLike α) :
    ∃ F : IdealLike α → Nat, IdealFactorizationLike hD I F := by
  rcases hD with ⟨⟨factorRel, norm⟩, hprops⟩
  rcases hprops with ⟨hex, huniq, hmulfac, hnormmul, hprimegt, hprimepow, hfin, hclass⟩
  have hI : ∃ F : IdealLike α → Nat, factorRel I F := hex I
  rcases hI with ⟨F, hF⟩
  refine ⟨F, ?_⟩
  exact hF

theorem factorization_unique {α : Type u} [DomainLike α]
    (hD : IsDedekindLike (α := α)) (I : IdealLike α)
    (F G : IdealLike α → Nat)
    (hF : IdealFactorizationLike hD I F)
    (hG : IdealFactorizationLike hD I G) :
    F = G := by
  rcases hD with ⟨⟨factorRel, norm⟩, hprops⟩
  rcases hprops with ⟨hex, huniq, hmulfac, hnormmul, hprimegt, hprimepow, hfin, hclass⟩
  have hEq : F = G := huniq I F G hF hG
  exact hEq

theorem norm_mul {α : Type u} [DomainLike α]
    (hD : IsDedekindLike (α := α)) (I J : IdealLike α) :
    IdealNormLike hD (DomainLike.idealMul I J) = IdealNormLike hD I * IdealNormLike hD J := by
  rcases hD with ⟨⟨factorRel, norm⟩, hprops⟩
  rcases hprops with ⟨hex, huniq, hmulfac, hnormmul, hprimegt, hprimepow, hfin, hclass⟩
  have hmul : norm (DomainLike.idealMul I J) = norm I * norm J := hnormmul I J
  simpa [IdealNormLike] using hmul

theorem norm_prime_power {α : Type u} [DomainLike α]
    (hD : IsDedekindLike (α := α))
    (P : IdealLike α) (hP : PrimeIdealLike P) (n : Nat) :
    ∃ Ipow : IdealLike α, ∃ F : IdealLike α → Nat,
      IdealFactorizationLike hD Ipow F ∧ F P = n ∧
      IdealNormLike hD Ipow = Nat.pow (IdealNormLike hD P) n := by
  rcases hD with ⟨⟨factorRel, norm⟩, hprops⟩
  rcases hprops with ⟨hex, huniq, hmulfac, hnormmul, hprimegt, hprimepow, hfin, hclass⟩
  have hpow :
      ∃ Ipow : IdealLike α, ∃ F : IdealLike α → Nat,
        factorRel Ipow F ∧ F P = n ∧ norm Ipow = Nat.pow (norm P) n :=
    hprimepow P hP n
  rcases hpow with ⟨Ipow, F, hFac, hExp, hNormPow⟩
  refine ⟨Ipow, F, ?_⟩
  constructor
  · exact hFac
  constructor
  · exact hExp
  · simpa [IdealNormLike] using hNormPow

theorem finite_ideal_quotient_like {α : Type u} [DomainLike α]
    (hD : IsDedekindLike (α := α)) :
    ∃ B : Nat, B > 0 := by
  rcases hD with ⟨⟨factorRel, norm⟩, hprops⟩
  rcases hprops with ⟨hex, huniq, hmulfac, hnormmul, hprimegt, hprimepow, hfin, hclass⟩
  rcases hfin with ⟨B, hB⟩
  refine ⟨B, ?_⟩
  exact hB

theorem class_group_finite_like {α : Type u} [DomainLike α]
    (hD : IsDedekindLike (α := α)) :
    ∃ h : Nat, h > 0 := by
  have hfiniteQ : ∃ B : Nat, B > 0 := finite_ideal_quotient_like hD
  rcases hfiniteQ with ⟨B, hB⟩
  rcases hD with ⟨⟨factorRel, norm⟩, hprops⟩
  rcases hprops with ⟨hex, huniq, hmulfac, hnormmul, hprimegt, hprimepow, hfin, hclass⟩
  rcases hclass with ⟨h, hh⟩
  have hh_pos : h > 0 := hh
  have _ : B > 0 := hB
  refine ⟨h, ?_⟩
  exact hh_pos
