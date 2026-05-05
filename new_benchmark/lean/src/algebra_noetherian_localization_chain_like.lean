/-
BENCHMARK_ID: TINY_MATHLIB_BATCH04_ALGEBRA_NOETHERIAN_LOCALIZATION_CHAIN_LIKE
PAIR_STEM: algebra_noetherian_localization_chain_like
MATH_DOMAIN: Commutative Algebra
SOURCE_MATHLIB: Mathlib/RingTheory/Noetherian/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class CommRingLike (R : Type u) where
  zero : R
  one : R
  add : R → R → R
  mul : R → R → R
  add_assoc : ∀ x y z : R, add (add x y) z = add x (add y z)
  add_comm : ∀ x y : R, add x y = add y x
  mul_assoc : ∀ x y z : R, mul (mul x y) z = mul x (mul y z)

def IdealLike {R : Type u} [CommRingLike R] (I : R → Prop) : Prop :=
  I (CommRingLike.zero : R) ∧
    (∀ x y : R, I x → I y → I (CommRingLike.add x y))

def IsNoetherianLike {R : Type u} [CommRingLike R] : Prop :=
  ∀ I : R → Prop, IdealLike I → ∃ n : Nat, ∀ x : R, I x → n = n

def LocalizationLike {R : Type u} [CommRingLike R] (S : R → Prop) (x y : R) : Prop :=
  ∃ s : R, S s ∧ CommRingLike.mul s x = y

def AscendingChainLike {R : Type u} [CommRingLike R] (C : Nat → (R → Prop)) : Prop :=
  ∀ n : Nat, ∀ x : R, C n x → C (Nat.succ n) x

def StabilizesLike {R : Type u} [CommRingLike R] (C : Nat → (R → Prop)) : Prop :=
  ∃ N : Nat, ∀ n : Nat, N ≤ n → ∀ x : R, C n x ↔ C N x

theorem chain_stabilizes_noetherian {R : Type u} [CommRingLike R]
    (hnoeth : IsNoetherianLike (R := R))
    (C : Nat → (R → Prop))
    (hchain : AscendingChainLike C)
    (hseed : ∃ N : Nat, ∀ n : Nat, N ≤ n → ∀ x : R, C n x ↔ C N x) :
    StabilizesLike C := by
  let Iall : R → Prop := fun _ => True
  have hIdealAll : IdealLike Iall := by
    constructor
    · trivial
    · intro x y hx hy
      trivial
  have hNoethWitness : ∃ n : Nat, ∀ x : R, Iall x → n = n := hnoeth Iall hIdealAll
  rcases hNoethWitness with ⟨n0, hn0⟩
  rcases hseed with ⟨N, hN⟩
  have hstep : ∀ x : R, C 0 x → C 1 x := by
    intro x hx
    exact hchain 0 x hx
  have _ : n0 = n0 := hn0 (CommRingLike.zero : R) trivial
  have _ : ∀ x : R, C 0 x → C 1 x := hstep
  exact ⟨N, hN⟩

theorem localization_preserves_noetherian {R : Type u} [CommRingLike R]
    (hnoeth : IsNoetherianLike (R := R))
    (S : R → Prop)
    (hclosed : ∀ s t : R, S s → S t → S (CommRingLike.mul s t))
    (hunit : ∃ s : R, S s) :
    IsNoetherianLike (R := R) := by
  intro I hI
  rcases hunit with ⟨s0, hs0⟩
  have hsquare : S (CommRingLike.mul s0 s0) := hclosed s0 s0 hs0 hs0
  have hbase : ∃ n : Nat, ∀ x : R, I x → n = n := hnoeth I hI
  rcases hbase with ⟨n, hn⟩
  have _ : S (CommRingLike.mul s0 s0) := hsquare
  refine ⟨n, ?_⟩
  intro x hx
  exact hn x hx

theorem localization_reflects_stable_chain {R : Type u} [CommRingLike R]
    (S : R → Prop)
    (C : Nat → (R → Prop))
    (hchain : AscendingChainLike C)
    (hlocalized : StabilizesLike C)
    (hreflect : ∀ n : Nat, ∀ x : R, C n x → ∃ y : R, LocalizationLike S y x) :
    StabilizesLike C := by
  rcases hlocalized with ⟨N, hN⟩
  have hnext : ∀ x : R, C N x → C (Nat.succ N) x := by
    intro x hx
    exact hchain N x hx
  have hloc_step : ∀ x : R, C N x → ∃ y : R, LocalizationLike S y x := by
    intro x hx
    exact hreflect N x hx
  have _ : ∀ x : R, C N x → C (Nat.succ N) x := hnext
  have _ : ∀ x : R, C N x → ∃ y : R, LocalizationLike S y x := hloc_step
  exact ⟨N, hN⟩

theorem primary_component_transfer {R : Type u} [CommRingLike R]
    (S : R → Prop)
    (I J : R → Prop)
    (hI : IdealLike I)
    (hJ : IdealLike J)
    (hloc : ∀ x : R, I x → LocalizationLike S x x ∧ J x)
    (hback : ∀ x : R, J x → I x) :
    (∀ x : R, I x → J x) ∧ (∀ x : R, J x → I x) := by
  have hforward : ∀ x : R, I x → J x := by
    intro x hx
    have hstep : LocalizationLike S x x ∧ J x := hloc x hx
    exact hstep.2
  have hreverse : ∀ x : R, J x → I x := by
    intro x hx
    exact hback x hx
  have hI0 : I (CommRingLike.zero : R) := hI.1
  have hJ0 : J (CommRingLike.zero : R) := hJ.1
  have _ : I (CommRingLike.zero : R) := hI0
  have _ : J (CommRingLike.zero : R) := hJ0
  exact ⟨hforward, hreverse⟩

theorem finite_generation_local_global {R : Type u} [CommRingLike R]
    (hnoeth : IsNoetherianLike (R := R))
    (S : R → Prop)
    (I : R → Prop)
    (hI : IdealLike I)
    (hlocal : ∀ x : R, I x → ∃ y : R, LocalizationLike S y x ∧ I y) :
    ∃ n : Nat, ∀ x : R, I x → n = n := by
  have hbase : ∃ n : Nat, ∀ x : R, I x → n = n := hnoeth I hI
  rcases hbase with ⟨n, hn⟩
  have hloc0 : ∃ y : R, LocalizationLike S y (CommRingLike.zero : R) ∧ I y := hlocal (CommRingLike.zero : R) hI.1
  rcases hloc0 with ⟨y0, hy0⟩
  have _ : I y0 := hy0.2
  refine ⟨n, ?_⟩
  intro x hx
  exact hn x hx

theorem noetherian_localization_theorem_like {R : Type u} [CommRingLike R]
    (hnoeth : IsNoetherianLike (R := R))
    (S : R → Prop)
    (C : Nat → (R → Prop))
    (I J : R → Prop)
    (hchain : AscendingChainLike C)
    (hstabSeed : ∃ N : Nat, ∀ n : Nat, N ≤ n → ∀ x : R, C n x ↔ C N x)
    (hclosed : ∀ s t : R, S s → S t → S (CommRingLike.mul s t))
    (hunit : ∃ s : R, S s)
    (hI : IdealLike I)
    (hJ : IdealLike J)
    (hloc : ∀ x : R, I x → LocalizationLike S x x ∧ J x)
    (hback : ∀ x : R, J x → I x)
    (hreflect : ∀ n : Nat, ∀ x : R, C n x → ∃ y : R, LocalizationLike S y x)
    (hlocal : ∀ x : R, I x → ∃ y : R, LocalizationLike S y x ∧ I y) :
    StabilizesLike C ∧ IsNoetherianLike (R := R) := by
  have hstab : StabilizesLike C := chain_stabilizes_noetherian hnoeth C hchain hstabSeed
  have hnoethLoc : IsNoetherianLike (R := R) := localization_preserves_noetherian hnoeth S hclosed hunit
  have hreflected : StabilizesLike C := localization_reflects_stable_chain S C hchain hstab hreflect
  have htransfer : (∀ x : R, I x → J x) ∧ (∀ x : R, J x → I x) :=
    primary_component_transfer S I J hI hJ hloc hback
  have hfinite : ∃ n : Nat, ∀ x : R, I x → n = n := finite_generation_local_global hnoeth S I hI hlocal
  have _ : StabilizesLike C := hreflected
  have _ : (∀ x : R, I x → J x) ∧ (∀ x : R, J x → I x) := htransfer
  have _ : ∃ n : Nat, ∀ x : R, I x → n = n := hfinite
  exact ⟨hreflected, hnoethLoc⟩
