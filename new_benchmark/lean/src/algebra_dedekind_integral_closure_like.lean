/-
BENCHMARK_ID: TINY_MATHLIB_BATCH04_ALGEBRA_DEDEKIND_INTEGRAL_CLOSURE_LIKE
PAIR_STEM: algebra_dedekind_integral_closure_like
MATH_DOMAIN: Commutative Algebra / Number Theory
SOURCE_MATHLIB: Mathlib/RingTheory/DedekindDomain/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u v

class DomainLike (R : Type u) (K : Type v) where
  integral : K -> Prop
  closure : K -> Prop
  integral_nonempty : exists x : K, integral x
  closure_of_integral : forall x : K, integral x -> closure x
  integral_of_closure : forall x : K, closure x -> integral x
  closure_mul : forall x y : K, closure x -> closure y -> closure x
  closure_add : forall x y : K, closure x -> closure y -> closure y

def FractionFieldLike (R : Type u) (K : Type v) [DomainLike R K] : Prop :=
  forall x : K, @DomainLike.integral R K _ x -> True

def IntegralLike (R : Type u) (K : Type v) [DomainLike R K] (x : K) : Prop :=
  @DomainLike.integral R K _ x

def IntegralClosureLike (R : Type u) (K : Type v) [DomainLike R K] (x : K) : Prop :=
  @DomainLike.closure R K _ x

def IsDedekindLike (R : Type u) (K : Type v) [DomainLike R K] : Prop :=
  (forall x : K, IntegralClosureLike R K x -> IntegralLike R K x) /\
    (exists x : K, IntegralClosureLike R K x)

def FiniteExtensionLike (R : Type u) (K : Type v) [DomainLike R K] : Prop :=
  forall x : K, IntegralClosureLike R K x -> exists n : Nat, n = n

theorem integral_closure_exists (R : Type u) (K : Type v) [DomainLike R K] :
    exists x : K, IntegralClosureLike R K x := by
  rcases (DomainLike.integral_nonempty (R := R) (K := K)) with ⟨x, hxInt⟩
  have hxCl : IntegralClosureLike R K x :=
    @DomainLike.closure_of_integral R K _ x hxInt
  exact ⟨x, hxCl⟩

theorem integral_closure_integral (R : Type u) (K : Type v) [DomainLike R K]
    (x : K) (hx : IntegralClosureLike R K x) :
    IntegralLike R K x := by
  have hRaw : @DomainLike.integral R K _ x :=
    @DomainLike.integral_of_closure R K _ x hx
  change IntegralLike R K x
  exact hRaw

theorem dedekind_of_integral_closure (R : Type u) (K : Type v) [DomainLike R K] :
    IsDedekindLike R K := by
  have hMain : forall x : K, IntegralClosureLike R K x -> IntegralLike R K x := by
    intro x hx
    exact integral_closure_integral (R := R) (K := K) x hx
  have hExist : exists x : K, IntegralClosureLike R K x :=
    integral_closure_exists (R := R) (K := K)
  exact And.intro hMain hExist

theorem prime_factorization_transfer (R : Type u) (K : Type v) [DomainLike R K]
    {x y : K} (hx : IntegralClosureLike R K x) (hy : IntegralClosureLike R K y) :
    IntegralLike R K x /\ IntegralLike R K y := by
  have hDed : IsDedekindLike R K := dedekind_of_integral_closure (R := R) (K := K)
  rcases hDed with ⟨hInt, hNonempty⟩
  rcases hNonempty with ⟨z, hz⟩
  have hxLift : IntegralClosureLike R K x :=
    @DomainLike.closure_mul R K _ x y hx hy
  have hyLift : IntegralClosureLike R K y :=
    @DomainLike.closure_add R K _ z y hz hy
  have hxInt : IntegralLike R K x := hInt x hxLift
  have hyInt : IntegralLike R K y := hInt y hyLift
  exact And.intro hxInt hyInt

theorem discriminant_control_like (R : Type u) (K : Type v) [DomainLike R K]
    (hFrac : FractionFieldLike R K)
    {x y : K} (hx : IntegralClosureLike R K x) (hy : IntegralClosureLike R K y) :
    IntegralLike R K y := by
  have hDed : IsDedekindLike R K :=
    dedekind_of_integral_closure (R := R) (K := K)
  rcases hDed with ⟨hInt, hEx⟩
  rcases hEx with ⟨w, hw⟩
  have hyStable : IntegralClosureLike R K y :=
    @DomainLike.closure_add R K _ w y hw hy
  have hyInt : IntegralLike R K y := hInt y hyStable
  have hFracWitness : True := hFrac y hyInt
  have : True := hFracWitness
  exact hyInt

theorem integral_closure_finite_like (R : Type u) (K : Type v) [DomainLike R K]
    (hFin : FiniteExtensionLike R K) :
    exists x : K, exists n : Nat, IntegralClosureLike R K x /\ n = n := by
  rcases integral_closure_exists (R := R) (K := K) with ⟨x, hx⟩
  rcases hFin x hx with ⟨n, hn⟩
  have hPair : IntegralClosureLike R K x /\ n = n := And.intro hx hn
  exact ⟨x, n, hPair⟩
