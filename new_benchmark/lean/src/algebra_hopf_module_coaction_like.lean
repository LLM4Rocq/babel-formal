/-
BENCHMARK_ID: TINY_MATHLIB_BATCH04_ALGEBRA_HOPF_MODULE_COACTION_LIKE
PAIR_STEM: algebra_hopf_module_coaction_like
MATH_DOMAIN: Algebra / Category-style Algebra
SOURCE_MATHLIB: Mathlib/Algebra/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u v

class BialgebraLike (H : Type u) (M : Type v) where
  coact : M -> H -> M -> Prop
  counit : H -> Prop
  smul : H -> M -> M
  coassoc :
    forall (m : M) (h1 : H) (m1 : M) (h2 : H) (m2 : M),
      coact m h1 m1 -> coact m1 h2 m2 -> coact m h1 m2
  counit_law :
    forall (m : M) (h : H) (m1 : M),
      coact m h m1 -> counit h -> m1 = m
  module_coact :
    forall (h : H) (m : M), counit h -> coact m h (smul h m)

def ComoduleLike (H : Type u) (M : Type v) [BialgebraLike H M] : Prop :=
  forall m : M, exists h : H, exists m1 : M, @BialgebraLike.coact H M _ m h m1

def ModuleLike (H : Type u) (M : Type v) [BialgebraLike H M] : Prop :=
  forall (h : H) (m : M), @BialgebraLike.counit H M _ h ->
    @BialgebraLike.smul H M _ h m = @BialgebraLike.smul H M _ h m

def HopfModuleLike (H : Type u) (M : Type v) [BialgebraLike H M] : Prop :=
  ComoduleLike H M /\ ModuleLike H M

def CoinvariantLike (H : Type u) (M : Type v) [BialgebraLike H M] (m : M) : Prop :=
  forall (h : H) (m1 : M), @BialgebraLike.coact H M _ m h m1 -> @BialgebraLike.counit H M _ h -> m1 = m

def TensorOverLike (H : Type u) (M : Type v) [BialgebraLike H M] : Type (max u v) :=
  H × M

theorem coaction_coassoc_like (H : Type u) (M : Type v) [BialgebraLike H M]
    {m : M} {h1 : H} {m1 : M} {h2 : H} {m2 : M}
    (hco1 : @BialgebraLike.coact H M _ m h1 m1)
    (hco2 : @BialgebraLike.coact H M _ m1 h2 m2) :
    exists h : H, @BialgebraLike.coact H M _ m h m2 := by
  have hCompose : @BialgebraLike.coact H M _ m h1 m2 :=
    @BialgebraLike.coassoc H M _ m h1 m1 h2 m2 hco1 hco2
  exact ⟨h1, hCompose⟩

theorem coaction_counit_like (H : Type u) (M : Type v) [BialgebraLike H M]
    {m : M} {h : H} {m1 : M}
    (hco : @BialgebraLike.coact H M _ m h m1)
    (hc : @BialgebraLike.counit H M _ h) :
    m1 = m := by
  have hEq : m1 = m := @BialgebraLike.counit_law H M _ m h m1 hco hc
  exact hEq

theorem coinvariant_submodule_like (H : Type u) (M : Type v) [BialgebraLike H M]
    (m : M) (hcoinv : CoinvariantLike H M m) :
    forall (h : H) (m1 : M),
      @BialgebraLike.coact H M _ m h m1 -> @BialgebraLike.counit H M _ h -> CoinvariantLike H M m1 := by
  intro h m1 hco hmCounit
  have hm1Eqm : m1 = m := hcoinv h m1 hco hmCounit
  intro h' m2 hco' hCounit'
  have hm2Eqm1 : m2 = m1 := @BialgebraLike.counit_law H M _ m1 h' m2 hco' hCounit'
  have hm2Eqm : m2 = m := by
    calc
      m2 = m1 := hm2Eqm1
      _ = m := hm1Eqm
  calc
    m2 = m := hm2Eqm
    _ = m1 := Eq.symm hm1Eqm

theorem fundamental_map_like (H : Type u) (M : Type v) [BialgebraLike H M]
    (h : H) (m : M) (hc : @BialgebraLike.counit H M _ h) :
    exists t : TensorOverLike H M, t.1 = h /\ @BialgebraLike.coact H M _ m h t.2 := by
  let t : TensorOverLike H M := (h, @BialgebraLike.smul H M _ h m)
  have htEq : t.1 = h := by
    rfl
  have hco : @BialgebraLike.coact H M _ m h t.2 := by
    change @BialgebraLike.coact H M _ m h (@BialgebraLike.smul H M _ h m)
    exact @BialgebraLike.module_coact H M _ h m hc
  exact ⟨t, And.intro htEq hco⟩

theorem fundamental_inverse_like (H : Type u) (M : Type v) [BialgebraLike H M]
    (m : M) (hcoinv : CoinvariantLike H M m)
    (h : H) (hc : @BialgebraLike.counit H M _ h) :
    (let t : TensorOverLike H M := (h, @BialgebraLike.smul H M _ h m); t.2 = m) := by
  have hco : @BialgebraLike.coact H M _ m h (@BialgebraLike.smul H M _ h m) :=
    @BialgebraLike.module_coact H M _ h m hc
  have hEq : @BialgebraLike.smul H M _ h m = m := hcoinv h (@BialgebraLike.smul H M _ h m) hco hc
  dsimp
  exact hEq

theorem hopf_module_decomposition_like (H : Type u) (M : Type v) [BialgebraLike H M]
    (m : M) (hcoinv : CoinvariantLike H M m) :
    forall h : H, @BialgebraLike.counit H M _ h ->
      exists t : TensorOverLike H M, t.1 = h /\ t.2 = m := by
  intro h hc
  have hMap : exists t : TensorOverLike H M, t.1 = h /\ @BialgebraLike.coact H M _ m h t.2 :=
    fundamental_map_like (H := H) (M := M) h m hc
  rcases hMap with ⟨t, ht₁, htCo⟩
  have hAnchor : t.1 = h := ht₁
  have hAnchorCoact : @BialgebraLike.coact H M _ m h t.2 := by
    simpa [hAnchor] using htCo
  have hInv :
      (let s : TensorOverLike H M := (h, @BialgebraLike.smul H M _ h m); s.2 = m) :=
    fundamental_inverse_like (H := H) (M := M) m hcoinv h hc
  have hDrop : True := by
    have : @BialgebraLike.coact H M _ m h t.2 := hAnchorCoact
    trivial
  have hUse : True := hDrop
  refine ⟨(h, @BialgebraLike.smul H M _ h m), rfl, ?_⟩
  calc
    ((h, @BialgebraLike.smul H M _ h m) : TensorOverLike H M).2 = m := by
      simpa using hInv
