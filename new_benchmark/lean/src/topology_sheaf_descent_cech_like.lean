/-
BENCHMARK_ID: TINY_MATHLIB_BATCH04_TOPOLOGY_SHEAF_DESCENT_CECH
PAIR_STEM: topology_sheaf_descent_cech_like
MATH_DOMAIN: Topology / Sheaf Theory
SOURCE_MATHLIB: Mathlib/Topology/Sheaves/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u v

class SiteLike (X : Type u) where
  le : X → X → Prop
  le_refl : ∀ U : X, le U U
  le_trans : ∀ {U V W : X}, le W V → le V U → le W U
  Cover : X → (X → Prop) → Prop
  cover_mem : ∀ {U : X} {I : X → Prop}, Cover U I → ∀ V : X, I V → le V U
  cover_refl : ∀ U : X, Cover U (fun V : X => V = U)
  cover_refine :
    ∀ {U : X} {I J : X → Prop},
      Cover U I →
      (∀ V : X, I V → Cover V J) →
      Cover U (fun W : X => ∃ V : X, I V ∧ J W)

structure PresheafLike (X : Type u) [SiteLike X] where
  Sect : Type v
  res : X → X → Sect → Sect
  res_id : ∀ U : X, ∀ s : Sect, res U U s = s
  res_comp :
    ∀ U V W : X,
      ∀ s : Sect,
      SiteLike.le W V → SiteLike.le V U →
      res W V (res V U s) = res W U s

def CompatibleFamilyLike {X : Type u} [SiteLike X]
    (F : PresheafLike X)
    (I : X → Prop)
    (σ : X → F.Sect) : Prop :=
  ∀ V W : X,
    I V → I W →
    ∀ T : X,
      SiteLike.le T V → SiteLike.le T W →
      F.res T V (σ V) = F.res T W (σ W)

def MatchingObjectLike {X : Type u} [SiteLike X]
    (F : PresheafLike X)
    (I : X → Prop) : Prop :=
  ∃ σ : X → F.Sect, CompatibleFamilyLike F I σ

def DescentDataLike {X : Type u} [SiteLike X]
    (F : PresheafLike X)
    (U : X) (I : X → Prop) : Prop :=
  ∀ σ : X → F.Sect,
    CompatibleFamilyLike F I σ →
    ∃ s : F.Sect, ∀ V : X, I V → F.res V U s = σ V

def CechComplexLike {X : Type u} [SiteLike X]
    (F : PresheafLike X)
    (U : X) (I : X → Prop) : Prop :=
  DescentDataLike F U I ∧
  (∀ s t : F.Sect, (∀ V : X, I V → F.res V U s = F.res V U t) → s = t)

theorem sheaf_condition_matching {X : Type u} [SiteLike X]
    (F : PresheafLike X)
    (U : X) (I : X → Prop)
    (hdesc : DescentDataLike F U I) :
    ∀ σ : X → F.Sect,
      CompatibleFamilyLike F I σ →
      ∃ s : F.Sect, ∀ V : X, I V → F.res V U s = σ V := by
  intro σ hσ
  have hglue : ∃ s : F.Sect, ∀ V : X, I V → F.res V U s = σ V := hdesc σ hσ
  rcases hglue with ⟨s, hs⟩
  exact ⟨s, hs⟩

theorem descent_gluing_exists {X : Type u} [SiteLike X]
    (F : PresheafLike X)
    (U : X) (I : X → Prop)
    (hdesc : DescentDataLike F U I)
    (hmatch : MatchingObjectLike F I) :
    ∃ s : F.Sect,
      ∃ σ : X → F.Sect,
        CompatibleFamilyLike F I σ ∧
        (∀ V : X, I V → F.res V U s = σ V) := by
  rcases hmatch with ⟨σ, hσ⟩
  have hglue : ∃ s : F.Sect, ∀ V : X, I V → F.res V U s = σ V := hdesc σ hσ
  rcases hglue with ⟨s, hs⟩
  exact ⟨s, σ, hσ, hs⟩

theorem descent_gluing_unique {X : Type u} [SiteLike X]
    (F : PresheafLike X)
    (U : X) (I : X → Prop)
    (hcech : CechComplexLike F U I)
    (s t : F.Sect)
    (hEq : ∀ V : X, I V → F.res V U s = F.res V U t) :
    s = t := by
  have huniq : ∀ a b : F.Sect, (∀ V : X, I V → F.res V U a = F.res V U b) → a = b := hcech.2
  exact huniq s t hEq

theorem cech_exactness_degree1 {X : Type u} [SiteLike X]
    (F : PresheafLike X)
    (U : X) (I : X → Prop)
    (hcech : CechComplexLike F U I) :
    ∀ σ : X → F.Sect,
      CompatibleFamilyLike F I σ →
      ∃ s : F.Sect,
        (∀ V : X, I V → F.res V U s = σ V) ∧
        (∀ t : F.Sect,
          (∀ V : X, I V → F.res V U t = σ V) →
          t = s) := by
  intro σ hσ
  have hdesc : DescentDataLike F U I := hcech.1
  have huniq : ∀ a b : F.Sect, (∀ V : X, I V → F.res V U a = F.res V U b) → a = b := hcech.2
  rcases hdesc σ hσ with ⟨s, hs⟩
  refine ⟨s, ?_⟩
  refine And.intro hs ?_
  intro t ht
  have hts : t = s := huniq t s (by
    intro V hV
    exact Eq.trans (ht V hV) (Eq.symm (hs V hV)))
  exact hts

theorem cech_descent_equivalence {X : Type u} [SiteLike X]
    (F : PresheafLike X)
    (U : X) (I : X → Prop)
    (hcech : CechComplexLike F U I) :
    DescentDataLike F U I ∧
    (∀ σ : X → F.Sect,
      CompatibleFamilyLike F I σ →
      ∃ s : F.Sect,
        (∀ V : X, I V → F.res V U s = σ V) ∧
        (∀ t : F.Sect,
          (∀ V : X, I V → F.res V U t = σ V) →
          t = s)) := by
  refine And.intro hcech.1 ?_
  intro σ hσ
  have hdesc : DescentDataLike F U I := hcech.1
  have huniq : ∀ a b : F.Sect, (∀ V : X, I V → F.res V U a = F.res V U b) → a = b := hcech.2
  rcases hdesc σ hσ with ⟨s, hs⟩
  refine ⟨s, ?_⟩
  refine And.intro hs ?_
  intro t ht
  exact huniq t s (by
    intro V hV
    exact Eq.trans (ht V hV) (Eq.symm (hs V hV)))

theorem hypercover_refinement_transfer {X : Type u} [SiteLike X]
    (F : PresheafLike X)
    (U : X)
    (I J : X → Prop)
    (hsub : ∀ V : X, J V → I V)
    (hlift :
      ∀ σJ : X → F.Sect,
        CompatibleFamilyLike F J σJ →
        ∃ σI : X → F.Sect,
          CompatibleFamilyLike F I σI ∧
          (∀ V : X, J V → σI V = σJ V))
    (hdescI : DescentDataLike F U I) :
    DescentDataLike F U J := by
  intro σJ hσJ
  rcases hlift σJ hσJ with ⟨σI, hσI, hagree⟩
  rcases hdescI σI hσI with ⟨s, hsI⟩
  refine ⟨s, ?_⟩
  intro V hV
  have hIV : I V := hsub V hV
  have hresI : F.res V U s = σI V := hsI V hIV
  have hEq : σI V = σJ V := hagree V hV
  exact Eq.trans hresI hEq
