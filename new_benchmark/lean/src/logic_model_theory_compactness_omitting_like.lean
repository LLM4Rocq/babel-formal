/-
BENCHMARK_ID: TINY_MATHLIB_BATCH04_LOGIC_MODEL_THEORY_COMPACTNESS_OMITTING_LIKE
PAIR_STEM: logic_model_theory_compactness_omitting_like
MATH_DOMAIN: Model Theory
SOURCE_MATHLIB: Mathlib/ModelTheory
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class LanguageLike (L : Type u) where
  Formula : Type u
  Model : Type u
  Sat : Model → Formula → Prop

def TheoryLike {L : Type u} [LanguageLike L] : Type u :=
  LanguageLike.Formula (L := L) → Prop

def TypeLike {L : Type u} [LanguageLike L] : Type u :=
  LanguageLike.Formula (L := L) → Prop

def SatisfiableLike {L : Type u} [LanguageLike L] (T : TheoryLike (L := L)) : Prop :=
  ∃ M : LanguageLike.Model (L := L),
    ∀ φ : LanguageLike.Formula (L := L), T φ → LanguageLike.Sat (L := L) M φ

def OmitsLike {L : Type u} [LanguageLike L]
    (M : LanguageLike.Model (L := L)) (p : TypeLike (L := L)) : Prop :=
  ∀ φ : LanguageLike.Formula (L := L),
    p φ → ¬ LanguageLike.Sat (L := L) M φ

def ElementaryChainLike {L : Type u} [LanguageLike L]
    (Ms : Nat → LanguageLike.Model (L := L)) (T : TheoryLike (L := L)) : Prop :=
  ∀ n : Nat,
    ∀ φ : LanguageLike.Formula (L := L),
      T φ → LanguageLike.Sat (L := L) (Ms n) φ → LanguageLike.Sat (L := L) (Ms (n + 1)) φ

theorem finite_satisfiable_compact {L : Type u} [LanguageLike L]
    (T : TheoryLike (L := L))
    (S : Nat → LanguageLike.Formula (L := L) → Prop)
    (hcompact :
      (∀ n : Nat,
        ∃ M : LanguageLike.Model (L := L),
          ∀ φ : LanguageLike.Formula (L := L), T φ → S n φ → LanguageLike.Sat (L := L) M φ) →
      SatisfiableLike (L := L) T)
    (hfin :
      ∀ n : Nat,
        ∃ M : LanguageLike.Model (L := L),
          ∀ φ : LanguageLike.Formula (L := L), T φ → S n φ → LanguageLike.Sat (L := L) M φ) :
    SatisfiableLike (L := L) T := by
  have hseed :
      ∀ n : Nat,
        ∃ M : LanguageLike.Model (L := L),
          ∀ φ : LanguageLike.Formula (L := L), T φ → S n φ → LanguageLike.Sat (L := L) M φ := hfin
  have hsat : SatisfiableLike (L := L) T := hcompact hseed
  have _ :
      ∀ n : Nat,
        ∃ M : LanguageLike.Model (L := L),
          ∀ φ : LanguageLike.Formula (L := L), T φ → S n φ → LanguageLike.Sat (L := L) M φ := hseed
  exact hsat

theorem henkin_extension_step {L : Type u} [LanguageLike L]
    (T T' : TheoryLike (L := L))
    (hincl : ∀ φ : LanguageLike.Formula (L := L), T φ → T' φ)
    (hsat : SatisfiableLike (L := L) T)
    (hhenkin :
      ∀ M : LanguageLike.Model (L := L),
        (∀ φ : LanguageLike.Formula (L := L), T φ → LanguageLike.Sat (L := L) M φ) →
        ∀ ψ : LanguageLike.Formula (L := L), T' ψ → LanguageLike.Sat (L := L) M ψ) :
    SatisfiableLike (L := L) T' := by
  rcases hsat with ⟨M, hM⟩
  have hbase :
      ∀ φ : LanguageLike.Formula (L := L), T φ → LanguageLike.Sat (L := L) M φ := hM
  have hext :
      ∀ ψ : LanguageLike.Formula (L := L), T' ψ → LanguageLike.Sat (L := L) M ψ :=
    hhenkin M hbase
  have _ : ∀ φ : LanguageLike.Formula (L := L), T φ → T' φ := hincl
  exact ⟨M, hext⟩

theorem chain_model_union_like {L : Type u} [LanguageLike L]
    (Ms : Nat → LanguageLike.Model (L := L))
    (T : TheoryLike (L := L))
    (hchain : ElementaryChainLike (L := L) Ms T)
    (hroot : ∀ φ : LanguageLike.Formula (L := L), T φ → LanguageLike.Sat (L := L) (Ms 0) φ)
    (hunion :
      ∀ φ : LanguageLike.Formula (L := L),
        T φ → (∃ n : Nat, LanguageLike.Sat (L := L) (Ms n) φ) → LanguageLike.Sat (L := L) (Ms 0) φ) :
    SatisfiableLike (L := L) T := by
  have hwitness :
      ∀ φ : LanguageLike.Formula (L := L), T φ → ∃ n : Nat, LanguageLike.Sat (L := L) (Ms n) φ := by
    intro φ hφ
    refine ⟨0, ?_⟩
    exact hroot φ hφ
  have hmodel :
      ∀ φ : LanguageLike.Formula (L := L), T φ → LanguageLike.Sat (L := L) (Ms 0) φ := by
    intro φ hφ
    exact hunion φ hφ (hwitness φ hφ)
  have _ : ElementaryChainLike (L := L) Ms T := hchain
  exact ⟨Ms 0, hmodel⟩

theorem omitting_types_step {L : Type u} [LanguageLike L]
    (T : TheoryLike (L := L))
    (p : TypeLike (L := L))
    (hsat : SatisfiableLike (L := L) T)
    (homit :
      ∀ M : LanguageLike.Model (L := L),
        (∀ φ : LanguageLike.Formula (L := L), T φ → LanguageLike.Sat (L := L) M φ) →
        ∀ φ : LanguageLike.Formula (L := L), p φ → ¬ LanguageLike.Sat (L := L) M φ) :
    ∃ M : LanguageLike.Model (L := L),
      (∀ φ : LanguageLike.Formula (L := L), T φ → LanguageLike.Sat (L := L) M φ) ∧
      OmitsLike (L := L) M p := by
  rcases hsat with ⟨M, hM⟩
  have hT :
      ∀ φ : LanguageLike.Formula (L := L), T φ → LanguageLike.Sat (L := L) M φ := hM
  have hO : OmitsLike (L := L) M p := homit M hT
  exact ⟨M, hT, hO⟩

theorem complete_theory_model_exists {L : Type u} [LanguageLike L]
    (T T' : TheoryLike (L := L))
    (hincl : ∀ φ : LanguageLike.Formula (L := L), T φ → T' φ)
    (hsat' : SatisfiableLike (L := L) T') :
    SatisfiableLike (L := L) T := by
  rcases hsat' with ⟨M, hM'⟩
  have hM :
      ∀ φ : LanguageLike.Formula (L := L), T φ → LanguageLike.Sat (L := L) M φ := by
    intro φ hφ
    have hφ' : T' φ := hincl φ hφ
    exact hM' φ hφ'
  exact ⟨M, hM⟩

theorem compactness_omitting_types_like {L : Type u} [LanguageLike L]
    (T T' : TheoryLike (L := L))
    (p : TypeLike (L := L))
    (S : Nat → LanguageLike.Formula (L := L) → Prop)
    (hcompact :
      (∀ n : Nat,
        ∃ M : LanguageLike.Model (L := L),
          ∀ φ : LanguageLike.Formula (L := L), T' φ → S n φ → LanguageLike.Sat (L := L) M φ) →
      SatisfiableLike (L := L) T')
    (hfin :
      ∀ n : Nat,
        ∃ M : LanguageLike.Model (L := L),
          ∀ φ : LanguageLike.Formula (L := L), T' φ → S n φ → LanguageLike.Sat (L := L) M φ)
    (hincl : ∀ φ : LanguageLike.Formula (L := L), T φ → T' φ)
    (homit :
      ∀ M : LanguageLike.Model (L := L),
        (∀ φ : LanguageLike.Formula (L := L), T' φ → LanguageLike.Sat (L := L) M φ) →
        ∀ φ : LanguageLike.Formula (L := L), p φ → ¬ LanguageLike.Sat (L := L) M φ) :
    ∃ M : LanguageLike.Model (L := L),
      (∀ φ : LanguageLike.Formula (L := L), T φ → LanguageLike.Sat (L := L) M φ) ∧
      OmitsLike (L := L) M p := by
  have hsat' : SatisfiableLike (L := L) T' :=
    finite_satisfiable_compact T' S hcompact hfin
  have hsatT : SatisfiableLike (L := L) T :=
    complete_theory_model_exists T T' hincl hsat'
  have homitModel :
      ∃ M : LanguageLike.Model (L := L),
        (∀ φ : LanguageLike.Formula (L := L), T' φ → LanguageLike.Sat (L := L) M φ) ∧
        OmitsLike (L := L) M p :=
    omitting_types_step T' p hsat' homit
  rcases homitModel with ⟨M, hM' , hOmits⟩
  have hM : ∀ φ : LanguageLike.Formula (L := L), T φ → LanguageLike.Sat (L := L) M φ := by
    intro φ hφ
    exact hM' φ (hincl φ hφ)
  have _ : SatisfiableLike (L := L) T := hsatT
  exact ⟨M, hM, hOmits⟩
