/-
BENCHMARK_ID: TINY_MATHLIB_BATCH04_ANALYSIS_SOBOLEV_COMPACT_EMBEDDING_AXIOMATIC_LIKE
PAIR_STEM: analysis_sobolev_compact_embedding_axiomatic_like
MATH_DOMAIN: Analysis / PDE
SOURCE_MATHLIB: Mathlib/Analysis/NormedSpace/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class NormedSpaceLike (E : Type u) where
  norm : E → Nat

def SobolevLike {E : Type u} [NormedSpaceLike E] (u : E) : Prop :=
  ∃ C : Nat, NormedSpaceLike.norm u ≤ C

def WeakConvergenceLike {E : Type u} [NormedSpaceLike E] (seq : Nat → E) (u : E) : Prop :=
  ∀ ε : Nat, ∃ N : Nat, ∀ n : Nat, N ≤ n → NormedSpaceLike.norm (seq n) ≤ NormedSpaceLike.norm u + ε

def RelCompactLike {E : Type u} [NormedSpaceLike E] (A : (Nat → E) → Prop) : Prop :=
  ∀ seq : Nat → E, A seq →
    ∃ φ : Nat → Nat, (∀ n : Nat, n ≤ φ n) ∧ ∃ u : E, WeakConvergenceLike (fun n => seq (φ n)) u

def EmbeddingLike {E : Type u} [NormedSpaceLike E] (T : E → E) : Prop :=
  ∀ u : E, SobolevLike u → SobolevLike (T u)

def BoundedSequenceLike {E : Type u} [NormedSpaceLike E] (seq : Nat → E) : Prop :=
  ∃ C : Nat, ∀ n : Nat, NormedSpaceLike.norm (seq n) ≤ C

theorem reflexive_subsequence_step {E : Type u} [NormedSpaceLike E]
    (A : (Nat → E) → Prop)
    (hrel : RelCompactLike A)
    (seq : Nat → E)
    (hA : A seq) :
    ∃ φ : Nat → Nat, ∃ u : E,
      (∀ n : Nat, n ≤ φ n) ∧ WeakConvergenceLike (fun n => seq (φ n)) u := by
  have hextract : ∃ φ : Nat → Nat, (∀ n : Nat, n ≤ φ n) ∧ ∃ u : E, WeakConvergenceLike (fun n => seq (φ n)) u :=
    hrel seq hA
  rcases hextract with ⟨φ, hφ, hrest⟩
  rcases hrest with ⟨u, hweak⟩
  have hmono : ∀ n : Nat, n ≤ φ n := by
    intro n
    exact hφ n
  have hdiag : WeakConvergenceLike (fun n => seq (φ n)) u := hweak
  exact ⟨φ, u, hmono, hdiag⟩

theorem compactness_on_bounded_sets {E : Type u} [NormedSpaceLike E]
    (T : E → E)
    (A : (Nat → E) → Prop)
    (seq : Nat → E)
    (hemb : EmbeddingLike T)
    (hrel : RelCompactLike A)
    (hclosed : ∀ s : Nat → E, A s → A (fun n => T (s n)))
    (hA : A seq)
    (hsob : ∀ n : Nat, SobolevLike (seq n)) :
    ∃ φ : Nat → Nat, ∃ u : E,
      (∀ n : Nat, n ≤ φ n) ∧ WeakConvergenceLike (fun n => T (seq (φ n))) u := by
  have hAimage : A (fun n => T (seq n)) := hclosed seq hA
  have hsobImage : ∀ n : Nat, SobolevLike (T (seq n)) := by
    intro n
    exact hemb (seq n) (hsob n)
  have hextract : ∃ φ : Nat → Nat, (∀ n : Nat, n ≤ φ n) ∧ ∃ u : E, WeakConvergenceLike (fun n => T (seq (φ n))) u :=
    hrel (fun n => T (seq n)) hAimage
  rcases hextract with ⟨φ, hφ, hrest⟩
  rcases hrest with ⟨u, hweak⟩
  have hs0 : SobolevLike (T (seq (φ 0))) := hsobImage (φ 0)
  have _ : SobolevLike (T (seq (φ 0))) := hs0
  exact ⟨φ, u, hφ, hweak⟩

theorem rellich_step_like {E : Type u} [NormedSpaceLike E]
    (A : (Nat → E) → Prop)
    (seq : Nat → E)
    (hbounded : BoundedSequenceLike seq)
    (hbridge : ∀ s : Nat → E, BoundedSequenceLike s → A s)
    (hrel : RelCompactLike A) :
    ∃ φ : Nat → Nat, ∃ u : E,
      (∀ n : Nat, n ≤ φ n) ∧ WeakConvergenceLike (fun n => seq (φ n)) u := by
  have hA : A seq := hbridge seq hbounded
  have hstep :
      ∃ φ : Nat → Nat, ∃ u : E,
        (∀ n : Nat, n ≤ φ n) ∧ WeakConvergenceLike (fun n => seq (φ n)) u :=
    reflexive_subsequence_step A hrel seq hA
  rcases hstep with ⟨φ, u, hpair⟩
  rcases hpair with ⟨hmono, hweak⟩
  have hpack : WeakConvergenceLike (fun n => seq (φ n)) u := hweak
  exact ⟨φ, u, hmono, hpack⟩

theorem compact_embedding_core {E : Type u} [NormedSpaceLike E]
    (T : E → E)
    (A : (Nat → E) → Prop)
    (seq : Nat → E)
    (hemb : EmbeddingLike T)
    (hbounded : BoundedSequenceLike seq)
    (hbridge : ∀ s : Nat → E, BoundedSequenceLike s → A s)
    (hrel : RelCompactLike A)
    (hclosed : ∀ s : Nat → E, A s → A (fun n => T (s n)))
    (hsob : ∀ n : Nat, SobolevLike (seq n)) :
    ∃ φ : Nat → Nat, ∃ u : E,
      (∀ n : Nat, n ≤ φ n) ∧ WeakConvergenceLike (fun n => T (seq (φ n))) u := by
  have hA : A seq := hbridge seq hbounded
  have hcore :
      ∃ φ : Nat → Nat, ∃ u : E,
        (∀ n : Nat, n ≤ φ n) ∧ WeakConvergenceLike (fun n => T (seq (φ n))) u :=
    compactness_on_bounded_sets T A seq hemb hrel hclosed hA hsob
  rcases hcore with ⟨φ, u, hpair⟩
  rcases hpair with ⟨hmono, hweak⟩
  have hs0 : SobolevLike (T (seq (φ 0))) := hemb (seq (φ 0)) (hsob (φ 0))
  have _ : SobolevLike (T (seq (φ 0))) := hs0
  exact ⟨φ, u, hmono, hweak⟩

theorem strong_convergence_extraction {E : Type u} [NormedSpaceLike E]
    (seq : Nat → E)
    (u : E)
    (hweak : WeakConvergenceLike seq u)
    (hupgrade : ∀ ε : Nat, ∃ N : Nat, ∀ n : Nat, N ≤ n → NormedSpaceLike.norm (seq n) ≤ NormedSpaceLike.norm u + ε) :
    ∃ N : Nat, ∀ n : Nat, N ≤ n → NormedSpaceLike.norm (seq n) ≤ NormedSpaceLike.norm u + 1 := by
  have hweakOne : ∃ N : Nat, ∀ n : Nat, N ≤ n → NormedSpaceLike.norm (seq n) ≤ NormedSpaceLike.norm u + 1 :=
    hweak 1
  rcases hweakOne with ⟨Nw, hNw⟩
  have hupOne : ∃ N : Nat, ∀ n : Nat, N ≤ n → NormedSpaceLike.norm (seq n) ≤ NormedSpaceLike.norm u + 1 :=
    hupgrade 1
  rcases hupOne with ⟨Nu, hNu⟩
  let N := Nat.max Nw Nu
  refine ⟨N, ?_⟩
  intro n hn
  have hNu_le : Nu ≤ n := Nat.le_trans (Nat.le_max_right Nw Nu) hn
  have hNuBound : NormedSpaceLike.norm (seq n) ≤ NormedSpaceLike.norm u + 1 := hNu n hNu_le
  have hNw_le : Nw ≤ n := Nat.le_trans (Nat.le_max_left Nw Nu) hn
  have hNwBound : NormedSpaceLike.norm (seq n) ≤ NormedSpaceLike.norm u + 1 := hNw n hNw_le
  have _ : NormedSpaceLike.norm (seq n) ≤ NormedSpaceLike.norm u + 1 := hNwBound
  exact hNuBound

theorem sobolev_compact_embedding_like {E : Type u} [NormedSpaceLike E]
    (T : E → E)
    (A : (Nat → E) → Prop)
    (seq : Nat → E)
    (hemb : EmbeddingLike T)
    (hbounded : BoundedSequenceLike seq)
    (hbridge : ∀ s : Nat → E, BoundedSequenceLike s → A s)
    (hrel : RelCompactLike A)
    (hclosed : ∀ s : Nat → E, A s → A (fun n => T (s n)))
    (hsob : ∀ n : Nat, SobolevLike (seq n))
    (hupgrade :
      ∀ φ : Nat → Nat, ∀ u : E,
        WeakConvergenceLike (fun n => T (seq (φ n))) u →
          ∀ ε : Nat, ∃ N : Nat, ∀ n : Nat, N ≤ n →
            NormedSpaceLike.norm (T (seq (φ n))) ≤ NormedSpaceLike.norm u + ε) :
    ∃ φ : Nat → Nat, ∃ u : E, ∃ N : Nat,
      (∀ n : Nat, n ≤ φ n) ∧
      (∀ n : Nat, N ≤ n → NormedSpaceLike.norm (T (seq (φ n))) ≤ NormedSpaceLike.norm u + 1) := by
  have hcompact :
      ∃ φ : Nat → Nat, ∃ u : E,
        (∀ n : Nat, n ≤ φ n) ∧ WeakConvergenceLike (fun n => T (seq (φ n))) u :=
    compact_embedding_core T A seq hemb hbounded hbridge hrel hclosed hsob
  rcases hcompact with ⟨φ, u, hpair⟩
  rcases hpair with ⟨hmono, hweak⟩
  have hup : ∀ ε : Nat, ∃ N : Nat, ∀ n : Nat, N ≤ n → NormedSpaceLike.norm (T (seq (φ n))) ≤ NormedSpaceLike.norm u + ε :=
    hupgrade φ u hweak
  have hstrong :
      ∃ N : Nat, ∀ n : Nat, N ≤ n → NormedSpaceLike.norm (T (seq (φ n))) ≤ NormedSpaceLike.norm u + 1 :=
    strong_convergence_extraction (fun n => T (seq (φ n))) u hweak hup
  rcases hstrong with ⟨N, hN⟩
  exact ⟨φ, u, N, hmono, hN⟩
