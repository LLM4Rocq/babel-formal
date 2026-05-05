/-
BENCHMARK_ID: TINY_MATHLIB_BATCH04_NUM_CLASS_GROUP_FINITENESS_LIKE
PAIR_STEM: number_theory_class_group_finiteness_like
MATH_DOMAIN: Algebraic Number Theory
SOURCE_MATHLIB: Mathlib/NumberTheory/ClassGroup
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u v

class DedekindDomainLike (K : Type u) where
  Ideal : Type v
  classOf : Ideal → Nat
  norm : Ideal → Nat
  principal : Ideal → Prop
  inClass : Nat → Ideal → Prop
  reduced : Nat → Ideal → Prop
  reduced_witness : ∀ c : Nat, ∃ I : Ideal, inClass c I ∧ reduced c I
  reduced_norm_bound : ∃ B : Nat, ∀ c : Nat, ∀ I : Ideal, reduced c I → norm I ≤ B
  class_of_inClass : ∀ c : Nat, ∀ I : Ideal, inClass c I → classOf I = c
  principal_class_zero : ∀ I : Ideal, principal I → classOf I = 0
  class_has_torsion : ∃ n : Nat, n > 0 ∧ ∀ c : Nat, n * c = 0 → ∃ I : Ideal, inClass c I
  finite_classes_from_norm : ∀ B : Nat, ∃ N : Nat,
    ∀ c : Nat, (∃ I : Ideal, inClass c I ∧ norm I ≤ B) → c < N

def FractionalIdealLike (K : Type u) [DedekindDomainLike K] : Type v :=
  DedekindDomainLike.Ideal (K := K)

def PrincipalLike {K : Type u} [DedekindDomainLike K]
    (I : FractionalIdealLike K) : Prop :=
  DedekindDomainLike.principal I

def ClassGroupLike (K : Type u) [DedekindDomainLike K] : Type :=
  Nat

def MinkowskiBoundLike {K : Type u} [DedekindDomainLike K]
    (B : Nat) : Prop :=
  ∀ c : ClassGroupLike K, ∀ I : FractionalIdealLike K,
    DedekindDomainLike.reduced c I → DedekindDomainLike.norm I ≤ B

def ReducedIdealLike {K : Type u} [DedekindDomainLike K]
    (c : ClassGroupLike K) (I : FractionalIdealLike K) : Prop :=
  DedekindDomainLike.reduced c I ∧ DedekindDomainLike.inClass c I

theorem reduced_ideal_exists {K : Type u} [DedekindDomainLike K]
    (c : ClassGroupLike K) :
    ∃ I : FractionalIdealLike K, ReducedIdealLike c I := by
  rcases DedekindDomainLike.reduced_witness (K := K) c with ⟨I, hIn, hRed⟩
  have hPack : ReducedIdealLike c I := by
    constructor
    · exact hRed
    · exact hIn
  exact ⟨I, hPack⟩

theorem reduced_ideal_finite_set {K : Type u} [DedekindDomainLike K] :
    ∃ B : Nat, MinkowskiBoundLike (K := K) B := by
  rcases DedekindDomainLike.reduced_norm_bound (K := K) with ⟨B, hB⟩
  refine ⟨B, ?_⟩
  intro c I hRed
  have hnorm : DedekindDomainLike.norm I ≤ B := hB c I hRed
  exact hnorm

theorem every_class_has_reduced_rep {K : Type u} [DedekindDomainLike K]
    (c : ClassGroupLike K) :
    ∃ I : FractionalIdealLike K, ReducedIdealLike c I ∧ DedekindDomainLike.classOf I = c := by
  rcases reduced_ideal_exists (K := K) c with ⟨I, hRedI⟩
  have hIn : DedekindDomainLike.inClass c I := hRedI.2
  have hClass : DedekindDomainLike.classOf I = c :=
    DedekindDomainLike.class_of_inClass (K := K) c I hIn
  refine ⟨I, ?_⟩
  constructor
  · exact hRedI
  · exact hClass

theorem class_group_generated_finitely {K : Type u} [DedekindDomainLike K] :
    ∃ B : Nat,
      ∀ c : Nat,
        ∃ I : FractionalIdealLike K,
          DedekindDomainLike.inClass c I ∧ DedekindDomainLike.norm I ≤ B := by
  rcases reduced_ideal_finite_set (K := K) with ⟨B, hB⟩
  refine ⟨B, ?_⟩
  intro c
  rcases every_class_has_reduced_rep (K := K) c with ⟨I, hReducedI, hClassI⟩
  have hIn : DedekindDomainLike.inClass c I := hReducedI.2
  have hNorm : DedekindDomainLike.norm I ≤ B := hB c I hReducedI.1
  have _ : DedekindDomainLike.classOf I = c := hClassI
  exact ⟨I, hIn, hNorm⟩

theorem class_group_torsion_like {K : Type u} [DedekindDomainLike K] :
    ∃ n : Nat,
      n > 0 ∧
      ∀ c : Nat,
        n * c = 0 → ∃ I : FractionalIdealLike K, DedekindDomainLike.inClass c I := by
  rcases DedekindDomainLike.class_has_torsion (K := K) with ⟨n, hnPos, hnKill⟩
  refine ⟨n, ?_⟩
  constructor
  · exact hnPos
  · intro c hc
    have hRep : ∃ I : FractionalIdealLike K, DedekindDomainLike.inClass c I := hnKill c hc
    rcases hRep with ⟨I, hI⟩
    exact ⟨I, hI⟩

theorem class_group_finite_like {K : Type u} [DedekindDomainLike K] :
    ∃ N : Nat, ∀ c : Nat, c < N := by
  rcases class_group_generated_finitely (K := K) with ⟨B, hGen⟩
  rcases DedekindDomainLike.finite_classes_from_norm (K := K) B with ⟨N, hN⟩
  refine ⟨N, ?_⟩
  intro c
  rcases hGen c with ⟨I, hIn, hNorm⟩
  have hWitness : ∃ J : FractionalIdealLike K, DedekindDomainLike.inClass c J ∧ DedekindDomainLike.norm J ≤ B :=
    ⟨I, hIn, hNorm⟩
  have hBound : c < N := hN c hWitness
  exact hBound
