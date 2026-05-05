/-
BENCHMARK_ID: TINY_MATHLIB_BATCH05_COMBINATORICS_HYPERGRAPH_CONTAINER_AXIOMATIC_LIKE
PAIR_STEM: combinatorics_hypergraph_container_axiomatic_like
MATH_DOMAIN: Combinatorics
SOURCE_MATHLIB: Mathlib/Combinatorics/Extremal
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u v

class CombStruct_hypergraph_container (V : Type u) where
  Hypergraph : Type v
  entropy : Hypergraph → Nat
  density : Hypergraph → Nat
  regularity : Hypergraph → Nat
  container : Hypergraph → Hypergraph
  partition : Hypergraph → Hypergraph
  container_step_axiom : ∀ H : Hypergraph,
    density (container H) ≤ density H
  entropy_step_axiom : ∀ H : Hypergraph,
    entropy (container H) + density (container H) ≤ entropy H + density H
  flag_density_axiom : ∀ H : Hypergraph,
    density (partition H) ≤ density (container H)
  sparse_regularity_axiom : ∀ H : Hypergraph,
    regularity (partition H) ≤ regularity H + density H
  tverberg_axiom : ∀ H : Hypergraph,
    regularity (partition (container H)) ≤ regularity H + density H
  counting_axiom : ∀ H : Hypergraph,
    entropy (partition H) ≤ entropy H + regularity H
  extremal_axiom : ∀ H : Hypergraph,
    density H = 0 → entropy H ≤ regularity H
  add_mono_axiom : ∀ a b c d : Nat, a ≤ b → c ≤ d → a + c ≤ b + d

def HypergraphObj_hypergraph_container {V : Type u}
    (C : CombStruct_hypergraph_container V) : Type v :=
  C.Hypergraph

def EntropyObj_hypergraph_container {V : Type u}
    (C : CombStruct_hypergraph_container V)
    (H : HypergraphObj_hypergraph_container C) : Nat :=
  C.entropy H

def DensityObj_hypergraph_container {V : Type u}
    (C : CombStruct_hypergraph_container V)
    (H : HypergraphObj_hypergraph_container C) : Nat :=
  C.density H

def RegularityObj_hypergraph_container {V : Type u}
    (C : CombStruct_hypergraph_container V)
    (H : HypergraphObj_hypergraph_container C) : Nat :=
  C.regularity H

theorem container_step_hypergraph_container {V : Type u}
    (C : CombStruct_hypergraph_container V)
    (H : HypergraphObj_hypergraph_container C) :
    DensityObj_hypergraph_container C (C.container H) ≤ DensityObj_hypergraph_container C H := by
  have hCore : C.density (C.container H) ≤ C.density H := C.container_step_axiom H
  change C.density (C.container H) ≤ C.density H
  exact hCore

theorem entropy_lemma_step_hypergraph_container {V : Type u}
    (C : CombStruct_hypergraph_container V)
    (H : HypergraphObj_hypergraph_container C) :
    EntropyObj_hypergraph_container C (C.container H) +
      DensityObj_hypergraph_container C (C.container H)
      ≤ EntropyObj_hypergraph_container C H + DensityObj_hypergraph_container C H := by
  have hEntropy : C.entropy (C.container H) + C.density (C.container H)
      ≤ C.entropy H + C.density H := C.entropy_step_axiom H
  change C.entropy (C.container H) + C.density (C.container H)
      ≤ C.entropy H + C.density H
  exact hEntropy

theorem flag_density_step_hypergraph_container {V : Type u}
    (C : CombStruct_hypergraph_container V)
    (H : HypergraphObj_hypergraph_container C) :
    DensityObj_hypergraph_container C (C.partition H)
      ≤ DensityObj_hypergraph_container C (C.container H) := by
  have hFlag : C.density (C.partition H) ≤ C.density (C.container H) := C.flag_density_axiom H
  change C.density (C.partition H) ≤ C.density (C.container H)
  exact hFlag

theorem sparse_regularity_step_hypergraph_container {V : Type u}
    (C : CombStruct_hypergraph_container V)
    (H : HypergraphObj_hypergraph_container C) :
    RegularityObj_hypergraph_container C (C.partition H)
      ≤ RegularityObj_hypergraph_container C H + DensityObj_hypergraph_container C H := by
  have hSparse : C.regularity (C.partition H) ≤ C.regularity H + C.density H :=
    C.sparse_regularity_axiom H
  change C.regularity (C.partition H) ≤ C.regularity H + C.density H
  exact hSparse

theorem tverberg_partition_step_hypergraph_container {V : Type u}
    (C : CombStruct_hypergraph_container V)
    (H : HypergraphObj_hypergraph_container C) :
    RegularityObj_hypergraph_container C (C.partition (C.container H))
      ≤ RegularityObj_hypergraph_container C H + DensityObj_hypergraph_container C H := by
  have hTv : C.regularity (C.partition (C.container H)) ≤ C.regularity H + C.density H :=
    C.tverberg_axiom H
  change C.regularity (C.partition (C.container H)) ≤ C.regularity H + C.density H
  exact hTv

theorem counting_upgrade_hypergraph_container {V : Type u}
    (C : CombStruct_hypergraph_container V)
    (H : HypergraphObj_hypergraph_container C) :
    EntropyObj_hypergraph_container C (C.partition H)
      + DensityObj_hypergraph_container C (C.partition H)
      ≤ EntropyObj_hypergraph_container C H
        + RegularityObj_hypergraph_container C H
        + DensityObj_hypergraph_container C (C.container H) := by
  have hCount : C.entropy (C.partition H) ≤ C.entropy H + C.regularity H :=
    C.counting_axiom H
  have hFlag : C.density (C.partition H) ≤ C.density (C.container H) :=
    C.flag_density_axiom H
  have hAdd : C.entropy (C.partition H) + C.density (C.partition H)
      ≤ (C.entropy H + C.regularity H) + C.density (C.container H) :=
    C.add_mono_axiom
      (C.entropy (C.partition H))
      (C.entropy H + C.regularity H)
      (C.density (C.partition H))
      (C.density (C.container H))
      hCount hFlag
  have hAssoc : (C.entropy H + C.regularity H) + C.density (C.container H)
      = C.entropy H + C.regularity H + C.density (C.container H) := by
    rw [Nat.add_assoc]
  calc
    EntropyObj_hypergraph_container C (C.partition H)
        + DensityObj_hypergraph_container C (C.partition H)
        = C.entropy (C.partition H) + C.density (C.partition H) := by
          rfl
    _ ≤ (C.entropy H + C.regularity H) + C.density (C.container H) := hAdd
    _ = C.entropy H + C.regularity H + C.density (C.container H) := hAssoc
    _ = EntropyObj_hypergraph_container C H
          + RegularityObj_hypergraph_container C H
          + DensityObj_hypergraph_container C (C.container H) := by
          rfl

theorem extremal_conclusion_hypergraph_container {V : Type u}
    (C : CombStruct_hypergraph_container V)
    (H : HypergraphObj_hypergraph_container C)
    (hzero : DensityObj_hypergraph_container C (C.container H) = 0) :
    EntropyObj_hypergraph_container C (C.container H)
      ≤ RegularityObj_hypergraph_container C (C.container H) := by
  have hExt : C.entropy (C.container H) ≤ C.regularity (C.container H) :=
    C.extremal_axiom (C.container H) hzero
  have hEntropy : EntropyObj_hypergraph_container C (C.container H) = C.entropy (C.container H) := by
    rfl
  have hRegularity : RegularityObj_hypergraph_container C (C.container H) = C.regularity (C.container H) := by
    rfl
  calc
    EntropyObj_hypergraph_container C (C.container H) = C.entropy (C.container H) := hEntropy
    _ ≤ C.regularity (C.container H) := hExt
    _ = RegularityObj_hypergraph_container C (C.container H) := by
      symm
      exact hRegularity
