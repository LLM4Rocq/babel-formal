/-
BENCHMARK_ID: TINY_MATHLIB_BATCH04_COMB_SZEMEREDI_REGULARITY_AXIOMATIC_LIKE
PAIR_STEM: combinatorics_szemeredi_regularity_axiomatic_like
MATH_DOMAIN: Combinatorics
SOURCE_MATHLIB: Mathlib/Combinatorics/SimpleGraph/Regularity
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class FiniteGraphLike (V : Type u) where
  edgeDensity : (V → Prop) → (V → Prop) → Nat
  energy : (Nat → V → Prop) → Nat
  regular : Nat → (V → Prop) → (V → Prop) → Prop
  refinement_energy_mono :
    ∀ P Q : Nat → V → Prop,
      (∀ j : Nat, ∀ v : V, Q j v → ∃ i : Nat, P i v) →
      energy P ≤ energy Q
  nonregular_witness :
    ∀ ε : Nat, ∀ P : Nat → V → Prop,
      (¬ (∀ i j : Nat, regular ε (P i) (P j))) →
      ∃ i j : Nat, ¬ regular ε (P i) (P j)
  refinement_step :
    ∀ ε : Nat, ∀ P : Nat → V → Prop, ∀ i j : Nat,
      ¬ regular ε (P i) (P j) →
      ∃ Q : Nat → V → Prop,
        (∀ k : Nat, ∀ v : V, Q k v → ∃ t : Nat, P t v) ∧
        energy P + 1 ≤ energy Q
  bounded_iteration_axiom :
    ∀ M start : Nat,
      (∀ n : Nat, n < M → ∃ P : Nat → V → Prop, start + n ≤ energy P) →
      ∃ n : Nat, n ≤ M
  regular_partition_axiom :
    ∀ ε : Nat, ∀ P0 : Nat → V → Prop,
      ∃ P : Nat → V → Prop,
        (∀ j : Nat, ∀ v : V, P j v → ∃ i : Nat, P0 i v) ∧
        (∀ i j : Nat, regular ε (P i) (P j))
  counting_axiom :
    ∀ ε : Nat, ∀ P : Nat → V → Prop, ∀ A B : V → Prop,
      regular ε A B →
      ∃ m : Nat, energy P ≤ m ∧ edgeDensity A B ≤ m

def EdgeDensityLike {V : Type u} [FiniteGraphLike V]
    (A B : V → Prop) : Nat :=
  FiniteGraphLike.edgeDensity A B

def PartitionLike {V : Type u} [FiniteGraphLike V]
    (P : Nat → V → Prop) : Prop :=
  ∀ v : V, ∃ i : Nat, P i v

def RegularPairLike {V : Type u} [FiniteGraphLike V]
    (ε : Nat) (A B : V → Prop) : Prop :=
  FiniteGraphLike.regular ε A B

def EnergyLike {V : Type u} [FiniteGraphLike V]
    (P : Nat → V → Prop) : Nat :=
  FiniteGraphLike.energy P

def RefinementLike {V : Type u} [FiniteGraphLike V]
    (P Q : Nat → V → Prop) : Prop :=
  ∀ j : Nat, ∀ v : V, Q j v → ∃ i : Nat, P i v

theorem energy_monotone_refinement {V : Type u} [FiniteGraphLike V]
    (P Q : Nat → V → Prop)
    (hRef : RefinementLike P Q) :
    EnergyLike P ≤ EnergyLike Q := by
  have hMono :
      FiniteGraphLike.energy P ≤ FiniteGraphLike.energy Q :=
    FiniteGraphLike.refinement_energy_mono (P := P) (Q := Q) hRef
  have hLeft : EnergyLike P = FiniteGraphLike.energy P := rfl
  have hRight : EnergyLike Q = FiniteGraphLike.energy Q := rfl
  rw [hLeft, hRight]
  exact hMono

theorem irregular_pair_witness {V : Type u} [FiniteGraphLike V]
    (ε : Nat) (P : Nat → V → Prop)
    (hNotRegular : ¬ (∀ i j : Nat, RegularPairLike ε (P i) (P j))) :
    ∃ i j : Nat, ¬ RegularPairLike ε (P i) (P j) := by
  have hWitness :
      ∃ i j : Nat, ¬ FiniteGraphLike.regular ε (P i) (P j) :=
    FiniteGraphLike.nonregular_witness ε P hNotRegular
  rcases hWitness with ⟨i, j, hij⟩
  refine ⟨i, j, ?_⟩
  exact hij

theorem energy_increment_step {V : Type u} [FiniteGraphLike V]
    (ε : Nat) (P : Nat → V → Prop)
    (hIrreg : ∃ i j : Nat, ¬ RegularPairLike ε (P i) (P j)) :
    ∃ Q : Nat → V → Prop,
      RefinementLike P Q ∧ EnergyLike P + 1 ≤ EnergyLike Q := by
  rcases hIrreg with ⟨i, j, hij⟩
  have hStep :
      ∃ Q : Nat → V → Prop,
        (∀ k : Nat, ∀ v : V, Q k v → ∃ t : Nat, P t v) ∧
        FiniteGraphLike.energy P + 1 ≤ FiniteGraphLike.energy Q :=
    FiniteGraphLike.refinement_step ε P i j hij
  rcases hStep with ⟨Q, hRefQ, hIncQ⟩
  have hELeft : EnergyLike P = FiniteGraphLike.energy P := rfl
  have hERight : EnergyLike Q = FiniteGraphLike.energy Q := rfl
  refine ⟨Q, ?_⟩
  constructor
  · exact hRefQ
  · rw [hELeft, hERight]
    exact hIncQ

theorem bounded_iteration_step {V : Type u} [FiniteGraphLike V]
    (M start : Nat)
    (hIter : ∀ n : Nat, n < M → ∃ P : Nat → V → Prop, start + n ≤ EnergyLike P) :
    ∃ n : Nat, n ≤ M := by
  have hLifted :
      ∀ n : Nat, n < M → ∃ P : Nat → V → Prop, start + n ≤ FiniteGraphLike.energy P := by
    intro n hn
    rcases hIter n hn with ⟨P, hP⟩
    have hEq : EnergyLike P = FiniteGraphLike.energy P := rfl
    rw [hEq] at hP
    exact ⟨P, hP⟩
  have hBound : ∃ n : Nat, n ≤ M :=
    FiniteGraphLike.bounded_iteration_axiom M start hLifted
  rcases hBound with ⟨n, hn⟩
  exact ⟨n, hn⟩

theorem regular_partition_exists_like {V : Type u} [FiniteGraphLike V]
    (ε : Nat) (P0 : Nat → V → Prop) :
    ∃ P : Nat → V → Prop,
      RefinementLike P0 P ∧ ∀ i j : Nat, RegularPairLike ε (P i) (P j) := by
  rcases FiniteGraphLike.regular_partition_axiom ε P0 with ⟨P, hRef, hReg⟩
  have hReg' : ∀ i j : Nat, RegularPairLike ε (P i) (P j) := by
    intro i j
    have hij : FiniteGraphLike.regular ε (P i) (P j) := hReg i j
    exact hij
  exact ⟨P, hRef, hReg'⟩

theorem counting_lemma_interface {V : Type u} [FiniteGraphLike V]
    (ε : Nat) (P : Nat → V → Prop) (A B : V → Prop)
    (hReg : RegularPairLike ε A B) :
    ∃ m : Nat, EnergyLike P ≤ m ∧ EdgeDensityLike A B ≤ m := by
  have hCount :
      ∃ m : Nat,
        FiniteGraphLike.energy P ≤ m ∧ FiniteGraphLike.edgeDensity A B ≤ m :=
    FiniteGraphLike.counting_axiom ε P A B hReg
  rcases hCount with ⟨m, hEnergy, hDensity⟩
  have hEP : EnergyLike P = FiniteGraphLike.energy P := rfl
  have hDAB : EdgeDensityLike A B = FiniteGraphLike.edgeDensity A B := rfl
  refine ⟨m, ?_⟩
  constructor
  · rw [hEP]
    exact hEnergy
  · rw [hDAB]
    exact hDensity
