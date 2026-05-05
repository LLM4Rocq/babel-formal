(***
BENCHMARK_ID: TINY_MATHLIB_BATCH04_COMB_SZEMEREDI_REGULARITY_AXIOMATIC_LIKE
PAIR_STEM: combinatorics_szemeredi_regularity_axiomatic_like
MATH_DOMAIN: Combinatorics
SOURCE_MATHLIB: Mathlib/Combinatorics/SimpleGraph/Regularity
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
***)

Set Universe Polymorphism.
Set Implicit Arguments.

Class FiniteGraphLike (V : Type) := {
  edgeDensity : (V -> Prop) -> (V -> Prop) -> nat;
  energy : (nat -> V -> Prop) -> nat;
  regular : nat -> (V -> Prop) -> (V -> Prop) -> Prop;
  refinement_energy_mono :
    forall P Q : nat -> V -> Prop,
      (forall j : nat, forall v : V, Q j v -> exists i : nat, P i v) ->
      energy P <= energy Q;
  nonregular_witness :
    forall eps : nat, forall P : nat -> V -> Prop,
      (~ (forall i j : nat, regular eps (P i) (P j))) ->
      exists i j : nat, ~ regular eps (P i) (P j);
  refinement_step :
    forall eps : nat, forall P : nat -> V -> Prop, forall i j : nat,
      ~ regular eps (P i) (P j) ->
      exists Q : nat -> V -> Prop,
        (forall k : nat, forall v : V, Q k v -> exists t : nat, P t v) /\
        energy P + 1 <= energy Q;
  bounded_iteration_axiom :
    forall M start : nat,
      (forall n : nat, n < M -> exists P : nat -> V -> Prop, start + n <= energy P) ->
      exists n : nat, n <= M;
  regular_partition_axiom :
    forall eps : nat, forall P0 : nat -> V -> Prop,
      exists P : nat -> V -> Prop,
        (forall j : nat, forall v : V, P j v -> exists i : nat, P0 i v) /\
        (forall i j : nat, regular eps (P i) (P j));
  counting_axiom :
    forall eps : nat, forall P : nat -> V -> Prop, forall A B : V -> Prop,
      regular eps A B ->
      exists m : nat, energy P <= m /\ edgeDensity A B <= m
}.

Definition EdgeDensityLike {V : Type} `{FiniteGraphLike V}
    (A B : V -> Prop) : nat :=
  edgeDensity A B.

Definition PartitionLike {V : Type} `{FiniteGraphLike V}
    (P : nat -> V -> Prop) : Prop :=
  forall v : V, exists i : nat, P i v.

Definition RegularPairLike {V : Type} `{FiniteGraphLike V}
    (eps : nat) (A B : V -> Prop) : Prop :=
  regular eps A B.

Definition EnergyLike {V : Type} `{FiniteGraphLike V}
    (P : nat -> V -> Prop) : nat :=
  energy P.

Definition RefinementLike {V : Type} `{FiniteGraphLike V}
    (P Q : nat -> V -> Prop) : Prop :=
  forall j : nat, forall v : V, Q j v -> exists i : nat, P i v.

Lemma energy_monotone_refinement {V : Type} `{FiniteGraphLike V}
    (P Q : nat -> V -> Prop)
    (hRef : RefinementLike P Q) :
    EnergyLike P <= EnergyLike Q.
Proof.
  assert (hMono : energy P <= energy Q).
  { apply (refinement_energy_mono P Q hRef). }
  assert (hLeft : EnergyLike P = energy P).
  { reflexivity. }
  assert (hRight : EnergyLike Q = energy Q).
  { reflexivity. }
  rewrite hLeft.
  rewrite hRight.
  exact hMono.
Qed.

Lemma irregular_pair_witness {V : Type} `{FiniteGraphLike V}
    (eps : nat) (P : nat -> V -> Prop)
    (hNotRegular : ~ (forall i j : nat, RegularPairLike eps (P i) (P j))) :
    exists i j : nat, ~ RegularPairLike eps (P i) (P j).
Proof.
  assert (hWitness : exists i j : nat, ~ regular eps (P i) (P j)).
  { apply (nonregular_witness eps P hNotRegular). }
  destruct hWitness as [i [j hij]].
  exists i.
  exists j.
  exact hij.
Qed.

Lemma energy_increment_step {V : Type} `{FiniteGraphLike V}
    (eps : nat) (P : nat -> V -> Prop)
    (hIrreg : exists i j : nat, ~ RegularPairLike eps (P i) (P j)) :
    exists Q : nat -> V -> Prop,
      RefinementLike P Q /\ EnergyLike P + 1 <= EnergyLike Q.
Proof.
  destruct hIrreg as [i [j hij]].
  assert (hStep :
      exists Q : nat -> V -> Prop,
        (forall k : nat, forall v : V, Q k v -> exists t : nat, P t v) /\
        energy P + 1 <= energy Q).
  { apply (refinement_step eps P i j hij). }
  destruct hStep as [Q [hRefQ hIncQ]].
  assert (hELeft : EnergyLike P = energy P).
  { reflexivity. }
  assert (hERight : EnergyLike Q = energy Q).
  { reflexivity. }
  exists Q.
  split.
  - exact hRefQ.
  - rewrite hELeft.
    rewrite hERight.
    exact hIncQ.
Qed.

Lemma bounded_iteration_step {V : Type} `{FiniteGraphLike V}
    (M start : nat)
    (hIter : forall n : nat, n < M -> exists P : nat -> V -> Prop, start + n <= EnergyLike P) :
    exists n : nat, n <= M.
Proof.
  assert (hLifted :
      forall n : nat, n < M -> exists P : nat -> V -> Prop, start + n <= energy P).
  {
    intros n hn.
    destruct (hIter n hn) as [P hP].
    assert (hEq : EnergyLike P = energy P).
    { reflexivity. }
    rewrite hEq in hP.
    exists P.
    exact hP.
  }
  assert (hBound : exists n : nat, n <= M).
  { apply (@bounded_iteration_axiom V _ M start hLifted). }
  destruct hBound as [n hn].
  exists n.
  exact hn.
Qed.

Lemma regular_partition_exists_like {V : Type} `{FiniteGraphLike V}
    (eps : nat) (P0 : nat -> V -> Prop) :
    exists P : nat -> V -> Prop,
      RefinementLike P0 P /\ forall i j : nat, RegularPairLike eps (P i) (P j).
Proof.
  destruct (regular_partition_axiom eps P0) as [P [hRef hReg]].
  assert (hReg' : forall i j : nat, RegularPairLike eps (P i) (P j)).
  {
    intros i j.
    assert (hij : regular eps (P i) (P j)).
    { apply (hReg i j). }
    exact hij.
  }
  exists P.
  split.
  - exact hRef.
  - exact hReg'.
Qed.

Lemma counting_lemma_interface {V : Type} `{FiniteGraphLike V}
    (eps : nat) (P : nat -> V -> Prop) (A B : V -> Prop)
    (hReg : RegularPairLike eps A B) :
    exists m : nat, EnergyLike P <= m /\ EdgeDensityLike A B <= m.
Proof.
  assert (hCount :
      exists m : nat,
        energy P <= m /\ edgeDensity A B <= m).
  { apply (counting_axiom eps P A B hReg). }
  destruct hCount as [m [hEnergy hDensity]].
  assert (hEP : EnergyLike P = energy P).
  { reflexivity. }
  assert (hDAB : EdgeDensityLike A B = edgeDensity A B).
  { reflexivity. }
  exists m.
  split.
  - rewrite hEP.
    exact hEnergy.
  - rewrite hDAB.
    exact hDensity.
Qed.
