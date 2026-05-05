(*
BENCHMARK_ID: TINY_MATHLIB_BATCH05_COMBINATORICS_HYPERGRAPH_CONTAINER_AXIOMATIC_LIKE
PAIR_STEM: combinatorics_hypergraph_container_axiomatic_like
MATH_DOMAIN: Combinatorics
SOURCE_MATHLIB: Mathlib/Combinatorics/Extremal
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class CombStruct_hypergraph_container (V : Type) := {
  Hypergraph : Type;
  entropy : Hypergraph -> nat;
  density : Hypergraph -> nat;
  regularity : Hypergraph -> nat;
  container : Hypergraph -> Hypergraph;
  partition : Hypergraph -> Hypergraph;
  container_step_axiom : forall H : Hypergraph,
    density (container H) <= density H;
  entropy_step_axiom : forall H : Hypergraph,
    entropy (container H) + density (container H) <= entropy H + density H;
  flag_density_axiom : forall H : Hypergraph,
    density (partition H) <= density (container H);
  sparse_regularity_axiom : forall H : Hypergraph,
    regularity (partition H) <= regularity H + density H;
  tverberg_axiom : forall H : Hypergraph,
    regularity (partition (container H)) <= regularity H + density H;
  counting_axiom : forall H : Hypergraph,
    entropy (partition H) <= entropy H + regularity H;
  extremal_axiom : forall H : Hypergraph,
    density H = 0 -> entropy H <= regularity H;
  add_mono_axiom : forall a b c d : nat, a <= b -> c <= d -> a + c <= b + d
}.

Definition HypergraphObj_hypergraph_container {V : Type}
    (C : CombStruct_hypergraph_container V) : Type :=
  Hypergraph.

Definition EntropyObj_hypergraph_container {V : Type}
    (C : CombStruct_hypergraph_container V)
    (H : HypergraphObj_hypergraph_container C) : nat :=
  entropy H.

Definition DensityObj_hypergraph_container {V : Type}
    (C : CombStruct_hypergraph_container V)
    (H : HypergraphObj_hypergraph_container C) : nat :=
  density H.

Definition RegularityObj_hypergraph_container {V : Type}
    (C : CombStruct_hypergraph_container V)
    (H : HypergraphObj_hypergraph_container C) : nat :=
  regularity H.

Lemma container_step_hypergraph_container {V : Type}
    (C : CombStruct_hypergraph_container V)
    (H : HypergraphObj_hypergraph_container C) :
    DensityObj_hypergraph_container C (container H) <= DensityObj_hypergraph_container C H.
Proof.
  assert (hCore : density (container H) <= density H).
  { apply container_step_axiom. }
  change (density (container H) <= density H).
  exact hCore.
Qed.

Lemma entropy_lemma_step_hypergraph_container {V : Type}
    (C : CombStruct_hypergraph_container V)
    (H : HypergraphObj_hypergraph_container C) :
    EntropyObj_hypergraph_container C (container H) +
      DensityObj_hypergraph_container C (container H)
      <= EntropyObj_hypergraph_container C H + DensityObj_hypergraph_container C H.
Proof.
  assert (hEntropy : entropy (container H) + density (container H)
      <= entropy H + density H).
  { apply entropy_step_axiom. }
  change (entropy (container H) + density (container H)
      <= entropy H + density H).
  exact hEntropy.
Qed.

Lemma flag_density_step_hypergraph_container {V : Type}
    (C : CombStruct_hypergraph_container V)
    (H : HypergraphObj_hypergraph_container C) :
    DensityObj_hypergraph_container C (partition H)
      <= DensityObj_hypergraph_container C (container H).
Proof.
  assert (hFlag : density (partition H) <= density (container H)).
  { apply flag_density_axiom. }
  change (density (partition H) <= density (container H)).
  exact hFlag.
Qed.

Lemma sparse_regularity_step_hypergraph_container {V : Type}
    (C : CombStruct_hypergraph_container V)
    (H : HypergraphObj_hypergraph_container C) :
    RegularityObj_hypergraph_container C (partition H)
      <= RegularityObj_hypergraph_container C H + DensityObj_hypergraph_container C H.
Proof.
  assert (hSparse : regularity (partition H) <= regularity H + density H).
  { apply sparse_regularity_axiom. }
  change (regularity (partition H) <= regularity H + density H).
  exact hSparse.
Qed.

Lemma tverberg_partition_step_hypergraph_container {V : Type}
    (C : CombStruct_hypergraph_container V)
    (H : HypergraphObj_hypergraph_container C) :
    RegularityObj_hypergraph_container C (partition (container H))
      <= RegularityObj_hypergraph_container C H + DensityObj_hypergraph_container C H.
Proof.
  assert (hTv : regularity (partition (container H)) <= regularity H + density H).
  { apply tverberg_axiom. }
  change (regularity (partition (container H)) <= regularity H + density H).
  exact hTv.
Qed.

Lemma counting_upgrade_hypergraph_container {V : Type}
    (C : CombStruct_hypergraph_container V)
    (H : HypergraphObj_hypergraph_container C) :
    EntropyObj_hypergraph_container C (partition H)
      + DensityObj_hypergraph_container C (partition H)
      <= EntropyObj_hypergraph_container C H
        + RegularityObj_hypergraph_container C H
        + DensityObj_hypergraph_container C (container H).
Proof.
  assert (hCount : entropy (partition H) <= entropy H + regularity H).
  { apply counting_axiom. }
  assert (hFlag : density (partition H) <= density (container H)).
  { apply flag_density_axiom. }
  assert (hAdd : entropy (partition H) + density (partition H)
      <= (entropy H + regularity H) + density (container H)).
  {
    apply (add_mono_axiom (a := entropy (partition H))
      (b := entropy H + regularity H)
      (c := density (partition H))
      (d := density (container H)));
    assumption.
  }
  change (entropy (partition H) + density (partition H)
      <= entropy H + regularity H + density (container H)).
  exact hAdd.
Qed.

Lemma extremal_conclusion_hypergraph_container {V : Type}
    (C : CombStruct_hypergraph_container V)
    (H : HypergraphObj_hypergraph_container C)
    (hzero : DensityObj_hypergraph_container C (container H) = 0) :
    EntropyObj_hypergraph_container C (container H)
      <= RegularityObj_hypergraph_container C (container H).
Proof.
  assert (hExt : entropy (container H) <= regularity (container H)).
  { apply extremal_axiom. exact hzero. }
  assert (hEntropy : EntropyObj_hypergraph_container C (container H) = entropy (container H)).
  { reflexivity. }
  assert (hRegularity : RegularityObj_hypergraph_container C (container H) = regularity (container H)).
  { reflexivity. }
  rewrite hEntropy.
  rewrite hRegularity.
  exact hExt.
Qed.
