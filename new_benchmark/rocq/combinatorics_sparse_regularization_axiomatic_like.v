(***
BENCHMARK_ID: TINY_MATHLIB_BATCH05_COMBINATORICS_SPARSE_REGULARIZATION_AXIOMATIC_LIKE
PAIR_STEM: combinatorics_sparse_regularization_axiomatic_like
MATH_DOMAIN: Combinatorics
SOURCE_MATHLIB: Mathlib/Combinatorics/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
***)

Set Universe Polymorphism.
Set Implicit Arguments.

Class CombStruct_sparse_regularization (V : Type) := {
  weight : V -> nat;
  density : (V -> nat) -> nat;
  entropy : (V -> nat) -> nat;
  regularize : (V -> nat) -> (V -> nat);
  container : (V -> nat) -> (V -> nat);
  density_mono : forall {f g : V -> nat},
      (forall v, f v <= g v) -> density f <= density g;
  entropy_nonneg : forall f : V -> nat, 0 <= entropy f;
  regularize_majorizes : forall (f : V -> nat) (v : V), f v <= regularize f v;
  regularize_idem : forall (f : V -> nat) (v : V),
      regularize (regularize f) v = regularize f v;
  container_controls : forall (f : V -> nat) (v : V),
      regularize f v <= container f v;
  counting_axiom : forall f : V -> nat,
      density (regularize f) <= density (container f) + entropy f;
  nat_le_trans_axiom : forall a b c : nat, a <= b -> b <= c -> a <= c
}.

Definition HypergraphObj_sparse_regularization {V : Type}
    `{CombStruct_sparse_regularization V} : Type :=
  V -> nat.

Definition EntropyObj_sparse_regularization {V : Type}
    `{CombStruct_sparse_regularization V}
    (f : HypergraphObj_sparse_regularization) : nat :=
  entropy f.

Definition DensityObj_sparse_regularization {V : Type}
    `{CombStruct_sparse_regularization V}
    (f : HypergraphObj_sparse_regularization) : nat :=
  density f.

Definition RegularityObj_sparse_regularization {V : Type}
    `{CombStruct_sparse_regularization V}
    (f : HypergraphObj_sparse_regularization) : HypergraphObj_sparse_regularization :=
  regularize f.

Lemma container_step_sparse_regularization {V : Type}
    `{CombStruct_sparse_regularization V}
    (f : HypergraphObj_sparse_regularization) :
    forall v : V,
      RegularityObj_sparse_regularization f v <=
        container f v.
Proof.
  intro v.
  assert (hraw : regularize f v <= container f v).
  { apply container_controls. }
  exact hraw.
Qed.

Lemma entropy_lemma_step_sparse_regularization {V : Type}
    `{CombStruct_sparse_regularization V}
    (f : HypergraphObj_sparse_regularization) :
    0 <= EntropyObj_sparse_regularization f /\
      DensityObj_sparse_regularization f <=
        DensityObj_sparse_regularization (RegularityObj_sparse_regularization f).
Proof.
  assert (hEntropy : 0 <= entropy f).
  { apply entropy_nonneg. }
  assert (hPoint : forall v : V, f v <= regularize f v).
  { apply regularize_majorizes. }
  assert (hDense : density f <= density (regularize f)).
  { apply density_mono. exact hPoint. }
  split.
  - exact hEntropy.
  - exact hDense.
Qed.

Lemma flag_density_step_sparse_regularization {V : Type}
    `{CombStruct_sparse_regularization V}
    (f : HypergraphObj_sparse_regularization) :
    forall v : V,
      f v <= RegularityObj_sparse_regularization f v.
Proof.
  intro v.
  assert (hMajor : f v <= regularize f v).
  { apply regularize_majorizes. }
  exact hMajor.
Qed.

Lemma sparse_regularity_step_sparse_regularization {V : Type}
    `{CombStruct_sparse_regularization V}
    (f : HypergraphObj_sparse_regularization) :
    forall v : V,
      f v <= RegularityObj_sparse_regularization f v /\
      RegularityObj_sparse_regularization f v <=
        container f v.
Proof.
  intro v.
  assert (hMajor : f v <= regularize f v).
  { apply regularize_majorizes. }
  assert (hCont : regularize f v <= container f v).
  { apply container_controls. }
  split.
  - exact hMajor.
  - exact hCont.
Qed.

Lemma tverberg_partition_step_sparse_regularization {V : Type}
    `{CombStruct_sparse_regularization V}
    (f : HypergraphObj_sparse_regularization) :
    forall v : V,
      RegularityObj_sparse_regularization
        (RegularityObj_sparse_regularization f) v =
      RegularityObj_sparse_regularization f v.
Proof.
  intro v.
  assert (hidem : regularize (regularize f) v = regularize f v).
  { apply regularize_idem. }
  exact hidem.
Qed.

Lemma counting_upgrade_sparse_regularization {V : Type}
    `{CombStruct_sparse_regularization V}
    (f : HypergraphObj_sparse_regularization) :
    DensityObj_sparse_regularization (RegularityObj_sparse_regularization f) <=
      DensityObj_sparse_regularization (container f) +
        EntropyObj_sparse_regularization f.
Proof.
  assert (hcount : density (regularize f) <= density (container f) + entropy f).
  { apply counting_axiom. }
  exact hcount.
Qed.

Lemma extremal_conclusion_sparse_regularization {V : Type}
    `{CombStruct_sparse_regularization V}
    (f : HypergraphObj_sparse_regularization) :
    DensityObj_sparse_regularization f <=
      DensityObj_sparse_regularization (container f) +
        EntropyObj_sparse_regularization f.
Proof.
  assert (hSeed :
      DensityObj_sparse_regularization f <=
        DensityObj_sparse_regularization (RegularityObj_sparse_regularization f)).
  { exact (proj2 (entropy_lemma_step_sparse_regularization f)). }
  assert (hCount :
      DensityObj_sparse_regularization (RegularityObj_sparse_regularization f) <=
      DensityObj_sparse_regularization (container f) +
        EntropyObj_sparse_regularization f).
  { apply counting_upgrade_sparse_regularization. }
  apply (nat_le_trans_axiom
    (a := DensityObj_sparse_regularization f)
    (b := DensityObj_sparse_regularization (RegularityObj_sparse_regularization f))
    (c := DensityObj_sparse_regularization (container f) +
      EntropyObj_sparse_regularization f));
  assumption.
Qed.
