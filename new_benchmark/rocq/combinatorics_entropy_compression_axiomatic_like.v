(*
BENCHMARK_ID: TINY_MATHLIB_BATCH05_COMBINATORICS_ENTROPY_COMPRESSION_AXIOMATIC_LIKE
PAIR_STEM: combinatorics_entropy_compression_axiomatic_like
MATH_DOMAIN: Combinatorics
SOURCE_MATHLIB: Mathlib/Combinatorics/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class CombStruct_entropy_compression (V : Type) := {
  weight : V -> nat;
  compress : V -> V;
  encode : nat -> nat;
  decode : nat -> nat;
  density : V -> nat;
  regularize : V -> V;
  complexity : V -> nat;
  decode_encode_axiom : forall n : nat, decode (encode n) = n;
  encode_mono_axiom : forall {a b : nat}, a <= b -> encode a <= encode b;
  compress_weight_axiom : forall v : V, weight (compress v) <= weight v;
  density_regularize_axiom : forall v : V, density (regularize v) <= density v;
  complexity_bound_axiom : forall v : V, complexity v <= weight v + density v
}.

Definition HypergraphObj_entropy_compression (V : Type) : Type :=
  V.

Definition EntropyObj_entropy_compression {V : Type}
    {CV : CombStruct_entropy_compression V} (v : V) : nat :=
  @encode V CV (@weight V CV v).

Definition DensityObj_entropy_compression {V : Type}
    {CV : CombStruct_entropy_compression V} (v : V) : nat :=
  @density V CV v.

Definition RegularityObj_entropy_compression {V : Type}
    {CV : CombStruct_entropy_compression V} (v : V) : V :=
  @regularize V CV v.

Lemma container_step_entropy_compression {V : Type}
    {CV : CombStruct_entropy_compression V}
    (v : V) :
    @weight V CV (@compress V CV v) <=
      @weight V CV v.
Proof.
  exact (compress_weight_axiom v).
Qed.

Lemma entropy_lemma_step_entropy_compression {V : Type}
    {CV : CombStruct_entropy_compression V}
    (v : V) :
    @decode V CV
      (EntropyObj_entropy_compression v) =
      @weight V CV v.
Proof.
  assert (hDecode :
      @decode V CV
          (@encode V CV (@weight V CV v)) =
        @weight V CV v).
  { apply decode_encode_axiom. }
  unfold EntropyObj_entropy_compression.
  exact hDecode.
Qed.

Lemma flag_density_step_entropy_compression {V : Type}
    {CV : CombStruct_entropy_compression V}
    (v : V) :
    DensityObj_entropy_compression (RegularityObj_entropy_compression v) <=
      DensityObj_entropy_compression v.
Proof.
  assert (hRaw :
      @density V CV (@regularize V CV v) <=
        @density V CV v).
  { apply density_regularize_axiom. }
  unfold DensityObj_entropy_compression, RegularityObj_entropy_compression.
  exact hRaw.
Qed.

Lemma sparse_regularity_step_entropy_compression {V : Type}
    {CV : CombStruct_entropy_compression V}
    (v : V) :
    @complexity V CV
        (RegularityObj_entropy_compression v) <=
      @weight V CV
        (RegularityObj_entropy_compression v) +
        DensityObj_entropy_compression v.
Proof.
  assert (hBound :
      @complexity V CV
          (@regularize V CV v) <=
        @weight V CV
          (@regularize V CV v) +
          @density V CV
            (@regularize V CV v)).
  { apply complexity_bound_axiom. }
  assert (hDensity :
      @density V CV
          (@regularize V CV v) <=
        @density V CV v).
  { apply density_regularize_axiom. }
  assert (hAdd :
      @weight V CV (@regularize V CV v) +
          @density V CV (@regularize V CV v) <=
      @weight V CV (@regularize V CV v) +
          @density V CV v).
  {
    assert (hLift : forall w a b : nat, a <= b -> w + a <= w + b).
    {
      intro w.
      induction w as [| w ih].
      - intros a b hab. simpl. exact hab.
      - intros a b hab. simpl. apply le_n_S. apply ih. exact hab.
    }
    apply (hLift (@weight V CV (@regularize V CV v))
      (@density V CV (@regularize V CV v))
      (@density V CV v)).
    exact hDensity.
  }
  assert (hTrans :
      @complexity V CV (@regularize V CV v) <=
        @weight V CV (@regularize V CV v) +
          @density V CV v).
  {
    assert (hTransNat : forall a b c : nat, a <= b -> b <= c -> a <= c).
    {
      intros a b c hab hbc.
      induction hbc.
      - exact hab.
      - apply le_S. apply IHhbc.
    }
    exact (hTransNat _ _ _ hBound hAdd).
  }
  unfold RegularityObj_entropy_compression, DensityObj_entropy_compression.
  exact hTrans.
Qed.

Lemma tverberg_partition_step_entropy_compression {V : Type}
    {CV : CombStruct_entropy_compression V}
    (v : V) :
    EntropyObj_entropy_compression
      (@compress V CV v) <=
      EntropyObj_entropy_compression v.
Proof.
  assert (hWeight :
      @weight V CV
          (@compress V CV v) <=
        @weight V CV v).
  { apply compress_weight_axiom. }
  assert (hEncode :
      @encode V CV
          (@weight V CV (@compress V CV v)) <=
        @encode V CV
          (@weight V CV v)).
  { apply encode_mono_axiom. exact hWeight. }
  unfold EntropyObj_entropy_compression.
  exact hEncode.
Qed.

Lemma counting_upgrade_entropy_compression {V : Type}
    {CV : CombStruct_entropy_compression V}
    (v : V) :
    @decode V CV
      (EntropyObj_entropy_compression
        (@compress V CV v)) <=
      @weight V CV v.
Proof.
  assert (hDecodeCompress :
      @decode V CV
          (EntropyObj_entropy_compression
            (@compress V CV v)) =
        @weight V CV
          (@compress V CV v)).
  { apply entropy_lemma_step_entropy_compression. }
  assert (hWeight :
      @weight V CV
          (@compress V CV v) <=
        @weight V CV v).
  { apply container_step_entropy_compression. }
  rewrite hDecodeCompress.
  exact hWeight.
Qed.

Lemma extremal_conclusion_entropy_compression {V : Type}
    {CV : CombStruct_entropy_compression V}
    (v : V) :
    @complexity V CV v <=
      @decode V CV
        (EntropyObj_entropy_compression v) +
        DensityObj_entropy_compression v.
Proof.
  assert (hBound :
      @complexity V CV v <=
        @weight V CV v +
          @density V CV v).
  { apply complexity_bound_axiom. }
  assert (hDecode :
      @decode V CV
          (EntropyObj_entropy_compression v) =
        @weight V CV v).
  { apply entropy_lemma_step_entropy_compression. }
  rewrite hDecode.
  unfold DensityObj_entropy_compression.
  exact hBound.
Qed.
