/-
BENCHMARK_ID: TINY_MATHLIB_BATCH05_COMBINATORICS_ENTROPY_COMPRESSION_AXIOMATIC_LIKE
PAIR_STEM: combinatorics_entropy_compression_axiomatic_like
MATH_DOMAIN: Combinatorics
SOURCE_MATHLIB: Mathlib/Combinatorics/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class CombStruct_entropy_compression (V : Type u) where
  weight : V -> Nat
  compress : V -> V
  encode : Nat -> Nat
  decode : Nat -> Nat
  density : V -> Nat
  regularize : V -> V
  complexity : V -> Nat
  decode_encode_axiom : forall n : Nat, decode (encode n) = n
  encode_mono_axiom : forall {a b : Nat}, a <= b -> encode a <= encode b
  compress_weight_axiom : forall v : V, weight (compress v) <= weight v
  density_regularize_axiom : forall v : V, density (regularize v) <= density v
  complexity_bound_axiom : forall v : V, complexity v <= weight v + density v

def HypergraphObj_entropy_compression (V : Type u) : Type u :=
  V

def EntropyObj_entropy_compression {V : Type u}
    [CombStruct_entropy_compression V] (v : V) : Nat :=
  CombStruct_entropy_compression.encode (V := V)
    (CombStruct_entropy_compression.weight (V := V) v)

def DensityObj_entropy_compression {V : Type u}
    [CombStruct_entropy_compression V] (v : V) : Nat :=
  CombStruct_entropy_compression.density (V := V) v

def RegularityObj_entropy_compression {V : Type u}
    [CombStruct_entropy_compression V] (v : V) : V :=
  CombStruct_entropy_compression.regularize (V := V) v

theorem container_step_entropy_compression {V : Type u}
    [CombStruct_entropy_compression V]
    (v : V) :
    CombStruct_entropy_compression.weight (V := V)
      (CombStruct_entropy_compression.compress (V := V) v) <=
      CombStruct_entropy_compression.weight (V := V) v := by
  have hRaw :
      CombStruct_entropy_compression.weight (V := V)
          (CombStruct_entropy_compression.compress (V := V) v) <=
        CombStruct_entropy_compression.weight (V := V) v :=
    CombStruct_entropy_compression.compress_weight_axiom (V := V) v
  exact hRaw

theorem entropy_lemma_step_entropy_compression {V : Type u}
    [CombStruct_entropy_compression V]
    (v : V) :
    CombStruct_entropy_compression.decode (V := V)
      (EntropyObj_entropy_compression v) =
      CombStruct_entropy_compression.weight (V := V) v := by
  have hDecode :
      CombStruct_entropy_compression.decode (V := V)
          (CombStruct_entropy_compression.encode (V := V)
            (CombStruct_entropy_compression.weight (V := V) v)) =
        CombStruct_entropy_compression.weight (V := V) v :=
    CombStruct_entropy_compression.decode_encode_axiom (V := V)
      (CombStruct_entropy_compression.weight (V := V) v)
  calc
    CombStruct_entropy_compression.decode (V := V)
        (EntropyObj_entropy_compression v)
        = CombStruct_entropy_compression.decode (V := V)
            (CombStruct_entropy_compression.encode (V := V)
              (CombStruct_entropy_compression.weight (V := V) v)) := by
          rfl
    _ = CombStruct_entropy_compression.weight (V := V) v := hDecode

theorem flag_density_step_entropy_compression {V : Type u}
    [CombStruct_entropy_compression V]
    (v : V) :
    DensityObj_entropy_compression (RegularityObj_entropy_compression v) <=
      DensityObj_entropy_compression v := by
  have hRaw :
      CombStruct_entropy_compression.density (V := V)
          (CombStruct_entropy_compression.regularize (V := V) v) <=
        CombStruct_entropy_compression.density (V := V) v :=
    CombStruct_entropy_compression.density_regularize_axiom (V := V) v
  calc
    DensityObj_entropy_compression (RegularityObj_entropy_compression v)
        = CombStruct_entropy_compression.density (V := V)
            (CombStruct_entropy_compression.regularize (V := V) v) := by
          rfl
    _ <= CombStruct_entropy_compression.density (V := V) v := hRaw
    _ = DensityObj_entropy_compression v := by
          rfl

theorem sparse_regularity_step_entropy_compression {V : Type u}
    [CombStruct_entropy_compression V]
    (v : V) :
    CombStruct_entropy_compression.complexity (V := V)
        (RegularityObj_entropy_compression v) <=
      CombStruct_entropy_compression.weight (V := V)
        (RegularityObj_entropy_compression v) +
        DensityObj_entropy_compression v := by
  have hBound :
      CombStruct_entropy_compression.complexity (V := V)
          (CombStruct_entropy_compression.regularize (V := V) v) <=
        CombStruct_entropy_compression.weight (V := V)
          (CombStruct_entropy_compression.regularize (V := V) v) +
          CombStruct_entropy_compression.density (V := V)
            (CombStruct_entropy_compression.regularize (V := V) v) :=
    CombStruct_entropy_compression.complexity_bound_axiom (V := V)
      (CombStruct_entropy_compression.regularize (V := V) v)
  have hDensity :
      CombStruct_entropy_compression.density (V := V)
          (CombStruct_entropy_compression.regularize (V := V) v) <=
        CombStruct_entropy_compression.density (V := V) v :=
    CombStruct_entropy_compression.density_regularize_axiom (V := V) v
  have hAdd :
      CombStruct_entropy_compression.weight (V := V)
          (CombStruct_entropy_compression.regularize (V := V) v) +
          CombStruct_entropy_compression.density (V := V)
            (CombStruct_entropy_compression.regularize (V := V) v) <=
      CombStruct_entropy_compression.weight (V := V)
          (CombStruct_entropy_compression.regularize (V := V) v) +
          CombStruct_entropy_compression.density (V := V) v :=
    Nat.add_le_add_left hDensity _
  have hTrans :
      CombStruct_entropy_compression.complexity (V := V)
          (CombStruct_entropy_compression.regularize (V := V) v) <=
        CombStruct_entropy_compression.weight (V := V)
          (CombStruct_entropy_compression.regularize (V := V) v) +
          CombStruct_entropy_compression.density (V := V) v :=
    Nat.le_trans hBound hAdd
  calc
    CombStruct_entropy_compression.complexity (V := V)
        (RegularityObj_entropy_compression v)
        = CombStruct_entropy_compression.complexity (V := V)
            (CombStruct_entropy_compression.regularize (V := V) v) := by
          rfl
    _ <= CombStruct_entropy_compression.weight (V := V)
          (CombStruct_entropy_compression.regularize (V := V) v) +
          CombStruct_entropy_compression.density (V := V) v := hTrans
    _ = CombStruct_entropy_compression.weight (V := V)
          (RegularityObj_entropy_compression v) +
          DensityObj_entropy_compression v := by
          rfl

theorem tverberg_partition_step_entropy_compression {V : Type u}
    [CombStruct_entropy_compression V]
    (v : V) :
    EntropyObj_entropy_compression
      (CombStruct_entropy_compression.compress (V := V) v) <=
      EntropyObj_entropy_compression v := by
  have hWeight :
      CombStruct_entropy_compression.weight (V := V)
          (CombStruct_entropy_compression.compress (V := V) v) <=
        CombStruct_entropy_compression.weight (V := V) v :=
    CombStruct_entropy_compression.compress_weight_axiom (V := V) v
  have hEncode :
      CombStruct_entropy_compression.encode (V := V)
          (CombStruct_entropy_compression.weight (V := V)
            (CombStruct_entropy_compression.compress (V := V) v)) <=
        CombStruct_entropy_compression.encode (V := V)
          (CombStruct_entropy_compression.weight (V := V) v) :=
    CombStruct_entropy_compression.encode_mono_axiom (V := V) hWeight
  calc
    EntropyObj_entropy_compression
        (CombStruct_entropy_compression.compress (V := V) v)
        = CombStruct_entropy_compression.encode (V := V)
            (CombStruct_entropy_compression.weight (V := V)
              (CombStruct_entropy_compression.compress (V := V) v)) := by
          rfl
    _ <= CombStruct_entropy_compression.encode (V := V)
          (CombStruct_entropy_compression.weight (V := V) v) := hEncode
    _ = EntropyObj_entropy_compression v := by
          rfl

theorem counting_upgrade_entropy_compression {V : Type u}
    [CombStruct_entropy_compression V]
    (v : V) :
    CombStruct_entropy_compression.decode (V := V)
      (EntropyObj_entropy_compression
        (CombStruct_entropy_compression.compress (V := V) v)) <=
      CombStruct_entropy_compression.weight (V := V) v := by
  have hDecodeCompress :
      CombStruct_entropy_compression.decode (V := V)
          (EntropyObj_entropy_compression
            (CombStruct_entropy_compression.compress (V := V) v)) =
        CombStruct_entropy_compression.weight (V := V)
          (CombStruct_entropy_compression.compress (V := V) v) :=
    entropy_lemma_step_entropy_compression
      (CombStruct_entropy_compression.compress (V := V) v)
  have hWeight :
      CombStruct_entropy_compression.weight (V := V)
          (CombStruct_entropy_compression.compress (V := V) v) <=
        CombStruct_entropy_compression.weight (V := V) v :=
    container_step_entropy_compression v
  calc
    CombStruct_entropy_compression.decode (V := V)
        (EntropyObj_entropy_compression
          (CombStruct_entropy_compression.compress (V := V) v))
        = CombStruct_entropy_compression.weight (V := V)
            (CombStruct_entropy_compression.compress (V := V) v) :=
          hDecodeCompress
    _ <= CombStruct_entropy_compression.weight (V := V) v := hWeight

theorem extremal_conclusion_entropy_compression {V : Type u}
    [CombStruct_entropy_compression V]
    (v : V) :
    CombStruct_entropy_compression.complexity (V := V) v <=
      CombStruct_entropy_compression.decode (V := V)
        (EntropyObj_entropy_compression v) +
        DensityObj_entropy_compression v := by
  have hBound :
      CombStruct_entropy_compression.complexity (V := V) v <=
        CombStruct_entropy_compression.weight (V := V) v +
          CombStruct_entropy_compression.density (V := V) v :=
    CombStruct_entropy_compression.complexity_bound_axiom (V := V) v
  have hDecode :
      CombStruct_entropy_compression.decode (V := V)
          (EntropyObj_entropy_compression v) =
        CombStruct_entropy_compression.weight (V := V) v :=
    entropy_lemma_step_entropy_compression v
  calc
    CombStruct_entropy_compression.complexity (V := V) v
        <= CombStruct_entropy_compression.weight (V := V) v +
            CombStruct_entropy_compression.density (V := V) v := hBound
    _ = CombStruct_entropy_compression.decode (V := V)
          (EntropyObj_entropy_compression v) +
          CombStruct_entropy_compression.density (V := V) v := by
          rw [hDecode]
    _ = CombStruct_entropy_compression.decode (V := V)
          (EntropyObj_entropy_compression v) +
          DensityObj_entropy_compression v := by
          rfl
