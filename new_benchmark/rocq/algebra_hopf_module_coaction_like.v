(*
BENCHMARK_ID: TINY_MATHLIB_BATCH04_ALGEBRA_HOPF_MODULE_COACTION_LIKE
PAIR_STEM: algebra_hopf_module_coaction_like
MATH_DOMAIN: Algebra / Category-style Algebra
SOURCE_MATHLIB: Mathlib/Algebra/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class BialgebraLike (H M : Type) := {
  coact : M -> H -> M -> Prop;
  counit : H -> Prop;
  smul : H -> M -> M;
  coassoc :
    forall (m : M) (h1 : H) (m1 : M) (h2 : H) (m2 : M),
      coact m h1 m1 -> coact m1 h2 m2 -> coact m h1 m2;
  counit_law :
    forall (m : M) (h : H) (m1 : M),
      coact m h m1 -> counit h -> m1 = m;
  module_coact :
    forall (h : H) (m : M), counit h -> coact m h (smul h m)
}.

Definition ComoduleLike (H M : Type) {B : BialgebraLike H M} : Prop :=
  forall m : M, exists h : H, exists m1 : M, coact m h m1.

Definition ModuleLike (H M : Type) {B : BialgebraLike H M} : Prop :=
  forall (h : H) (m : M), counit h -> smul h m = smul h m.

Arguments ComoduleLike H M {B}.
Arguments ModuleLike H M {B}.

Definition HopfModuleLike (H M : Type) {B : BialgebraLike H M} : Prop :=
  ComoduleLike H M /\ ModuleLike H M.

Definition CoinvariantLike (H M : Type) {B : BialgebraLike H M} (m : M) : Prop :=
  forall (h : H) (m1 : M), coact m h m1 -> counit h -> m1 = m.

Arguments HopfModuleLike H M {B}.
Arguments CoinvariantLike H M {B} m.

Definition TensorOverLike (H M : Type) {B : BialgebraLike H M} : Type :=
  (H * M)%type.

Arguments TensorOverLike H M {B}.

Lemma coaction_coassoc_like (H M : Type) {B : BialgebraLike H M}
    {m : M} {h1 : H} {m1 : M} {h2 : H} {m2 : M}
    (hco1 : coact m h1 m1)
    (hco2 : coact m1 h2 m2) :
    exists h : H, coact m h m2.
Proof.
  assert (hCompose : coact m h1 m2).
  { exact (coassoc m h1 m1 h2 m2 hco1 hco2). }
  exists h1.
  exact hCompose.
Qed.

Lemma coaction_counit_like (H M : Type) {B : BialgebraLike H M}
    {m : M} {h : H} {m1 : M}
    (hco : coact m h m1)
    (hc : counit h) :
    m1 = m.
Proof.
  assert (hEq : m1 = m).
  { exact (counit_law m h m1 hco hc). }
  exact hEq.
Qed.

Lemma coinvariant_submodule_like (H M : Type) {B : BialgebraLike H M}
    (m : M) (hcoinv : CoinvariantLike H M m) :
    forall (h : H) (m1 : M),
      coact m h m1 -> counit h -> CoinvariantLike H M m1.
Proof.
  intros h m1 hco hmCounit.
  assert (hm1Eqm : m1 = m).
  { exact (hcoinv h m1 hco hmCounit). }
  intros h' m2 hco' hCounit'.
  assert (hm2Eqm1 : m2 = m1).
  { exact (counit_law m1 h' m2 hco' hCounit'). }
  assert (hm2Eqm : m2 = m).
  {
    rewrite hm2Eqm1.
    exact hm1Eqm.
  }
  rewrite hm1Eqm.
  exact hm2Eqm.
Qed.

Lemma fundamental_map_like (H M : Type) {B : BialgebraLike H M}
    (h : H) (m : M) (hc : counit h) :
    exists t : TensorOverLike H M, fst t = h /\ coact m h (snd t).
Proof.
  set (t := (h, smul h m)).
  assert (htEq : fst t = h).
  { reflexivity. }
  assert (hco : coact m h (snd t)).
  {
    unfold t.
    simpl.
    exact (module_coact h m hc).
  }
  exists t.
  split.
  - exact htEq.
  - exact hco.
Qed.

Lemma fundamental_inverse_like (H M : Type) {B : BialgebraLike H M}
    (m : M) (hcoinv : CoinvariantLike H M m)
    (h : H) (hc : counit h) :
    let t : TensorOverLike H M := (h, smul h m)
    in snd t = m.
Proof.
  assert (hco : coact m h (smul h m)).
  { exact (module_coact h m hc). }
  assert (hEq : smul h m = m).
  { exact (hcoinv h (smul h m) hco hc). }
  simpl.
  exact hEq.
Qed.

Lemma hopf_module_decomposition_like (H M : Type) {B : BialgebraLike H M}
    (m : M) (hcoinv : CoinvariantLike H M m) :
    forall h : H, counit h ->
      exists t : TensorOverLike H M, fst t = h /\ snd t = m.
Proof.
  intros h hc.
  assert (hMap : exists t : TensorOverLike H M, fst t = h /\ coact m h (snd t)).
  { exact (@fundamental_map_like H M B h m hc). }
  destruct hMap as [t [ht1 htCo]].
  assert (hInv : (let s : TensorOverLike H M := (h, smul h m) in snd s = m)).
  { exact (@fundamental_inverse_like H M B m hcoinv h hc). }
  exists (h, smul h m).
  split.
  - reflexivity.
  - simpl in hInv.
    exact hInv.
Qed.
