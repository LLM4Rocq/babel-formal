(*
BENCHMARK_ID: TINY_MATHLIB_BATCH06_TOPOLOGY_STONE_CECH_COMPACTIFICATION_LIKE
PAIR_STEM: topology_stone_cech_compactification_like
MATH_DOMAIN: Topology
SOURCE_MATHLIB: Mathlib/Topology/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class CompactificationStruct_stone_cech (X betaX Y : Type) := {
  embed : X -> betaX;
  dense_image : Prop;
  compact_space : Prop;
  extend : (X -> Y) -> betaX -> Y;
  extend_on_embed :
    forall (f : X -> Y) (x : X),
      extend f (embed x) = f x;
  extend_unique :
    forall (f : X -> Y) (g h : betaX -> Y),
      (forall x : X, g (embed x) = f x) ->
      (forall x : X, h (embed x) = f x) ->
      g = h;
  extend_comp :
    forall (f : X -> Y) (k : Y -> Y) (b : betaX),
      extend (fun x : X => k (f x)) b = k (extend f b);
  core : (betaX -> Prop) -> Prop;
  core_from_dense :
    forall p : betaX -> Prop,
      (forall x : X, p (embed x)) ->
      core p;
  core_mono :
    forall p q : betaX -> Prop,
      core p ->
      (forall b : betaX, p b -> q b) ->
      core q
}.

Record ExtensionData_stone_cech_compactification
    (X betaX Y : Type)
    `{CompactificationStruct_stone_cech X betaX Y} := {
  base_map : X -> Y;
  extended_map : betaX -> Y;
  agree_on_dense :
    forall x : X,
      extended_map (embed x) = base_map x
}.

Definition dense_embedding_stone_cech_compactification
    {X betaX Y : Type}
    `{CompactificationStruct_stone_cech X betaX Y} :
    X -> betaX :=
  embed.

Definition extension_map_stone_cech_compactification
    {X betaX Y : Type}
    `{CompactificationStruct_stone_cech X betaX Y}
    (f : X -> Y) :
    betaX -> Y :=
  extend f.

Definition compact_core_stone_cech_compactification
    {X betaX Y : Type}
    `{CompactificationStruct_stone_cech X betaX Y}
    (p : betaX -> Prop) :
    Prop :=
  core p.

Lemma extension_exists_stone_cech_compactification
    {X betaX Y : Type}
    `{CompactificationStruct_stone_cech X betaX Y}
    (f : X -> Y) :
    exists g : betaX -> Y,
      (forall x : X,
        g (dense_embedding_stone_cech_compactification x) = f x) /\
      (forall b : betaX,
        g b = extension_map_stone_cech_compactification f b).
Proof.
  exists (extension_map_stone_cech_compactification f).
  split.
  - intro x.
    assert (hAgree : extend f (embed x) = f x).
    { exact (extend_on_embed f x). }
    exact hAgree.
  - intro b.
    assert (hDef : extension_map_stone_cech_compactification f b = extend f b).
    { reflexivity. }
    symmetry.
    exact hDef.
Qed.

Lemma extension_unique_stone_cech_compactification
    {X betaX Y : Type}
    `{CompactificationStruct_stone_cech X betaX Y}
    (f : X -> Y)
    (g1 g2 : betaX -> Y)
    (hg1 : forall x : X,
      g1 (dense_embedding_stone_cech_compactification x) = f x)
    (hg2 : forall x : X,
      g2 (dense_embedding_stone_cech_compactification x) = f x) :
    g1 = g2 /\
      forall b : betaX, g1 b = g2 b.
Proof.
  assert (hEq : g1 = g2).
  { apply (extend_unique f g1 g2); assumption. }
  assert (hPointwise : forall b : betaX, g1 b = g2 b).
  {
    intro b.
    exact (f_equal (fun g : betaX -> Y => g b) hEq).
  }
  split.
  - exact hEq.
  - exact hPointwise.
Qed.

Lemma extension_respects_comp_stone_cech_compactification
    {X betaX Y : Type}
    `{CompactificationStruct_stone_cech X betaX Y}
    (f : X -> Y)
    (k : Y -> Y) :
    (forall x : X,
      extension_map_stone_cech_compactification
        (fun t : X => k (f t))
        (dense_embedding_stone_cech_compactification x) = k (f x)) /\
    (forall b : betaX,
      extension_map_stone_cech_compactification
        (fun t : X => k (f t)) b =
      k (extension_map_stone_cech_compactification f b)).
Proof.
  split.
  - intro x.
    assert (hStep : extend (fun t : X => k (f t)) (embed x) = k (f x)).
    {
      transitivity ((fun t : X => k (f t)) x).
      - exact (extend_on_embed (fun t : X => k (f t)) x).
      - reflexivity.
    }
    exact hStep.
  - intro b.
    assert (hComp : extend (fun t : X => k (f t)) b = k (extend f b)).
    { exact (extend_comp f k b). }
    exact hComp.
Qed.

Lemma dense_image_universal_stone_cech_compactification
    {X betaX Y : Type}
    `{CompactificationStruct_stone_cech X betaX Y}
    (p : betaX -> Prop)
    (hp : forall x : X,
      p (dense_embedding_stone_cech_compactification x)) :
    compact_core_stone_cech_compactification p /\
      forall q : betaX -> Prop,
        (forall b : betaX, p b -> q b) ->
        compact_core_stone_cech_compactification q.
Proof.
  assert (hCoreP : core p).
  { apply (core_from_dense p). exact hp. }
  split.
  - exact hCoreP.
  - intros q hpq.
    assert (hCoreQ : core q).
    { apply (core_mono p q hCoreP). exact hpq. }
    exact hCoreQ.
Qed.

Lemma compact_core_minimal_stone_cech_compactification
    {X betaX Y : Type}
    `{CompactificationStruct_stone_cech X betaX Y}
    (p q r : betaX -> Prop)
    (hp : compact_core_stone_cech_compactification p)
    (hpq : forall b : betaX, p b -> q b)
    (hqr : forall b : betaX, q b -> r b) :
    compact_core_stone_cech_compactification r /\
      compact_core_stone_cech_compactification q.
Proof.
  assert (hCoreQ : compact_core_stone_cech_compactification q).
  { apply (core_mono p q hp). exact hpq. }
  assert (hCoreR : compact_core_stone_cech_compactification r).
  { apply (core_mono q r hCoreQ). exact hqr. }
  split.
  - exact hCoreR.
  - exact hCoreQ.
Qed.

Lemma factorization_through_core_stone_cech_compactification
    {X betaX Y : Type}
    `{CompactificationStruct_stone_cech X betaX Y}
    (data : @ExtensionData_stone_cech_compactification X betaX Y _)
    (p : betaX -> Prop)
    (hpDense : forall x : X,
      p (dense_embedding_stone_cech_compactification x))
    (hconst : forall b1 b2 : betaX, p b1 -> p b2 -> extended_map data b1 = extended_map data b2)
    (hnonempty : exists b0 : betaX, p b0) :
    exists y0 : Y,
      (forall b : betaX, p b -> extended_map data b = y0) /\
      compact_core_stone_cech_compactification p.
Proof.
  assert (hCoreP : compact_core_stone_cech_compactification p).
  { apply (core_from_dense p). exact hpDense. }
  destruct hnonempty as [b0 hb0].
  exists (extended_map data b0).
  split.
  - intros b hb.
    apply (hconst b b0 hb hb0).
  - exact hCoreP.
Qed.

Lemma universal_property_stone_cech_compactification
    {X betaX Y : Type}
    `{CompactificationStruct_stone_cech X betaX Y}
    (f : X -> Y) :
    (exists g : betaX -> Y,
      forall x : X,
        g (dense_embedding_stone_cech_compactification x) = f x) /\
    (forall g1 g2 : betaX -> Y,
      (forall x : X,
        g1 (dense_embedding_stone_cech_compactification x) = f x) ->
      (forall x : X,
        g2 (dense_embedding_stone_cech_compactification x) = f x) ->
      forall b : betaX, g1 b = g2 b).
Proof.
  assert (hExistStrong :
      exists g : betaX -> Y,
        (forall x : X,
          g (dense_embedding_stone_cech_compactification x) = f x) /\
        (forall b : betaX,
          g b = extension_map_stone_cech_compactification f b)).
  { apply (extension_exists_stone_cech_compactification f). }
  assert (hExist :
      exists g : betaX -> Y,
        forall x : X,
          g (dense_embedding_stone_cech_compactification x) = f x).
  {
    destruct hExistStrong as [g [hg _]].
    exists g.
    exact hg.
  }
  split.
  - exact hExist.
  - intros g1 g2 hg1 hg2 b.
    assert (hEq : g1 = g2).
    { apply (extend_unique f g1 g2); assumption. }
    exact (f_equal (fun g : betaX -> Y => g b) hEq).
Qed.
