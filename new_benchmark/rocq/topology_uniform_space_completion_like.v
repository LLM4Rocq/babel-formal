(*
BENCHMARK_ID: TINY_MATHLIB_BATCH04_TOPOLOGY_UNIFORM_COMPLETION
PAIR_STEM: topology_uniform_space_completion_like
MATH_DOMAIN: Topology / Uniform Spaces
SOURCE_MATHLIB: Mathlib/Topology/UniformSpace/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class UniformSpaceLike (A : Type) := {
  entourage : (A -> A -> Prop) -> Prop;
  entourage_refl : forall V : A -> A -> Prop, entourage V -> forall x : A, V x x;
  entourage_symm :
    forall V : A -> A -> Prop, entourage V ->
      exists W : A -> A -> Prop, entourage W /\ (forall x y : A, W x y -> V y x);
  entourage_comp :
    forall V : A -> A -> Prop, entourage V ->
      exists W : A -> A -> Prop, entourage W /\
        (forall x y z : A, W x y -> W y z -> V x z);
  entourage_mono :
    forall V W : A -> A -> Prop, entourage V ->
      (forall x y : A, W x y -> V x y) -> entourage W
}.

Definition CauchyLike {A : Type} `{UniformSpaceLike A} (F : (A -> Prop) -> Prop) : Prop :=
  forall V : A -> A -> Prop,
    entourage V ->
      exists s : A -> Prop, F s /\ forall x y : A, s x -> s y -> V x y.

Definition CompleteLike {A : Type} `{UniformSpaceLike A} : Prop :=
  forall F : (A -> Prop) -> Prop,
    CauchyLike F ->
      exists x : A,
        forall V : A -> A -> Prop,
          entourage V ->
            exists s : A -> Prop, F s /\ forall y : A, s y -> V y x.

Definition DenseLike {A B : Type} `{UniformSpaceLike B} (f : A -> B) : Prop :=
  forall y : B,
    forall V : B -> B -> Prop,
      entourage V ->
        exists x : A, V (f x) y.

Record CompletionLike (A : Type) (UA : UniformSpaceLike A) := {
  B : Type;
  uniformB :> UniformSpaceLike B;
  emb : A -> B;
  emb_dense : @DenseLike A B uniformB emb;
  emb_uniform :
    forall V : B -> B -> Prop,
      @entourage B uniformB V ->
        exists W : A -> A -> Prop,
          @entourage A UA W /\
            (forall x y : A, W x y -> V (emb x) (emb y));
  completeB : @CompleteLike B uniformB
}.

Definition LiftLike {A : Type} `{UniformSpaceLike A}
    (c : @CompletionLike A _)
    {G : Type} `{UniformSpaceLike G}
    (f : A -> G) : Prop :=
  exists g : B c -> G,
    (forall x : A, g (emb c x) = f x) /\
      (forall V : G -> G -> Prop,
        entourage V ->
          exists W : B c -> B c -> Prop,
            @entourage (B c) (uniformB c) W /\
              (forall x y : B c, W x y -> V (g x) (g y))).

Lemma completion_map_dense {A : Type} `{UniformSpaceLike A}
    (c : @CompletionLike A _) :
    @DenseLike A (B c) (uniformB c) (emb c).
Proof.
  intros y V hV.
  assert (hDense : @DenseLike A (B c) (uniformB c) (emb c)).
  {
    exact (emb_dense c).
  }
  assert (hWitness : exists x : A, V (emb c x) y).
  {
    apply hDense.
    exact hV.
  }
  destruct hWitness as [x hx].
  exists x.
  exact hx.
Qed.

Lemma completion_map_uniform {A : Type} `{UniformSpaceLike A}
    (c : @CompletionLike A _) :
    forall V : B c -> B c -> Prop,
      @entourage (B c) (uniformB c) V ->
        exists W : A -> A -> Prop,
          entourage W /\
            (forall x y : A, W x y -> V (emb c x) (emb c y)).
Proof.
  intros V hV.
  assert (hRaw :
      exists W : A -> A -> Prop,
        entourage W /\
          (forall x y : A, W x y -> V (emb c x) (emb c y))).
  {
    apply (emb_uniform c).
    exact hV.
  }
  destruct hRaw as [W [hW hWmap]].
  exists W.
  split.
  - exact hW.
  - intros x y hxy.
    apply hWmap.
    exact hxy.
Qed.

Lemma completion_extension_exists {A : Type} `{UniformSpaceLike A}
    (c : @CompletionLike A _)
    {G : Type} `{UniformSpaceLike G}
    (f : A -> G)
    (hLift : LiftLike c f) :
    exists g : B c -> G, forall x : A, g (emb c x) = f x.
Proof.
  destruct hLift as [g [hg hunif]].
  assert (hgraph : forall x : A, g (emb c x) = f x).
  {
    intro x.
    apply hg.
  }
  assert (hkeep : forall V : G -> G -> Prop,
      entourage V ->
        exists W : B c -> B c -> Prop,
          @entourage (B c) (uniformB c) W /\
            (forall x y : B c, W x y -> V (g x) (g y))).
  {
    exact hunif.
  }
  exists g.
  intro x.
  apply hgraph.
Qed.

Lemma completion_extension_unique {A : Type} `{UniformSpaceLike A}
    (c : @CompletionLike A _)
    {G : Type} `{UniformSpaceLike G}
    (f : A -> G)
    (hDenseExt :
      forall g1 g2 : B c -> G,
        (forall x : A, g1 (emb c x) = g2 (emb c x)) ->
        g1 = g2)
    (hLift : LiftLike c f)
    (g1 g2 : B c -> G)
    (hg1 : forall x : A, g1 (emb c x) = f x)
    (hg2 : forall x : A, g2 (emb c x) = f x) :
    g1 = g2.
Proof.
  destruct hLift as [g [hg hunif]].
  assert (hEq1 : g1 = g).
  {
    apply hDenseExt.
    intro x.
    transitivity (f x).
    - apply hg1.
    - symmetry.
      apply hg.
  }
  assert (hEq2 : g2 = g).
  {
    apply hDenseExt.
    intro x.
    transitivity (f x).
    - apply hg2.
    - symmetry.
      apply hg.
  }
  assert (hkeep : forall V : G -> G -> Prop,
      entourage V ->
        exists W : B c -> B c -> Prop,
          @entourage (B c) (uniformB c) W /\
            (forall x y : B c, W x y -> V (g x) (g y))).
  {
    exact hunif.
  }
  rewrite hEq1.
  rewrite hEq2.
  reflexivity.
Qed.

Lemma complete_of_completion {A : Type} `{UniformSpaceLike A}
    (c : @CompletionLike A _) :
    @CompleteLike (B c) (uniformB c).
Proof.
  assert (hcomp : @CompleteLike (B c) (uniformB c)).
  {
    exact (completeB c).
  }
  assert (hkeep : @CompleteLike (B c) (uniformB c)).
  {
    exact hcomp.
  }
  exact hkeep.
Qed.

Lemma completion_idempotent_like {A : Type} `{UniformSpaceLike A}
    (c : @CompletionLike A _)
    (c2 : @CompletionLike (B c) (uniformB c))
    (hLift : @LiftLike (B c) (uniformB c) c2 (B c) (uniformB c) (fun x : B c => x))
    (hRetrUniq :
      forall r1 r2 : B c2 -> B c,
        (forall x : B c, r1 (emb c2 x) = x) ->
        (forall x : B c, r2 (emb c2 x) = x) ->
        r1 = r2) :
    exists r : B c2 -> B c,
      (forall x : B c, r (emb c2 x) = x) /\
      (forall s : B c2 -> B c, (forall x : B c, s (emb c2 x) = x) -> s = r).
Proof.
  destruct hLift as [r [hr hunif]].
  assert (hr_id : forall x : B c, r (emb c2 x) = x).
  {
    intro x.
    assert (hraw : r (emb c2 x) = (fun y : B c => y) x).
    {
      apply hr.
    }
    exact hraw.
  }
  exists r.
  split.
  - exact hr_id.
  - intros s hs.
    assert (hsr : s = r).
    {
      apply hRetrUniq.
      + exact hs.
      + exact hr_id.
    }
  assert (hkeep : forall V : B c -> B c -> Prop,
      @entourage (B c) (uniformB c) V ->
        exists W : B c2 -> B c2 -> Prop,
          @entourage (B c2) (uniformB c2) W /\
            (forall x y : B c2, W x y -> V (r x) (r y))).
    {
      exact hunif.
    }
    exact hsr.
Qed.
