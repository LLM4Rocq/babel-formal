(*
BENCHMARK_ID: TINY_MATHLIB_BATCH03_TOPOLOGY_ULTRAFILTER_COMPACT
PAIR_STEM: topology_ultrafilter_compact_like
MATH_DOMAIN: Topology
SOURCE_MATHLIB: Mathlib/Topology/Compactness
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class TopologicalSpaceLike (A : Type) := {
  IsNeighborhood : A -> (A -> Prop) -> Prop;
  nhds_univ : forall x : A, IsNeighborhood x (fun _ => True);
  nhds_inter :
    forall (x : A) (s t : A -> Prop),
      IsNeighborhood x s -> IsNeighborhood x t -> IsNeighborhood x (fun y => s y /\ t y);
  nhds_mono :
    forall (x : A) (s t : A -> Prop),
      IsNeighborhood x s -> (forall y : A, s y -> t y) -> IsNeighborhood x t
}.

Record FilterLike (A : Type) := {
  sets : (A -> Prop) -> Prop;
  univ_sets : sets (fun _ => True);
  sets_of_superset :
    forall s t : A -> Prop, sets s -> (forall x : A, s x -> t x) -> sets t;
  inter_sets :
    forall s t : A -> Prop, sets s -> sets t -> sets (fun x => s x /\ t x)
}.

Arguments sets {A} _ _.
Arguments univ_sets {A} _.
Arguments sets_of_superset {A} _ _ _ _ _.
Arguments inter_sets {A} _ _ _ _ _. 

Definition TendstoLike {A B : Type}
    (f : A -> B) (F : FilterLike A) (G : FilterLike B) : Prop :=
  forall s : B -> Prop, sets G s -> sets F (fun x => s (f x)).

Definition ClusterPtLike {A : Type} `{TopologicalSpaceLike A}
    (F : FilterLike A) (x : A) : Prop :=
  forall s : A -> Prop, IsNeighborhood x s -> sets F s.

Definition CompactLike {A : Type} `{TopologicalSpaceLike A} (K : A -> Prop) : Prop :=
  forall U : FilterLike A,
    (forall s : A -> Prop, sets U s \/ sets U (fun x => ~ s x)) ->
    (forall s : A -> Prop, sets U s -> exists x : A, K x /\ s x) ->
    exists x : A, K x /\ ClusterPtLike U x.

Definition UltrafilterLike {A : Type} (U : FilterLike A) : Prop :=
  forall s : A -> Prop, sets U s \/ sets U (fun x => ~ s x).

Lemma compact_of_ultrafilter_cluster {A : Type} `{TopologicalSpaceLike A}
    (K : A -> Prop)
    (hcluster :
      forall U : FilterLike A,
        UltrafilterLike U ->
        (forall s : A -> Prop, sets U s -> exists x : A, K x /\ s x) ->
        exists x : A, K x /\ ClusterPtLike U x) :
    CompactLike K.
Proof.
  intros U hUltraRaw hMeet.
  assert (hUltra : UltrafilterLike U).
  {
    intro s.
    exact (hUltraRaw s).
  }
  assert (hWitness : exists x : A, K x /\ ClusterPtLike U x).
  {
    apply hcluster.
    - exact hUltra.
    - exact hMeet.
  }
  exact hWitness.
Qed.

Lemma ultrafilter_refines {A : Type} `{TopologicalSpaceLike A}
    (F U : FilterLike A)
    (hU : UltrafilterLike U)
    (href : forall s : A -> Prop, sets U s -> sets F s) :
    forall s : A -> Prop, sets U s -> sets F s /\ (sets U s \/ sets U (fun x => ~ s x)).
Proof.
  intros s hs.
  assert (hFs : sets F s).
  {
    apply href.
    exact hs.
  }
  assert (hDec : sets U s \/ sets U (fun x => ~ s x)).
  {
    apply hU.
  }
  split.
  - exact hFs.
  - exact hDec.
Qed.

Lemma cluster_of_refinement {A : Type} `{TopologicalSpaceLike A}
    (F U : FilterLike A) (x : A)
    (hClusterU : ClusterPtLike U x)
    (href : forall s : A -> Prop, sets U s -> sets F s) :
    ClusterPtLike F x.
Proof.
  intros s hsNhds.
  assert (hsU : sets U s).
  {
    apply hClusterU.
    exact hsNhds.
  }
  assert (hsF : sets F s).
  {
    apply href.
    exact hsU.
  }
  exact hsF.
Qed.

Lemma compact_image_like {A B : Type}
    `{TopologicalSpaceLike A} `{TopologicalSpaceLike B}
    (f : A -> B) (K : A -> Prop)
    (himage :
      forall U : FilterLike B,
        UltrafilterLike U ->
        (forall s : B -> Prop, sets U s -> exists y : B, (exists x : A, K x /\ y = f x) /\ s y) ->
        exists y : B, (exists x : A, K x /\ y = f x) /\ ClusterPtLike U y) :
    CompactLike (fun y : B => exists x : A, K x /\ y = f x).
Proof.
  assert (hcluster :
      forall U : FilterLike B,
        UltrafilterLike U ->
        (forall s : B -> Prop, sets U s -> exists y : B, (exists x : A, K x /\ y = f x) /\ s y) ->
        exists y : B, (exists x : A, K x /\ y = f x) /\ ClusterPtLike U y).
  {
    intros U hU hSat.
    apply himage.
    - exact hU.
    - exact hSat.
  }
  apply compact_of_ultrafilter_cluster.
  exact hcluster.
Qed.

Lemma compact_finite_intersection {A : Type} `{TopologicalSpaceLike A}
    (K L : A -> Prop)
    (hinter :
      forall U : FilterLike A,
        UltrafilterLike U ->
        (forall s : A -> Prop, sets U s -> exists x : A, (K x /\ L x) /\ s x) ->
        exists x : A, (K x /\ L x) /\ ClusterPtLike U x) :
    CompactLike (fun x : A => K x /\ L x).
Proof.
  assert (hcluster :
      forall U : FilterLike A,
        UltrafilterLike U ->
        (forall s : A -> Prop, sets U s -> exists x : A, (K x /\ L x) /\ s x) ->
        exists x : A, (K x /\ L x) /\ ClusterPtLike U x).
  {
    intros U hU hSat.
    apply hinter.
    - exact hU.
    - exact hSat.
  }
  apply compact_of_ultrafilter_cluster.
  exact hcluster.
Qed.

Lemma compact_closed_subspace {A : Type} `{TopologicalSpaceLike A}
    (K C : A -> Prop)
    (hclosed : forall x : A, C x -> IsNeighborhood x C)
    (hsub :
      forall U : FilterLike A,
        UltrafilterLike U ->
        (forall s : A -> Prop, sets U s -> exists x : A, (K x /\ C x) /\ s x) ->
        exists x : A, (K x /\ C x) /\ ClusterPtLike U x) :
    CompactLike (fun x : A => K x /\ C x).
Proof.
  assert (hclosed_on_subset : forall x : A, (K x /\ C x) -> IsNeighborhood x C).
  {
    intros x hx.
    apply hclosed.
    exact (proj2 hx).
  }
  assert (hcluster :
      forall U : FilterLike A,
        UltrafilterLike U ->
        (forall s : A -> Prop, sets U s -> exists x : A, (K x /\ C x) /\ s x) ->
        exists x : A, (K x /\ C x) /\ ClusterPtLike U x).
  {
    intros U hU hSat.
    pose proof hclosed_on_subset as hclosed_copy.
    apply hsub.
    - exact hU.
    - exact hSat.
  }
  apply compact_of_ultrafilter_cluster.
  exact hcluster.
Qed.
