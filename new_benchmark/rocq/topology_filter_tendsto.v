(*
BENCHMARK_ID: TINY_MATHLIB_BATCH04_TOPO_FILTER_TENDSTO
PAIR_STEM: topology_filter_tendsto
MATH_DOMAIN: Topology
SOURCE_MATHLIB: Mathlib/Order/Filter
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 14
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Module TopologyFilterTendsto.

Record Filter (A : Type) := {
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

Definition preimage {A B : Type} (f : A -> B) (s : B -> Prop) : A -> Prop :=
  fun x => s (f x).

Definition map {A B : Type} (f : A -> B) (F : Filter A) : Filter B.
Proof.
  refine {| sets := fun s => sets F (preimage f s) |}.
  - assert (hpre : preimage f (fun _ : B => True) = (fun _ : A => True)).
    { reflexivity. }
    rewrite hpre.
    exact (univ_sets F).
  - intros s t hs hst.
    apply (sets_of_superset F (preimage f s) (preimage f t)).
    + exact hs.
    + intros x hx.
      apply hst.
      exact hx.
  - intros s t hs ht.
    assert (hinter : sets F (fun x : A => preimage f s x /\ preimage f t x)).
    { apply (inter_sets F (preimage f s) (preimage f t)); assumption. }
    assert (hpre :
      preimage f (fun y : B => s y /\ t y) = (fun x : A => preimage f s x /\ preimage f t x)).
    { reflexivity. }
    rewrite hpre.
    exact hinter.
Defined.

Definition Tendsto {A B : Type} (f : A -> B) (F : Filter A) (G : Filter B) : Prop :=
  forall s : B -> Prop, sets G s -> sets F (preimage f s).

Lemma tendsto_id {A : Type} (F : Filter A) :
  Tendsto (fun x : A => x) F F.
Proof.
  intros s hs.
  assert (hpre : preimage (fun x : A => x) s = s).
  { reflexivity. }
  rewrite hpre.
  exact hs.
Qed.

Lemma tendsto_comp {A B C : Type}
    {f : A -> B} {g : B -> C} {F : Filter A} {G : Filter B} {H : Filter C}
    (hg : Tendsto g G H) (hf : Tendsto f F G) :
    Tendsto (fun x => g (f x)) F H.
Proof.
  intros s hs.
  assert (hgs : sets G (preimage g s)).
  { apply hg. exact hs. }
  assert (hfs : sets F (preimage f (preimage g s))).
  { apply hf. exact hgs. }
  assert (hpre :
    preimage (fun x => g (f x)) s = preimage f (preimage g s)).
  { reflexivity. }
  rewrite hpre.
  exact hfs.
Qed.

Lemma tendsto_const {A B : Type}
    (F : Filter A) (G : Filter B) (c : B)
    (hGc : forall s : B -> Prop, sets G s -> s c) :
    Tendsto (fun _ : A => c) F G.
Proof.
  intros s hs.
  assert (hsc : s c).
  { apply hGc. exact hs. }
  assert (huniv : sets F (fun _ : A => True)).
  { exact (univ_sets F). }
  assert (hsub : forall x : A, True -> preimage (fun _ : A => c) s x).
  {
    intros x hx.
    exact hsc.
  }
  assert (hpre : sets F (preimage (fun _ : A => c) s)).
  { apply (sets_of_superset F (fun _ : A => True) (preimage (fun _ : A => c) s)); assumption. }
  exact hpre.
Qed.

Lemma tendsto_mono {A B : Type}
    {f : A -> B} {F : Filter A} {G H : Filter B}
    (hFG : Tendsto f F G)
    (hGH : forall s : B -> Prop, sets H s -> sets G s) :
    Tendsto f F H.
Proof.
  intros s hsH.
  assert (hsG : sets G s).
  { apply hGH. exact hsH. }
  assert (hsF : sets F (preimage f s)).
  { apply hFG. exact hsG. }
  exact hsF.
Qed.

Lemma map_id {A : Type} (F : Filter A) :
  forall s : A -> Prop, sets (map (fun x : A => x) F) s <-> sets F s.
Proof.
  intro s.
  split.
  - intro hs.
    assert (hdef : sets (map (fun x : A => x) F) s = sets F (preimage (fun x : A => x) s)).
    { reflexivity. }
    assert (hpre : preimage (fun x : A => x) s = s).
    { reflexivity. }
    rewrite hdef in hs.
    rewrite hpre in hs.
    exact hs.
  - intro hs.
    assert (hpre : preimage (fun x : A => x) s = s).
    { reflexivity. }
    assert (hdef : sets (map (fun x : A => x) F) s = sets F (preimage (fun x : A => x) s)).
    { reflexivity. }
    rewrite hdef.
    rewrite hpre.
    exact hs.
Qed.

Lemma map_comp {A B C : Type}
    (g : B -> C) (f : A -> B) (F : Filter A) :
    forall s : C -> Prop, sets (map g (map f F)) s <-> sets (map (fun x => g (f x)) F) s.
Proof.
  intro s.
  split.
  - intro hs.
    assert (hleft1 : sets (map g (map f F)) s = sets (map f F) (preimage g s)).
    { reflexivity. }
    assert (hleft2 : sets (map f F) (preimage g s) = sets F (preimage f (preimage g s))).
    { reflexivity. }
    assert (hpre : preimage f (preimage g s) = preimage (fun x => g (f x)) s).
    { reflexivity. }
    assert (hright : sets (map (fun x => g (f x)) F) s = sets F (preimage (fun x => g (f x)) s)).
    { reflexivity. }
    rewrite hleft1 in hs.
    rewrite hleft2 in hs.
    rewrite hpre in hs.
    rewrite hright.
    exact hs.
  - intro hs.
    assert (hleft1 : sets (map g (map f F)) s = sets (map f F) (preimage g s)).
    { reflexivity. }
    assert (hleft2 : sets (map f F) (preimage g s) = sets F (preimage f (preimage g s))).
    { reflexivity. }
    assert (hpre : preimage f (preimage g s) = preimage (fun x => g (f x)) s).
    { reflexivity. }
    assert (hright : sets (map (fun x => g (f x)) F) s = sets F (preimage (fun x => g (f x)) s)).
    { reflexivity. }
    rewrite hright in hs.
    rewrite <- hpre in hs.
    rewrite <- hleft2 in hs.
    rewrite hleft1.
    exact hs.
Qed.

Lemma tendsto_map {A B : Type}
    (f : A -> B) (F : Filter A) :
    Tendsto f F (map f F).
Proof.
  intros s hs.
  assert (hs' : sets F (preimage f s)).
  { exact hs. }
  exact hs'.
Qed.

Lemma tendsto_of_eq {A B : Type}
    {f g : A -> B} {F : Filter A} {G : Filter B}
    (hfg : Tendsto f F G) (heq : f = g) :
    Tendsto g F G.
Proof.
  intros s hs.
  assert (hpre_f : sets F (preimage f s)).
  { apply hfg. exact hs. }
  assert (hpre_eq : preimage g s = preimage f s).
  {
    rewrite <- heq.
    reflexivity.
  }
  rewrite hpre_eq.
  exact hpre_f.
Qed.

Lemma tendsto_inter {A B : Type}
    {f : A -> B} {F : Filter A} {G : Filter B}
    (hFG : Tendsto f F G) (s t : B -> Prop)
    (hs : sets G s) (ht : sets G t) :
    sets F (preimage f (fun y => s y /\ t y)).
Proof.
  assert (hsF : sets F (preimage f s)).
  { apply hFG. exact hs. }
  assert (htF : sets F (preimage f t)).
  { apply hFG. exact ht. }
  assert (hinter : sets F (fun x => preimage f s x /\ preimage f t x)).
  { apply (inter_sets F (preimage f s) (preimage f t)); assumption. }
  assert (hpre :
    preimage f (fun y => s y /\ t y) = (fun x => preimage f s x /\ preimage f t x)).
  { reflexivity. }
  rewrite hpre.
  exact hinter.
Qed.

Lemma tendsto_principal_like {A B : Type}
    {f : A -> B} {F : Filter A} {G : Filter B} (c : B)
    (hGc : forall s : B -> Prop, sets G s <-> s c) :
    Tendsto f F G <-> forall s : B -> Prop, s c -> sets F (preimage f s).
Proof.
  split.
  - intros hT s hs.
    assert (hsG : sets G s).
    { apply (proj2 (hGc s)). exact hs. }
    assert (hsF : sets F (preimage f s)).
    { apply hT. exact hsG. }
    exact hsF.
  - intros hpc s hsG.
    assert (hsc : s c).
    { apply (proj1 (hGc s)). exact hsG. }
    assert (hsF : sets F (preimage f s)).
    { apply hpc. exact hsc. }
    exact hsF.
Qed.

End TopologyFilterTendsto.
