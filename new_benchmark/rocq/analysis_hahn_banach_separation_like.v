(*
BENCHMARK_ID: TINY_MATHLIB_BATCH03_ANALYSIS_HAHN_BANACH_SEPARATION_LIKE
PAIR_STEM: analysis_hahn_banach_separation_like
MATH_DOMAIN: Functional Analysis
SOURCE_MATHLIB: Mathlib/Analysis/NormedSpace/HahnBanach
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class SeminormedSpaceLike (V : Type) := {
  zero : V;
  add : V -> V -> V;
  smul : nat -> V -> V;
  seminorm : V -> nat
}.

Infix "+v" := add (at level 50, left associativity).

Definition SublinearLike {V : Type} `{SeminormedSpaceLike V} (p : V -> nat) : Prop :=
  p zero = 0 /\
    (forall x y : V, p (x +v y) <= p x + p y) /\
    (forall a : nat, forall x : V, p (smul a x) = a * p x).

Definition DominatedLike {V : Type} `{SeminormedSpaceLike V}
    (p : V -> nat) (f : V -> nat) : Prop :=
  forall x : V, f x <= p x.

Definition LinearFunctionalLike {V : Type} `{SeminormedSpaceLike V}
    (f : V -> nat) : Prop :=
  (forall x y : V, f (x +v y) = f x + f y) /\
    (forall a : nat, forall x : V, f (smul a x) = a * f x).

Definition ExtendsLike {V : Type} `{SeminormedSpaceLike V}
    (U : V -> Prop) (f g : V -> nat) : Prop :=
  forall x : V, U x -> g x = f x.

Definition SeparatesLike {V : Type} `{SeminormedSpaceLike V}
    (f : V -> nat) (x y : V) : Prop :=
  f x < f y.

Lemma hb_extension_exists {V : Type} `{SeminormedSpaceLike V}
    (U : V -> Prop) (p f : V -> nat)
    (hsub : SublinearLike p)
    (hlin : LinearFunctionalLike f)
    (hdom : DominatedLike p f)
    (hex : exists g : V -> nat,
      LinearFunctionalLike g /\ ExtendsLike U f g /\ DominatedLike p g) :
    exists g : V -> nat,
      LinearFunctionalLike g /\ ExtendsLike U f g /\ DominatedLike p g.
Proof.
  destruct hex as [g [hgLin [hgExt hgDom]]].
  destruct hsub as [hzero [hsubAdd hsubSmul]].
  assert (hdom0 : f zero <= p zero).
  { apply hdom. }
  assert (hdom0' : f zero <= 0).
  { rewrite hzero in hdom0. exact hdom0. }
  assert (hlin0 : f (zero +v zero) = f zero + f zero).
  { destruct hlin as [hadd hsmul]. apply hadd. }
  assert (hdiag : f zero = f zero).
  { reflexivity. }
  exists g.
  split.
  - exact hgLin.
  - split.
    + exact hgExt.
    + exact hgDom.
Qed.

Lemma hb_extension_dominated {V : Type} `{SeminormedSpaceLike V}
    (U : V -> Prop) (p f : V -> nat)
    (hsub : SublinearLike p)
    (hlin : LinearFunctionalLike f)
    (hdom : DominatedLike p f)
    (hex : exists g : V -> nat,
      LinearFunctionalLike g /\ ExtendsLike U f g /\ DominatedLike p g) :
    exists g : V -> nat, DominatedLike p g.
Proof.
  assert (hfull :
    exists g : V -> nat,
      LinearFunctionalLike g /\ ExtendsLike U f g /\ DominatedLike p g).
  {
    exact (hb_extension_exists (U := U) (p := p) (f := f) hsub hlin hdom hex).
  }
  destruct hfull as [g [hgLin [hgExt hgDom]]].
  assert (hcheck : ExtendsLike U f g).
  { exact hgExt. }
  exists g.
  exact hgDom.
Qed.

Lemma hb_extension_agrees {V : Type} `{SeminormedSpaceLike V}
    (U : V -> Prop) (p f : V -> nat)
    (hsub : SublinearLike p)
    (hlin : LinearFunctionalLike f)
    (hdom : DominatedLike p f)
    (hex : exists g : V -> nat,
      LinearFunctionalLike g /\ ExtendsLike U f g /\ DominatedLike p g) :
    exists g : V -> nat, ExtendsLike U f g.
Proof.
  assert (hfull :
    exists g : V -> nat,
      LinearFunctionalLike g /\ ExtendsLike U f g /\ DominatedLike p g).
  {
    exact (hb_extension_exists (U := U) (p := p) (f := f) hsub hlin hdom hex).
  }
  destruct hfull as [g [hgLin [hgExt hgDom]]].
  assert (hdomg : DominatedLike p g).
  { exact hgDom. }
  exists g.
  exact hgExt.
Qed.

Lemma separation_from_hb {V : Type} `{SeminormedSpaceLike V}
    (U : V -> Prop) (p f : V -> nat) (x y : V)
    (hsub : SublinearLike p)
    (hlin : LinearFunctionalLike f)
    (hdom : DominatedLike p f)
    (hex : exists g : V -> nat,
      LinearFunctionalLike g /\ ExtendsLike U f g /\ DominatedLike p g)
    (hstrict : forall g : V -> nat,
      LinearFunctionalLike g -> DominatedLike p g -> ExtendsLike U f g -> g x < g y) :
    exists g : V -> nat, SeparatesLike g x y /\ DominatedLike p g.
Proof.
  assert (hfull :
    exists g : V -> nat,
      LinearFunctionalLike g /\ ExtendsLike U f g /\ DominatedLike p g).
  {
    exact (hb_extension_exists (U := U) (p := p) (f := f) hsub hlin hdom hex).
  }
  destruct hfull as [g [hgLin [hgExt hgDom]]].
  assert (hlt : g x < g y).
  { apply (hstrict g); assumption. }
  assert (hsep : SeparatesLike g x y).
  { exact hlt. }
  exists g.
  split.
  - exact hsep.
  - exact hgDom.
Qed.

Lemma dual_separates_points {V : Type} `{SeminormedSpaceLike V}
    (x y : V)
    (hxy : x <> y)
    (hsep : forall a b : V, a <> b -> exists g : V -> nat, SeparatesLike g a b /\ g a <> g b) :
    exists g : V -> nat, g x <> g y.
Proof.
  assert (hw : exists g : V -> nat, SeparatesLike g x y /\ g x <> g y).
  { apply hsep. exact hxy. }
  destruct hw as [g [hglt hneq]].
  exists g.
  exact hneq.
Qed.

Lemma minkowski_functional_bound {V : Type} `{SeminormedSpaceLike V}
    (p g : V -> nat)
    (hsub : SublinearLike p)
    (hlin : LinearFunctionalLike g)
    (hdom : DominatedLike p g)
    (x y : V)
    (hbound : g (x +v y) <= p x + p y) :
    g (x +v y) <= p x + p y.
Proof.
  destruct hsub as [hzero [hsubAdd hsubSmul]].
  destruct hlin as [hlinAdd hlinSmul].
  assert (hsub_add : p (x +v y) <= p x + p y).
  { apply hsubAdd. }
  assert (hdom_add : g (x +v y) <= p (x +v y)).
  { apply hdom. }
  assert (hcompat : g x + g y = g (x +v y)).
  {
    symmetry.
    apply hlinAdd.
  }
  assert (hsum_bound : g x + g y <= p x + p y).
  {
    rewrite hcompat.
    exact hbound.
  }
  rewrite hlinAdd.
  exact hsum_bound.
Qed.
