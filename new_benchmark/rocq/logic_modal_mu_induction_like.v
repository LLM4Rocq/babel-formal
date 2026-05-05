(*
BENCHMARK_ID: TINY_MATHLIB_BATCH03_LOGIC_MODAL_MU_INDUCTION
PAIR_STEM: logic_modal_mu_induction_like
MATH_DOMAIN: Logic
SOURCE_MATHLIB: Mathlib/Order/FixedPoints
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class CompleteBooleanAlgebraLike (A : Type) := {
  leq : A -> A -> Prop;
  compl : A -> A;
  box : A -> A;
  diamond : A -> A;
  mu : (A -> A) -> A;
  nu : (A -> A) -> A;
  leq_refl : forall a : A, leq a a;
  leq_trans : forall a b c : A, leq a b -> leq b c -> leq a c;
  compl_antitone : forall a b : A, leq a b -> leq (compl b) (compl a);
  compl_involutive : forall a : A, compl (compl a) = a;
  box_mono : forall a b : A, leq a b -> leq (box a) (box b);
  diamond_mono : forall a b : A, leq a b -> leq (diamond a) (diamond b);
  modal_dual : forall a : A, leq (diamond a) (compl (box (compl a)));
  mu_unfold_axiom :
    forall f : A -> A,
      (forall a b : A, leq a b -> leq (f a) (f b)) ->
      leq (f (mu f)) (mu f);
  mu_least_axiom :
    forall f : A -> A,
      (forall a b : A, leq a b -> leq (f a) (f b)) ->
      forall x : A, leq (f x) x -> leq (mu f) x;
  nu_unfold_axiom :
    forall f : A -> A,
      (forall a b : A, leq a b -> leq (f a) (f b)) ->
      leq (nu f) (f (nu f));
  nu_greatest_axiom :
    forall f : A -> A,
      (forall a b : A, leq a b -> leq (f a) (f b)) ->
      forall x : A, leq x (f x) -> leq x (nu f);
  mu_nu_dual_axiom :
    forall f : A -> A,
      (forall a b : A, leq a b -> leq (f a) (f b)) ->
      leq (mu f) (compl (nu (fun x => compl (f (compl x)))))
}.

Infix "<=" := leq (at level 70).

Definition monotone {A : Type} `{CompleteBooleanAlgebraLike A} (f : A -> A) : Prop :=
  forall a b : A, a <= b -> f a <= f b.

Definition boxLike {A : Type} `{CompleteBooleanAlgebraLike A} (x : A) : A :=
  box x.

Definition diamondLike {A : Type} `{CompleteBooleanAlgebraLike A} (x : A) : A :=
  diamond x.

Definition muLike {A : Type} `{CompleteBooleanAlgebraLike A} (f : A -> A) : A :=
  mu f.

Definition nuLike {A : Type} `{CompleteBooleanAlgebraLike A} (f : A -> A) : A :=
  nu f.

Lemma mu_unfold {A : Type} `{CompleteBooleanAlgebraLike A}
    (f : A -> A) (hmono : monotone f) :
    f (muLike f) <= muLike f.
Proof.
  assert (hmono_explicit : forall a b : A, leq a b -> leq (f a) (f b)).
  {
    intros a b hab.
    exact (hmono a b hab).
  }
  assert (hcore : leq (f (mu f)) (mu f)).
  {
    apply mu_unfold_axiom.
    exact hmono_explicit.
  }
  exact hcore.
Qed.

Lemma mu_induction {A : Type} `{CompleteBooleanAlgebraLike A}
    (f : A -> A) (hmono : monotone f) (x : A) (hx : f x <= x) :
    muLike f <= x.
Proof.
  assert (hmono_explicit : forall a b : A, leq a b -> leq (f a) (f b)).
  {
    intros a b hab.
    exact (hmono a b hab).
  }
  assert (hleast : leq (mu f) x).
  {
    apply mu_least_axiom.
    - exact hmono_explicit.
    - exact hx.
  }
  assert (hself : muLike f <= muLike f).
  {
    apply leq_refl.
  }
  assert (hchain : muLike f <= x).
  {
    exact (leq_trans _ _ _ hself hleast).
  }
  exact hchain.
Qed.

Lemma nu_unfold {A : Type} `{CompleteBooleanAlgebraLike A}
    (f : A -> A) (hmono : monotone f) :
    nuLike f <= f (nuLike f).
Proof.
  assert (hmono_explicit : forall a b : A, leq a b -> leq (f a) (f b)).
  {
    intros a b hab.
    exact (hmono a b hab).
  }
  assert (hcore : leq (nu f) (f (nu f))).
  {
    apply nu_unfold_axiom.
    exact hmono_explicit.
  }
  exact hcore.
Qed.

Lemma nu_coinduction {A : Type} `{CompleteBooleanAlgebraLike A}
    (f : A -> A) (hmono : monotone f) (x : A) (hx : x <= f x) :
    x <= nuLike f.
Proof.
  assert (hmono_explicit : forall a b : A, leq a b -> leq (f a) (f b)).
  {
    intros a b hab.
    exact (hmono a b hab).
  }
  assert (hgreatest : leq x (nu f)).
  {
    apply nu_greatest_axiom.
    - exact hmono_explicit.
    - exact hx.
  }
  assert (hself : x <= x).
  {
    apply leq_refl.
  }
  assert (hchain : x <= nuLike f).
  {
    exact (leq_trans _ _ _ hself hgreatest).
  }
  exact hchain.
Qed.

Lemma bekic_split_like {A : Type} `{CompleteBooleanAlgebraLike A}
    (f g : A -> A)
    (hmono_f : monotone f) (hmono_g : monotone g)
    (x y : A) (hx : f x <= x) (hy : y <= g y) :
    muLike f <= x /\ y <= nuLike g.
Proof.
  assert (hmu : muLike f <= x).
  {
    exact (@mu_induction A H f hmono_f x hx).
  }
  assert (hnu : y <= nuLike g).
  {
    exact (@nu_coinduction A H g hmono_g y hy).
  }
  split.
  - exact hmu.
  - exact hnu.
Qed.

Lemma modal_mu_duality {A : Type} `{CompleteBooleanAlgebraLike A}
    (f : A -> A) (hmono : monotone f) :
    muLike f <= compl (nuLike (fun x => compl (f (compl x)))).
Proof.
  set (dualOp := fun x : A => compl (f (compl x))).
  assert (hdual_mono : monotone dualOp).
  {
    intros a b hab.
    unfold dualOp.
    assert (hcbca : compl b <= compl a).
    {
      apply compl_antitone.
      exact hab.
    }
    assert (hfbfa : f (compl b) <= f (compl a)).
    {
      apply hmono.
      exact hcbca.
    }
    apply compl_antitone.
    exact hfbfa.
  }
  assert (hmono_explicit : forall a b : A, leq a b -> leq (f a) (f b)).
  {
    intros a b hab.
    exact (hmono a b hab).
  }
  assert (hcore : leq (mu f) (compl (nu (fun x : A => compl (f (compl x)))))).
  {
    apply mu_nu_dual_axiom.
    exact hmono_explicit.
  }
  assert (hdual_used : monotone dualOp).
  {
    exact hdual_mono.
  }
  exact hcore.
Qed.
