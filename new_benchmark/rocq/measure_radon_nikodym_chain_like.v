(*
BENCHMARK_ID: TINY_MATHLIB_BATCH03_MEASURE_RN_CHAIN_LIKE
PAIR_STEM: measure_radon_nikodym_chain_like
MATH_DOMAIN: Measure Theory
SOURCE_MATHLIB: Mathlib/MeasureTheory/Decomposition/RadonNikodym
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class MeasurableSpaceLike (A : Type) := {
  measurable : (A -> Prop) -> Prop
}.

Class MeasureLike (A : Type) (M : Type) `{MeasurableSpaceLike A} := {
  integral : M -> (A -> nat) -> nat
}.

Definition AbsolutelyContinuous {A : Type} {M : Type}
    `{MeasurableSpaceLike A} `{MeasureLike A M}
    (mu nu : M) : Prop :=
  forall f : A -> nat,
    integral nu f = 0 ->
      integral mu f = 0.

Definition RNDerivative {A : Type} {M : Type}
    `{MeasurableSpaceLike A} `{MeasureLike A M}
    (mu nu : M) (f : A -> nat) : Prop :=
  forall g : A -> nat,
    integral mu g = integral nu (fun x => g x * f x).

Definition IntegrableLike {A : Type} {M : Type}
    `{MeasurableSpaceLike A} `{MeasureLike A M}
    (nu : M) (f : A -> nat) : Prop :=
  exists n : nat, integral nu f = n.

Definition IntegralLike {A : Type} {M : Type}
    `{MeasurableSpaceLike A} `{MeasureLike A M}
    (nu : M) (f : A -> nat) : nat :=
  integral nu f.

Lemma rn_spec {A : Type} {M : Type}
    `{MeasurableSpaceLike A} `{MeasureLike A M}
    (mu nu : M) (f : A -> nat)
    (hder : RNDerivative mu nu f) :
    IntegrableLike nu f /\
      (forall g : A -> nat, IntegralLike mu g = IntegralLike nu (fun x => g x * f x)).
Proof.
  split.
  - exists (IntegralLike nu f).
    reflexivity.
  - intro g.
    assert (hraw : integral mu g = integral nu (fun x => g x * f x)).
    { apply hder. }
    exact hraw.
Qed.

Lemma rn_unique {A : Type} {M : Type}
    `{MeasurableSpaceLike A} `{MeasureLike A M}
    (mu nu : M) (f g : A -> nat)
    (hf : RNDerivative mu nu f) (hg : RNDerivative mu nu g)
    (hsep : forall f0 g0 : A -> nat,
      (forall h : A -> nat,
        IntegralLike nu (fun x => h x * f0 x) = IntegralLike nu (fun x => h x * g0 x)) ->
      f0 = g0) :
    f = g.
Proof.
  apply (hsep f g).
  intro h.
  assert (hf_h : IntegralLike mu h = IntegralLike nu (fun x => h x * f x)).
  { apply hf. }
  assert (hg_h : IntegralLike mu h = IntegralLike nu (fun x => h x * g x)).
  { apply hg. }
  transitivity (IntegralLike mu h).
  - symmetry.
    exact hf_h.
  - exact hg_h.
Qed.

Lemma rn_linear_combo {A : Type} {M : Type}
    `{MeasurableSpaceLike A} `{MeasureLike A M}
    (mu1 mu2 musum nu : M) (f g : A -> nat)
    (hf : RNDerivative mu1 nu f) (hg : RNDerivative mu2 nu g)
    (hsum : forall h : A -> nat,
      IntegralLike musum h = IntegralLike mu1 h + IntegralLike mu2 h)
    (hmul_add : forall h f0 g0 : A -> nat,
      IntegralLike nu (fun x => h x * (f0 x + g0 x)) =
        IntegralLike nu (fun x => h x * f0 x) + IntegralLike nu (fun x => h x * g0 x)) :
    RNDerivative musum nu (fun x => f x + g x).
Proof.
  intro h.
  assert (hsum_h : integral musum h = integral mu1 h + integral mu2 h).
  { apply hsum. }
  assert (hf_h : integral mu1 h = integral nu (fun x => h x * f x)).
  { apply hf. }
  assert (hg_h : integral mu2 h = integral nu (fun x => h x * g x)).
  { apply hg. }
  rewrite hsum_h.
  rewrite hf_h.
  rewrite hg_h.
  symmetry.
  apply hmul_add.
Qed.

Lemma rn_chain_rule_like {A : Type} {M : Type}
    `{MeasurableSpaceLike A} `{MeasureLike A M}
    (mu nu rho : M) (f g : A -> nat)
    (hmunu : RNDerivative mu nu f) (hnurho : RNDerivative nu rho g)
    (hmul_assoc : forall h f0 g0 : A -> nat,
      IntegralLike rho (fun x => h x * (f0 x * g0 x)) =
        IntegralLike rho (fun x => (h x * f0 x) * g0 x)) :
    RNDerivative mu rho (fun x => f x * g x).
Proof.
  intro h.
  assert (hmu : IntegralLike mu h = IntegralLike nu (fun x => h x * f x)).
  { apply hmunu. }
  assert (hnu : IntegralLike nu (fun x => h x * f x) =
    IntegralLike rho (fun x => (h x * f x) * g x)).
  { apply hnurho. }
  transitivity (IntegralLike nu (fun x => h x * f x)).
  - exact hmu.
  - transitivity (IntegralLike rho (fun x => (h x * f x) * g x)).
    + exact hnu.
    + symmetry.
      apply hmul_assoc.
Qed.

Lemma rn_restrict_like {A : Type} {M : Type}
    `{MeasurableSpaceLike A} `{MeasureLike A M}
    (mu mur nu : M) (r f : A -> nat)
    (hf : RNDerivative mu nu f)
    (hrestrict : forall h : A -> nat,
      IntegralLike mur h = IntegralLike mu (fun x => r x * h x))
    (hswap : forall h f0 : A -> nat,
      IntegralLike nu (fun x => (r x * h x) * f0 x) =
        IntegralLike nu (fun x => h x * (r x * f0 x))) :
    RNDerivative mur nu (fun x => r x * f x).
Proof.
  intro h.
  assert (hmur : IntegralLike mur h = IntegralLike mu (fun x => r x * h x)).
  { apply hrestrict. }
  assert (hmu : IntegralLike mu (fun x => r x * h x) =
    IntegralLike nu (fun x => (r x * h x) * f x)).
  { apply hf. }
  assert (hnu : IntegralLike nu (fun x => (r x * h x) * f x) =
    IntegralLike nu (fun x => h x * (r x * f x))).
  { apply hswap. }
  transitivity (IntegralLike mu (fun x => r x * h x)).
  - exact hmur.
  - transitivity (IntegralLike nu (fun x => (r x * h x) * f x)).
    + exact hmu.
    + exact hnu.
Qed.

Lemma rn_zero_of_singular_like {A : Type} {M : Type}
    `{MeasurableSpaceLike A} `{MeasureLike A M}
    (mu nu : M)
    (hsing : forall h : A -> nat, IntegralLike mu h = 0)
    (hzero : forall h : A -> nat, IntegralLike nu (fun x => h x * 0) = 0) :
    RNDerivative mu nu (fun _ : A => 0).
Proof.
  intro h.
  assert (hmu0 : integral mu h = 0).
  { apply hsing. }
  assert (hnu0 : integral nu (fun x => h x * 0) = 0).
  { apply hzero. }
  rewrite hmu0.
  symmetry.
  exact hnu0.
Qed.
