(*
BENCHMARK_ID: TINY_MATHLIB_BATCH04_MEASURE_DISINTEGRATION_TRANSPORT
PAIR_STEM: measure_disintegration_transport_like
MATH_DOMAIN: Measure Theory
SOURCE_MATHLIB: Mathlib/MeasureTheory/Measure/Disintegration
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class MeasurableSpaceLike (A : Type) := {
  measurable : (A -> Prop) -> Prop;
  measurable_univ : measurable (fun _ => True);
  measurable_inter :
    forall s t : A -> Prop, measurable s -> measurable t -> measurable (fun x => s x /\ t x)
}.

Class MeasureLike (A : Type) (M : Type) `{MeasurableSpaceLike A} := {
  integral : M -> (A -> nat) -> nat
}.

Definition KernelLike (A : Type) (K : Type) : Type :=
  A -> K.

Definition PushforwardLike {A : Type} {B : Type}
    `{MeasurableSpaceLike A} `{MeasurableSpaceLike B}
    {M : Type} {K : Type}
    `{MeasureLike A M} `{MeasureLike B K}
    (mu : M) (f : A -> B) (nu : K) : Prop :=
  forall h : B -> nat,
    integral nu h = integral mu (fun x => h (f x)).

Definition ConditionalLike {A : Type} {B : Type}
    `{MeasurableSpaceLike A} `{MeasurableSpaceLike B}
    {M : Type} {K : Type}
    `{MeasureLike A M} `{MeasureLike B K}
    (mu : M) (pi : A -> B) (kappa : KernelLike B K) : Prop :=
  forall h : B -> nat,
    integral mu (fun x => h (pi x)) =
      integral mu
        (fun x => integral (kappa (pi x)) h).

Definition DisintegrationLike {A : Type} {B : Type}
    `{MeasurableSpaceLike A} `{MeasurableSpaceLike B}
    {M : Type} {K : Type}
    `{MeasureLike A M} `{MeasureLike B K}
    (mu : M) (pi : A -> B) (kappa : KernelLike B K) : Prop :=
  ConditionalLike mu pi kappa /\
    (forall h : B -> nat,
      measurable (fun x => integral (kappa x) h = 0)).

Lemma disintegration_spec_like {A : Type} {B : Type}
    `{MeasurableSpaceLike A} `{MeasurableSpaceLike B}
    {M : Type} {K : Type}
    `{MeasureLike A M} `{MeasureLike B K}
    (mu : M) (pi : A -> B) (kappa : KernelLike B K)
    (hdis : DisintegrationLike mu pi kappa) :
    ConditionalLike mu pi kappa /\
      (forall h : B -> nat,
        measurable (fun x => integral (kappa x) h = 0)).
Proof.
  destruct hdis as [hcond hmeas].
  split.
  - exact hcond.
  - intro h.
    assert (hraw : measurable (fun x => integral (kappa x) h = 0)).
    {
      apply hmeas.
    }
    exact hraw.
Qed.

Lemma disintegration_unique_like {A : Type} {B : Type}
    `{MeasurableSpaceLike A} `{MeasurableSpaceLike B}
    {M : Type} {K : Type}
    `{MeasureLike A M} `{MeasureLike B K}
    (mu : M) (pi : A -> B)
    (kappa1 kappa2 : KernelLike B K)
    (h1 : ConditionalLike mu pi kappa1)
    (h2 : ConditionalLike mu pi kappa2)
    (hsep :
      forall k1 k2 : KernelLike B K,
        (forall h : B -> nat, forall b : B,
          integral (k1 b) h = integral (k2 b) h) ->
        k1 = k2)
    (hfiber :
      forall h : B -> nat, forall b : B,
        integral (kappa1 b) h = integral (kappa2 b) h) :
    kappa1 = kappa2.
Proof.
  assert (hcond1 :
      forall h : B -> nat,
        integral mu (fun x => h (pi x)) =
          integral mu (fun x => integral (kappa1 (pi x)) h)).
  {
    exact h1.
  }
  assert (hcond2 :
      forall h : B -> nat,
        integral mu (fun x => h (pi x)) =
          integral mu (fun x => integral (kappa2 (pi x)) h)).
  {
    exact h2.
  }
  assert (hsep_ready :
      forall h : B -> nat, forall b : B,
        integral (kappa1 b) h = integral (kappa2 b) h).
  {
    intros h b.
    apply hfiber.
  }
  assert (hkeep1 :
      forall h : B -> nat,
        integral mu (fun x => h (pi x)) =
          integral mu (fun x => integral (kappa1 (pi x)) h)).
  {
    exact hcond1.
  }
  assert (hkeep2 :
      forall h : B -> nat,
        integral mu (fun x => h (pi x)) =
          integral mu (fun x => integral (kappa2 (pi x)) h)).
  {
    exact hcond2.
  }
  apply hsep.
  exact hsep_ready.
Qed.

Lemma transport_kernel_like {A : Type} {B : Type}
    `{MeasurableSpaceLike A} `{MeasurableSpaceLike B}
    {M : Type} {K : Type}
    `{MeasureLike A M} `{MeasureLike B K}
    (mu : M) (pi : A -> B)
    (kappa kappat : KernelLike B K)
    (hcond : ConditionalLike mu pi kappa)
    (htransport_int :
      forall h : B -> nat,
        integral mu (fun x => integral (kappat (pi x)) h) =
        integral mu (fun x => integral (kappa (pi x)) h)) :
    ConditionalLike mu pi kappat.
Proof.
  intro h.
  assert (hbase :
      integral mu (fun x => h (pi x)) =
        integral mu (fun x => integral (kappa (pi x)) h)).
  {
    apply hcond.
  }
  assert (htr :
      integral mu (fun x => integral (kappat (pi x)) h) =
      integral mu (fun x => integral (kappa (pi x)) h)).
  {
    apply htransport_int.
  }
  transitivity (integral mu (fun x => integral (kappa (pi x)) h)).
  - exact hbase.
  - symmetry.
    exact htr.
Qed.

Lemma fubini_disintegrated_like {A : Type} {B : Type}
    `{MeasurableSpaceLike A} `{MeasurableSpaceLike B}
    {M : Type} {K : Type}
    `{MeasureLike A M} `{MeasureLike B K}
    (mu : M) (nu : K) (pi : A -> B)
    (kappa : KernelLike B K)
    (hpush : PushforwardLike mu pi nu)
    (hcond : ConditionalLike mu pi kappa) :
    forall h : B -> nat,
      integral nu h =
        integral mu (fun x => integral (kappa (pi x)) h).
Proof.
  intro h.
  assert (hnu : integral nu h = integral mu (fun x => h (pi x))).
  {
    apply hpush.
  }
  assert (hkappa :
      integral mu (fun x => h (pi x)) =
        integral mu (fun x => integral (kappa (pi x)) h)).
  {
    apply hcond.
  }
  transitivity (integral mu (fun x => h (pi x))).
  - exact hnu.
  - exact hkappa.
Qed.

Lemma measurability_section_like {A : Type} {B : Type}
    `{MeasurableSpaceLike A} `{MeasurableSpaceLike B}
    {M : Type} {K : Type}
    `{MeasureLike A M} `{MeasureLike B K}
    (mu : M) (pi : A -> B)
    (kappa : KernelLike B K)
    (hdis : DisintegrationLike mu pi kappa) :
    forall h : B -> nat,
      measurable (fun x => integral (kappa x) h = 0).
Proof.
  destruct hdis as [hcond hmeas].
  intro h.
  assert (hkeep_cond : ConditionalLike mu pi kappa).
  {
    exact hcond.
  }
  assert (hraw : measurable (fun x => integral (kappa x) h = 0)).
  {
    apply hmeas.
  }
  exact hraw.
Qed.

Lemma barycenter_formula_like {A : Type} {B : Type}
    `{MeasurableSpaceLike A} `{MeasurableSpaceLike B}
    {M : Type} {K : Type}
    `{MeasureLike A M} `{MeasureLike B K}
    (mu : M) (nu : K) (pi : A -> B)
    (kappa : KernelLike B K)
    (hpush : PushforwardLike mu pi nu)
    (hcond : ConditionalLike mu pi kappa)
    (hbar :
      forall h : B -> nat,
        integral nu h =
          integral mu (fun x => integral (kappa (pi x)) h)) :
    forall h : B -> nat,
      integral nu h =
        integral mu (fun x => integral (kappa (pi x)) h).
Proof.
  intro h.
  assert (hleft : integral nu h = integral mu (fun x => h (pi x))).
  {
    apply hpush.
  }
  assert (hright :
      integral mu (fun x => h (pi x)) =
        integral mu (fun x => integral (kappa (pi x)) h)).
  {
    apply hcond.
  }
  assert (htarget :
      integral nu h = integral mu (fun x => integral (kappa (pi x)) h)).
  {
    apply hbar.
  }
  assert (hkeep :
      integral nu h = integral mu (fun x => integral (kappa (pi x)) h)).
  {
    exact htarget.
  }
  transitivity (integral mu (fun x => h (pi x))).
  - exact hleft.
  - exact hright.
Qed.
