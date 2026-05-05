(*
BENCHMARK_ID: TINY_MATHLIB_BATCH05_MEASURE_WASSERSTEIN_DUALITY_AXIOMATIC_LIKE
PAIR_STEM: measure_wasserstein_duality_axiomatic_like
MATH_DOMAIN: Measure Theory
SOURCE_MATHLIB: Mathlib/MeasureTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class MeasureStruct_wasserstein_duality (Omega M : Type) := {
  density : M -> Omega -> Prop;
  kernel : M -> M;
  entropy : M -> Prop;
  integral : (Omega -> Prop) -> M -> Prop;
  density_entropy_axiom :
    forall mu : M,
      forall x : Omega,
        density mu x -> entropy mu;
  integral_mono_axiom :
    forall f g : Omega -> Prop,
      forall mu : M,
        (forall x : Omega, f x -> g x) ->
        integral f mu ->
          integral g mu;
  transport_id_axiom :
    forall mu : M,
      kernel mu = mu ->
      forall x : Omega,
        density (kernel mu) x ->
          density mu x;
  chain_rule_axiom :
    forall mu : M,
      entropy mu -> entropy (kernel mu);
  dual_bound_axiom :
    forall f : Omega -> Prop,
      forall mu : M,
        (forall x : Omega, f x -> density mu x) ->
        integral f mu ->
          entropy mu;
  concentration_axiom :
    forall mu : M,
      entropy mu -> entropy (kernel (kernel mu));
  decomposition_axiom :
    forall mu : M,
      forall x : Omega,
        entropy (kernel mu) ->
        density (kernel mu) x ->
          density mu x
}.

Definition DensityFn_wasserstein_duality
    {Omega M : Type} `{MeasureStruct_wasserstein_duality Omega M} :
    M -> Omega -> Prop :=
  density.

Definition KernelMap_wasserstein_duality
    {Omega M : Type} `{MeasureStruct_wasserstein_duality Omega M} :
    M -> M :=
  kernel.

Definition EntropyLike_wasserstein_duality
    {Omega M : Type} `{MeasureStruct_wasserstein_duality Omega M} :
    M -> Prop :=
  entropy.

Definition IntegralForm_wasserstein_duality
    {Omega M : Type} `{MeasureStruct_wasserstein_duality Omega M} :
    (Omega -> Prop) -> M -> Prop :=
  integral.

Lemma density_nonneg_wasserstein_duality
    {Omega M : Type} `{MeasureStruct_wasserstein_duality Omega M}
    (mu : M)
    (x : Omega)
    (hDen : DensityFn_wasserstein_duality mu x) :
    EntropyLike_wasserstein_duality mu.
Proof.
  assert (hEntropy : entropy mu).
  { apply (density_entropy_axiom mu x). exact hDen. }
  assert (hPack : EntropyLike_wasserstein_duality mu).
  { exact hEntropy. }
  exact hPack.
Qed.

Lemma integral_mono_wasserstein_duality
    {Omega M : Type} `{MeasureStruct_wasserstein_duality Omega M}
    (f g : Omega -> Prop)
    (mu : M)
    (hfg : forall x : Omega, f x -> g x)
    (hInt : IntegralForm_wasserstein_duality f mu) :
    IntegralForm_wasserstein_duality g mu.
Proof.
  assert (hStep1 : integral g mu).
  { apply (integral_mono_axiom f g mu hfg hInt). }
  assert (hStep2 : IntegralForm_wasserstein_duality g mu).
  { exact hStep1. }
  exact hStep2.
Qed.

Lemma transport_identity_wasserstein_duality
    {Omega M : Type} `{MeasureStruct_wasserstein_duality Omega M}
    (mu : M)
    (hFix : KernelMap_wasserstein_duality mu = mu)
    (x : Omega)
    (hDenK : DensityFn_wasserstein_duality (KernelMap_wasserstein_duality mu) x) :
    DensityFn_wasserstein_duality mu x.
Proof.
  assert (hStep1 : density (kernel mu) x).
  { exact hDenK. }
  assert (hStep2 : density mu x).
  {
    eapply (transport_id_axiom (mu := mu)).
    - exact hFix.
    - exact hStep1.
  }
  assert (hPack : DensityFn_wasserstein_duality mu x).
  { exact hStep2. }
  exact hPack.
Qed.

Lemma chain_rule_measure_wasserstein_duality
    {Omega M : Type} `{MeasureStruct_wasserstein_duality Omega M}
    (mu : M)
    (hEnt : EntropyLike_wasserstein_duality mu) :
    EntropyLike_wasserstein_duality (KernelMap_wasserstein_duality mu) /\
    EntropyLike_wasserstein_duality (KernelMap_wasserstein_duality (KernelMap_wasserstein_duality mu)).
Proof.
  assert (hFirst : entropy (kernel mu)).
  { apply (chain_rule_axiom mu). exact hEnt. }
  assert (hSecond : entropy (kernel (kernel mu))).
  { apply (concentration_axiom mu). exact hEnt. }
  assert (hPack1 : EntropyLike_wasserstein_duality (KernelMap_wasserstein_duality mu)).
  { exact hFirst. }
  assert (hPack2 : EntropyLike_wasserstein_duality (KernelMap_wasserstein_duality (KernelMap_wasserstein_duality mu))).
  { exact hSecond. }
  split.
  - exact hPack1.
  - exact hPack2.
Qed.

Lemma dual_variational_bound_wasserstein_duality
    {Omega M : Type} `{MeasureStruct_wasserstein_duality Omega M}
    (f : Omega -> Prop)
    (mu : M)
    (hDom : forall x : Omega, f x -> DensityFn_wasserstein_duality mu x)
    (hInt : IntegralForm_wasserstein_duality f mu) :
    EntropyLike_wasserstein_duality mu.
Proof.
  assert (hRaw : entropy mu).
  { apply (dual_bound_axiom f mu hDom hInt). }
  assert (hKeep : EntropyLike_wasserstein_duality mu).
  { exact hRaw. }
  exact hKeep.
Qed.

Lemma concentration_step_wasserstein_duality
    {Omega M : Type} `{MeasureStruct_wasserstein_duality Omega M}
    (mu : M)
    (hEnt : EntropyLike_wasserstein_duality mu) :
    EntropyLike_wasserstein_duality (KernelMap_wasserstein_duality (KernelMap_wasserstein_duality mu)).
Proof.
  assert (hChain :
      EntropyLike_wasserstein_duality (KernelMap_wasserstein_duality mu) /\
      EntropyLike_wasserstein_duality (KernelMap_wasserstein_duality (KernelMap_wasserstein_duality mu))).
  { apply (chain_rule_measure_wasserstein_duality mu hEnt). }
  exact (proj2 hChain).
Qed.

Lemma decomposition_formula_wasserstein_duality
    {Omega M : Type} `{MeasureStruct_wasserstein_duality Omega M}
    (mu : M)
    (hFix : KernelMap_wasserstein_duality mu = mu)
    (x : Omega)
    (hEntK : EntropyLike_wasserstein_duality (KernelMap_wasserstein_duality mu))
    (hDenK : DensityFn_wasserstein_duality (KernelMap_wasserstein_duality mu) x) :
    DensityFn_wasserstein_duality mu x.
Proof.
  assert (hViaTransport : DensityFn_wasserstein_duality mu x).
  {
    apply (transport_identity_wasserstein_duality (mu := mu)).
    - exact hFix.
    - exact hDenK.
  }
  assert (hViaDecomposition : DensityFn_wasserstein_duality mu x).
  { apply (decomposition_axiom mu x hEntK hDenK). }
  assert (hKeep : DensityFn_wasserstein_duality mu x).
  { exact hViaTransport. }
  exact hViaDecomposition.
Qed.
