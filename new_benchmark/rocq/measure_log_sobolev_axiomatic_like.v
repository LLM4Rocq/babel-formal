(*
BENCHMARK_ID: TINY_MATHLIB_BATCH05_MEASURE_LOG_SOBOLEV_AXIOMATIC_LIKE
PAIR_STEM: measure_log_sobolev_axiomatic_like
MATH_DOMAIN: Measure Theory
SOURCE_MATHLIB: Mathlib/MeasureTheory/Integral
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class MeasureStruct_log_sobolev (Omega : Type) := {
  density : Omega -> nat;
  kernel : Omega -> Omega -> nat;
  entropy : (Omega -> nat) -> nat;
  integral : (Omega -> nat) -> nat;
  density_nonneg_axiom : forall x : Omega, 0 <= density x;
  integral_mono_axiom :
    forall f g : Omega -> nat,
      (forall x : Omega, f x <= g x) ->
      integral f <= integral g;
  transport_identity_axiom :
    forall x : Omega, kernel x x = density x;
  chain_rule_axiom :
    forall f : Omega -> nat,
      entropy f + integral f = integral (fun x : Omega => f x + density x);
  dual_variational_axiom :
    forall f : Omega -> nat,
      entropy f <= integral f;
  concentration_axiom :
    forall f : Omega -> nat,
      forall n : nat,
        integral f <= n ->
        entropy f <= n;
  decomposition_axiom :
    forall f g : Omega -> nat,
      integral (fun x : Omega => f x + g x) = integral f + integral g
}.

Definition DensityFn_log_sobolev
    (Omega : Type) : Type :=
  Omega -> nat.

Definition KernelMap_log_sobolev
    (Omega : Type) : Type :=
  Omega -> Omega -> nat.

Definition EntropyLike_log_sobolev
    {Omega : Type} `{MeasureStruct_log_sobolev Omega}
    (f : DensityFn_log_sobolev Omega) : nat :=
  entropy f.

Definition IntegralForm_log_sobolev
    {Omega : Type} `{MeasureStruct_log_sobolev Omega}
    (f : DensityFn_log_sobolev Omega) : nat :=
  integral f.

Lemma density_nonneg_log_sobolev
    {Omega : Type} `{MeasureStruct_log_sobolev Omega}
    (x : Omega) :
    exists n : nat, kernel x x = n /\ 0 <= n.
Proof.
  exists (density x).
  split.
  - apply transport_identity_axiom.
  - apply density_nonneg_axiom.
Qed.

Lemma integral_mono_log_sobolev
    {Omega : Type} `{MeasureStruct_log_sobolev Omega}
    (f g : DensityFn_log_sobolev Omega)
    (hfg : forall x : Omega, f x <= g x) :
    IntegralForm_log_sobolev f <= IntegralForm_log_sobolev g /\
    IntegralForm_log_sobolev (fun x : Omega => f x + g x) =
      IntegralForm_log_sobolev f + IntegralForm_log_sobolev g.
Proof.
  assert (hMono : integral f <= integral g).
  {
    apply (integral_mono_axiom f g).
    exact hfg.
  }
  assert (hDecomp : integral (fun x : Omega => f x + g x) = integral f + integral g).
  { apply (decomposition_axiom f g). }
  split.
  - exact hMono.
  - exact hDecomp.
Qed.

Lemma transport_identity_log_sobolev
    {Omega : Type} `{MeasureStruct_log_sobolev Omega}
    (x : Omega) :
    kernel x x = density x /\
    EntropyLike_log_sobolev (fun _ : Omega => 0) <=
      IntegralForm_log_sobolev (fun _ : Omega => 0).
Proof.
  assert (hDiag : kernel x x = density x).
  { apply transport_identity_axiom. }
  assert (hZero : entropy (fun _ : Omega => 0) <= integral (fun _ : Omega => 0)).
  { apply dual_variational_axiom. }
  split.
  - exact hDiag.
  - exact hZero.
Qed.

Lemma chain_rule_measure_log_sobolev
    {Omega : Type} `{MeasureStruct_log_sobolev Omega}
    (f : DensityFn_log_sobolev Omega) :
    IntegralForm_log_sobolev (fun x : Omega => f x + density x) =
      EntropyLike_log_sobolev f + IntegralForm_log_sobolev f.
Proof.
  assert (hChain : entropy f + integral f = integral (fun x : Omega => f x + density x)).
  { apply chain_rule_axiom. }
  symmetry.
  exact hChain.
Qed.

Lemma dual_variational_bound_log_sobolev
    {Omega : Type} `{MeasureStruct_log_sobolev Omega}
    (f : DensityFn_log_sobolev Omega) :
    EntropyLike_log_sobolev (fun x : Omega => kernel x x) <=
      IntegralForm_log_sobolev (fun x : Omega => kernel x x).
Proof.
  apply dual_variational_axiom.
Qed.

Lemma concentration_step_log_sobolev
    {Omega : Type} `{MeasureStruct_log_sobolev Omega}
    (f : DensityFn_log_sobolev Omega)
    (n : nat)
    (hBound : IntegralForm_log_sobolev f <= n) :
    exists m : nat, m = n /\ EntropyLike_log_sobolev f <= m.
Proof.
  assert (hConc : entropy f <= n).
  { exact (@concentration_axiom Omega H f n hBound). }
  exists n.
  split.
  - reflexivity.
  - exact hConc.
Qed.

Lemma decomposition_formula_log_sobolev
    {Omega : Type} `{MeasureStruct_log_sobolev Omega}
    (f g : DensityFn_log_sobolev Omega) :
    IntegralForm_log_sobolev (fun x : Omega => f x + g x) =
      IntegralForm_log_sobolev f + IntegralForm_log_sobolev g.
Proof.
  apply (decomposition_axiom f g).
Qed.
