(*
BENCHMARK_ID: TINY_MATHLIB_BATCH05_MEASURE_TRANSPORT_ENTROPY_INEQUALITY_LIKE
PAIR_STEM: measure_transport_entropy_inequality_like
MATH_DOMAIN: Measure Theory
SOURCE_MATHLIB: Mathlib/MeasureTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class MeasureStruct_transport_entropy_inequality (Omega : Type) := {
  density : Omega -> nat;
  kernel : Omega -> Omega -> nat;
  entropy : (Omega -> nat) -> nat;
  integral : (Omega -> nat) -> nat;
  density_nonneg_axiom : forall x : Omega, 0 <= density x;
  integral_mono_axiom :
    forall f g : Omega -> nat,
      (forall x : Omega, f x <= g x) ->
        integral f <= integral g;
  kernel_identity_axiom :
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

Definition DensityFn_transport_entropy_inequality
    (Omega : Type) : Type :=
  Omega -> nat.

Definition KernelMap_transport_entropy_inequality
    (Omega : Type) : Type :=
  Omega -> Omega -> nat.

Definition EntropyLike_transport_entropy_inequality
    {Omega : Type} `{MeasureStruct_transport_entropy_inequality Omega}
    (f : DensityFn_transport_entropy_inequality Omega) : nat :=
  entropy f.

Definition IntegralForm_transport_entropy_inequality
    {Omega : Type} `{MeasureStruct_transport_entropy_inequality Omega}
    (f : DensityFn_transport_entropy_inequality Omega) : nat :=
  integral f.

Lemma density_nonneg_transport_entropy_inequality
    {Omega : Type} `{MeasureStruct_transport_entropy_inequality Omega}
    (x : Omega) :
    0 <= density x /\ kernel x x = density x.
Proof.
  assert (hNonneg : 0 <= density x).
  { apply density_nonneg_axiom. }
  assert (hDiag : kernel x x = density x).
  { apply kernel_identity_axiom. }
  split.
  - exact hNonneg.
  - exact hDiag.
Qed.

Lemma integral_mono_transport_entropy_inequality
    {Omega : Type} `{MeasureStruct_transport_entropy_inequality Omega}
    (f g : DensityFn_transport_entropy_inequality Omega)
    (hfg : forall x : Omega, f x <= g x) :
    IntegralForm_transport_entropy_inequality f <=
      IntegralForm_transport_entropy_inequality g /\
    EntropyLike_transport_entropy_inequality f <=
      IntegralForm_transport_entropy_inequality g.
Proof.
  assert (hMono : integral f <= integral g).
  { apply (integral_mono_axiom f g). exact hfg. }
  assert (hConc : entropy f <= integral g).
  { exact (@concentration_axiom Omega H f (integral g) hMono). }
  split.
  - exact hMono.
  - exact hConc.
Qed.

Lemma transport_identity_transport_entropy_inequality
    {Omega : Type} `{MeasureStruct_transport_entropy_inequality Omega}
    (x : Omega) :
    kernel x x = density x.
Proof.
  apply kernel_identity_axiom.
Qed.

Lemma chain_rule_measure_transport_entropy_inequality
    {Omega : Type} `{MeasureStruct_transport_entropy_inequality Omega}
    (f : DensityFn_transport_entropy_inequality Omega) :
    EntropyLike_transport_entropy_inequality f +
        IntegralForm_transport_entropy_inequality f =
      IntegralForm_transport_entropy_inequality (fun x : Omega => f x + density x) /\
    EntropyLike_transport_entropy_inequality f <=
      IntegralForm_transport_entropy_inequality f.
Proof.
  assert (hChain : entropy f + integral f = integral (fun x : Omega => f x + density x)).
  { apply chain_rule_axiom. }
  assert (hDual : entropy f <= integral f).
  { apply dual_variational_axiom. }
  split.
  - exact hChain.
  - exact hDual.
Qed.

Lemma dual_variational_bound_transport_entropy_inequality
    {Omega : Type} `{MeasureStruct_transport_entropy_inequality Omega}
    (f : DensityFn_transport_entropy_inequality Omega) :
    EntropyLike_transport_entropy_inequality f <=
      IntegralForm_transport_entropy_inequality f /\
    EntropyLike_transport_entropy_inequality (fun x : Omega => f x + density x) <=
      IntegralForm_transport_entropy_inequality (fun x : Omega => f x + density x).
Proof.
  assert (hMain : entropy f <= integral f).
  { apply dual_variational_axiom. }
  assert (hShift : entropy (fun x : Omega => f x + density x) <= integral (fun x : Omega => f x + density x)).
  { apply dual_variational_axiom. }
  split.
  - exact hMain.
  - exact hShift.
Qed.

Lemma concentration_step_transport_entropy_inequality
    {Omega : Type} `{MeasureStruct_transport_entropy_inequality Omega}
    (f : DensityFn_transport_entropy_inequality Omega)
    (n : nat)
    (hBound : IntegralForm_transport_entropy_inequality f <= n) :
    EntropyLike_transport_entropy_inequality f <= n /\
    EntropyLike_transport_entropy_inequality f <=
      IntegralForm_transport_entropy_inequality f.
Proof.
  assert (hConc : entropy f <= n).
  { exact (@concentration_axiom Omega H f n hBound). }
  assert (hDual : entropy f <= integral f).
  { apply dual_variational_axiom. }
  split.
  - exact hConc.
  - exact hDual.
Qed.

Lemma decomposition_formula_transport_entropy_inequality
    {Omega : Type} `{MeasureStruct_transport_entropy_inequality Omega}
    (f g : DensityFn_transport_entropy_inequality Omega) :
    IntegralForm_transport_entropy_inequality (fun x : Omega => f x + g x) =
      IntegralForm_transport_entropy_inequality f +
        IntegralForm_transport_entropy_inequality g /\
    IntegralForm_transport_entropy_inequality (fun x : Omega => g x + f x) =
      IntegralForm_transport_entropy_inequality g +
        IntegralForm_transport_entropy_inequality f.
Proof.
  assert (hFG : integral (fun x : Omega => f x + g x) = integral f + integral g).
  { apply decomposition_axiom. }
  assert (hGF : integral (fun x : Omega => g x + f x) = integral g + integral f).
  { apply (decomposition_axiom g f). }
  split.
  - exact hFG.
  - exact hGF.
Qed.
