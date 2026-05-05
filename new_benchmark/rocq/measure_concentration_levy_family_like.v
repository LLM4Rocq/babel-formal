(*
BENCHMARK_ID: TINY_MATHLIB_BATCH05_MEASURE_CONCENTRATION_LEVY_FAMILY_LIKE
PAIR_STEM: measure_concentration_levy_family_like
MATH_DOMAIN: Measure Theory
SOURCE_MATHLIB: Mathlib/MeasureTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class MeasureStruct_concentration_levy_family (Omega : Type) := {
  density : Omega -> nat;
  kernel : Omega -> Omega -> nat;
  entropy : (Omega -> nat) -> nat;
  integral : (Omega -> nat) -> nat;
  nat_le_trans : forall a b c : nat, a <= b -> b <= c -> a <= c;
  nat_add_mono : forall a b c d : nat, a <= b -> c <= d -> a + c <= b + d;
  nat_le_refl : forall a : nat, a <= a;
  density_nonneg_axiom : forall x : Omega, 0 <= density x;
  integral_mono_axiom :
    forall f g : Omega -> nat,
      (forall x : Omega, f x <= g x) ->
        integral f <= integral g;
  kernel_identity_axiom :
    forall x : Omega,
      kernel x x = density x;
  chain_rule_axiom :
    forall f g : Omega -> nat,
      entropy f + integral g <= entropy (fun x : Omega => f x + g x) + integral f;
  dual_variational_axiom :
    forall f : Omega -> nat,
      entropy f <= integral f + integral (fun x : Omega => density x);
  concentration_axiom :
    forall f : Omega -> nat,
      forall n : nat,
        integral f <= n ->
          entropy f <= n + integral (fun x : Omega => density x);
  decomposition_axiom :
    forall f g : Omega -> nat,
      integral (fun x : Omega => f x + g x) = integral f + integral g
}.

Definition DensityFn_concentration_levy_family (Omega : Type) : Type :=
  Omega -> nat.

Definition KernelMap_concentration_levy_family (Omega : Type) : Type :=
  Omega -> Omega -> nat.

Definition EntropyLike_concentration_levy_family
    {Omega : Type} `{MeasureStruct_concentration_levy_family Omega}
    (f : DensityFn_concentration_levy_family Omega) : nat :=
  entropy f.

Definition IntegralForm_concentration_levy_family
    {Omega : Type} `{MeasureStruct_concentration_levy_family Omega}
    (f : DensityFn_concentration_levy_family Omega) : nat :=
  integral f.

Lemma density_nonneg_concentration_levy_family
    {Omega : Type} `{MeasureStruct_concentration_levy_family Omega}
    (x : Omega) :
    (forall n : nat, n = density x -> 0 <= n) /\
    (kernel x x = density x -> 0 <= kernel x x).
Proof.
  split.
  - intros n hn.
    rewrite hn.
    exact (density_nonneg_axiom x).
  - intro hDiag.
    rewrite hDiag.
    exact (density_nonneg_axiom x).
Qed.

Lemma integral_mono_concentration_levy_family
    {Omega : Type} `{MeasureStruct_concentration_levy_family Omega}
    (f g : DensityFn_concentration_levy_family Omega)
    (hfg : forall x : Omega, f x <= g x) :
    (forall n : nat,
      n = IntegralForm_concentration_levy_family f ->
        n <= IntegralForm_concentration_levy_family g) /\
    IntegralForm_concentration_levy_family (fun x : Omega => f x + f x) <=
      IntegralForm_concentration_levy_family (fun x : Omega => g x + g x).
Proof.
  assert (hMono : integral f <= integral g).
  { apply (integral_mono_axiom f g). exact hfg. }
  assert (hFF : integral (fun x : Omega => f x + f x) = integral f + integral f).
  { apply (decomposition_axiom f f). }
  assert (hGG : integral (fun x : Omega => g x + g x) = integral g + integral g).
  { apply (decomposition_axiom g g). }
  assert (hLift : integral f + integral f <= integral g + integral g).
  {
    apply (@nat_add_mono Omega H (integral f) (integral g) (integral f) (integral g)).
    - exact hMono.
    - exact hMono.
  }
  split.
  - intros n hn.
    rewrite hn.
    exact hMono.
  - change (integral (fun x : Omega => f x + f x) <= integral (fun x : Omega => g x + g x)).
    rewrite hFF.
    rewrite hGG.
    exact hLift.
Qed.

Lemma transport_identity_concentration_levy_family
    {Omega : Type} `{MeasureStruct_concentration_levy_family Omega}
    (x : Omega) :
    forall y : Omega, y = x -> kernel y y = density y.
Proof.
  intros y hy.
  rewrite hy.
  apply kernel_identity_axiom.
Qed.

Lemma chain_rule_measure_concentration_levy_family
    {Omega : Type} `{MeasureStruct_concentration_levy_family Omega}
    (f g : DensityFn_concentration_levy_family Omega) :
    EntropyLike_concentration_levy_family f +
        IntegralForm_concentration_levy_family g <=
      EntropyLike_concentration_levy_family (fun x : Omega => f x + g x) +
        IntegralForm_concentration_levy_family f /\
    (EntropyLike_concentration_levy_family f +
        IntegralForm_concentration_levy_family g) +
        IntegralForm_concentration_levy_family g <=
      (EntropyLike_concentration_levy_family (fun x : Omega => f x + g x) +
        IntegralForm_concentration_levy_family f) +
          IntegralForm_concentration_levy_family g.
Proof.
  assert (hChain :
      entropy f + integral g <= entropy (fun x : Omega => f x + g x) + integral f).
  { apply (chain_rule_axiom f g). }
  assert (hReflG : integral g <= integral g).
  { apply nat_le_refl. }
  assert (hLift :
      (entropy f + integral g) + integral g <=
      (entropy (fun x : Omega => f x + g x) + integral f) + integral g).
  {
    apply (@nat_add_mono Omega H
      (entropy f + integral g)
      (entropy (fun x : Omega => f x + g x) + integral f)
      (integral g)
      (integral g)).
    - exact hChain.
    - exact hReflG.
  }
  split.
  - exact hChain.
  - exact hLift.
Qed.

Lemma dual_variational_bound_concentration_levy_family
    {Omega : Type} `{MeasureStruct_concentration_levy_family Omega}
    (f : DensityFn_concentration_levy_family Omega) :
    EntropyLike_concentration_levy_family f <=
      IntegralForm_concentration_levy_family f +
        IntegralForm_concentration_levy_family (fun x : Omega => density x) /\
    EntropyLike_concentration_levy_family f +
        IntegralForm_concentration_levy_family f <=
      (IntegralForm_concentration_levy_family f +
        IntegralForm_concentration_levy_family (fun x : Omega => density x)) +
          IntegralForm_concentration_levy_family f.
Proof.
  assert (hDual : entropy f <= integral f + integral (fun x : Omega => density x)).
  { apply dual_variational_axiom. }
  assert (hReflF : integral f <= integral f).
  { apply nat_le_refl. }
  assert (hLift :
      entropy f + integral f <=
      (integral f + integral (fun x : Omega => density x)) + integral f).
  {
    apply (@nat_add_mono Omega H
      (entropy f)
      (integral f + integral (fun x : Omega => density x))
      (integral f)
      (integral f)).
    - exact hDual.
    - exact hReflF.
  }
  split.
  - exact hDual.
  - exact hLift.
Qed.

Lemma concentration_step_concentration_levy_family
    {Omega : Type} `{MeasureStruct_concentration_levy_family Omega}
    (f : DensityFn_concentration_levy_family Omega)
    (n : nat)
    (hBound : IntegralForm_concentration_levy_family f <= n) :
    EntropyLike_concentration_levy_family f <=
      n + IntegralForm_concentration_levy_family (fun x : Omega => density x) /\
    EntropyLike_concentration_levy_family f +
        IntegralForm_concentration_levy_family f <=
      (n + IntegralForm_concentration_levy_family (fun x : Omega => density x)) +
        IntegralForm_concentration_levy_family f.
Proof.
  assert (hConc : entropy f <= n + integral (fun x : Omega => density x)).
  { exact (@concentration_axiom Omega H f n hBound). }
  assert (hReflF : integral f <= integral f).
  { apply nat_le_refl. }
  assert (hLift :
      entropy f + integral f <=
      (n + integral (fun x : Omega => density x)) + integral f).
  {
    apply (@nat_add_mono Omega H
      (entropy f)
      (n + integral (fun x : Omega => density x))
      (integral f)
      (integral f)).
    - exact hConc.
    - exact hReflF.
  }
  split.
  - exact hConc.
  - exact hLift.
Qed.

Lemma decomposition_formula_concentration_levy_family
    {Omega : Type} `{MeasureStruct_concentration_levy_family Omega}
    (f g : DensityFn_concentration_levy_family Omega) :
    exists s t : nat,
      s = IntegralForm_concentration_levy_family (fun x : Omega => f x + g x) /\
      t = IntegralForm_concentration_levy_family (fun x : Omega => g x + f x) /\
      s = IntegralForm_concentration_levy_family f +
            IntegralForm_concentration_levy_family g /\
      t = IntegralForm_concentration_levy_family g +
            IntegralForm_concentration_levy_family f.
Proof.
  assert (hFG : integral (fun x : Omega => f x + g x) = integral f + integral g).
  { apply (decomposition_axiom f g). }
  assert (hGF : integral (fun x : Omega => g x + f x) = integral g + integral f).
  { apply (decomposition_axiom g f). }
  exists (integral (fun x : Omega => f x + g x)).
  exists (integral (fun x : Omega => g x + f x)).
  split.
  - reflexivity.
  - split.
    + reflexivity.
    + split.
      * exact hFG.
      * exact hGF.
Qed.
