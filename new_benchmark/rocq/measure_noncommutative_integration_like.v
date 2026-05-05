(***
BENCHMARK_ID: TINY_MATHLIB_BATCH05_MEASURE_NONCOMMUTATIVE_INTEGRATION_LIKE
PAIR_STEM: measure_noncommutative_integration_like
MATH_DOMAIN: Measure Theory
SOURCE_MATHLIB: Mathlib/MeasureTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
***)

Set Universe Polymorphism.
Set Implicit Arguments.

Class MeasureStruct_noncommutative_integration (Omega : Type) := {
  density : Omega -> nat;
  kernel : Omega -> Omega -> nat;
  entropy : (Omega -> nat) -> nat;
  integral : (Omega -> nat) -> nat;
  density_nonneg_axiom : forall x : Omega, 0 <= density x;
  integral_mono_axiom :
    forall f g : Omega -> nat,
      (forall x : Omega, f x <= g x) ->
        integral f <= integral g;
  integral_congr_axiom :
    forall f g : Omega -> nat,
      (forall x : Omega, f x = g x) ->
      integral f = integral g;
  nat_le_trans_axiom :
    forall a b c : nat,
      a <= b ->
      b <= c ->
        a <= c;
  nat_add_mono_right_axiom :
    forall a b c : nat,
      a <= b ->
      a + c <= b + c;
  integral_add_axiom :
    forall f g : Omega -> nat,
      integral (fun x : Omega => f x + g x) = integral f + integral g;
  integral_zero_axiom :
    integral (fun _ : Omega => 0) = 0;
  kernel_diag_axiom :
    forall x : Omega, kernel x x = density x;
  entropy_le_integral_axiom :
    forall f : Omega -> nat, entropy f <= integral f;
  entropy_subadd_axiom :
    forall f g : Omega -> nat,
      entropy (fun x : Omega => f x + g x) <= entropy f + integral g;
  transport_axiom :
    forall f : Omega -> nat,
      integral (fun x : Omega => kernel x x + f x) =
        integral (fun x : Omega => kernel x x) + integral f
}.

Definition DensityFn_noncommutative_integration
    (Omega : Type) : Type :=
  Omega -> nat.

Definition KernelMap_noncommutative_integration
    (Omega : Type) : Type :=
  Omega -> Omega -> nat.

Definition EntropyLike_noncommutative_integration
    {Omega : Type} `{MeasureStruct_noncommutative_integration Omega}
    (f : DensityFn_noncommutative_integration Omega) : nat :=
  entropy f.

Definition IntegralForm_noncommutative_integration
    {Omega : Type} `{MeasureStruct_noncommutative_integration Omega}
    (f : DensityFn_noncommutative_integration Omega) : nat :=
  integral f.

Lemma density_nonneg_noncommutative_integration
    {Omega : Type} `{MeasureStruct_noncommutative_integration Omega}
    (x : Omega) :
    0 <= density x /\ IntegralForm_noncommutative_integration (fun _ : Omega => 0) = 0.
Proof.
  assert (hNonneg : 0 <= density x).
  { apply density_nonneg_axiom. }
  assert (hZero : integral (fun _ : Omega => 0) = 0).
  { apply integral_zero_axiom. }
  split.
  - exact hNonneg.
  - exact hZero.
Qed.

Lemma integral_mono_noncommutative_integration
    {Omega : Type} `{MeasureStruct_noncommutative_integration Omega}
    (f g : DensityFn_noncommutative_integration Omega)
    (hfg : forall x : Omega, f x <= g x) :
    IntegralForm_noncommutative_integration f <=
      IntegralForm_noncommutative_integration g /\
    IntegralForm_noncommutative_integration f +
      IntegralForm_noncommutative_integration g <=
        IntegralForm_noncommutative_integration g +
          IntegralForm_noncommutative_integration g.
Proof.
  assert (hMono : integral f <= integral g).
  { apply (integral_mono_axiom f g). exact hfg. }
  assert (hLift : integral f + integral g <= integral g + integral g).
  { apply (@nat_add_mono_right_axiom Omega H (integral f) (integral g) (integral g)). exact hMono. }
  split.
  - exact hMono.
  - exact hLift.
Qed.

Lemma transport_identity_noncommutative_integration
    {Omega : Type} `{MeasureStruct_noncommutative_integration Omega} :
    IntegralForm_noncommutative_integration (fun x : Omega => kernel x x) =
      IntegralForm_noncommutative_integration (fun x : Omega => density x).
Proof.
  apply (integral_congr_axiom (fun x : Omega => kernel x x) (fun x : Omega => density x)).
  intro x.
  exact (kernel_diag_axiom x).
Qed.

Lemma chain_rule_measure_noncommutative_integration
    {Omega : Type} `{MeasureStruct_noncommutative_integration Omega}
    (f g : DensityFn_noncommutative_integration Omega) :
    IntegralForm_noncommutative_integration (fun x : Omega => f x + g x) =
      IntegralForm_noncommutative_integration f +
        IntegralForm_noncommutative_integration g /\
    IntegralForm_noncommutative_integration (fun x : Omega => kernel x x + f x) =
      IntegralForm_noncommutative_integration (fun x : Omega => kernel x x) +
        IntegralForm_noncommutative_integration f.
Proof.
  assert (hAdd : integral (fun x : Omega => f x + g x) = integral f + integral g).
  { apply (integral_add_axiom f g). }
  assert (hTransport : integral (fun x : Omega => kernel x x + f x) =
      integral (fun x : Omega => kernel x x) + integral f).
  { apply transport_axiom. }
  split.
  - exact hAdd.
  - exact hTransport.
Qed.

Lemma dual_variational_bound_noncommutative_integration
    {Omega : Type} `{MeasureStruct_noncommutative_integration Omega}
    (f : DensityFn_noncommutative_integration Omega) :
    EntropyLike_noncommutative_integration f <=
      IntegralForm_noncommutative_integration f /\
    EntropyLike_noncommutative_integration (fun x : Omega => f x + 0) <=
      IntegralForm_noncommutative_integration (fun x : Omega => f x + 0).
Proof.
  assert (hMain : entropy f <= integral f).
  { apply entropy_le_integral_axiom. }
  assert (hShift : entropy (fun x : Omega => f x + 0) <= integral (fun x : Omega => f x + 0)).
  { apply entropy_le_integral_axiom. }
  split.
  - exact hMain.
  - exact hShift.
Qed.

Lemma concentration_step_noncommutative_integration
    {Omega : Type} `{MeasureStruct_noncommutative_integration Omega}
    (f g : DensityFn_noncommutative_integration Omega)
    (hfg : forall x : Omega, f x <= g x) :
    EntropyLike_noncommutative_integration f <=
      IntegralForm_noncommutative_integration g /\
    EntropyLike_noncommutative_integration f +
      IntegralForm_noncommutative_integration g <=
        IntegralForm_noncommutative_integration g +
          IntegralForm_noncommutative_integration g.
Proof.
  assert (hEntropyToInt : entropy f <= integral f).
  { apply entropy_le_integral_axiom. }
  assert (hIntMono : integral f <= integral g).
  { apply (integral_mono_axiom f g). exact hfg. }
  assert (hMain : entropy f <= integral g).
  { apply (@nat_le_trans_axiom Omega H (entropy f) (integral f) (integral g)); assumption. }
  assert (hLift : entropy f + integral g <= integral g + integral g).
  { apply (@nat_add_mono_right_axiom Omega H (entropy f) (integral g) (integral g)). exact hMain. }
  split.
  - exact hMain.
  - exact hLift.
Qed.

Lemma decomposition_formula_noncommutative_integration
    {Omega : Type} `{MeasureStruct_noncommutative_integration Omega}
    (f g : DensityFn_noncommutative_integration Omega)
    (hf : forall x : Omega, f x <= kernel x x) :
    EntropyLike_noncommutative_integration (fun x : Omega => f x + g x) <=
      IntegralForm_noncommutative_integration (fun x : Omega => kernel x x) +
        IntegralForm_noncommutative_integration g /\
    IntegralForm_noncommutative_integration (fun x : Omega => kernel x x + f x) =
      IntegralForm_noncommutative_integration (fun x : Omega => kernel x x) +
        IntegralForm_noncommutative_integration f.
Proof.
  assert (hSubadd :
      entropy (fun x : Omega => f x + g x) <= entropy f + integral g).
  { apply (entropy_subadd_axiom f g). }
  assert (hDual : entropy f <= integral f).
  { apply entropy_le_integral_axiom. }
  assert (hMonoDiag : integral f <= integral (fun x : Omega => kernel x x)).
  { apply (integral_mono_axiom f (fun x : Omega => kernel x x)). exact hf. }
  assert (hEntropyToDiag : entropy f <= integral (fun x : Omega => kernel x x)).
  {
    apply (@nat_le_trans_axiom Omega H (entropy f) (integral f) (integral (fun x : Omega => kernel x x))).
    - exact hDual.
    - exact hMonoDiag.
  }
  assert (hLift : entropy f + integral g <= integral (fun x : Omega => kernel x x) + integral g).
  { apply (@nat_add_mono_right_axiom Omega H (entropy f) (integral (fun x : Omega => kernel x x)) (integral g)). exact hEntropyToDiag. }
  assert (hMain : entropy (fun x : Omega => f x + g x) <=
      integral (fun x : Omega => kernel x x) + integral g).
  {
    apply (@nat_le_trans_axiom Omega H
      (entropy (fun x : Omega => f x + g x))
      (entropy f + integral g)
      (integral (fun x : Omega => kernel x x) + integral g)).
    - exact hSubadd.
    - exact hLift.
  }
  assert (hTransport : integral (fun x : Omega => kernel x x + f x) =
      integral (fun x : Omega => kernel x x) + integral f).
  { apply transport_axiom. }
  split.
  - exact hMain.
  - exact hTransport.
Qed.
