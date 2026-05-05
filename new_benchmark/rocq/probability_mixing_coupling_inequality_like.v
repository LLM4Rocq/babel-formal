(*
BENCHMARK_ID: TINY_MATHLIB_BATCH05_PROBABILITY_MIXING_COUPLING_INEQUALITY_LIKE
PAIR_STEM: probability_mixing_coupling_inequality_like
MATH_DOMAIN: Probability Theory
SOURCE_MATHLIB: Mathlib/Probability/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class ProbStruct_mixing_coupling_inequality (Omega : Type) := {
  energy : Omega -> nat;
  step : Omega -> Omega;
  condExp : (Omega -> nat) -> Omega -> nat;
  adapted : (Omega -> nat) -> Prop;
  adapted_step : forall {f : Omega -> nat}, adapted f -> adapted (fun w => condExp f (step w));
  mono_condExp : forall {f g : Omega -> nat},
      (forall w, f w <= g w) -> forall w, condExp f w <= condExp g w;
  tower_condExp : forall (f : Omega -> nat) (w : Omega),
      condExp (fun x => condExp f (step x)) w = condExp f (step (step w));
  condExp_bound : forall {f : Omega -> nat} {n : nat},
      (forall w, f w <= n) -> forall w, condExp f w <= n;
  coupling_core : forall (f g : Omega -> nat) (w : Omega),
      condExp f w <= condExp g w + energy w;
  additivity_core : forall (f g : Omega -> nat) (w : Omega),
      condExp (fun x => f x + g x) w = condExp f w + condExp g w
}.

Definition Filtration_mixing_coupling_inequality {Omega : Type}
    `{ProbStruct_mixing_coupling_inequality Omega}
    (f : Omega -> nat) (n : nat) : Prop :=
  adapted f /\
    (forall w, f w <= n + energy w).

Definition MartingaleStep_mixing_coupling_inequality {Omega : Type}
    `{ProbStruct_mixing_coupling_inequality Omega}
    (f : Omega -> nat) : Omega -> nat :=
  fun w => condExp f (step w).

Definition DriftShift_mixing_coupling_inequality {Omega : Type}
    `{ProbStruct_mixing_coupling_inequality Omega}
    (f : Omega -> nat) (c : nat) : Omega -> nat :=
  fun w => f w.

Definition RateFunc_mixing_coupling_inequality {Omega : Type}
    `{ProbStruct_mixing_coupling_inequality Omega}
    (f : Omega -> nat) (n : nat) : Prop :=
  forall w, f w <= n.

Lemma adaptivity_rule_mixing_coupling_inequality {Omega : Type}
    `{ProbStruct_mixing_coupling_inequality Omega}
    {f : Omega -> nat} {n : nat}
    (hfil : Filtration_mixing_coupling_inequality f n) :
    adapted (MartingaleStep_mixing_coupling_inequality f).
Proof.
  destruct hfil as [hadapt hbound].
  assert (hstep : adapted (fun w => condExp f (step w))).
  { apply adapted_step. exact hadapt. }
  assert (hkeep : forall w, f w <= n + energy w).
  { exact hbound. }
  assert (htriv : True).
  { trivial. }
  exact hstep.
Qed.

Lemma tower_property_mixing_coupling_inequality {Omega : Type}
    `{ProbStruct_mixing_coupling_inequality Omega}
    (f : Omega -> nat) (w : Omega) :
    condExp (MartingaleStep_mixing_coupling_inequality f) w =
    MartingaleStep_mixing_coupling_inequality f (step w).
Proof.
  assert (htower :
      condExp (fun x => condExp f (step x)) w =
      condExp f (step (step w))).
  { apply tower_condExp. }
  assert (hRight :
      MartingaleStep_mixing_coupling_inequality f (step w) =
      condExp f (step (step w))).
  { reflexivity. }
  unfold MartingaleStep_mixing_coupling_inequality.
  rewrite htower.
  symmetry.
  exact hRight.
Qed.

Lemma change_measure_step_mixing_coupling_inequality {Omega : Type}
    `{ProbStruct_mixing_coupling_inequality Omega}
    {f g : Omega -> nat}
    (hfg : forall w, f w <= g w) :
    forall w,
      MartingaleStep_mixing_coupling_inequality f w <=
        MartingaleStep_mixing_coupling_inequality g w.
Proof.
  intro w.
  assert (hmono :
      forall xi, condExp f xi <= condExp g xi).
  { apply mono_condExp. exact hfg. }
  assert (hAtStep :
      condExp f (step w) <= condExp g (step w)).
  { apply hmono. }
  exact hAtStep.
Qed.

Lemma stopping_control_mixing_coupling_inequality {Omega : Type}
    `{ProbStruct_mixing_coupling_inequality Omega}
    {f : Omega -> nat} {n c : nat}
    (hRate : RateFunc_mixing_coupling_inequality f n) :
    RateFunc_mixing_coupling_inequality
      (DriftShift_mixing_coupling_inequality f c) n.
Proof.
  intro w.
  assert (hbase : f w <= n).
  { apply hRate. }
  unfold DriftShift_mixing_coupling_inequality.
  exact hbase.
Qed.

Lemma ld_upper_bound_mixing_coupling_inequality {Omega : Type}
    `{ProbStruct_mixing_coupling_inequality Omega}
    {f : Omega -> nat} {n : nat}
    (hfil : Filtration_mixing_coupling_inequality f n)
    (hRate : RateFunc_mixing_coupling_inequality f n) :
    RateFunc_mixing_coupling_inequality
      (MartingaleStep_mixing_coupling_inequality f) n.
Proof.
  intro w.
  assert (hBoundCond : forall xi, condExp f xi <= n).
  { apply condExp_bound. exact hRate. }
  assert (hAtStep : condExp f (step w) <= n).
  { apply hBoundCond. }
  assert (hkeep : adapted f).
  { exact (proj1 hfil). }
  exact hAtStep.
Qed.

Lemma coupling_estimate_mixing_coupling_inequality {Omega : Type}
    `{ProbStruct_mixing_coupling_inequality Omega}
    (f g : Omega -> nat) :
    forall w,
      MartingaleStep_mixing_coupling_inequality f w <=
        MartingaleStep_mixing_coupling_inequality g w +
          energy (step w).
Proof.
  intro w.
  assert (hcore : condExp f (step w) <= condExp g (step w) + energy (step w)).
  { apply coupling_core. }
  exact hcore.
Qed.

Lemma stochastic_fubini_rule_mixing_coupling_inequality {Omega : Type}
    `{ProbStruct_mixing_coupling_inequality Omega}
    (f g : Omega -> nat) :
    forall w,
      MartingaleStep_mixing_coupling_inequality
        (fun x => f x + g x) w =
      MartingaleStep_mixing_coupling_inequality f w +
        MartingaleStep_mixing_coupling_inequality g w.
Proof.
  intro w.
  assert (hadd :
      condExp (fun x => f x + g x) (step w) =
      condExp f (step w) + condExp g (step w)).
  { apply additivity_core. }
  unfold MartingaleStep_mixing_coupling_inequality.
  exact hadd.
Qed.
