(*
BENCHMARK_ID: TINY_MATHLIB_BATCH05_PROBABILITY_LARGE_DEVIATION_RATE_FUNCTION_LIKE
PAIR_STEM: probability_large_deviation_rate_function_like
MATH_DOMAIN: Probability Theory
SOURCE_MATHLIB: Mathlib/Probability/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class ProbStruct_large_deviation_rate (V : Type) := {
  le : V -> V -> Prop;
  le_refl : forall x : V, le x x;
  le_trans : forall {x y z : V}, le x y -> le y z -> le x z;
  add : V -> V -> V;
  cond : V -> V;
  drift : V -> V;
  rate : V -> V;
  cond_mono : forall {x y : V}, le x y -> le (cond x) (cond y);
  drift_mono : forall {x y : V}, le x y -> le (drift x) (drift y);
  rate_mono : forall {x y : V}, le x y -> le (rate x) (rate y);
  tower_axiom : forall x : V, le (cond (cond x)) (cond x);
  change_measure_axiom : forall x : V, le (drift x) (add x (rate x));
  stopping_axiom : forall x : V, le (cond x) x;
  coupling_axiom : forall x y : V, le (add (rate x) (rate y)) (rate (add x y));
  fubini_axiom : forall x y : V, le (cond (add x y)) (add (cond x) (cond y))
}.

Arguments le {V} {_} _ _.
Arguments add {V} {_} _ _. 
Arguments cond {V} {_} _. 
Arguments drift {V} {_} _. 
Arguments rate {V} {_} _. 

Infix "<~" := le (at level 70).
Infix "⊞" := add (at level 50, left associativity).

Definition Filtration_large_deviation_rate
    {V : Type} `{ProbStruct_large_deviation_rate V} (x : V) : V :=
  cond x.

Definition MartingaleStep_large_deviation_rate
    {V : Type} `{ProbStruct_large_deviation_rate V} (x : V) : V :=
  Filtration_large_deviation_rate x.

Definition DriftShift_large_deviation_rate
    {V : Type} `{ProbStruct_large_deviation_rate V} (x : V) : V :=
  drift x.

Definition RateFunc_large_deviation_rate
    {V : Type} `{ProbStruct_large_deviation_rate V} (x : V) : V :=
  rate x.

Lemma adaptivity_rule_large_deviation_rate
    {V : Type} `{ProbStruct_large_deviation_rate V}
    (x : V) :
    MartingaleStep_large_deviation_rate x <~ x.
Proof.
  assert (hstop : cond x <~ x).
  { exact (stopping_axiom x). }
  unfold MartingaleStep_large_deviation_rate, Filtration_large_deviation_rate.
  exact hstop.
Qed.

Lemma tower_property_large_deviation_rate
    {V : Type} `{ProbStruct_large_deviation_rate V}
    (x : V) :
    MartingaleStep_large_deviation_rate
      (MartingaleStep_large_deviation_rate x) <~
      MartingaleStep_large_deviation_rate x /\
    MartingaleStep_large_deviation_rate x <~ x.
Proof.
  assert (htower : cond (cond x) <~ cond x).
  { exact (tower_axiom x). }
  assert (hadapt : MartingaleStep_large_deviation_rate x <~ x).
  { apply adaptivity_rule_large_deviation_rate. }
  split.
  - unfold MartingaleStep_large_deviation_rate, Filtration_large_deviation_rate.
    exact htower.
  - exact hadapt.
Qed.

Lemma change_measure_step_large_deviation_rate
    {V : Type} `{ProbStruct_large_deviation_rate V}
    (x : V) :
    DriftShift_large_deviation_rate x <~
      x ⊞ RateFunc_large_deviation_rate x.
Proof.
  assert (hraw : drift x <~ add x (rate x)).
  { exact (change_measure_axiom x). }
  unfold DriftShift_large_deviation_rate, RateFunc_large_deviation_rate.
  exact hraw.
Qed.

Lemma stopping_control_large_deviation_rate
    {V : Type} `{ProbStruct_large_deviation_rate V}
    (x : V) :
    exists z : V,
      z = DriftShift_large_deviation_rate x /\
      MartingaleStep_large_deviation_rate
        (DriftShift_large_deviation_rate x) <~ z /\
      z <~ DriftShift_large_deviation_rate x.
Proof.
  assert (hstop : cond (drift x) <~ drift x).
  { exact (stopping_axiom (drift x)). }
  assert (hreflex : DriftShift_large_deviation_rate x <~ DriftShift_large_deviation_rate x).
  { apply le_refl. }
  exists (DriftShift_large_deviation_rate x).
  split.
  - reflexivity.
  - split.
    + unfold MartingaleStep_large_deviation_rate, Filtration_large_deviation_rate.
      unfold DriftShift_large_deviation_rate.
      exact hstop.
    + exact hreflex.
Qed.

Lemma ld_upper_bound_large_deviation_rate
    {V : Type} `{ProbStruct_large_deviation_rate V}
    (x : V) :
    MartingaleStep_large_deviation_rate
      (DriftShift_large_deviation_rate x) <~
      x ⊞ RateFunc_large_deviation_rate x.
Proof.
  assert (h1 : MartingaleStep_large_deviation_rate (DriftShift_large_deviation_rate x) <~
              DriftShift_large_deviation_rate x).
  {
    destruct (stopping_control_large_deviation_rate x) as [z [hz [hstep hback]]].
    subst z.
    exact hstep.
  }
  assert (h2 : DriftShift_large_deviation_rate x <~ x ⊞ RateFunc_large_deviation_rate x).
  { apply change_measure_step_large_deviation_rate. }
  exact (le_trans h1 h2).
Qed.

Lemma coupling_estimate_large_deviation_rate
    {V : Type} `{ProbStruct_large_deviation_rate V}
    (x y : V) :
    RateFunc_large_deviation_rate x ⊞
      RateFunc_large_deviation_rate y <~
      RateFunc_large_deviation_rate (x ⊞ y).
Proof.
  assert (hcouple : add (rate x) (rate y) <~ rate (add x y)).
  { exact (coupling_axiom x y). }
  unfold RateFunc_large_deviation_rate.
  exact hcouple.
Qed.

Lemma stochastic_fubini_rule_large_deviation_rate
    {V : Type} `{ProbStruct_large_deviation_rate V}
    (x y : V) :
    MartingaleStep_large_deviation_rate (x ⊞ y) <~
      MartingaleStep_large_deviation_rate x ⊞
        MartingaleStep_large_deviation_rate y.
Proof.
  assert (hfub : cond (add x y) <~ add (cond x) (cond y)).
  { exact (fubini_axiom x y). }
  unfold MartingaleStep_large_deviation_rate, Filtration_large_deviation_rate.
  exact hfub.
Qed.
