(***
BENCHMARK_ID: TINY_MATHLIB_BATCH05_ANALYSIS_SEMIGROUP_GENERATOR_HILLE_YOSIDA_LIKE
PAIR_STEM: analysis_semigroup_generator_hille_yosida_like
MATH_DOMAIN: Analysis
SOURCE_MATHLIB: Mathlib/Analysis/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
***)

Set Universe Polymorphism.
Set Implicit Arguments.

Class AnalysisStruct_semigroup_generator_hille (E : Type) := {
  zero : E;
  add : E -> E -> E;
  norm : E -> nat;
  chart : E -> E;
  transform : nat -> E -> E;
  generator : E -> E;
  norm_add_le : forall x y : E, norm (add x y) <= norm x + norm y;
  chart_bound : forall x : E, norm (chart x) <= norm x;
  chart_idem : forall x : E, chart (chart x) = chart x;
  transform_zero : forall n : nat, transform n zero = zero;
  transform_add : forall n : nat, forall x y : E, transform n (add x y) = add (transform n x) (transform n y);
  semigroup_axiom : forall m n : nat, forall x : E, transform (m + n) x = transform m (transform n x);
  generator_def : forall x : E, generator x = transform 1 x;
  isometry_one : forall x : E, norm (transform 1 x) = norm x;
  nat_le_trans_axiom :
    forall a b c : nat,
      a <= b ->
      b <= c ->
        a <= c;
  nat_le_refl_axiom :
    forall a : nat, a <= a;
  step_bound : forall n : nat, forall x : E, norm (transform (n + 1) x) <= norm (transform n x) + norm x;
  generator_regularity : forall x : E, norm (generator (chart x)) <= norm (chart x)
}.

Infix "+h" := add (at level 50, left associativity).

Definition NormCtrl_semigroup_generator_hille
    (E : Type) : Type :=
  E -> nat.

Definition LocalChart_semigroup_generator_hille
    (E : Type) : Type :=
  E -> E.

Definition Transform_semigroup_generator_hille
    (E : Type) : Type :=
  nat -> E -> E.

Definition Generator_semigroup_generator_hille
    (E : Type) : Type :=
  E -> E.

Lemma local_estimate_semigroup_generator_hille
    {E : Type} `{AnalysisStruct_semigroup_generator_hille E}
    (x : E) :
    norm (chart x) <= norm x.
Proof.
  apply chart_bound.
Qed.

Lemma patching_estimate_semigroup_generator_hille
    {E : Type} `{AnalysisStruct_semigroup_generator_hille E}
    (x y : E) :
    norm (transform 1 (x +h y)) <= norm x + norm y.
Proof.
  assert (hExpand : transform 1 (x +h y) = add (transform 1 x) (transform 1 y)).
  { apply transform_add. }
  assert (hNormAdd :
      norm (add (transform 1 x) (transform 1 y)) <=
        norm (transform 1 x) + norm (transform 1 y)).
  { apply norm_add_le. }
  assert (hIsoX : norm (transform 1 x) = norm x).
  { apply isometry_one. }
  assert (hIsoY : norm (transform 1 y) = norm y).
  { apply isometry_one. }
  rewrite hExpand.
  eapply nat_le_trans_axiom.
  - exact hNormAdd.
  - rewrite hIsoX. rewrite hIsoY. apply nat_le_refl_axiom.
Qed.

Lemma transform_isometry_semigroup_generator_hille
    {E : Type} `{AnalysisStruct_semigroup_generator_hille E}
    (x : E) :
    norm (transform 1 x) = norm x.
Proof.
  apply isometry_one.
Qed.

Lemma decomposition_bound_semigroup_generator_hille
    {E : Type} `{AnalysisStruct_semigroup_generator_hille E}
    (x y : E) :
    norm (generator (x +h y)) <= norm x + norm y.
Proof.
  assert (hGen : generator (x +h y) = transform 1 (x +h y)).
  { apply generator_def. }
  rewrite hGen.
  apply patching_estimate_semigroup_generator_hille.
Qed.

Lemma symbol_composition_semigroup_generator_hille
    {E : Type} `{AnalysisStruct_semigroup_generator_hille E}
    (m n : nat)
    (x : E) :
    transform (m + n) (chart x) = transform m (transform n (chart x)).
Proof.
  apply semigroup_axiom.
Qed.

Lemma semigroup_generation_semigroup_generator_hille
    {E : Type} `{AnalysisStruct_semigroup_generator_hille E}
    (x : E) :
    norm (generator x) <= norm x.
Proof.
  assert (hDef : generator x = transform 1 x).
  { apply generator_def. }
  assert (hIso : norm (transform 1 x) = norm x).
  { apply isometry_one. }
  rewrite hDef.
  rewrite hIso.
  apply nat_le_refl_axiom.
Qed.

Lemma regularity_upgrade_semigroup_generator_hille
    {E : Type} `{AnalysisStruct_semigroup_generator_hille E}
    (x : E) :
    norm (generator (chart x)) <= norm x.
Proof.
  assert (hReg : norm (generator (chart x)) <= norm (chart x)).
  { apply generator_regularity. }
  assert (hChart : norm (chart x) <= norm x).
  { apply chart_bound. }
  eapply nat_le_trans_axiom.
  - exact hReg.
  - exact hChart.
Qed.
