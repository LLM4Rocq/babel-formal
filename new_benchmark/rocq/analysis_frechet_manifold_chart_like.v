(*
BENCHMARK_ID: TINY_MATHLIB_BATCH05_ANALYSIS_FRECHET_MANIFOLD_CHART_LIKE
PAIR_STEM: analysis_frechet_manifold_chart_like
MATH_DOMAIN: Analysis
SOURCE_MATHLIB: Mathlib/Analysis/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class AnalysisStruct_frechet_manifold_chart (M : Type) := {
  norm : M -> nat;
  add : M -> M -> M;
  chart : M -> M;
  transform : M -> M -> M;
  generator : nat -> M -> M;
  transform_bound :
    forall x y : M,
      norm (transform x y) <= norm x + norm y;
  chart_idem :
    forall x : M,
      chart (chart x) = chart x;
  chart_bound :
    forall x : M,
      norm (chart x) <= norm x;
  nat_add_mono :
    forall a b c d : nat,
      a <= b ->
      c <= d ->
        a + c <= b + d;
  nat_le_trans :
    forall a b c : nat,
      a <= b ->
      b <= c ->
        a <= c;
  nat_add_comm :
    forall a b : nat,
      a + b = b + a;
  nat_le_refl :
    forall a : nat,
      a <= a;
  generator_step :
    forall n : nat,
      forall x : M,
        norm (generator (n + 1) x) <= norm (generator n x) + norm x;
  generator_zero :
    forall x : M,
      generator 0 x = x;
  semigroup_axiom :
    forall m n : nat,
      forall x : M,
        generator (m + n) x = generator m (generator n x);
  regularity_axiom :
    forall x : M,
      norm (generator 1 (chart x)) <= norm (chart x)
}.

Definition NormCtrl_frechet_manifold_chart
    {M : Type} `{AnalysisStruct_frechet_manifold_chart M} : Prop :=
  forall x : M, norm (chart x) <= norm x.

Definition LocalChart_frechet_manifold_chart
    (M : Type) : Type :=
  M -> M.

Definition Transform_frechet_manifold_chart
    (M : Type) : Type :=
  M -> M -> M.

Definition Generator_frechet_manifold_chart
    (M : Type) : Type :=
  nat -> M -> M.

Lemma local_estimate_frechet_manifold_chart
    {M : Type} `{AnalysisStruct_frechet_manifold_chart M} :
    NormCtrl_frechet_manifold_chart.
Proof.
  intro x.
  assert (hLocal : norm (chart x) <= norm x).
  { apply chart_bound. }
  assert (hStable : norm (chart (chart x)) <= norm (chart x)).
  { apply chart_bound. }
  assert (hTrans : norm (chart (chart x)) <= norm x).
  { exact (@nat_le_trans M H _ _ _ hStable hLocal). }
  assert (hKeep : norm (chart x) <= norm x).
  { exact hLocal. }
  exact hKeep.
Qed.

Lemma patching_estimate_frechet_manifold_chart
    {M : Type} `{AnalysisStruct_frechet_manifold_chart M}
    (x y : M) :
    norm (transform (chart x) y) <= norm x + norm y.
Proof.
  assert (hTrans : norm (transform (chart x) y) <= norm (chart x) + norm y).
  { apply transform_bound. }
  assert (hx : norm (chart x) <= norm x).
  { apply chart_bound. }
  assert (hy : norm y <= norm y).
  { apply nat_le_refl. }
  assert (hLift : norm (chart x) + norm y <= norm x + norm y).
  { exact (@nat_add_mono M H _ _ _ _ hx hy). }
  exact (@nat_le_trans M H _ _ _ hTrans hLift).
Qed.

Lemma transform_isometry_frechet_manifold_chart
    {M : Type} `{AnalysisStruct_frechet_manifold_chart M}
    (x y : M)
    (hIso : norm (transform x y) = norm x + norm y) :
    norm (transform x y) = norm y + norm x /\
    norm (chart (chart x)) <= norm x.
Proof.
  assert (hComm : norm x + norm y = norm y + norm x).
  { apply nat_add_comm. }
  assert (hEq : norm (transform x y) = norm y + norm x).
  {
    rewrite hIso.
    exact hComm.
  }
  assert (hChart1 : norm (chart (chart x)) <= norm (chart x)).
  { apply chart_bound. }
  assert (hChart2 : norm (chart x) <= norm x).
  { apply chart_bound. }
  assert (hChart : norm (chart (chart x)) <= norm x).
  { exact (@nat_le_trans M H _ _ _ hChart1 hChart2). }
  split.
  - exact hEq.
  - exact hChart.
Qed.

Lemma decomposition_bound_frechet_manifold_chart
    {M : Type} `{AnalysisStruct_frechet_manifold_chart M}
    (x : M) :
    norm (generator 1 x) <= norm x + norm x.
Proof.
  assert (hStep : norm (generator (0 + 1) x) <= norm (generator 0 x) + norm x).
  { apply (generator_step 0 x). }
  assert (hZero : generator 0 x = x).
  { apply generator_zero. }
  assert (hRewrite : norm (generator 1 x) <= norm x + norm x).
  {
    assert (hStep1 : norm (generator 1 x) <= norm (generator 0 x) + norm x).
    { exact hStep. }
    assert (hStep2 : norm (generator 0 x) + norm x <= norm x + norm x).
    {
      rewrite hZero.
      apply nat_le_refl.
    }
    exact (@nat_le_trans M H _ _ _ hStep1 hStep2).
  }
  exact hRewrite.
Qed.

Lemma symbol_composition_frechet_manifold_chart
    {M : Type} `{AnalysisStruct_frechet_manifold_chart M}
    (x y : M) :
    norm (chart (transform (chart x) (chart y))) <= norm x + norm y.
Proof.
  assert (hFirst : norm (transform (chart x) (chart y)) <= norm (chart x) + norm (chart y)).
  { apply transform_bound. }
  assert (hx : norm (chart x) <= norm x).
  { apply chart_bound. }
  assert (hy : norm (chart y) <= norm y).
  { apply chart_bound. }
  assert (hAdd : norm (chart x) + norm (chart y) <= norm x + norm y).
  { exact (@nat_add_mono M H _ _ _ _ hx hy). }
  assert (hPatch : norm (transform (chart x) (chart y)) <= norm x + norm y).
  { exact (@nat_le_trans M H _ _ _ hFirst hAdd). }
  assert (hChart : norm (chart (transform (chart x) (chart y))) <= norm (transform (chart x) (chart y))).
  { apply chart_bound. }
  exact (@nat_le_trans M H _ _ _ hChart hPatch).
Qed.

Lemma semigroup_generation_frechet_manifold_chart
    {M : Type} `{AnalysisStruct_frechet_manifold_chart M}
    (m n : nat)
    (x : M) :
    generator (m + n) x = generator m (generator n x) /\
    norm (chart (generator (m + n) x)) <= norm (generator m (generator n x)).
Proof.
  assert (hSemi : generator (m + n) x = generator m (generator n x)).
  { apply semigroup_axiom. }
  assert (hChart : norm (chart (generator m (generator n x))) <= norm (generator m (generator n x))).
  { apply chart_bound. }
  assert (hTransport : norm (chart (generator (m + n) x)) <= norm (generator m (generator n x))).
  {
    rewrite hSemi.
    exact hChart.
  }
  split.
  - exact hSemi.
  - exact hTransport.
Qed.

Lemma regularity_upgrade_frechet_manifold_chart
    {M : Type} `{AnalysisStruct_frechet_manifold_chart M}
    (x : M) :
    norm (generator 1 (chart x)) <= norm x /\
    norm (chart (generator 1 (chart x))) <= norm x.
Proof.
  assert (hReg : norm (generator 1 (chart x)) <= norm (chart x)).
  { apply regularity_axiom. }
  assert (hChart : norm (chart x) <= norm x).
  { apply chart_bound. }
  assert (hFirst : norm (generator 1 (chart x)) <= norm x).
  { exact (@nat_le_trans M H _ _ _ hReg hChart). }
  assert (hOuter : norm (chart (generator 1 (chart x))) <= norm (generator 1 (chart x))).
  { apply chart_bound. }
  assert (hSecond : norm (chart (generator 1 (chart x))) <= norm x).
  { exact (@nat_le_trans M H _ _ _ hOuter hFirst). }
  split.
  - exact hFirst.
  - exact hSecond.
Qed.
