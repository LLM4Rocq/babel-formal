(*
BENCHMARK_ID: TINY_MATHLIB_BATCH05_ANALYSIS_PSEUDO_DIFFERENTIAL_SYMBOL_LIKE
PAIR_STEM: analysis_pseudo_differential_symbol_like
MATH_DOMAIN: Analysis
SOURCE_MATHLIB: Mathlib/Analysis/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 18
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class AnalysisStruct_pseudo_differential_symbol (V : Type) := {
  seminorm : nat -> V -> nat;
  chart : V -> V;
  transform : V -> V;
  compose : V -> V -> V;
  generator : nat -> V -> V;
  smooth : nat -> V -> Prop;
  smooth_lift : forall k : nat, forall x : V, smooth k x -> smooth (k + 1) x;
  chart_smooth : forall k : nat, forall x : V, smooth k x -> smooth k (chart x);
  transform_smooth : forall k : nat, forall x : V, smooth (k + 1) x -> smooth k (transform x);
  compose_smooth :
    forall k : nat,
      forall x y : V,
        smooth k x ->
        smooth k y ->
          smooth k (compose x y);
  chart_bound : forall k : nat, forall x : V, seminorm k (chart x) <= seminorm k x;
  transform_shift : forall k : nat, forall x : V, seminorm k (transform x) <= seminorm (k + 1) x;
  compose_bound :
    forall k : nat,
      forall x y : V,
        seminorm k (compose x y) <= seminorm k x + seminorm k y;
  nat_add_mono : forall a b c d : nat, a <= b -> c <= d -> a + c <= b + d;
  nat_le_trans : forall a b c : nat, a <= b -> b <= c -> a <= c;
  nat_le_refl : forall a : nat, a <= a;
  nat_add_comm : forall a b : nat, a + b = b + a;
  generator_step :
    forall k n : nat,
      forall x : V,
        seminorm k (generator (n + 1) x) <= seminorm k (generator n x) + seminorm k x;
  generator_zero : forall x : V, generator 0 x = x;
  semigroup_axiom :
    forall m n : nat,
      forall x : V,
        generator (m + n) x = generator m (generator n x);
  symbol_comm_axiom :
    forall n : nat,
      forall x : V,
        transform (generator n x) = generator n (transform x);
  regularity_axiom :
    forall k : nat,
      forall x : V,
        smooth (k + 1) x ->
          seminorm k (generator 1 x) <= seminorm (k + 1) x
}.

Definition NormCtrl_pseudo_differential_symbol
    {V : Type} `{AnalysisStruct_pseudo_differential_symbol V} : Prop :=
  forall k : nat, forall x : V, seminorm k (chart x) <= seminorm k x.

Definition LocalChart_pseudo_differential_symbol (V : Type) : Type :=
  V -> V.

Definition Transform_pseudo_differential_symbol (V : Type) : Type :=
  V -> V.

Definition Generator_pseudo_differential_symbol (V : Type) : Type :=
  nat -> V -> V.

Lemma local_estimate_pseudo_differential_symbol
    {V : Type} `{AnalysisStruct_pseudo_differential_symbol V} :
    NormCtrl_pseudo_differential_symbol.
Proof.
  intros k x.
  assert (hBound : seminorm k (chart x) <= seminorm k x).
  { apply chart_bound. }
  assert (hSelf : smooth k x -> smooth k (chart x)).
  {
    intro hs.
    exact (chart_smooth k x hs).
  }
  assert (hKeepSmooth : smooth k x -> smooth k (chart x)).
  { exact hSelf. }
  exact hBound.
Qed.

Lemma patching_estimate_pseudo_differential_symbol
    {V : Type} `{AnalysisStruct_pseudo_differential_symbol V}
    (k : nat)
    (x : V)
    (hs : smooth (k + 1) x) :
    seminorm k (transform (chart x)) <= seminorm (k + 1) x /\
    smooth k (transform (chart x)).
Proof.
  assert (hSmoothChart : smooth (k + 1) (chart x)).
  { apply (chart_smooth (k + 1) x hs). }
  assert (hSmoothTrans : smooth k (transform (chart x))).
  { apply (transform_smooth k (chart x) hSmoothChart). }
  assert (hShift : seminorm k (transform (chart x)) <= seminorm (k + 1) (chart x)).
  { apply transform_shift. }
  assert (hChart : seminorm (k + 1) (chart x) <= seminorm (k + 1) x).
  { apply chart_bound. }
  assert (hBound : seminorm k (transform (chart x)) <= seminorm (k + 1) x).
  { exact (@nat_le_trans V H _ _ _ hShift hChart). }
  split.
  - exact hBound.
  - exact hSmoothTrans.
Qed.

Lemma transform_isometry_pseudo_differential_symbol
    {V : Type} `{AnalysisStruct_pseudo_differential_symbol V}
    (k : nat)
    (x : V)
    (hIso :
      seminorm k (transform x) =
        seminorm k (chart x) + seminorm (k + 1) x) :
    seminorm k (transform x) =
      seminorm (k + 1) x + seminorm k (chart x) /\
    seminorm k (chart x) <= seminorm k x.
Proof.
  assert (hComm :
      seminorm k (chart x) + seminorm (k + 1) x =
        seminorm (k + 1) x + seminorm k (chart x)).
  { apply nat_add_comm. }
  assert (hEq :
      seminorm k (transform x) =
        seminorm (k + 1) x + seminorm k (chart x)).
  {
    rewrite hIso.
    exact hComm.
  }
  assert (hChart : seminorm k (chart x) <= seminorm k x).
  { apply chart_bound. }
  split.
  - exact hEq.
  - exact hChart.
Qed.

Lemma decomposition_bound_pseudo_differential_symbol
    {V : Type} `{AnalysisStruct_pseudo_differential_symbol V}
    (k : nat)
    (x y : V)
    (hx : smooth k x)
    (hy : smooth k y) :
    seminorm k (compose (transform x) (chart y)) <=
      seminorm (k + 1) x + seminorm k y /\
    smooth k (compose (transform x) (chart y)).
Proof.
  assert (hx1 : smooth (k + 1) x).
  { apply (smooth_lift k x hx). }
  assert (hx' : smooth k (transform x)).
  { apply (transform_smooth k x hx1). }
  assert (hy' : smooth k (chart y)).
  { apply (chart_smooth k y hy). }
  assert (hSmooth : smooth k (compose (transform x) (chart y))).
  { apply (compose_smooth k (transform x) (chart y) hx' hy'). }
  assert (hComp :
      seminorm k (compose (transform x) (chart y)) <=
        seminorm k (transform x) + seminorm k (chart y)).
  { apply (compose_bound k (transform x) (chart y)). }
  assert (hLeft : seminorm k (transform x) <= seminorm (k + 1) x).
  { apply (transform_shift k x). }
  assert (hRight : seminorm k (chart y) <= seminorm k y).
  { apply (chart_bound k y). }
  assert (hAdd :
      seminorm k (transform x) + seminorm k (chart y) <=
        seminorm (k + 1) x + seminorm k y).
  { exact (@nat_add_mono V H _ _ _ _ hLeft hRight). }
  assert (hBound :
      seminorm k (compose (transform x) (chart y)) <=
        seminorm (k + 1) x + seminorm k y).
  { exact (@nat_le_trans V H _ _ _ hComp hAdd). }
  split.
  - exact hBound.
  - exact hSmooth.
Qed.

Lemma symbol_composition_pseudo_differential_symbol
    {V : Type} `{AnalysisStruct_pseudo_differential_symbol V}
    (n : nat)
    (x : V) :
    transform (generator n x) = generator n (transform x) /\
    transform (generator (n + 0) x) = generator (n + 0) (transform x).
Proof.
  assert (hMain : transform (generator n x) = generator n (transform x)).
  { apply (symbol_comm_axiom n x). }
  assert (hShift : transform (generator (n + 0) x) = generator (n + 0) (transform x)).
  { apply (symbol_comm_axiom (n + 0) x). }
  split.
  - exact hMain.
  - exact hShift.
Qed.

Lemma semigroup_generation_pseudo_differential_symbol
    {V : Type} `{AnalysisStruct_pseudo_differential_symbol V}
    (m n : nat)
    (x : V) :
    generator (m + n) x = generator m (generator n x) /\
    seminorm 0 (generator (m + 1) x) <= seminorm 0 (generator m x) + seminorm 0 x.
Proof.
  assert (hEq : generator (m + n) x = generator m (generator n x)).
  { apply (semigroup_axiom m n x). }
  assert (hStep : seminorm 0 (generator (m + 1) x) <= seminorm 0 (generator m x) + seminorm 0 x).
  { apply (generator_step 0 m x). }
  split.
  - exact hEq.
  - exact hStep.
Qed.

Lemma regularity_upgrade_pseudo_differential_symbol
    {V : Type} `{AnalysisStruct_pseudo_differential_symbol V}
    (k : nat)
    (x : V)
    (hs : smooth (k + 1) x) :
    seminorm k (generator 1 x) <= seminorm (k + 1) x /\
    smooth k (transform x).
Proof.
  assert (hNorm : seminorm k (generator 1 x) <= seminorm (k + 1) x).
  { apply (regularity_axiom k x hs). }
  assert (hSmooth : smooth k (transform x)).
  { apply (transform_smooth k x hs). }
  split.
  - exact hNorm.
  - exact hSmooth.
Qed.

Lemma generator_two_step_bound_pseudo_differential_symbol
    {V : Type} `{AnalysisStruct_pseudo_differential_symbol V}
    (k n : nat)
    (x : V) :
    seminorm k (generator ((n + 1) + 1) x) <=
      seminorm k (generator n x) + seminorm k x + seminorm k x.
Proof.
  assert (hStep1 :
      seminorm k (generator ((n + 1) + 1) x) <=
        seminorm k (generator (n + 1) x) + seminorm k x).
  { apply (generator_step k (n + 1) x). }
  assert (hStep0 :
      seminorm k (generator (n + 1) x) <=
        seminorm k (generator n x) + seminorm k x).
  { apply (generator_step k n x). }
  assert (hLift :
      seminorm k (generator (n + 1) x) + seminorm k x <=
        (seminorm k (generator n x) + seminorm k x) + seminorm k x).
  { apply (@nat_add_mono V H _ _ _ _ hStep0 (nat_le_refl (seminorm k x))). }
  exact (@nat_le_trans V H _ _ _ hStep1 hLift).
Qed.

Lemma symbol_semigroup_transport_pseudo_differential_symbol
    {V : Type} `{AnalysisStruct_pseudo_differential_symbol V}
    (m n : nat)
    (x : V) :
    transform (generator (m + n) x) =
      generator m (generator n (transform x)) /\
    generator (m + n) (transform x) =
      generator m (generator n (transform x)).
Proof.
  assert (hComm :
      transform (generator (m + n) x) =
        generator (m + n) (transform x)).
  { apply (symbol_comm_axiom (m + n) x). }
  assert (hSem :
      generator (m + n) (transform x) =
        generator m (generator n (transform x))).
  { apply (semigroup_axiom m n (transform x)). }
  assert (hMain :
      transform (generator (m + n) x) =
        generator m (generator n (transform x))).
  { rewrite hComm. exact hSem. }
  split.
  - exact hMain.
  - exact hSem.
Qed.

Lemma regularity_chart_generator_pseudo_differential_symbol
    {V : Type} `{AnalysisStruct_pseudo_differential_symbol V}
    (k n : nat)
    (x : V)
    (hs : smooth (k + 1) (generator n x)) :
    seminorm k (generator 1 (chart (generator n x))) <=
      seminorm (k + 1) (generator n x) /\
    smooth k (transform (chart (generator n x))).
Proof.
  assert (hSmoothChart :
      smooth (k + 1) (chart (generator n x))).
  { apply (chart_smooth (k + 1) (generator n x) hs). }
  assert (hRegChart :
      seminorm k (generator 1 (chart (generator n x))) <=
        seminorm (k + 1) (chart (generator n x))).
  { apply (regularity_axiom k (chart (generator n x)) hSmoothChart). }
  assert (hChartBound :
      seminorm (k + 1) (chart (generator n x)) <=
        seminorm (k + 1) (generator n x)).
  { apply (chart_bound (k + 1) (generator n x)). }
  assert (hReg :
      seminorm k (generator 1 (chart (generator n x))) <=
        seminorm (k + 1) (generator n x)).
  { exact (@nat_le_trans V H _ _ _ hRegChart hChartBound). }
  assert (hSmoothTrans :
      smooth k (transform (chart (generator n x)))).
  { apply (transform_smooth k (chart (generator n x)) hSmoothChart). }
  split.
  - exact hReg.
  - exact hSmoothTrans.
Qed.

Lemma transform_generator_zero_chart_pseudo_differential_symbol
    {V : Type} `{AnalysisStruct_pseudo_differential_symbol V}
    (k : nat)
    (x : V)
    (hs : smooth (k + 1) x) :
    seminorm k (transform (generator 0 (chart x))) <=
      seminorm (k + 1) x /\
    smooth k (transform (generator 0 (chart x))).
Proof.
  assert (hZero : generator 0 (chart x) = chart x).
  { apply (generator_zero (chart x)). }
  assert (hSmoothChart : smooth (k + 1) (chart x)).
  { apply (chart_smooth (k + 1) x hs). }
  assert (hSmoothRaw : smooth k (transform (chart x))).
  { apply (transform_smooth k (chart x) hSmoothChart). }
  assert (hShift :
      seminorm k (transform (chart x)) <=
        seminorm (k + 1) (chart x)).
  { apply (transform_shift k (chart x)). }
  assert (hChart :
      seminorm (k + 1) (chart x) <= seminorm (k + 1) x).
  { apply (chart_bound (k + 1) x). }
  assert (hBoundRaw :
      seminorm k (transform (chart x)) <= seminorm (k + 1) x).
  { exact (@nat_le_trans V H _ _ _ hShift hChart). }
  assert (hBound :
      seminorm k (transform (generator 0 (chart x))) <=
        seminorm (k + 1) x).
  { rewrite hZero. exact hBoundRaw. }
  assert (hSmooth :
      smooth k (transform (generator 0 (chart x)))).
  { rewrite hZero. exact hSmoothRaw. }
  split.
  - exact hBound.
  - exact hSmooth.
Qed.

Lemma compose_chart_transform_generator_bound_pseudo_differential_symbol
    {V : Type} `{AnalysisStruct_pseudo_differential_symbol V}
    (k m n : nat)
    (x y : V)
    (hx : smooth k (generator m x))
    (hy : smooth (k + 1) (generator n y)) :
    seminorm k
        (compose (chart (generator m x)) (transform (generator n y))) <=
      seminorm k (generator m x) + seminorm (k + 1) (generator n y) /\
    smooth k
        (compose (chart (generator m x)) (transform (generator n y))).
Proof.
  assert (hLeftSmooth : smooth k (chart (generator m x))).
  { apply (chart_smooth k (generator m x) hx). }
  assert (hRightSmooth : smooth k (transform (generator n y))).
  { apply (transform_smooth k (generator n y) hy). }
  assert (hSmooth :
      smooth k
        (compose (chart (generator m x)) (transform (generator n y)))).
  { apply (compose_smooth k (chart (generator m x)) (transform (generator n y)) hLeftSmooth hRightSmooth). }
  assert (hComp :
      seminorm k
          (compose (chart (generator m x)) (transform (generator n y))) <=
        seminorm k (chart (generator m x)) +
          seminorm k (transform (generator n y))).
  { apply (compose_bound k (chart (generator m x)) (transform (generator n y))). }
  assert (hLeftBound :
      seminorm k (chart (generator m x)) <= seminorm k (generator m x)).
  { apply (chart_bound k (generator m x)). }
  assert (hRightBound :
      seminorm k (transform (generator n y)) <=
        seminorm (k + 1) (generator n y)).
  { apply (transform_shift k (generator n y)). }
  assert (hAdd :
      seminorm k (chart (generator m x)) +
          seminorm k (transform (generator n y)) <=
        seminorm k (generator m x) +
          seminorm (k + 1) (generator n y)).
  { exact (@nat_add_mono V H _ _ _ _ hLeftBound hRightBound). }
  assert (hBound :
      seminorm k
          (compose (chart (generator m x)) (transform (generator n y))) <=
        seminorm k (generator m x) +
          seminorm (k + 1) (generator n y)).
  { exact (@nat_le_trans V H _ _ _ hComp hAdd). }
  split.
  - exact hBound.
  - exact hSmooth.
Qed.

Lemma generator_semigroup_successor_bound_pseudo_differential_symbol
    {V : Type} `{AnalysisStruct_pseudo_differential_symbol V}
    (k m n : nat)
    (x : V) :
    seminorm k (generator ((m + n) + 1) x) <=
      seminorm k (generator m (generator n x)) + seminorm k x /\
    transform (generator (m + n) x) =
      generator m (generator n (transform x)).
Proof.
  assert (hStep :
      seminorm k (generator ((m + n) + 1) x) <=
        seminorm k (generator (m + n) x) + seminorm k x).
  { apply (generator_step k (m + n) x). }
  assert (hSem :
      generator (m + n) x = generator m (generator n x)).
  { apply (semigroup_axiom m n x). }
  assert (hBound :
      seminorm k (generator ((m + n) + 1) x) <=
        seminorm k (generator m (generator n x)) + seminorm k x).
  { rewrite <- hSem. exact hStep. }
  assert (hComm :
      transform (generator (m + n) x) =
        generator (m + n) (transform x)).
  { apply (symbol_comm_axiom (m + n) x). }
  assert (hSemTrans :
      generator (m + n) (transform x) =
        generator m (generator n (transform x))).
  { apply (semigroup_axiom m n (transform x)). }
  assert (hEq :
      transform (generator (m + n) x) =
        generator m (generator n (transform x))).
  { rewrite hComm. exact hSemTrans. }
  split.
  - exact hBound.
  - exact hEq.
Qed.
