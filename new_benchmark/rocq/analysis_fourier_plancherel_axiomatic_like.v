(*
BENCHMARK_ID: TINY_MATHLIB_BATCH05_ANALYSIS_FOURIER_PLANCHEREL_AXIOMATIC_LIKE
PAIR_STEM: analysis_fourier_plancherel_axiomatic_like
MATH_DOMAIN: Analysis
SOURCE_MATHLIB: Mathlib/Analysis/Fourier
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 17
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class AnalysisStruct_fourier_plancherel (V : Type) := {
  norm : V -> nat;
  chart : V -> V;
  transform : V -> V;
  generator : V -> V;
  nat_le_trans : forall a b c : nat, a <= b -> b <= c -> a <= c;
  nat_add_mono : forall a b c d : nat, a <= b -> c <= d -> a + c <= b + d;
  nat_le_refl : forall a : nat, a <= a;
  nat_add_comm : forall a b : nat, a + b = b + a;
  nat_le_add_right : forall a b : nat, a <= a + b;
  local_estimate_axiom :
    forall x : V,
      norm (transform x) <= norm (chart x) + norm x;
  patching_estimate_axiom :
    forall x y : V,
      norm (chart x) <= norm (chart y) + norm x + norm y;
  isometry_axiom :
    forall x : V,
      norm (transform x) = norm x;
  decomposition_axiom :
    forall x y : V,
      norm (transform x) + norm (transform y) = norm x + norm y;
  symbol_composition_axiom :
    forall x : V,
      transform (generator x) = generator (transform x);
  semigroup_axiom :
    forall x : V,
      norm (generator x) <= norm x + norm (transform x);
  regularity_axiom :
    forall x : V,
      norm (chart (generator x)) <= norm (generator x) + norm (chart x)
}.

Definition NormCtrl_fourier_plancherel
    (V : Type) : Type :=
  V -> nat.

Definition LocalChart_fourier_plancherel
    (V : Type) : Type :=
  V -> V.

Definition Transform_fourier_plancherel
    (V : Type) : Type :=
  V -> V.

Definition Generator_fourier_plancherel
    (V : Type) : Type :=
  V -> V.

Lemma local_estimate_fourier_plancherel
    {V : Type} `{AnalysisStruct_fourier_plancherel V}
    (x : V) :
    norm (transform x) <= norm (chart x) + norm x /\
    norm (chart x) <= norm (chart x).
Proof.
  assert (hLoc : norm (transform x) <= norm (chart x) + norm x).
  { apply local_estimate_axiom. }
  assert (hDiag : norm (chart x) <= norm (chart x)).
  { apply nat_le_refl. }
  split.
  - exact hLoc.
  - exact hDiag.
Qed.

Lemma patching_estimate_fourier_plancherel
    {V : Type} `{AnalysisStruct_fourier_plancherel V}
    (x y : V) :
    norm (chart x) <= norm (chart y) + norm x + norm y /\
    norm (transform y) = norm y.
Proof.
  assert (hPatch : norm (chart x) <= norm (chart y) + norm x + norm y).
  { apply (patching_estimate_axiom x y). }
  assert (hIsoY : norm (transform y) = norm y).
  { apply (isometry_axiom y). }
  split.
  - exact hPatch.
  - exact hIsoY.
Qed.

Lemma transform_isometry_fourier_plancherel
    {V : Type} `{AnalysisStruct_fourier_plancherel V}
    (x : V) :
    norm (transform x) = norm x /\
    norm (transform x) <= norm x + norm x.
Proof.
  assert (hIso : norm (transform x) = norm x).
  { apply isometry_axiom. }
  assert (hBound : norm x <= norm x + norm x).
  {
    exact (nat_le_add_right (norm x) (norm x)).
  }
  assert (hFinal : norm (transform x) <= norm x + norm x).
  {
    rewrite hIso.
    exact hBound.
  }
  split.
  - exact hIso.
  - exact hFinal.
Qed.

Lemma decomposition_bound_fourier_plancherel
    {V : Type} `{AnalysisStruct_fourier_plancherel V}
    (x y : V) :
    norm (transform x) + norm (transform y) = norm x + norm y /\
    norm (transform x) + norm (transform y) = norm y + norm x.
Proof.
  assert (hDec : norm (transform x) + norm (transform y) = norm x + norm y).
  { apply (decomposition_axiom x y). }
  assert (hComm : norm x + norm y = norm y + norm x).
  { apply (nat_add_comm (norm x) (norm y)). }
  assert (hSwap : norm (transform x) + norm (transform y) = norm y + norm x).
  {
    rewrite hDec.
    exact hComm.
  }
  split.
  - exact hDec.
  - exact hSwap.
Qed.

Lemma symbol_composition_fourier_plancherel
    {V : Type} `{AnalysisStruct_fourier_plancherel V}
    (x : V) :
    transform (generator x) = generator (transform x) /\
    norm (generator x) <= norm x + norm (transform x).
Proof.
  assert (hEq : transform (generator x) = generator (transform x)).
  { apply symbol_composition_axiom. }
  assert (hBound : norm (generator x) <= norm x + norm (transform x)).
  { apply semigroup_axiom. }
  split.
  - exact hEq.
  - exact hBound.
Qed.

Lemma semigroup_generation_fourier_plancherel
    {V : Type} `{AnalysisStruct_fourier_plancherel V}
    (x : V) :
    norm (generator x) <= norm x + norm x.
Proof.
  assert (hBase : norm (generator x) <= norm x + norm (transform x)).
  { apply semigroup_axiom. }
  assert (hIso : norm (transform x) = norm x).
  { apply isometry_axiom. }
  assert (hRewrite : norm (generator x) <= norm x + norm x).
  {
    rewrite hIso in hBase.
    exact hBase.
  }
  exact hRewrite.
Qed.

Lemma regularity_upgrade_fourier_plancherel
    {V : Type} `{AnalysisStruct_fourier_plancherel V}
    (x : V) :
    norm (chart (generator x)) <= (norm x + norm x) + norm (chart x).
Proof.
  assert (hReg : norm (chart (generator x)) <= norm (generator x) + norm (chart x)).
  { apply regularity_axiom. }
  assert (hGen : norm (generator x) <= norm x + norm x).
  { apply semigroup_generation_fourier_plancherel. }
  assert (hLift : norm (generator x) + norm (chart x) <= (norm x + norm x) + norm (chart x)).
  { exact (@nat_add_mono V H _ _ _ _ hGen (@nat_le_refl V H (norm (chart x)))). }
  exact (@nat_le_trans V H _ _ _ hReg hLift).
Qed.

Lemma chart_generator_patching_bound_fourier_plancherel
    {V : Type} `{AnalysisStruct_fourier_plancherel V}
    (x y : V) :
    norm (chart (generator x)) <= (norm (chart y) + (norm x + norm x)) + norm y.
Proof.
  assert (hPatch : norm (chart (generator x)) <= norm (chart y) + norm (generator x) + norm y).
  { apply (patching_estimate_axiom (generator x) y). }
  assert (hGen : norm (generator x) <= norm x + norm x).
  { apply semigroup_generation_fourier_plancherel. }
  assert (hHead : norm (chart y) + norm (generator x) <= norm (chart y) + (norm x + norm x)).
  { exact (@nat_add_mono V H _ _ _ _ (@nat_le_refl V H (norm (chart y))) hGen). }
  assert (hTail : (norm (chart y) + norm (generator x)) + norm y <= (norm (chart y) + (norm x + norm x)) + norm y).
  { exact (@nat_add_mono V H _ _ _ _ hHead (@nat_le_refl V H (norm y))). }
  exact (@nat_le_trans V H _ _ _ hPatch hTail).
Qed.

Lemma symbol_generator_isometry_bound_fourier_plancherel
    {V : Type} `{AnalysisStruct_fourier_plancherel V}
    (x : V) :
    transform (generator x) = generator (transform x) /\
    norm (transform (generator x)) <= norm x + norm x.
Proof.
  assert (hSym : transform (generator x) = generator (transform x)).
  { apply symbol_composition_axiom. }
  assert (hIsoGen : norm (transform (generator x)) = norm (generator x)).
  { apply (isometry_axiom (generator x)). }
  assert (hGen : norm (generator x) <= norm x + norm x).
  { apply semigroup_generation_fourier_plancherel. }
  assert (hBound : norm (transform (generator x)) <= norm x + norm x).
  {
    rewrite hIsoGen.
    exact hGen.
  }
  split.
  - exact hSym.
  - exact hBound.
Qed.

Lemma decomposition_generator_mix_bound_fourier_plancherel
    {V : Type} `{AnalysisStruct_fourier_plancherel V}
    (x y : V) :
    norm (transform (generator x)) + norm (transform y) <=
      (norm x + norm x) + norm y.
Proof.
  assert (hDec : norm (transform (generator x)) + norm (transform y) = norm (generator x) + norm y).
  { apply (decomposition_axiom (generator x) y). }
  assert (hGen : norm (generator x) <= norm x + norm x).
  { apply semigroup_generation_fourier_plancherel. }
  assert (hLift : norm (generator x) + norm y <= (norm x + norm x) + norm y).
  { exact (@nat_add_mono V H _ _ _ _ hGen (@nat_le_refl V H (norm y))). }
  assert (hFinal : norm (transform (generator x)) + norm (transform y) <= (norm x + norm x) + norm y).
  {
    rewrite hDec.
    exact hLift.
  }
  exact hFinal.
Qed.

Lemma semigroup_transform_input_bound_fourier_plancherel
    {V : Type} `{AnalysisStruct_fourier_plancherel V}
    (x : V) :
    norm (generator (transform x)) <= norm x + norm x.
Proof.
  assert (hGenT : norm (generator (transform x)) <= norm (transform x) + norm (transform x)).
  { apply semigroup_generation_fourier_plancherel. }
  assert (hIso : norm (transform x) = norm x).
  { apply isometry_axiom. }
  assert (hFinal : norm (generator (transform x)) <= norm x + norm x).
  {
    rewrite hIso in hGenT.
    exact hGenT.
  }
  exact hFinal.
Qed.

Lemma double_transform_local_patching_bound_fourier_plancherel
    {V : Type} `{AnalysisStruct_fourier_plancherel V}
    (x y : V) :
    norm (transform (transform x)) <=
      (norm (chart y) + norm x + norm y) + norm x.
Proof.
  assert (hLocal : norm (transform (transform x)) <= norm (chart (transform x)) + norm (transform x)).
  { apply (local_estimate_axiom (transform x)). }
  assert (hPatch : norm (chart (transform x)) <= norm (chart y) + norm (transform x) + norm y).
  { apply (patching_estimate_axiom (transform x) y). }
  assert (hIso : norm (transform x) = norm x).
  { apply (isometry_axiom x). }
  assert (hPatch' : norm (chart (transform x)) <= norm (chart y) + norm x + norm y).
  {
    rewrite hIso in hPatch.
    exact hPatch.
  }
  assert (hLift :
    norm (chart (transform x)) + norm (transform x) <=
      (norm (chart y) + norm x + norm y) + norm (transform x)).
  { exact (@nat_add_mono V H _ _ _ _ hPatch' (@nat_le_refl V H (norm (transform x)))). }
  assert (hIsoLe : norm (transform x) <= norm x).
  {
    rewrite hIso.
    apply nat_le_refl.
  }
  assert (hTail :
    (norm (chart y) + norm x + norm y) + norm (transform x) <=
      (norm (chart y) + norm x + norm y) + norm x).
  { exact (@nat_add_mono V H _ _ _ _ (@nat_le_refl V H (norm (chart y) + norm x + norm y)) hIsoLe). }
  assert (hMid :
    norm (chart (transform x)) + norm (transform x) <=
      (norm (chart y) + norm x + norm y) + norm x).
  { exact (@nat_le_trans V H _ _ _ hLift hTail). }
  exact (@nat_le_trans V H _ _ _ hLocal hMid).
Qed.
