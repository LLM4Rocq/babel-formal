/-
BENCHMARK_ID: TINY_MATHLIB_BATCH05_ANALYSIS_PSEUDO_DIFFERENTIAL_SYMBOL_LIKE
PAIR_STEM: analysis_pseudo_differential_symbol_like
MATH_DOMAIN: Analysis
SOURCE_MATHLIB: Mathlib/Analysis/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 18
-/

universe u

class AnalysisStruct_pseudo_differential_symbol (V : Type u) where
  seminorm : Nat → V → Nat
  chart : V → V
  transform : V → V
  compose : V → V → V
  generator : Nat → V → V
  smooth : Nat → V → Prop
  smooth_lift : ∀ k : Nat, ∀ x : V, smooth k x → smooth (k + 1) x
  chart_smooth : ∀ k : Nat, ∀ x : V, smooth k x → smooth k (chart x)
  transform_smooth : ∀ k : Nat, ∀ x : V, smooth (k + 1) x → smooth k (transform x)
  compose_smooth :
    ∀ k : Nat,
      ∀ x y : V,
        smooth k x →
        smooth k y →
          smooth k (compose x y)
  chart_bound : ∀ k : Nat, ∀ x : V, seminorm k (chart x) ≤ seminorm k x
  transform_shift : ∀ k : Nat, ∀ x : V, seminorm k (transform x) ≤ seminorm (k + 1) x
  compose_bound :
    ∀ k : Nat,
      ∀ x y : V,
        seminorm k (compose x y) ≤ seminorm k x + seminorm k y
  nat_add_mono : ∀ a b c d : Nat, a ≤ b → c ≤ d → a + c ≤ b + d
  nat_le_trans : ∀ a b c : Nat, a ≤ b → b ≤ c → a ≤ c
  nat_le_refl : ∀ a : Nat, a ≤ a
  nat_add_comm : ∀ a b : Nat, a + b = b + a
  generator_step :
    ∀ k n : Nat,
      ∀ x : V,
        seminorm k (generator (n + 1) x) ≤ seminorm k (generator n x) + seminorm k x
  generator_zero : ∀ x : V, generator 0 x = x
  semigroup_axiom :
    ∀ m n : Nat,
      ∀ x : V,
        generator (m + n) x = generator m (generator n x)
  symbol_comm_axiom :
    ∀ n : Nat,
      ∀ x : V,
        transform (generator n x) = generator n (transform x)
  regularity_axiom :
    ∀ k : Nat,
      ∀ x : V,
        smooth (k + 1) x →
          seminorm k (generator 1 x) ≤ seminorm (k + 1) x

def NormCtrl_pseudo_differential_symbol
    {V : Type u} [h : AnalysisStruct_pseudo_differential_symbol V] : Prop :=
  ∀ k : Nat, ∀ x : V, h.seminorm k (h.chart x) ≤ h.seminorm k x

def LocalChart_pseudo_differential_symbol (V : Type u) : Type u :=
  V → V

def Transform_pseudo_differential_symbol (V : Type u) : Type u :=
  V → V

def Generator_pseudo_differential_symbol (V : Type u) : Type u :=
  Nat → V → V

theorem local_estimate_pseudo_differential_symbol
    {V : Type u} [h : AnalysisStruct_pseudo_differential_symbol V] :
    NormCtrl_pseudo_differential_symbol (V := V) := by
  intro k x
  have hBound : h.seminorm k (h.chart x) ≤ h.seminorm k x := h.chart_bound k x
  have hSelf : h.smooth k x → h.smooth k (h.chart x) := fun hs => h.chart_smooth k x hs
  have _ : h.smooth k x → h.smooth k (h.chart x) := hSelf
  exact hBound

theorem patching_estimate_pseudo_differential_symbol
    {V : Type u} [h : AnalysisStruct_pseudo_differential_symbol V]
    (k : Nat)
    (x : V)
    (hs : h.smooth (k + 1) x) :
    h.seminorm k (h.transform (h.chart x)) ≤ h.seminorm (k + 1) x ∧
    h.smooth k (h.transform (h.chart x)) := by
  have hSmoothChart : h.smooth (k + 1) (h.chart x) := h.chart_smooth (k + 1) x hs
  have hSmoothTrans : h.smooth k (h.transform (h.chart x)) :=
    h.transform_smooth k (h.chart x) hSmoothChart
  have hShift : h.seminorm k (h.transform (h.chart x)) ≤ h.seminorm (k + 1) (h.chart x) :=
    h.transform_shift k (h.chart x)
  have hChart : h.seminorm (k + 1) (h.chart x) ≤ h.seminorm (k + 1) x :=
    h.chart_bound (k + 1) x
  have hBound : h.seminorm k (h.transform (h.chart x)) ≤ h.seminorm (k + 1) x :=
    h.nat_le_trans _ _ _ hShift hChart
  exact And.intro hBound hSmoothTrans

theorem transform_isometry_pseudo_differential_symbol
    {V : Type u} [h : AnalysisStruct_pseudo_differential_symbol V]
    (k : Nat)
    (x : V)
    (hIso :
      h.seminorm k (h.transform x) =
        h.seminorm k (h.chart x) + h.seminorm (k + 1) x) :
    h.seminorm k (h.transform x) =
      h.seminorm (k + 1) x + h.seminorm k (h.chart x) ∧
    h.seminorm k (h.chart x) ≤ h.seminorm k x := by
  have hComm :
      h.seminorm k (h.chart x) + h.seminorm (k + 1) x =
        h.seminorm (k + 1) x + h.seminorm k (h.chart x) :=
    h.nat_add_comm (h.seminorm k (h.chart x)) (h.seminorm (k + 1) x)
  have hEq :
      h.seminorm k (h.transform x) =
        h.seminorm (k + 1) x + h.seminorm k (h.chart x) := by
    calc
      h.seminorm k (h.transform x) = h.seminorm k (h.chart x) + h.seminorm (k + 1) x := hIso
      _ = h.seminorm (k + 1) x + h.seminorm k (h.chart x) := hComm
  have hChart : h.seminorm k (h.chart x) ≤ h.seminorm k x := h.chart_bound k x
  exact And.intro hEq hChart

theorem decomposition_bound_pseudo_differential_symbol
    {V : Type u} [h : AnalysisStruct_pseudo_differential_symbol V]
    (k : Nat)
    (x y : V)
    (hx : h.smooth k x)
    (hy : h.smooth k y) :
    h.seminorm k (h.compose (h.transform x) (h.chart y)) ≤
      h.seminorm (k + 1) x + h.seminorm k y ∧
    h.smooth k (h.compose (h.transform x) (h.chart y)) := by
  have hx' : h.smooth k (h.transform x) :=
    h.transform_smooth k x (h.smooth_lift k x hx)
  have hy' : h.smooth k (h.chart y) := h.chart_smooth k y hy
  have hSmooth : h.smooth k (h.compose (h.transform x) (h.chart y)) :=
    h.compose_smooth k (h.transform x) (h.chart y) hx' hy'
  have hComp :
      h.seminorm k (h.compose (h.transform x) (h.chart y)) ≤
        h.seminorm k (h.transform x) + h.seminorm k (h.chart y) :=
    h.compose_bound k (h.transform x) (h.chart y)
  have hLeft : h.seminorm k (h.transform x) ≤ h.seminorm (k + 1) x :=
    h.transform_shift k x
  have hRight : h.seminorm k (h.chart y) ≤ h.seminorm k y :=
    h.chart_bound k y
  have hAdd :
      h.seminorm k (h.transform x) + h.seminorm k (h.chart y) ≤
        h.seminorm (k + 1) x + h.seminorm k y :=
    h.nat_add_mono _ _ _ _ hLeft hRight
  have hBound :
      h.seminorm k (h.compose (h.transform x) (h.chart y)) ≤
        h.seminorm (k + 1) x + h.seminorm k y :=
    h.nat_le_trans _ _ _ hComp hAdd
  exact And.intro hBound hSmooth

theorem symbol_composition_pseudo_differential_symbol
    {V : Type u} [h : AnalysisStruct_pseudo_differential_symbol V]
    (n : Nat)
    (x : V) :
    h.transform (h.generator n x) = h.generator n (h.transform x) ∧
    h.transform (h.generator (n + 0) x) = h.generator (n + 0) (h.transform x) := by
  have hMain : h.transform (h.generator n x) = h.generator n (h.transform x) :=
    h.symbol_comm_axiom n x
  have hShift : h.transform (h.generator (n + 0) x) = h.generator (n + 0) (h.transform x) :=
    h.symbol_comm_axiom (n + 0) x
  exact And.intro hMain hShift

theorem semigroup_generation_pseudo_differential_symbol
    {V : Type u} [h : AnalysisStruct_pseudo_differential_symbol V]
    (m n : Nat)
    (x : V) :
    h.generator (m + n) x = h.generator m (h.generator n x) ∧
    h.seminorm 0 (h.generator (m + 1) x) ≤ h.seminorm 0 (h.generator m x) + h.seminorm 0 x := by
  have hEq : h.generator (m + n) x = h.generator m (h.generator n x) :=
    h.semigroup_axiom m n x
  have hStep : h.seminorm 0 (h.generator (m + 1) x) ≤ h.seminorm 0 (h.generator m x) + h.seminorm 0 x :=
    h.generator_step 0 m x
  exact And.intro hEq hStep

theorem regularity_upgrade_pseudo_differential_symbol
    {V : Type u} [h : AnalysisStruct_pseudo_differential_symbol V]
    (k : Nat)
    (x : V)
    (hs : h.smooth (k + 1) x) :
    h.seminorm k (h.generator 1 x) ≤ h.seminorm (k + 1) x ∧
    h.smooth k (h.transform x) := by
  have hNorm : h.seminorm k (h.generator 1 x) ≤ h.seminorm (k + 1) x :=
    h.regularity_axiom k x hs
  have hSmooth : h.smooth k (h.transform x) :=
    h.transform_smooth k x hs
  exact And.intro hNorm hSmooth

theorem generator_two_step_bound_pseudo_differential_symbol
    {V : Type u} [h : AnalysisStruct_pseudo_differential_symbol V]
    (k n : Nat)
    (x : V) :
    h.seminorm k (h.generator ((n + 1) + 1) x) ≤
      h.seminorm k (h.generator n x) + h.seminorm k x + h.seminorm k x := by
  have hStep1 :
      h.seminorm k (h.generator ((n + 1) + 1) x) ≤
        h.seminorm k (h.generator (n + 1) x) + h.seminorm k x :=
    h.generator_step k (n + 1) x
  have hStep0 :
      h.seminorm k (h.generator (n + 1) x) ≤
        h.seminorm k (h.generator n x) + h.seminorm k x :=
    h.generator_step k n x
  have hLift :
      h.seminorm k (h.generator (n + 1) x) + h.seminorm k x ≤
        (h.seminorm k (h.generator n x) + h.seminorm k x) + h.seminorm k x :=
    h.nat_add_mono _ _ _ _ hStep0 (h.nat_le_refl (h.seminorm k x))
  exact h.nat_le_trans _ _ _ hStep1 hLift

theorem symbol_semigroup_transport_pseudo_differential_symbol
    {V : Type u} [h : AnalysisStruct_pseudo_differential_symbol V]
    (m n : Nat)
    (x : V) :
    h.transform (h.generator (m + n) x) =
      h.generator m (h.generator n (h.transform x)) ∧
    h.generator (m + n) (h.transform x) =
      h.generator m (h.generator n (h.transform x)) := by
  have hComm :
      h.transform (h.generator (m + n) x) =
        h.generator (m + n) (h.transform x) :=
    h.symbol_comm_axiom (m + n) x
  have hSem :
      h.generator (m + n) (h.transform x) =
        h.generator m (h.generator n (h.transform x)) :=
    h.semigroup_axiom m n (h.transform x)
  have hMain :
      h.transform (h.generator (m + n) x) =
        h.generator m (h.generator n (h.transform x)) :=
    Eq.trans hComm hSem
  exact And.intro hMain hSem

theorem regularity_chart_generator_pseudo_differential_symbol
    {V : Type u} [h : AnalysisStruct_pseudo_differential_symbol V]
    (k n : Nat)
    (x : V)
    (hs : h.smooth (k + 1) (h.generator n x)) :
    h.seminorm k (h.generator 1 (h.chart (h.generator n x))) ≤
      h.seminorm (k + 1) (h.generator n x) ∧
    h.smooth k (h.transform (h.chart (h.generator n x))) := by
  have hSmoothChart :
      h.smooth (k + 1) (h.chart (h.generator n x)) :=
    h.chart_smooth (k + 1) (h.generator n x) hs
  have hRegChart :
      h.seminorm k (h.generator 1 (h.chart (h.generator n x))) ≤
        h.seminorm (k + 1) (h.chart (h.generator n x)) :=
    h.regularity_axiom k (h.chart (h.generator n x)) hSmoothChart
  have hChartBound :
      h.seminorm (k + 1) (h.chart (h.generator n x)) ≤
        h.seminorm (k + 1) (h.generator n x) :=
    h.chart_bound (k + 1) (h.generator n x)
  have hReg :
      h.seminorm k (h.generator 1 (h.chart (h.generator n x))) ≤
        h.seminorm (k + 1) (h.generator n x) :=
    h.nat_le_trans _ _ _ hRegChart hChartBound
  have hSmoothTrans :
      h.smooth k (h.transform (h.chart (h.generator n x))) :=
    h.transform_smooth k (h.chart (h.generator n x)) hSmoothChart
  exact And.intro hReg hSmoothTrans

theorem transform_generator_zero_chart_pseudo_differential_symbol
    {V : Type u} [h : AnalysisStruct_pseudo_differential_symbol V]
    (k : Nat)
    (x : V)
    (hs : h.smooth (k + 1) x) :
    h.seminorm k (h.transform (h.generator 0 (h.chart x))) ≤
      h.seminorm (k + 1) x ∧
    h.smooth k (h.transform (h.generator 0 (h.chart x))) := by
  have hZero : h.generator 0 (h.chart x) = h.chart x :=
    h.generator_zero (h.chart x)
  have hSmoothChart : h.smooth (k + 1) (h.chart x) :=
    h.chart_smooth (k + 1) x hs
  have hSmoothRaw : h.smooth k (h.transform (h.chart x)) :=
    h.transform_smooth k (h.chart x) hSmoothChart
  have hShift :
      h.seminorm k (h.transform (h.chart x)) ≤
        h.seminorm (k + 1) (h.chart x) :=
    h.transform_shift k (h.chart x)
  have hChart :
      h.seminorm (k + 1) (h.chart x) ≤ h.seminorm (k + 1) x :=
    h.chart_bound (k + 1) x
  have hBoundRaw :
      h.seminorm k (h.transform (h.chart x)) ≤ h.seminorm (k + 1) x :=
    h.nat_le_trans _ _ _ hShift hChart
  have hBound :
      h.seminorm k (h.transform (h.generator 0 (h.chart x))) ≤
        h.seminorm (k + 1) x := by
    simpa [hZero] using hBoundRaw
  have hSmooth :
      h.smooth k (h.transform (h.generator 0 (h.chart x))) := by
    simpa [hZero] using hSmoothRaw
  exact And.intro hBound hSmooth

theorem compose_chart_transform_generator_bound_pseudo_differential_symbol
    {V : Type u} [h : AnalysisStruct_pseudo_differential_symbol V]
    (k m n : Nat)
    (x y : V)
    (hx : h.smooth k (h.generator m x))
    (hy : h.smooth (k + 1) (h.generator n y)) :
    h.seminorm k
        (h.compose (h.chart (h.generator m x)) (h.transform (h.generator n y))) ≤
      h.seminorm k (h.generator m x) + h.seminorm (k + 1) (h.generator n y) ∧
    h.smooth k
        (h.compose (h.chart (h.generator m x)) (h.transform (h.generator n y))) := by
  have hLeftSmooth : h.smooth k (h.chart (h.generator m x)) :=
    h.chart_smooth k (h.generator m x) hx
  have hRightSmooth : h.smooth k (h.transform (h.generator n y)) :=
    h.transform_smooth k (h.generator n y) hy
  have hSmooth :
      h.smooth k
        (h.compose (h.chart (h.generator m x)) (h.transform (h.generator n y))) :=
    h.compose_smooth k (h.chart (h.generator m x)) (h.transform (h.generator n y))
      hLeftSmooth hRightSmooth
  have hComp :
      h.seminorm k
          (h.compose (h.chart (h.generator m x)) (h.transform (h.generator n y))) ≤
        h.seminorm k (h.chart (h.generator m x)) +
          h.seminorm k (h.transform (h.generator n y)) :=
    h.compose_bound k (h.chart (h.generator m x)) (h.transform (h.generator n y))
  have hLeftBound :
      h.seminorm k (h.chart (h.generator m x)) ≤ h.seminorm k (h.generator m x) :=
    h.chart_bound k (h.generator m x)
  have hRightBound :
      h.seminorm k (h.transform (h.generator n y)) ≤
        h.seminorm (k + 1) (h.generator n y) :=
    h.transform_shift k (h.generator n y)
  have hAdd :
      h.seminorm k (h.chart (h.generator m x)) +
          h.seminorm k (h.transform (h.generator n y)) ≤
        h.seminorm k (h.generator m x) +
          h.seminorm (k + 1) (h.generator n y) :=
    h.nat_add_mono _ _ _ _ hLeftBound hRightBound
  have hBound :
      h.seminorm k
          (h.compose (h.chart (h.generator m x)) (h.transform (h.generator n y))) ≤
        h.seminorm k (h.generator m x) +
          h.seminorm (k + 1) (h.generator n y) :=
    h.nat_le_trans _ _ _ hComp hAdd
  exact And.intro hBound hSmooth

theorem generator_semigroup_successor_bound_pseudo_differential_symbol
    {V : Type u} [h : AnalysisStruct_pseudo_differential_symbol V]
    (k m n : Nat)
    (x : V) :
    h.seminorm k (h.generator ((m + n) + 1) x) ≤
      h.seminorm k (h.generator m (h.generator n x)) + h.seminorm k x ∧
    h.transform (h.generator (m + n) x) =
      h.generator m (h.generator n (h.transform x)) := by
  have hStep :
      h.seminorm k (h.generator ((m + n) + 1) x) ≤
        h.seminorm k (h.generator (m + n) x) + h.seminorm k x :=
    h.generator_step k (m + n) x
  have hSem :
      h.generator (m + n) x = h.generator m (h.generator n x) :=
    h.semigroup_axiom m n x
  have hBound :
      h.seminorm k (h.generator ((m + n) + 1) x) ≤
        h.seminorm k (h.generator m (h.generator n x)) + h.seminorm k x := by
    simpa [hSem] using hStep
  have hComm :
      h.transform (h.generator (m + n) x) =
        h.generator (m + n) (h.transform x) :=
    h.symbol_comm_axiom (m + n) x
  have hSemTrans :
      h.generator (m + n) (h.transform x) =
        h.generator m (h.generator n (h.transform x)) :=
    h.semigroup_axiom m n (h.transform x)
  have hEq :
      h.transform (h.generator (m + n) x) =
        h.generator m (h.generator n (h.transform x)) :=
    Eq.trans hComm hSemTrans
  exact And.intro hBound hEq
