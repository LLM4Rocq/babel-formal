/-
BENCHMARK_ID: TINY_MATHLIB_BATCH06_CATEGORY_VAN_KAMPEN_COLIMIT_LIKE
PAIR_STEM: category_van_kampen_colimit_like
MATH_DOMAIN: Category Theory
SOURCE_MATHLIB: Mathlib/CategoryTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u v

class VanKampenStruct_colimit (Obj : Type u) where
  Arrow : Type v
  pull : Arrow -> Obj -> Obj
  compare : Arrow -> Obj -> Obj
  IsColim : Obj -> Prop
  PullbackStable : Obj -> Prop
  DescentData : Obj -> Prop
  Reflective : Obj -> Prop
  colim_pull : forall (a : Arrow) (c : Obj), IsColim c -> IsColim (pull a c)
  compare_sound : forall (a : Arrow) (c : Obj), IsColim (pull a c) -> PullbackStable (compare a c)
  compare_complete :
    forall (a : Arrow) (c : Obj), PullbackStable (compare a c) -> IsColim (pull a c)
  descent_intro : forall (c : Obj), PullbackStable c -> DescentData c
  descent_elim : forall (c : Obj), DescentData c -> PullbackStable c
  reflection_intro : forall (c : Obj), DescentData c -> Reflective c
  reflection_elim : forall (c : Obj), Reflective c -> PullbackStable c
  universal_square_rule : forall (c : Obj), Reflective c -> PullbackStable c -> IsColim c
  colim_implies_stable : forall (c : Obj), IsColim c -> PullbackStable c
  pull_idempotent : forall (a : Arrow) (c : Obj), pull a (pull a c) = pull a c
  compare_transport : forall (a : Arrow) (c : Obj), compare a (pull a c) = compare a c

structure CoconeData_van_kampen_colimit
    (Obj : Type u) [VanKampenStruct_colimit Obj] where
  apex : Obj
  has_colimit : VanKampenStruct_colimit.IsColim apex
  has_stability : VanKampenStruct_colimit.PullbackStable apex
  has_reflection : VanKampenStruct_colimit.Reflective apex

def pullback_cocone_van_kampen_colimit
    {Obj : Type u} [VanKampenStruct_colimit Obj]
    (a : VanKampenStruct_colimit.Arrow (Obj := Obj))
    (K : CoconeData_van_kampen_colimit Obj) : Obj :=
  VanKampenStruct_colimit.pull a K.apex

def comparison_functor_van_kampen_colimit
    {Obj : Type u} [VanKampenStruct_colimit Obj]
    (a : VanKampenStruct_colimit.Arrow (Obj := Obj))
    (K : CoconeData_van_kampen_colimit Obj) : Obj :=
  VanKampenStruct_colimit.compare a K.apex

def descent_kernel_van_kampen_colimit
    {Obj : Type u} [VanKampenStruct_colimit Obj]
    (a : VanKampenStruct_colimit.Arrow (Obj := Obj))
    (K : CoconeData_van_kampen_colimit Obj) : Prop :=
  VanKampenStruct_colimit.DescentData (comparison_functor_van_kampen_colimit a K)

theorem pullback_preserves_colimit_van_kampen_colimit
    {Obj : Type u} [VanKampenStruct_colimit Obj]
    (a : VanKampenStruct_colimit.Arrow (Obj := Obj))
    (K : CoconeData_van_kampen_colimit Obj) :
    VanKampenStruct_colimit.IsColim (pullback_cocone_van_kampen_colimit a K) := by
  have hcolim_apex : VanKampenStruct_colimit.IsColim K.apex := K.has_colimit
  have hpull : VanKampenStruct_colimit.IsColim (VanKampenStruct_colimit.pull a K.apex) :=
    VanKampenStruct_colimit.colim_pull a K.apex hcolim_apex
  simpa [pullback_cocone_van_kampen_colimit] using hpull

theorem comparison_faithful_van_kampen_colimit
    {Obj : Type u} [VanKampenStruct_colimit Obj]
    (a : VanKampenStruct_colimit.Arrow (Obj := Obj))
    (K : CoconeData_van_kampen_colimit Obj) :
    VanKampenStruct_colimit.PullbackStable (comparison_functor_van_kampen_colimit a K) := by
  have hpull : VanKampenStruct_colimit.IsColim (pullback_cocone_van_kampen_colimit a K) :=
    pullback_preserves_colimit_van_kampen_colimit a K
  have hstable_cmp : VanKampenStruct_colimit.PullbackStable
      (VanKampenStruct_colimit.compare a K.apex) :=
    VanKampenStruct_colimit.compare_sound a K.apex (by
      simpa [pullback_cocone_van_kampen_colimit] using hpull)
  simpa [comparison_functor_van_kampen_colimit] using hstable_cmp

theorem comparison_full_van_kampen_colimit
    {Obj : Type u} [VanKampenStruct_colimit Obj]
    (a : VanKampenStruct_colimit.Arrow (Obj := Obj))
    (K : CoconeData_van_kampen_colimit Obj)
    (hstable_cmp : VanKampenStruct_colimit.PullbackStable
      (comparison_functor_van_kampen_colimit a K)) :
    VanKampenStruct_colimit.IsColim (pullback_cocone_van_kampen_colimit a K) := by
  have hstable_raw : VanKampenStruct_colimit.PullbackStable (VanKampenStruct_colimit.compare a K.apex) := by
    simpa [comparison_functor_van_kampen_colimit] using hstable_cmp
  have hcolim_raw : VanKampenStruct_colimit.IsColim (VanKampenStruct_colimit.pull a K.apex) :=
    VanKampenStruct_colimit.compare_complete a K.apex hstable_raw
  simpa [pullback_cocone_van_kampen_colimit] using hcolim_raw

theorem descent_effective_van_kampen_colimit
    {Obj : Type u} [VanKampenStruct_colimit Obj]
    (a : VanKampenStruct_colimit.Arrow (Obj := Obj))
    (K : CoconeData_van_kampen_colimit Obj) :
    descent_kernel_van_kampen_colimit a K := by
  have hstable_cmp : VanKampenStruct_colimit.PullbackStable
      (comparison_functor_van_kampen_colimit a K) :=
    comparison_faithful_van_kampen_colimit a K
  have hdescent_cmp : VanKampenStruct_colimit.DescentData
      (comparison_functor_van_kampen_colimit a K) :=
    VanKampenStruct_colimit.descent_intro
      (comparison_functor_van_kampen_colimit a K) hstable_cmp
  simpa [descent_kernel_van_kampen_colimit] using hdescent_cmp

theorem cocone_reflection_van_kampen_colimit
    {Obj : Type u} [VanKampenStruct_colimit Obj]
    (a : VanKampenStruct_colimit.Arrow (Obj := Obj))
    (K : CoconeData_van_kampen_colimit Obj) :
    VanKampenStruct_colimit.Reflective (comparison_functor_van_kampen_colimit a K) := by
  have hdescent : descent_kernel_van_kampen_colimit a K :=
    descent_effective_van_kampen_colimit a K
  have hreflect_raw : VanKampenStruct_colimit.Reflective
      (comparison_functor_van_kampen_colimit a K) :=
    VanKampenStruct_colimit.reflection_intro
      (comparison_functor_van_kampen_colimit a K)
      (by simpa [descent_kernel_van_kampen_colimit] using hdescent)
  exact hreflect_raw

theorem universal_square_van_kampen_colimit
    {Obj : Type u} [VanKampenStruct_colimit Obj]
    (a : VanKampenStruct_colimit.Arrow (Obj := Obj))
    (K : CoconeData_van_kampen_colimit Obj) :
    VanKampenStruct_colimit.IsColim (comparison_functor_van_kampen_colimit a K) := by
  have hreflect : VanKampenStruct_colimit.Reflective
      (comparison_functor_van_kampen_colimit a K) :=
    cocone_reflection_van_kampen_colimit a K
  have hstable_from_reflection : VanKampenStruct_colimit.PullbackStable
      (comparison_functor_van_kampen_colimit a K) :=
    VanKampenStruct_colimit.reflection_elim
      (comparison_functor_van_kampen_colimit a K) hreflect
  have hdescent : descent_kernel_van_kampen_colimit a K :=
    descent_effective_van_kampen_colimit a K
  have hstable_from_descent : VanKampenStruct_colimit.PullbackStable
      (comparison_functor_van_kampen_colimit a K) :=
    VanKampenStruct_colimit.descent_elim
      (comparison_functor_van_kampen_colimit a K)
      (by simpa [descent_kernel_van_kampen_colimit] using hdescent)
  have hcolim_cmp : VanKampenStruct_colimit.IsColim
      (comparison_functor_van_kampen_colimit a K) :=
    VanKampenStruct_colimit.universal_square_rule
      (comparison_functor_van_kampen_colimit a K)
      hreflect
      (by
        have hbridge := hstable_from_descent
        exact hbridge)
  have _ : VanKampenStruct_colimit.PullbackStable
      (comparison_functor_van_kampen_colimit a K) := hstable_from_reflection
  exact hcolim_cmp

theorem van_kampen_characterization_colimit
    {Obj : Type u} [VanKampenStruct_colimit Obj]
    (a : VanKampenStruct_colimit.Arrow (Obj := Obj))
    (K : CoconeData_van_kampen_colimit Obj) :
    VanKampenStruct_colimit.IsColim (pullback_cocone_van_kampen_colimit a K) ↔
      VanKampenStruct_colimit.IsColim (comparison_functor_van_kampen_colimit a K) := by
  constructor
  · intro hpull
    have hstable_cmp : VanKampenStruct_colimit.PullbackStable
        (comparison_functor_van_kampen_colimit a K) :=
      VanKampenStruct_colimit.compare_sound a K.apex (by
        simpa [pullback_cocone_van_kampen_colimit] using hpull)
    have hdescent_cmp : descent_kernel_van_kampen_colimit a K := by
      change VanKampenStruct_colimit.DescentData
        (comparison_functor_van_kampen_colimit a K)
      exact VanKampenStruct_colimit.descent_intro
        (comparison_functor_van_kampen_colimit a K) hstable_cmp
    have hreflect_cmp : VanKampenStruct_colimit.Reflective
        (comparison_functor_van_kampen_colimit a K) :=
      VanKampenStruct_colimit.reflection_intro
        (comparison_functor_van_kampen_colimit a K)
        (by simpa [descent_kernel_van_kampen_colimit] using hdescent_cmp)
    have hstable_again : VanKampenStruct_colimit.PullbackStable
        (comparison_functor_van_kampen_colimit a K) :=
      VanKampenStruct_colimit.reflection_elim
        (comparison_functor_van_kampen_colimit a K) hreflect_cmp
    exact VanKampenStruct_colimit.universal_square_rule
      (comparison_functor_van_kampen_colimit a K)
      hreflect_cmp hstable_again
  · intro hcmp
    have hstable_cmp : VanKampenStruct_colimit.PullbackStable
        (comparison_functor_van_kampen_colimit a K) :=
      VanKampenStruct_colimit.colim_implies_stable
        (comparison_functor_van_kampen_colimit a K) hcmp
    have hpull : VanKampenStruct_colimit.IsColim
        (pullback_cocone_van_kampen_colimit a K) :=
      comparison_full_van_kampen_colimit a K hstable_cmp
    exact hpull
