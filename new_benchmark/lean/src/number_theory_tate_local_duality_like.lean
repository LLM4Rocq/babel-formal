/-
BENCHMARK_ID: TINY_MATHLIB_BATCH06_NUMBER_THEORY_TATE_LOCAL_DUALITY_LIKE
PAIR_STEM: number_theory_tate_local_duality_like
MATH_DOMAIN: Number Theory
SOURCE_MATHLIB: Mathlib/NumberTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class LocalDualityStruct_tate (K : Type u) where
  H1 : Type u
  T : Type u
  zeroH1 : H1
  addH1 : H1 → H1 → H1
  negH1 : H1 → H1
  zeroT : T
  addT : T → T → T
  negT : T → T
  cup : H1 → H1 → T
  character : H1 → T
  annihilator : H1 → Prop
  addH1_assoc : ∀ x y z : H1, addH1 (addH1 x y) z = addH1 x (addH1 y z)
  addH1_zero : ∀ x : H1, addH1 x zeroH1 = x
  addH1_zero_left : ∀ x : H1, addH1 zeroH1 x = x
  addH1_neg : ∀ x : H1, addH1 x (negH1 x) = zeroH1
  addT_assoc : ∀ a b c : T, addT (addT a b) c = addT a (addT b c)
  addT_zero : ∀ a : T, addT a zeroT = a
  zeroT_add : ∀ a : T, addT zeroT a = a
  addT_neg : ∀ a : T, addT a (negT a) = zeroT
  cup_add_left : ∀ x y z : H1, cup (addH1 x y) z = addT (cup x z) (cup y z)
  cup_add_right : ∀ x y z : H1, cup x (addH1 y z) = addT (cup x y) (cup x z)
  cup_neg_left : ∀ x z : H1, cup (negH1 x) z = negT (cup x z)
  cup_zero_left : ∀ z : H1, cup zeroH1 z = zeroT
  character_add : ∀ x y : H1, character (addH1 x y) = addT (character x) (character y)
  character_zero : character zeroH1 = zeroT
  local_exact : ∀ x : H1, annihilator x ↔ character x = zeroT
  annihilator_closed_add : ∀ x y : H1, annihilator x → annihilator y → annihilator (addH1 x y)
  annihilator_closed_neg : ∀ x : H1, annihilator x → annihilator (negH1 x)
  finite_level : ∀ x : H1, annihilator x → ∃ y : H1, cup x y = zeroT ∧ cup y x = zeroT
  dual_step : ∀ x : H1, (∀ y : H1, cup x y = zeroT) → annihilator x
  limit_step :
    ∀ P : H1 → Prop,
      (∃ x : H1, P x) →
      (∀ x : H1, P x → annihilator x) →
      ∃ x : H1, P x ∧ annihilator x
  nondegenerate :
    ∀ x : H1,
      annihilator x →
      (∀ y : H1, cup y x = zeroT) →
      x = zeroH1

structure CohomologyData_tate_local_duality
    (K : Type u) [s : LocalDualityStruct_tate K] where
  left_class : s.H1
  right_class : s.H1
  left_ann : s.annihilator left_class
  right_ann : s.annihilator right_class

def cup_pairing_tate_local_duality
    {K : Type u} [s : LocalDualityStruct_tate K]
    (x y : s.H1) : s.T :=
  s.cup x y

def local_character_tate_local_duality
    {K : Type u} [s : LocalDualityStruct_tate K]
    (x : s.H1) : s.T :=
  s.character x

def annihilator_subgroup_tate_local_duality
    {K : Type u} [s : LocalDualityStruct_tate K]
    (x : s.H1) : Prop :=
  s.annihilator x

theorem cup_product_bilinear_tate_local_duality
    {K : Type u} [s : LocalDualityStruct_tate K]
    (x y z : s.H1) :
    cup_pairing_tate_local_duality (s.addH1 x y) z =
      s.addT (cup_pairing_tate_local_duality x z) (cup_pairing_tate_local_duality y z) ∧
    cup_pairing_tate_local_duality x (s.addH1 y z) =
      s.addT (cup_pairing_tate_local_duality x y) (cup_pairing_tate_local_duality x z) ∧
    cup_pairing_tate_local_duality (s.negH1 x) z =
      s.negT (cup_pairing_tate_local_duality x z) := by
  have hLeft :
      cup_pairing_tate_local_duality (s.addH1 x y) z =
        s.addT (cup_pairing_tate_local_duality x z) (cup_pairing_tate_local_duality y z) :=
    s.cup_add_left x y z
  have hRight :
      cup_pairing_tate_local_duality x (s.addH1 y z) =
        s.addT (cup_pairing_tate_local_duality x y) (cup_pairing_tate_local_duality x z) :=
    s.cup_add_right x y z
  have hNeg :
      cup_pairing_tate_local_duality (s.negH1 x) z =
        s.negT (cup_pairing_tate_local_duality x z) :=
    s.cup_neg_left x z
  constructor
  · exact hLeft
  · constructor
    · exact hRight
    · exact hNeg

theorem local_invariant_exactness_tate_local_duality
    {K : Type u} [s : LocalDualityStruct_tate K]
    (x y : s.H1)
    (hx : annihilator_subgroup_tate_local_duality x)
    (hy : annihilator_subgroup_tate_local_duality y) :
    local_character_tate_local_duality (s.addH1 x (s.negH1 y)) = s.zeroT ∧
    annihilator_subgroup_tate_local_duality (s.addH1 x (s.negH1 y)) := by
  have hxZero : local_character_tate_local_duality x = s.zeroT :=
    (s.local_exact x).1 hx
  have hyNegAnn : annihilator_subgroup_tate_local_duality (s.negH1 y) :=
    s.annihilator_closed_neg y hy
  have hyNegZero : local_character_tate_local_duality (s.negH1 y) = s.zeroT :=
    (s.local_exact (s.negH1 y)).1 hyNegAnn
  have hAddChar :
      local_character_tate_local_duality (s.addH1 x (s.negH1 y)) =
        s.addT (local_character_tate_local_duality x) (local_character_tate_local_duality (s.negH1 y)) :=
    s.character_add x (s.negH1 y)
  have hCharZero : local_character_tate_local_duality (s.addH1 x (s.negH1 y)) = s.zeroT := by
    calc
      local_character_tate_local_duality (s.addH1 x (s.negH1 y))
          = s.addT (local_character_tate_local_duality x) (local_character_tate_local_duality (s.negH1 y)) :=
        hAddChar
      _ = s.addT s.zeroT (local_character_tate_local_duality (s.negH1 y)) := by
        rw [hxZero]
      _ = local_character_tate_local_duality (s.negH1 y) :=
        s.zeroT_add (local_character_tate_local_duality (s.negH1 y))
      _ = s.zeroT :=
        hyNegZero
  have hAnn : annihilator_subgroup_tate_local_duality (s.addH1 x (s.negH1 y)) :=
    (s.local_exact (s.addH1 x (s.negH1 y))).2 hCharZero
  exact ⟨hCharZero, hAnn⟩

theorem pontryagin_dual_step_tate_local_duality
    {K : Type u} [s : LocalDualityStruct_tate K]
    (x : s.H1)
    (hker : ∀ y : s.H1, cup_pairing_tate_local_duality x y = s.zeroT) :
    annihilator_subgroup_tate_local_duality x ∧
    cup_pairing_tate_local_duality (s.negH1 x) s.zeroH1 = s.zeroT := by
  have hAnn : annihilator_subgroup_tate_local_duality x :=
    s.dual_step x hker
  have hCupZero : cup_pairing_tate_local_duality x s.zeroH1 = s.zeroT :=
    hker s.zeroH1
  have hNegCup :
      cup_pairing_tate_local_duality (s.negH1 x) s.zeroH1 =
        s.negT (cup_pairing_tate_local_duality x s.zeroH1) :=
    s.cup_neg_left x s.zeroH1
  have hNegZero : s.negT s.zeroT = s.zeroT := by
    have hZeroLeft : s.addT s.zeroT (s.negT s.zeroT) = s.negT s.zeroT :=
      s.zeroT_add (s.negT s.zeroT)
    have hAddNeg : s.addT s.zeroT (s.negT s.zeroT) = s.zeroT :=
      s.addT_neg s.zeroT
    calc
      s.negT s.zeroT = s.addT s.zeroT (s.negT s.zeroT) := by
        symm
        exact hZeroLeft
      _ = s.zeroT :=
        hAddNeg
  have hFinal : cup_pairing_tate_local_duality (s.negH1 x) s.zeroH1 = s.zeroT := by
    calc
      cup_pairing_tate_local_duality (s.negH1 x) s.zeroH1
          = s.negT (cup_pairing_tate_local_duality x s.zeroH1) :=
        hNegCup
      _ = s.negT s.zeroT := by
        rw [hCupZero]
      _ = s.zeroT :=
        hNegZero
  exact ⟨hAnn, hFinal⟩

theorem orthogonality_criterion_tate_local_duality
    {K : Type u} [s : LocalDualityStruct_tate K]
    (d : CohomologyData_tate_local_duality K) :
    annihilator_subgroup_tate_local_duality (s.addH1 d.left_class (s.negH1 d.right_class)) ∧
    local_character_tate_local_duality (s.addH1 d.left_class (s.negH1 d.right_class)) = s.zeroT := by
  have hExact :=
    local_invariant_exactness_tate_local_duality
      (x := d.left_class)
      (y := d.right_class)
      d.left_ann
      d.right_ann
  rcases hExact with ⟨hChar, hAnn⟩
  exact ⟨hAnn, hChar⟩

theorem finite_level_perfectness_tate_local_duality
    {K : Type u} [s : LocalDualityStruct_tate K]
    (x : s.H1)
    (hx : annihilator_subgroup_tate_local_duality x) :
    ∃ y : s.H1,
      cup_pairing_tate_local_duality x y = s.zeroT ∧
      cup_pairing_tate_local_duality y x = s.zeroT ∧
      (annihilator_subgroup_tate_local_duality y → local_character_tate_local_duality y = s.zeroT) := by
  rcases s.finite_level x hx with ⟨y, hxy, hyx⟩
  have hImp :
      annihilator_subgroup_tate_local_duality y →
      local_character_tate_local_duality y = s.zeroT := by
    intro hy
    exact (s.local_exact y).1 hy
  refine ⟨y, hxy, hyx, ?_⟩
  exact hImp

theorem passage_to_limit_tate_local_duality
    {K : Type u} [s : LocalDualityStruct_tate K]
    (P : s.H1 → Prop)
    (hnonempty : ∃ x : s.H1, P x)
    (hstable : ∀ x : s.H1, P x → annihilator_subgroup_tate_local_duality x) :
    ∃ x : s.H1,
      P x ∧
      local_character_tate_local_duality x = s.zeroT ∧
      ∃ y : s.H1, cup_pairing_tate_local_duality x y = s.zeroT := by
  rcases s.limit_step P hnonempty hstable with ⟨x, hxP, hxAnn⟩
  have hxChar : local_character_tate_local_duality x = s.zeroT :=
    (s.local_exact x).1 hxAnn
  rcases finite_level_perfectness_tate_local_duality (K := K) x hxAnn with
      ⟨y, hxy, hyx, hyCharFromAnn⟩
  have hyIfAnn :
      annihilator_subgroup_tate_local_duality y →
      local_character_tate_local_duality y = s.zeroT :=
    hyCharFromAnn
  have hSymmetricCup : cup_pairing_tate_local_duality y x = s.zeroT :=
    hyx
  have hAnnAgain : annihilator_subgroup_tate_local_duality x :=
    hxAnn
  have _ :
      (annihilator_subgroup_tate_local_duality y → local_character_tate_local_duality y = s.zeroT) :=
    hyIfAnn
  have _ : cup_pairing_tate_local_duality y x = s.zeroT :=
    hSymmetricCup
  have _ : annihilator_subgroup_tate_local_duality x :=
    hAnnAgain
  refine ⟨x, hxP, hxChar, ?_⟩
  exact ⟨y, hxy⟩

theorem perfect_pairing_final_tate_local_duality
    {K : Type u} [s : LocalDualityStruct_tate K]
    (P : s.H1 → Prop)
    (hnonempty : ∃ x : s.H1, P x)
    (hstable : ∀ x : s.H1, P x → annihilator_subgroup_tate_local_duality x)
    (horth : ∀ x : s.H1, P x → ∀ y : s.H1, cup_pairing_tate_local_duality y x = s.zeroT) :
    ∃ x : s.H1,
      P x ∧
      x = s.zeroH1 ∧
      local_character_tate_local_duality x = s.zeroT := by
  rcases passage_to_limit_tate_local_duality (K := K) P hnonempty hstable with
      ⟨x, hxP, hxChar, hExistsPartner⟩
  have hxAnn : annihilator_subgroup_tate_local_duality x :=
    (s.local_exact x).2 hxChar
  have hAllRight : ∀ y : s.H1, cup_pairing_tate_local_duality y x = s.zeroT :=
    horth x hxP
  have hZero : x = s.zeroH1 :=
    s.nondegenerate x hxAnn hAllRight
  rcases hExistsPartner with ⟨yWitness, hyWitness⟩
  have hWitnessed : cup_pairing_tate_local_duality x yWitness = s.zeroT :=
    hyWitness
  have _ : cup_pairing_tate_local_duality x yWitness = s.zeroT :=
    hWitnessed
  refine ⟨x, hxP, hZero, hxChar⟩
