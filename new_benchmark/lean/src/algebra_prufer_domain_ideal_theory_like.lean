/-
BENCHMARK_ID: TINY_MATHLIB_BATCH06_ALG_PRUFER_DOMAIN_IDEAL_THEORY_LIKE
PAIR_STEM: algebra_prufer_domain_ideal_theory_like
MATH_DOMAIN: Algebra
SOURCE_MATHLIB: Mathlib/Algebra/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u v

class PruferStruct_domain_ideal_theory (R : Type u) where
  Ideal : Type v
  rmul : R → R → R
  le : Ideal → Ideal → Prop
  top : Ideal
  mul : Ideal → Ideal → Ideal
  inter : Ideal → Ideal → Ideal
  inv : Ideal → Ideal
  hull : Ideal → Ideal
  localization : Ideal → Ideal
  content : R → Ideal
  principal : Ideal → Prop
  le_refl : ∀ I : Ideal, le I I
  le_trans : ∀ I J K : Ideal, le I J → le J K → le I K
  le_antisymm : ∀ I J : Ideal, le I J → le J I → I = J
  mul_assoc : ∀ I J K : Ideal, mul (mul I J) K = mul I (mul J K)
  mul_comm : ∀ I J : Ideal, mul I J = mul J I
  mul_top : ∀ I : Ideal, mul I top = I
  mul_mono_left : ∀ I J K : Ideal, le I J → le (mul K I) (mul K J)
  inter_left : ∀ I J : Ideal, le (inter I J) I
  inter_right : ∀ I J : Ideal, le (inter I J) J
  inter_glb : ∀ K I J : Ideal, le K I → le K J → le K (inter I J)
  hull_extensive : ∀ I : Ideal, le I (hull I)
  hull_idem : ∀ I : Ideal, hull (hull I) = hull I
  hull_mono : ∀ I J : Ideal, le I J → le (hull I) (hull J)
  localization_mono : ∀ I J : Ideal, le I J → le (localization I) (localization J)
  localization_hull : ∀ I : Ideal, le (hull I) (localization I)
  localization_principal : ∀ I : Ideal, principal (localization I)
  inv_mul_top : ∀ I : Ideal, le top (mul I (inv I))
  mul_inv_top : ∀ I : Ideal, le (mul I (inv I)) top
  content_mul_le : ∀ x y : R, le (content (rmul x y)) (mul (content x) (content y))
  content_mul_ge : ∀ x y : R, le (mul (content x) (content y)) (content (rmul x y))
  divisor_closed : ∀ I J : Ideal, le I J → le J (hull I)

structure IdealData_prufer_domain_ideal_theory (R : Type u)
    [P : PruferStruct_domain_ideal_theory R] where
  base_ideal : P.Ideal
  aux_ideal : P.Ideal
  coeff_left : R
  coeff_right : R

def invertible_hull_prufer_domain_ideal_theory {R : Type u}
    [P : PruferStruct_domain_ideal_theory R]
    (d : IdealData_prufer_domain_ideal_theory (R := R)) : P.Ideal :=
  P.hull d.base_ideal

def localization_ideal_prufer_domain_ideal_theory {R : Type u}
    [P : PruferStruct_domain_ideal_theory R]
    (d : IdealData_prufer_domain_ideal_theory (R := R)) : P.Ideal :=
  P.localization d.base_ideal

def content_ideal_prufer_domain_ideal_theory {R : Type u}
    [P : PruferStruct_domain_ideal_theory R]
    (d : IdealData_prufer_domain_ideal_theory (R := R)) : P.Ideal :=
  P.content d.coeff_left

theorem localization_principal_step_prufer_domain_ideal_theory {R : Type u}
    [P : PruferStruct_domain_ideal_theory R]
    (d : IdealData_prufer_domain_ideal_theory (R := R)) :
    P.principal (localization_ideal_prufer_domain_ideal_theory d) ∧
      P.le (invertible_hull_prufer_domain_ideal_theory d)
        (localization_ideal_prufer_domain_ideal_theory d) := by
  have h_principal : P.principal (P.localization d.base_ideal) :=
    P.localization_principal d.base_ideal
  have h_hull_to_loc : P.le (P.hull d.base_ideal) (P.localization d.base_ideal) :=
    P.localization_hull d.base_ideal
  have h_principal' : P.principal (localization_ideal_prufer_domain_ideal_theory d) := by
    simpa [localization_ideal_prufer_domain_ideal_theory] using h_principal
  have h_hull_to_loc' : P.le
      (invertible_hull_prufer_domain_ideal_theory d)
      (localization_ideal_prufer_domain_ideal_theory d) := by
    simpa [invertible_hull_prufer_domain_ideal_theory, localization_ideal_prufer_domain_ideal_theory]
      using h_hull_to_loc
  exact And.intro h_principal' h_hull_to_loc'

theorem invertible_times_inverse_prufer_domain_ideal_theory {R : Type u}
    [P : PruferStruct_domain_ideal_theory R]
    (d : IdealData_prufer_domain_ideal_theory (R := R)) :
    P.mul (invertible_hull_prufer_domain_ideal_theory d)
      (P.inv (invertible_hull_prufer_domain_ideal_theory d)) = P.top := by
  let I : P.Ideal := invertible_hull_prufer_domain_ideal_theory d
  have h_left : P.le P.top (P.mul I (P.inv I)) := P.inv_mul_top I
  have h_right : P.le (P.mul I (P.inv I)) P.top := P.mul_inv_top I
  have h_top_stable : P.mul P.top P.top = P.top := P.mul_top P.top
  have h_top_rewrite : P.top = P.mul P.top P.top := by
    symm
    exact h_top_stable
  have h_reflexive : P.le P.top P.top := by
    exact P.le_refl P.top
  have h_eq : P.mul I (P.inv I) = P.top :=
    P.le_antisymm (P.mul I (P.inv I)) P.top h_right h_left
  exact h_eq

theorem finite_intersection_stable_prufer_domain_ideal_theory {R : Type u}
    [P : PruferStruct_domain_ideal_theory R]
    (d : IdealData_prufer_domain_ideal_theory (R := R)) :
    P.le (P.inter (invertible_hull_prufer_domain_ideal_theory d)
      (content_ideal_prufer_domain_ideal_theory d))
      (invertible_hull_prufer_domain_ideal_theory d) ∧
    P.le (P.inter (invertible_hull_prufer_domain_ideal_theory d)
      (content_ideal_prufer_domain_ideal_theory d))
      (content_ideal_prufer_domain_ideal_theory d) ∧
    P.le (P.inter (invertible_hull_prufer_domain_ideal_theory d)
      (content_ideal_prufer_domain_ideal_theory d))
      (P.hull (invertible_hull_prufer_domain_ideal_theory d)) := by
  have h_left : P.le
      (P.inter (invertible_hull_prufer_domain_ideal_theory d)
        (content_ideal_prufer_domain_ideal_theory d))
      (invertible_hull_prufer_domain_ideal_theory d) :=
    P.inter_left (invertible_hull_prufer_domain_ideal_theory d)
      (content_ideal_prufer_domain_ideal_theory d)
  have h_right : P.le
      (P.inter (invertible_hull_prufer_domain_ideal_theory d)
        (content_ideal_prufer_domain_ideal_theory d))
      (content_ideal_prufer_domain_ideal_theory d) :=
    P.inter_right (invertible_hull_prufer_domain_ideal_theory d)
      (content_ideal_prufer_domain_ideal_theory d)
  have h_hull : P.le
      (invertible_hull_prufer_domain_ideal_theory d)
      (P.hull (invertible_hull_prufer_domain_ideal_theory d)) :=
    P.hull_extensive (invertible_hull_prufer_domain_ideal_theory d)
  have h_left_to_hull : P.le
      (P.inter (invertible_hull_prufer_domain_ideal_theory d)
        (content_ideal_prufer_domain_ideal_theory d))
      (P.hull (invertible_hull_prufer_domain_ideal_theory d)) :=
    P.le_trans _ _ _ h_left h_hull
  exact And.intro h_left (And.intro h_right h_left_to_hull)

theorem content_multiplicative_step_prufer_domain_ideal_theory {R : Type u}
    [P : PruferStruct_domain_ideal_theory R]
    (d : IdealData_prufer_domain_ideal_theory (R := R)) :
    P.content (P.rmul d.coeff_left d.coeff_right) =
      P.mul (content_ideal_prufer_domain_ideal_theory d) (P.content d.coeff_right) := by
  have h_le_raw :
      P.le (P.content (P.rmul d.coeff_left d.coeff_right))
        (P.mul (P.content d.coeff_left) (P.content d.coeff_right)) :=
    P.content_mul_le d.coeff_left d.coeff_right
  have h_ge_raw :
      P.le (P.mul (P.content d.coeff_left) (P.content d.coeff_right))
        (P.content (P.rmul d.coeff_left d.coeff_right)) :=
    P.content_mul_ge d.coeff_left d.coeff_right
  have h_le :
      P.le (P.content (P.rmul d.coeff_left d.coeff_right))
        (P.mul (content_ideal_prufer_domain_ideal_theory d) (P.content d.coeff_right)) := by
    simpa [content_ideal_prufer_domain_ideal_theory] using h_le_raw
  have h_ge :
      P.le (P.mul (content_ideal_prufer_domain_ideal_theory d) (P.content d.coeff_right))
        (P.content (P.rmul d.coeff_left d.coeff_right)) := by
    simpa [content_ideal_prufer_domain_ideal_theory] using h_ge_raw
  exact P.le_antisymm
    (P.content (P.rmul d.coeff_left d.coeff_right))
    (P.mul (content_ideal_prufer_domain_ideal_theory d) (P.content d.coeff_right))
    h_le h_ge

theorem divisor_closure_step_prufer_domain_ideal_theory {R : Type u}
    [P : PruferStruct_domain_ideal_theory R]
    (d : IdealData_prufer_domain_ideal_theory (R := R)) :
    P.le (localization_ideal_prufer_domain_ideal_theory d)
      (invertible_hull_prufer_domain_ideal_theory d) := by
  have h_local_step := localization_principal_step_prufer_domain_ideal_theory d
  have h_hull_to_loc :
      P.le (invertible_hull_prufer_domain_ideal_theory d)
        (localization_ideal_prufer_domain_ideal_theory d) :=
    h_local_step.2
  have h_divisor_raw :
      P.le (localization_ideal_prufer_domain_ideal_theory d)
        (P.hull (invertible_hull_prufer_domain_ideal_theory d)) :=
    P.divisor_closed
      (invertible_hull_prufer_domain_ideal_theory d)
      (localization_ideal_prufer_domain_ideal_theory d)
      h_hull_to_loc
  have h_hull_idem :
      P.hull (invertible_hull_prufer_domain_ideal_theory d) =
        invertible_hull_prufer_domain_ideal_theory d := by
    change P.hull (P.hull d.base_ideal) = P.hull d.base_ideal
    exact P.hull_idem d.base_ideal
  have h_divisor :
      P.le (localization_ideal_prufer_domain_ideal_theory d)
        (invertible_hull_prufer_domain_ideal_theory d) := by
    rw [← h_hull_idem]
    exact h_divisor_raw
  exact h_divisor

theorem invertible_factorization_prufer_domain_ideal_theory {R : Type u}
    [P : PruferStruct_domain_ideal_theory R]
    (d : IdealData_prufer_domain_ideal_theory (R := R)) :
    P.le (P.mul (localization_ideal_prufer_domain_ideal_theory d)
      (P.inv (invertible_hull_prufer_domain_ideal_theory d))) P.top := by
  let I : P.Ideal := invertible_hull_prufer_domain_ideal_theory d
  have h_div : P.le (localization_ideal_prufer_domain_ideal_theory d) I :=
    divisor_closure_step_prufer_domain_ideal_theory d
  have h_mono_raw :
      P.le (P.mul (P.inv I) (localization_ideal_prufer_domain_ideal_theory d))
        (P.mul (P.inv I) I) :=
    P.mul_mono_left
      (localization_ideal_prufer_domain_ideal_theory d) I (P.inv I) h_div
  have h_mono :
      P.le (P.mul (localization_ideal_prufer_domain_ideal_theory d) (P.inv I))
        (P.mul I (P.inv I)) := by
    simpa [P.mul_comm] using h_mono_raw
  have h_top : P.le (P.mul I (P.inv I)) P.top := P.mul_inv_top I
  have h_chain :
      P.le (P.mul (localization_ideal_prufer_domain_ideal_theory d) (P.inv I)) P.top :=
    P.le_trans _ _ _ h_mono h_top
  simpa [I] using h_chain

theorem ideal_classification_final_prufer_domain_ideal_theory {R : Type u}
    [P : PruferStruct_domain_ideal_theory R]
    (d : IdealData_prufer_domain_ideal_theory (R := R)) :
    P.principal (localization_ideal_prufer_domain_ideal_theory d) ∧
    localization_ideal_prufer_domain_ideal_theory d =
      invertible_hull_prufer_domain_ideal_theory d ∧
    P.le (P.mul (localization_ideal_prufer_domain_ideal_theory d)
      (P.inv (invertible_hull_prufer_domain_ideal_theory d))) P.top ∧
    P.mul (invertible_hull_prufer_domain_ideal_theory d)
      (P.inv (invertible_hull_prufer_domain_ideal_theory d)) = P.top := by
  have h_local := localization_principal_step_prufer_domain_ideal_theory d
  have h_principal : P.principal (localization_ideal_prufer_domain_ideal_theory d) := h_local.1
  have h_backward :
      P.le (invertible_hull_prufer_domain_ideal_theory d)
        (localization_ideal_prufer_domain_ideal_theory d) := h_local.2
  have h_forward :
      P.le (localization_ideal_prufer_domain_ideal_theory d)
        (invertible_hull_prufer_domain_ideal_theory d) :=
    divisor_closure_step_prufer_domain_ideal_theory d
  have h_eq :
      localization_ideal_prufer_domain_ideal_theory d =
        invertible_hull_prufer_domain_ideal_theory d :=
    P.le_antisymm
      (localization_ideal_prufer_domain_ideal_theory d)
      (invertible_hull_prufer_domain_ideal_theory d)
      h_forward h_backward
  have h_factor :
      P.le (P.mul (localization_ideal_prufer_domain_ideal_theory d)
        (P.inv (invertible_hull_prufer_domain_ideal_theory d))) P.top :=
    invertible_factorization_prufer_domain_ideal_theory d
  have h_unit :
      P.mul (invertible_hull_prufer_domain_ideal_theory d)
        (P.inv (invertible_hull_prufer_domain_ideal_theory d)) = P.top :=
    invertible_times_inverse_prufer_domain_ideal_theory d
  exact And.intro h_principal (And.intro h_eq (And.intro h_factor h_unit))
