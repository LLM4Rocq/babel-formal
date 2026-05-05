/-
BENCHMARK_ID: TINY_MATHLIB_BATCH03_LINEAR_TENSOR_HOM_CURRYING_LIKE
PAIR_STEM: linear_tensor_hom_currying_like
MATH_DOMAIN: Linear Algebra
SOURCE_MATHLIB: Mathlib/LinearAlgebra/TensorProduct
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u v w z

class RingLike (R : Type u) where
  zero : R
  add : R → R → R
  neg : R → R
  one : R
  mul : R → R → R
  add_assoc : ∀ a b c : R, add (add a b) c = add a (add b c)
  add_comm : ∀ a b : R, add a b = add b a
  add_zero : ∀ a : R, add a zero = a
  zero_add : ∀ a : R, add zero a = a
  add_left_neg : ∀ a : R, add (neg a) a = zero
  mul_assoc : ∀ a b c : R, mul (mul a b) c = mul a (mul b c)
  one_mul : ∀ a : R, mul one a = a
  mul_one : ∀ a : R, mul a one = a
  left_distrib : ∀ a b c : R, mul a (add b c) = add (mul a b) (mul a c)
  right_distrib : ∀ a b c : R, mul (add a b) c = add (mul a c) (mul b c)

class ModuleLike (R : Type u) (M : Type v) [RingLike R] where
  zero : M
  add : M → M → M
  neg : M → M
  smul : R → M → M
  add_assoc : ∀ x y z : M, add (add x y) z = add x (add y z)
  add_comm : ∀ x y : M, add x y = add y x
  add_zero : ∀ x : M, add x zero = x
  zero_add : ∀ x : M, add zero x = x
  add_left_neg : ∀ x : M, add (neg x) x = zero
  smul_add : ∀ a : R, ∀ x y : M, smul a (add x y) = add (smul a x) (smul a y)
  add_smul : ∀ a b : R, ∀ x : M,
      smul (RingLike.add a b) x = add (smul a x) (smul b x)
  one_smul : ∀ x : M, smul (RingLike.one (R := R)) x = x
  mul_smul : ∀ a b : R, ∀ x : M,
      smul (RingLike.mul a b) x = smul a (smul b x)
  zero_smul : ∀ x : M, smul (RingLike.zero (R := R)) x = zero

def BilinearLike {R : Type u} {M : Type v} {N : Type w} {P : Type z}
    [RingLike R] [ModuleLike R M] [ModuleLike R N] [ModuleLike R P]
    (b : M → N → P) : Prop :=
  (∀ x₁ x₂ : M, ∀ y : N,
      b (ModuleLike.add (R := R) (M := M) x₁ x₂) y
        = ModuleLike.add (R := R) (M := P) (b x₁ y) (b x₂ y)) ∧
  (∀ a : R, ∀ x : M, ∀ y : N,
      b (ModuleLike.smul (R := R) (M := M) a x) y
        = ModuleLike.smul (R := R) (M := P) a (b x y)) ∧
  (∀ x : M, ∀ y₁ y₂ : N,
      b x (ModuleLike.add (R := R) (M := N) y₁ y₂)
        = ModuleLike.add (R := R) (M := P) (b x y₁) (b x y₂)) ∧
  (∀ a : R, ∀ x : M, ∀ y : N,
      b x (ModuleLike.smul (R := R) (M := N) a y)
        = ModuleLike.smul (R := R) (M := P) a (b x y))

def TensorLike (R : Type u) (M : Type v) (N : Type w)
    [RingLike R] [ModuleLike R M] [ModuleLike R N] : Type (max v w) :=
  M × N

def TensorLiftLike {R : Type u} {M : Type v} {N : Type w} {P : Type z}
    [RingLike R] [ModuleLike R M] [ModuleLike R N] [ModuleLike R P]
    (b : M → N → P) (t : TensorLike R M N) : P :=
  b t.1 t.2

def CurryLike {R : Type u} {M : Type v} {N : Type w} {P : Type z}
    [RingLike R] [ModuleLike R M] [ModuleLike R N] [ModuleLike R P]
    (h : TensorLike R M N → P) : M → N → P :=
  fun x y => h (x, y)

theorem curry_linear {R : Type u} {M : Type v} {N : Type w} {P : Type z}
    [RingLike R] [ModuleLike R M] [ModuleLike R N] [ModuleLike R P]
    (h : TensorLike R M N → P)
    (hAdd : ∀ x₁ x₂ : M, ∀ y : N,
      h (ModuleLike.add (R := R) (M := M) x₁ x₂, y)
        = ModuleLike.add (R := R) (M := P) (h (x₁, y)) (h (x₂, y)))
    (hSmul : ∀ a : R, ∀ x : M, ∀ y : N,
      h (ModuleLike.smul (R := R) (M := M) a x, y)
        = ModuleLike.smul (R := R) (M := P) a (h (x, y))) :
    (∀ x₁ x₂ : M, ∀ y : N,
      CurryLike (R := R) (M := M) (N := N) (P := P) h (ModuleLike.add (R := R) (M := M) x₁ x₂) y
        = ModuleLike.add (R := R) (M := P) (CurryLike (R := R) (M := M) (N := N) (P := P) h x₁ y) (CurryLike (R := R) (M := M) (N := N) (P := P) h x₂ y)) ∧
    (∀ a : R, ∀ x : M, ∀ y : N,
      CurryLike (R := R) (M := M) (N := N) (P := P) h (ModuleLike.smul (R := R) (M := M) a x) y
        = ModuleLike.smul (R := R) (M := P) a (CurryLike (R := R) (M := M) (N := N) (P := P) h x y)) := by
  constructor
  · intro x₁ x₂ y
    have hStep : h (ModuleLike.add (R := R) (M := M) x₁ x₂, y)
        = ModuleLike.add (R := R) (M := P) (h (x₁, y)) (h (x₂, y)) :=
      hAdd x₁ x₂ y
    have hDefL :
        CurryLike (R := R) (M := M) (N := N) (P := P) h
          (ModuleLike.add (R := R) (M := M) x₁ x₂) y
          = h (ModuleLike.add (R := R) (M := M) x₁ x₂, y) := by
      rfl
    have hDefR1 :
        CurryLike (R := R) (M := M) (N := N) (P := P) h x₁ y = h (x₁, y) := by
      rfl
    have hDefR2 :
        CurryLike (R := R) (M := M) (N := N) (P := P) h x₂ y = h (x₂, y) := by
      rfl
    have hPost :
        ModuleLike.add (R := R) (M := P) (h (x₁, y)) (h (x₂, y))
          = ModuleLike.add (R := R) (M := P)
              (CurryLike (R := R) (M := M) (N := N) (P := P) h x₁ y)
              (CurryLike (R := R) (M := M) (N := N) (P := P) h x₂ y) := by
      rw [hDefR1, hDefR2]
    calc
      CurryLike (R := R) (M := M) (N := N) (P := P) h (ModuleLike.add (R := R) (M := M) x₁ x₂) y
          = h (ModuleLike.add (R := R) (M := M) x₁ x₂, y) := hDefL
      _ = ModuleLike.add (R := R) (M := P) (h (x₁, y)) (h (x₂, y)) := hStep
      _ = ModuleLike.add (R := R) (M := P)
            (CurryLike (R := R) (M := M) (N := N) (P := P) h x₁ y)
            (CurryLike (R := R) (M := M) (N := N) (P := P) h x₂ y) := hPost
  · intro a x y
    have hStep : h (ModuleLike.smul (R := R) (M := M) a x, y)
        = ModuleLike.smul (R := R) (M := P) a (h (x, y)) :=
      hSmul a x y
    have hDefL :
        CurryLike (R := R) (M := M) (N := N) (P := P) h
          (ModuleLike.smul (R := R) (M := M) a x) y
          = h (ModuleLike.smul (R := R) (M := M) a x, y) := by
      rfl
    have hDefR :
        CurryLike (R := R) (M := M) (N := N) (P := P) h x y = h (x, y) := by
      rfl
    have hPost :
        ModuleLike.smul (R := R) (M := P) a (h (x, y))
          = ModuleLike.smul (R := R) (M := P) a
              (CurryLike (R := R) (M := M) (N := N) (P := P) h x y) := by
      rw [hDefR]
    calc
      CurryLike (R := R) (M := M) (N := N) (P := P) h (ModuleLike.smul (R := R) (M := M) a x) y
          = h (ModuleLike.smul (R := R) (M := M) a x, y) := hDefL
      _ = ModuleLike.smul (R := R) (M := P) a (h (x, y)) := hStep
      _ = ModuleLike.smul (R := R) (M := P) a
            (CurryLike (R := R) (M := M) (N := N) (P := P) h x y) := hPost

theorem uncurry_linear {R : Type u} {M : Type v} {N : Type w} {P : Type z}
    [RingLike R] [ModuleLike R M] [ModuleLike R N] [ModuleLike R P]
    (b : M → N → P) (hb : BilinearLike (R := R) (M := M) (N := N) (P := P) b) :
    (∀ x₁ x₂ : M, ∀ y : N,
      (TensorLiftLike (R := R) (M := M) (N := N) (P := P) b (ModuleLike.add (R := R) (M := M) x₁ x₂, y) : P)
        = ModuleLike.add (R := R) (M := P) (TensorLiftLike (R := R) (M := M) (N := N) (P := P) b (x₁, y) : P) (TensorLiftLike (R := R) (M := M) (N := N) (P := P) b (x₂, y) : P)) ∧
    (∀ a : R, ∀ x : M, ∀ y : N,
      (TensorLiftLike (R := R) (M := M) (N := N) (P := P) b (ModuleLike.smul (R := R) (M := M) a x, y) : P)
        = ModuleLike.smul (R := R) (M := P) a (TensorLiftLike (R := R) (M := M) (N := N) (P := P) b (x, y) : P)) ∧
    (∀ x : M, ∀ y₁ y₂ : N,
      (TensorLiftLike (R := R) (M := M) (N := N) (P := P) b (x, ModuleLike.add (R := R) (M := N) y₁ y₂) : P)
        = ModuleLike.add (R := R) (M := P) (TensorLiftLike (R := R) (M := M) (N := N) (P := P) b (x, y₁) : P) (TensorLiftLike (R := R) (M := M) (N := N) (P := P) b (x, y₂) : P)) ∧
    (∀ a : R, ∀ x : M, ∀ y : N,
      (TensorLiftLike (R := R) (M := M) (N := N) (P := P) b (x, ModuleLike.smul (R := R) (M := N) a y) : P)
        = ModuleLike.smul (R := R) (M := P) a (TensorLiftLike (R := R) (M := M) (N := N) (P := P) b (x, y) : P)) := by
  rcases hb with ⟨hAddL, hSmulL, hAddR, hSmulR⟩
  constructor
  · intro x₁ x₂ y
    have hStep : b (ModuleLike.add (R := R) (M := M) x₁ x₂) y
        = ModuleLike.add (R := R) (M := P) (b x₁ y) (b x₂ y) :=
      hAddL x₁ x₂ y
    have hDefL :
        TensorLiftLike (R := R) (M := M) (N := N) (P := P) b
          (ModuleLike.add (R := R) (M := M) x₁ x₂, y)
          = b (ModuleLike.add (R := R) (M := M) x₁ x₂) y := by
      rfl
    have hDefR1 :
        TensorLiftLike (R := R) (M := M) (N := N) (P := P) b (x₁, y) = b x₁ y := by
      rfl
    have hDefR2 :
        TensorLiftLike (R := R) (M := M) (N := N) (P := P) b (x₂, y) = b x₂ y := by
      rfl
    have hPost :
        ModuleLike.add (R := R) (M := P) (b x₁ y) (b x₂ y)
          = ModuleLike.add (R := R) (M := P)
              (TensorLiftLike (R := R) (M := M) (N := N) (P := P) b (x₁, y))
              (TensorLiftLike (R := R) (M := M) (N := N) (P := P) b (x₂, y)) := by
      rw [hDefR1, hDefR2]
    calc
      TensorLiftLike (R := R) (M := M) (N := N) (P := P) b (ModuleLike.add (R := R) (M := M) x₁ x₂, y)
          = b (ModuleLike.add (R := R) (M := M) x₁ x₂) y := hDefL
      _ = ModuleLike.add (R := R) (M := P) (b x₁ y) (b x₂ y) := hStep
      _ = ModuleLike.add (R := R) (M := P)
            (TensorLiftLike (R := R) (M := M) (N := N) (P := P) b (x₁, y))
            (TensorLiftLike (R := R) (M := M) (N := N) (P := P) b (x₂, y)) := hPost
  constructor
  · intro a x y
    have hStep : b (ModuleLike.smul (R := R) (M := M) a x) y
        = ModuleLike.smul (R := R) (M := P) a (b x y) :=
      hSmulL a x y
    have hDefL :
        TensorLiftLike (R := R) (M := M) (N := N) (P := P) b
          (ModuleLike.smul (R := R) (M := M) a x, y)
          = b (ModuleLike.smul (R := R) (M := M) a x) y := by
      rfl
    have hDefR :
        TensorLiftLike (R := R) (M := M) (N := N) (P := P) b (x, y) = b x y := by
      rfl
    have hPost :
        ModuleLike.smul (R := R) (M := P) a (b x y)
          = ModuleLike.smul (R := R) (M := P) a
              (TensorLiftLike (R := R) (M := M) (N := N) (P := P) b (x, y)) := by
      rw [hDefR]
    calc
      TensorLiftLike (R := R) (M := M) (N := N) (P := P) b (ModuleLike.smul (R := R) (M := M) a x, y)
          = b (ModuleLike.smul (R := R) (M := M) a x) y := hDefL
      _ = ModuleLike.smul (R := R) (M := P) a (b x y) := hStep
      _ = ModuleLike.smul (R := R) (M := P) a
            (TensorLiftLike (R := R) (M := M) (N := N) (P := P) b (x, y)) := hPost
  constructor
  · intro x y₁ y₂
    have hStep : b x (ModuleLike.add (R := R) (M := N) y₁ y₂)
        = ModuleLike.add (R := R) (M := P) (b x y₁) (b x y₂) :=
      hAddR x y₁ y₂
    have hDefL :
        TensorLiftLike (R := R) (M := M) (N := N) (P := P) b
          (x, ModuleLike.add (R := R) (M := N) y₁ y₂)
          = b x (ModuleLike.add (R := R) (M := N) y₁ y₂) := by
      rfl
    have hDefR1 :
        TensorLiftLike (R := R) (M := M) (N := N) (P := P) b (x, y₁) = b x y₁ := by
      rfl
    have hDefR2 :
        TensorLiftLike (R := R) (M := M) (N := N) (P := P) b (x, y₂) = b x y₂ := by
      rfl
    have hPost :
        ModuleLike.add (R := R) (M := P) (b x y₁) (b x y₂)
          = ModuleLike.add (R := R) (M := P)
              (TensorLiftLike (R := R) (M := M) (N := N) (P := P) b (x, y₁))
              (TensorLiftLike (R := R) (M := M) (N := N) (P := P) b (x, y₂)) := by
      rw [hDefR1, hDefR2]
    calc
      TensorLiftLike (R := R) (M := M) (N := N) (P := P) b (x, ModuleLike.add (R := R) (M := N) y₁ y₂)
          = b x (ModuleLike.add (R := R) (M := N) y₁ y₂) := hDefL
      _ = ModuleLike.add (R := R) (M := P) (b x y₁) (b x y₂) := hStep
      _ = ModuleLike.add (R := R) (M := P)
            (TensorLiftLike (R := R) (M := M) (N := N) (P := P) b (x, y₁))
            (TensorLiftLike (R := R) (M := M) (N := N) (P := P) b (x, y₂)) := hPost
  · intro a x y
    have hStep : b x (ModuleLike.smul (R := R) (M := N) a y)
        = ModuleLike.smul (R := R) (M := P) a (b x y) :=
      hSmulR a x y
    have hDefL :
        TensorLiftLike (R := R) (M := M) (N := N) (P := P) b
          (x, ModuleLike.smul (R := R) (M := N) a y)
          = b x (ModuleLike.smul (R := R) (M := N) a y) := by
      rfl
    have hDefR :
        TensorLiftLike (R := R) (M := M) (N := N) (P := P) b (x, y) = b x y := by
      rfl
    have hPost :
        ModuleLike.smul (R := R) (M := P) a (b x y)
          = ModuleLike.smul (R := R) (M := P) a
              (TensorLiftLike (R := R) (M := M) (N := N) (P := P) b (x, y)) := by
      rw [hDefR]
    calc
      TensorLiftLike (R := R) (M := M) (N := N) (P := P) b (x, ModuleLike.smul (R := R) (M := N) a y)
          = b x (ModuleLike.smul (R := R) (M := N) a y) := hDefL
      _ = ModuleLike.smul (R := R) (M := P) a (b x y) := hStep
      _ = ModuleLike.smul (R := R) (M := P) a
            (TensorLiftLike (R := R) (M := M) (N := N) (P := P) b (x, y)) := hPost

theorem curry_uncurry {R : Type u} {M : Type v} {N : Type w} {P : Type z}
    [RingLike R] [ModuleLike R M] [ModuleLike R N] [ModuleLike R P]
    (b : M → N → P) :
    ∀ x : M, ∀ y : N,
      CurryLike (fun t : TensorLike R M N => TensorLiftLike (R := R) (M := M) (N := N) (P := P) b t) x y = b x y := by
  intro x y
  have hDefC :
      CurryLike (fun t : TensorLike R M N => TensorLiftLike (R := R) (M := M) (N := N) (P := P) b t) x y
        = (TensorLiftLike (R := R) (M := M) (N := N) (P := P) b (x, y) : P) := by
    rfl
  have hDefT :
      (TensorLiftLike (R := R) (M := M) (N := N) (P := P) b (x, y) : P) = b x y := by
    rfl
  calc
    CurryLike (fun t : TensorLike R M N => TensorLiftLike (R := R) (M := M) (N := N) (P := P) b t) x y
        = (TensorLiftLike (R := R) (M := M) (N := N) (P := P) b (x, y) : P) := hDefC
    _ = b x y := hDefT

theorem uncurry_curry {R : Type u} {M : Type v} {N : Type w} {P : Type z}
    [RingLike R] [ModuleLike R M] [ModuleLike R N] [ModuleLike R P]
    (h : TensorLike R M N → P) :
    ∀ t : TensorLike R M N,
      TensorLiftLike (R := R) (M := M) (N := N) (P := P) (CurryLike (R := R) (M := M) (N := N) (P := P) h) t = h t := by
  intro t
  cases t with
  | mk x y =>
      have hDefT :
          TensorLiftLike (R := R) (M := M) (N := N) (P := P) (CurryLike (R := R) (M := M) (N := N) (P := P) h) (x, y)
            = CurryLike (R := R) (M := M) (N := N) (P := P) h x y := by
        rfl
      have hDefC :
          CurryLike (R := R) (M := M) (N := N) (P := P) h x y = h (x, y) := by
        rfl
      calc
        TensorLiftLike (R := R) (M := M) (N := N) (P := P) (CurryLike (R := R) (M := M) (N := N) (P := P) h) (x, y)
            = CurryLike (R := R) (M := M) (N := N) (P := P) h x y := hDefT
        _ = h (x, y) := hDefC

theorem tensor_hom_adjunction_like {R : Type u} {M : Type v} {N : Type w} {P : Type z}
    [RingLike R] [ModuleLike R M] [ModuleLike R N] [ModuleLike R P] :
    (∀ h : TensorLike R M N → P, ∀ t : TensorLike R M N,
      TensorLiftLike (R := R) (M := M) (N := N) (P := P) (CurryLike (R := R) (M := M) (N := N) (P := P) h) t = h t) ∧
    (∀ b : M → N → P, ∀ x : M, ∀ y : N,
      CurryLike (fun t : TensorLike R M N => TensorLiftLike (R := R) (M := M) (N := N) (P := P) b t) x y = b x y) := by
  constructor
  · intro h t
    have hUncurry : TensorLiftLike (R := R) (M := M) (N := N) (P := P) (CurryLike (R := R) (M := M) (N := N) (P := P) h) t = h t :=
      uncurry_curry (h := h) t
    have hReflexive : TensorLiftLike (R := R) (M := M) (N := N) (P := P) (CurryLike (R := R) (M := M) (N := N) (P := P) h) t
        = TensorLiftLike (R := R) (M := M) (N := N) (P := P) (CurryLike (R := R) (M := M) (N := N) (P := P) h) t := by
      rfl
    calc
      TensorLiftLike (R := R) (M := M) (N := N) (P := P) (CurryLike (R := R) (M := M) (N := N) (P := P) h) t
          = TensorLiftLike (R := R) (M := M) (N := N) (P := P) (CurryLike (R := R) (M := M) (N := N) (P := P) h) t := hReflexive
      _ = h t := hUncurry
  · intro b x y
    have hCurry : CurryLike (fun t : TensorLike R M N => TensorLiftLike (R := R) (M := M) (N := N) (P := P) b t) x y = b x y :=
      curry_uncurry (b := b) x y
    have hReflexive :
        CurryLike (fun t : TensorLike R M N => TensorLiftLike (R := R) (M := M) (N := N) (P := P) b t) x y
          = CurryLike (fun t : TensorLike R M N => TensorLiftLike (R := R) (M := M) (N := N) (P := P) b t) x y := by
      rfl
    calc
      CurryLike (fun t : TensorLike R M N => TensorLiftLike (R := R) (M := M) (N := N) (P := P) b t) x y
          = CurryLike (fun t : TensorLike R M N => TensorLiftLike (R := R) (M := M) (N := N) (P := P) b t) x y := hReflexive
      _ = b x y := hCurry

theorem tensor_ext_like {R : Type u} {M : Type v} {N : Type w} {P : Type z}
    [RingLike R] [ModuleLike R M] [ModuleLike R N] [ModuleLike R P]
    (h₁ h₂ : TensorLike R M N → P)
    (hEq : ∀ x : M, ∀ y : N, h₁ (x, y) = h₂ (x, y)) :
    ∀ t : TensorLike R M N, h₁ t = h₂ t := by
  intro t
  cases t with
  | mk x y =>
      have hxy : h₁ (x, y) = h₂ (x, y) := hEq x y
      have hLift : h₁ (Prod.mk x y) = h₂ (Prod.mk x y) := by
        simpa using hxy
      exact hLift
