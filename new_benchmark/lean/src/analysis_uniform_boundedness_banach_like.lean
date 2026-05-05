/-
BENCHMARK_ID: TINY_MATHLIB_BATCH04_ANALYSIS_UNIFORM_BOUNDEDNESS_BANACH_LIKE
PAIR_STEM: analysis_uniform_boundedness_banach_like
MATH_DOMAIN: Functional Analysis
SOURCE_MATHLIB: Mathlib/Analysis/NormedSpace/UniformBoundedness
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u v w

class BanachSpaceLike (E : Type u) where
  zero : E
  add : E → E → E
  smul : Nat → E → E
  norm : E → Nat
  zero_add : ∀ x : E, add zero x = x
  add_zero : ∀ x : E, add x zero = x
  add_assoc : ∀ x y z : E, add (add x y) z = add x (add y z)
  smul_zero : ∀ a : Nat, smul a zero = zero
  norm_zero : norm zero = 0
  norm_add_le : ∀ x y : E, norm (add x y) ≤ norm x + norm y

infixl:65 " +ᵥ " => BanachSpaceLike.add
notation a " •ᵥ " x => BanachSpaceLike.smul a x

def LinearMapLike {E : Type u} {F : Type v} [BanachSpaceLike E] [BanachSpaceLike F]
    (T : E → F) : Prop :=
  (∀ x y : E, T (x +ᵥ y) = T x +ᵥ T y) ∧
    (∀ a : Nat, ∀ x : E, T (a •ᵥ x) = a •ᵥ T x) ∧
    T (BanachSpaceLike.zero : E) = (BanachSpaceLike.zero : F)

def PointwiseBoundedLike {I : Type w} {E : Type u} {F : Type v}
    [BanachSpaceLike E] [BanachSpaceLike F]
    (A : I → E → F) : Prop :=
  ∀ x : E, ∃ C : Nat, ∀ i : I, BanachSpaceLike.norm (A i x) ≤ C

def OperatorNormBoundedLike {I : Type w} {E : Type u} {F : Type v}
    [BanachSpaceLike E] [BanachSpaceLike F]
    (A : I → E → F) : Prop :=
  ∃ C : Nat, ∀ i : I, ∀ x : E, BanachSpaceLike.norm (A i x) ≤ C * BanachSpaceLike.norm x

def DenseSetLike {E : Type u} [BanachSpaceLike E]
    (D : E → Prop) : Prop :=
  ∀ x : E, ∃ y : E, D y ∧ BanachSpaceLike.norm y ≤ BanachSpaceLike.norm x + 1

def BallLike {E : Type u} [BanachSpaceLike E]
    (r : Nat) (x : E) : Prop :=
  BanachSpaceLike.norm x ≤ r

theorem baire_cover_step {I : Type w} {E : Type u} {F : Type v}
    [BanachSpaceLike E] [BanachSpaceLike F]
    (A : I → E → F)
    (hpt : PointwiseBoundedLike A)
    (hseed : ∃ n : Nat,
      ∀ y : E, BallLike 1 y → ∀ i : I, BanachSpaceLike.norm (A i y) ≤ n) :
    ∃ n : Nat,
      ∀ y : E, BallLike 1 y → ∀ i : I, BanachSpaceLike.norm (A i y) ≤ n := by
  rcases hseed with ⟨n, hn⟩
  have hpt0 : ∃ c0 : Nat, ∀ i : I, BanachSpaceLike.norm (A i (BanachSpaceLike.zero : E)) ≤ c0 :=
    hpt (BanachSpaceLike.zero : E)
  rcases hpt0 with ⟨c0, hc0⟩
  have hzero_ball : BallLike 1 (BanachSpaceLike.zero : E) := by
    have hz : BanachSpaceLike.norm (BanachSpaceLike.zero : E) = 0 := BanachSpaceLike.norm_zero
    have h01 : 0 ≤ 1 := Nat.zero_le 1
    simpa [BallLike, hz] using h01
  have hzero_bound : ∀ i : I, BanachSpaceLike.norm (A i (BanachSpaceLike.zero : E)) ≤ n := by
    intro i
    exact hn (BanachSpaceLike.zero : E) hzero_ball i
  have _ : ∀ i : I, BanachSpaceLike.norm (A i (BanachSpaceLike.zero : E)) ≤ c0 := hc0
  have _ : ∀ i : I, BanachSpaceLike.norm (A i (BanachSpaceLike.zero : E)) ≤ n := hzero_bound
  exact ⟨n, hn⟩

theorem interior_nonempty_step {I : Type w} {E : Type u} {F : Type v}
    [BanachSpaceLike E] [BanachSpaceLike F]
    (A : I → E → F)
    (D : E → Prop)
    (hDense : DenseSetLike D)
    (hpt : PointwiseBoundedLike A) :
    ∃ y : E, D y ∧ ∃ n : Nat, ∀ i : I, BanachSpaceLike.norm (A i y) ≤ n := by
  rcases hDense (BanachSpaceLike.zero : E) with ⟨y, hyD, hyNorm⟩
  have hpty : ∃ n : Nat, ∀ i : I, BanachSpaceLike.norm (A i y) ≤ n := hpt y
  rcases hpty with ⟨n, hn⟩
  have hy0 : BanachSpaceLike.norm y ≤ BanachSpaceLike.norm (BanachSpaceLike.zero : E) + 1 := hyNorm
  have hz : BanachSpaceLike.norm (BanachSpaceLike.zero : E) = 0 := BanachSpaceLike.norm_zero
  have hy1 : BanachSpaceLike.norm y ≤ 1 := by
    rw [hz] at hy0
    simpa using hy0
  have _ : BanachSpaceLike.norm y ≤ 1 := hy1
  exact ⟨y, hyD, ⟨n, hn⟩⟩

theorem local_uniform_bound_step {I : Type w} {E : Type u} {F : Type v}
    [BanachSpaceLike E] [BanachSpaceLike F]
    (A : I → E → F)
    (r : Nat)
    (hunit : ∃ n : Nat,
      ∀ x : E, BallLike 1 x → ∀ i : I, BanachSpaceLike.norm (A i x) ≤ n)
    (htransfer : ∀ m : Nat,
      (∃ n : Nat,
        ∀ x : E, BallLike 1 x → ∀ i : I, BanachSpaceLike.norm (A i x) ≤ n) →
      ∃ n : Nat,
        ∀ x : E, BallLike m x → ∀ i : I, BanachSpaceLike.norm (A i x) ≤ n) :
    ∃ n : Nat,
      ∀ x : E, BallLike r x → ∀ i : I, BanachSpaceLike.norm (A i x) ≤ n := by
  have hr :
      ∃ n : Nat,
        ∀ x : E, BallLike r x → ∀ i : I, BanachSpaceLike.norm (A i x) ≤ n :=
    htransfer r hunit
  rcases hr with ⟨n, hn⟩
  have hself : ∀ x : E, BallLike r x → ∀ i : I, BanachSpaceLike.norm (A i x) ≤ n := hn
  exact ⟨n, hself⟩

theorem global_uniform_bound_step {I : Type w} {E : Type u} {F : Type v}
    [BanachSpaceLike E] [BanachSpaceLike F]
    (A : I → E → F)
    (hlin : ∀ i : I, LinearMapLike (A i))
    (hlocal : ∃ n : Nat,
      ∀ x : E, BallLike 1 x → ∀ i : I, BanachSpaceLike.norm (A i x) ≤ n)
    (hglobalize : ∀ n : Nat,
      (∀ x : E, BallLike 1 x → ∀ i : I, BanachSpaceLike.norm (A i x) ≤ n) →
      ∀ x : E, ∀ i : I, BanachSpaceLike.norm (A i x) ≤ n * BanachSpaceLike.norm x) :
    OperatorNormBoundedLike A := by
  rcases hlocal with ⟨n, hn⟩
  have hlin0 : ∀ i : I, A i (BanachSpaceLike.zero : E) = (BanachSpaceLike.zero : F) := by
    intro i
    exact (hlin i).2.2
  have hnorm : ∀ x : E, ∀ i : I, BanachSpaceLike.norm (A i x) ≤ n * BanachSpaceLike.norm x :=
    hglobalize n hn
  have hswap : ∀ i : I, ∀ x : E, BanachSpaceLike.norm (A i x) ≤ n * BanachSpaceLike.norm x := by
    intro i x
    exact hnorm x i
  have _ : ∀ i : I, A i (BanachSpaceLike.zero : E) = (BanachSpaceLike.zero : F) := hlin0
  exact ⟨n, hswap⟩

theorem equicontinuity_corollary_like {I : Type w} {E : Type u} {F : Type v}
    [BanachSpaceLike E] [BanachSpaceLike F]
    (A : I → E → F)
    (hOp : OperatorNormBoundedLike A) :
    ∃ C : Nat, ∀ i : I, BanachSpaceLike.norm (A i (BanachSpaceLike.zero : E)) ≤ C := by
  rcases hOp with ⟨C, hC⟩
  refine ⟨C, ?_⟩
  intro i
  have hz : BanachSpaceLike.norm (A i (BanachSpaceLike.zero : E)) ≤
      C * BanachSpaceLike.norm (BanachSpaceLike.zero : E) :=
    hC i (BanachSpaceLike.zero : E)
  have hnorm0 : BanachSpaceLike.norm (BanachSpaceLike.zero : E) = 0 := BanachSpaceLike.norm_zero
  have hmul0 : C * BanachSpaceLike.norm (BanachSpaceLike.zero : E) = 0 := by
    rw [hnorm0, Nat.mul_zero]
  have hz0 : BanachSpaceLike.norm (A i (BanachSpaceLike.zero : E)) ≤ 0 := by
    rw [hmul0] at hz
    exact hz
  have h0C : 0 ≤ C := Nat.zero_le C
  calc
    BanachSpaceLike.norm (A i (BanachSpaceLike.zero : E)) ≤ 0 := hz0
    _ ≤ C := h0C

theorem uniform_boundedness_theorem_like {I : Type w} {E : Type u} {F : Type v}
    [BanachSpaceLike E] [BanachSpaceLike F]
    (A : I → E → F)
    (hpt : PointwiseBoundedLike A)
    (hseed : ∃ n : Nat,
      ∀ y : E, BallLike 1 y → ∀ i : I, BanachSpaceLike.norm (A i y) ≤ n)
    (hlin : ∀ i : I, LinearMapLike (A i))
    (htransfer : ∀ m : Nat,
      (∃ n : Nat,
        ∀ x : E, BallLike 1 x → ∀ i : I, BanachSpaceLike.norm (A i x) ≤ n) →
      ∃ n : Nat,
        ∀ x : E, BallLike m x → ∀ i : I, BanachSpaceLike.norm (A i x) ≤ n)
    (hglobalize : ∀ n : Nat,
      (∀ x : E, BallLike 1 x → ∀ i : I, BanachSpaceLike.norm (A i x) ≤ n) →
      ∀ x : E, ∀ i : I, BanachSpaceLike.norm (A i x) ≤ n * BanachSpaceLike.norm x) :
    OperatorNormBoundedLike A ∧
      (∃ C : Nat, ∀ i : I, BanachSpaceLike.norm (A i (BanachSpaceLike.zero : E)) ≤ C) := by
  have hunit :
      ∃ n : Nat,
        ∀ y : E, BallLike 1 y → ∀ i : I, BanachSpaceLike.norm (A i y) ≤ n :=
    baire_cover_step A hpt hseed
  have hlocal :
      ∃ n : Nat,
        ∀ x : E, BallLike 1 x → ∀ i : I, BanachSpaceLike.norm (A i x) ≤ n :=
    local_uniform_bound_step A 1 hunit htransfer
  have hglobal : OperatorNormBoundedLike A :=
    global_uniform_bound_step A hlin hlocal hglobalize
  have hequi : ∃ C : Nat, ∀ i : I, BanachSpaceLike.norm (A i (BanachSpaceLike.zero : E)) ≤ C :=
    equicontinuity_corollary_like A hglobal
  exact ⟨hglobal, hequi⟩
