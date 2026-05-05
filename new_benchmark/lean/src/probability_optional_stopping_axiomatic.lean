/-
BENCHMARK_ID: TINY_MATHLIB_BATCH03_PROB_OPTIONAL_STOPPING_AXIOMATIC
PAIR_STEM: probability_optional_stopping_axiomatic
MATH_DOMAIN: Probability Theory
SOURCE_MATHLIB: Mathlib/Probability/Martingale
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class ProbSpaceLike (Ω : Type u) where
  Expect : (Ω → Nat) → Nat

def FiltrationLike (Ω : Type u) : Type u :=
  Nat → (Ω → Prop) → Prop

def AdaptedLike {Ω : Type u} (F : FiltrationLike Ω) (X : Nat → Ω → Nat) : Prop :=
  ∀ n m : Nat, F n (fun ω => X m ω = X m ω)

def MartingaleLike {Ω : Type u} [ProbSpaceLike Ω]
    (F : FiltrationLike Ω) (X : Nat → Ω → Nat) : Prop :=
  AdaptedLike F X ∧
    (∀ n : Nat,
      ProbSpaceLike.Expect (X (Nat.succ n)) = ProbSpaceLike.Expect (X n))

def StoppingTimeLike {Ω : Type u} (F : FiltrationLike Ω) (τ : Nat) : Prop :=
  ∀ n : Nat, F n (fun _ : Ω => True)

def StoppedValue {Ω : Type u} (X : Nat → Ω → Nat) (τ : Nat) : Nat → Ω → Nat :=
  fun n ω => X (Nat.min τ n) ω

theorem stopped_adapted {Ω : Type u}
    (F : FiltrationLike Ω) (X : Nat → Ω → Nat) (τ : Nat)
    (hmono : ∀ m n : Nat, m ≤ n → ∀ s : Ω → Prop, F m s → F n s)
    (hinter : ∀ n : Nat, ∀ s t : Ω → Prop,
      F n s → F n t → F n (fun ω => s ω ∧ t ω))
    (hsuperset : ∀ n : Nat, ∀ s t : Ω → Prop,
      F n s → (∀ ω : Ω, s ω → t ω) → F n t)
    (hX : AdaptedLike F X)
    (hτ : StoppingTimeLike F τ) :
    AdaptedLike F (StoppedValue X τ) := by
  intro n
  intro m
  have hlift : F n (fun ω => X (Nat.min τ m) ω = X (Nat.min τ m) ω) :=
    hX n (Nat.min τ m)
  have hstop : F n (fun _ : Ω => True) := hτ n
  have hboth :
      F n (fun ω => (X (Nat.min τ m) ω = X (Nat.min τ m) ω) ∧ True) :=
    hinter n (fun ω => X (Nat.min τ m) ω = X (Nat.min τ m) ω) (fun _ => True) hlift hstop
  have hsub :
      ∀ ω : Ω,
        ((X (Nat.min τ m) ω = X (Nat.min τ m) ω) ∧ True) →
        StoppedValue X τ m ω = StoppedValue X τ m ω := by
    intro ω hω
    rcases hω with ⟨hEq, _⟩
    simpa [StoppedValue] using hEq
  exact hsuperset n (fun ω => (X (Nat.min τ m) ω = X (Nat.min τ m) ω) ∧ True)
    (fun ω => StoppedValue X τ m ω = StoppedValue X τ m ω) hboth hsub

theorem stopped_martingale {Ω : Type u} [ProbSpaceLike Ω]
    (F : FiltrationLike Ω) (X : Nat → Ω → Nat) (τ : Nat)
    (hmono : ∀ m n : Nat, m ≤ n → ∀ s : Ω → Prop, F m s → F n s)
    (hinter : ∀ n : Nat, ∀ s t : Ω → Prop,
      F n s → F n t → F n (fun ω => s ω ∧ t ω))
    (hsuperset : ∀ n : Nat, ∀ s t : Ω → Prop,
      F n s → (∀ ω : Ω, s ω → t ω) → F n t)
    (hτ : StoppingTimeLike F τ)
    (hM : MartingaleLike F X)
    (hstationary : ∀ n : Nat,
      ProbSpaceLike.Expect (X (Nat.min τ (Nat.succ n))) =
        ProbSpaceLike.Expect (X (Nat.min τ n))) :
    MartingaleLike F (StoppedValue X τ) := by
  constructor
  · exact stopped_adapted F X τ hmono hinter hsuperset hM.1 hτ
  · intro n
    have hstep :
        ProbSpaceLike.Expect (X (Nat.min τ (Nat.succ n))) =
          ProbSpaceLike.Expect (X (Nat.min τ n)) :=
      hstationary n
    have hleft :
        ProbSpaceLike.Expect ((StoppedValue X τ) (Nat.succ n)) =
          ProbSpaceLike.Expect (X (Nat.min τ (Nat.succ n))) := by
      rfl
    have hright :
        ProbSpaceLike.Expect ((StoppedValue X τ) n) =
          ProbSpaceLike.Expect (X (Nat.min τ n)) := by
      rfl
    calc
      ProbSpaceLike.Expect ((StoppedValue X τ) (Nat.succ n))
          = ProbSpaceLike.Expect (X (Nat.min τ (Nat.succ n))) := hleft
      _ = ProbSpaceLike.Expect (X (Nat.min τ n)) := hstep
      _ = ProbSpaceLike.Expect ((StoppedValue X τ) n) := by
        symm
        exact hright

theorem optional_stopping_submartingale_bound {Ω : Type u} [ProbSpaceLike Ω]
    (X : Nat → Ω → Nat) (τ : Nat)
    (hstop_eq : ∀ n : Nat,
      ProbSpaceLike.Expect ((StoppedValue X τ) (Nat.succ n)) =
        ProbSpaceLike.Expect ((StoppedValue X τ) n))
    (hbound0n : ∀ n : Nat,
      ProbSpaceLike.Expect ((StoppedValue X τ) 0) ≤ ProbSpaceLike.Expect (X n))
    (n : Nat) :
    ProbSpaceLike.Expect ((StoppedValue X τ) n) ≤ ProbSpaceLike.Expect (X n) := by
  have hconst : ∀ k : Nat,
      ProbSpaceLike.Expect ((StoppedValue X τ) k) =
        ProbSpaceLike.Expect ((StoppedValue X τ) 0) := by
    intro k
    induction k with
    | zero =>
      rfl
    | succ k ih =>
      calc
        ProbSpaceLike.Expect ((StoppedValue X τ) (Nat.succ k))
            = ProbSpaceLike.Expect ((StoppedValue X τ) k) := hstop_eq k
        _ = ProbSpaceLike.Expect ((StoppedValue X τ) 0) := ih
  calc
    ProbSpaceLike.Expect ((StoppedValue X τ) n)
        = ProbSpaceLike.Expect ((StoppedValue X τ) 0) := hconst n
    _ ≤ ProbSpaceLike.Expect (X n) := hbound0n n

theorem optional_stopping_eq_expectation {Ω : Type u} [ProbSpaceLike Ω]
    (F : FiltrationLike Ω) (X : Nat → Ω → Nat) (τ : Nat)
    (hM : MartingaleLike F (StoppedValue X τ)) :
    ∀ n : Nat,
      ProbSpaceLike.Expect ((StoppedValue X τ) n) =
        ProbSpaceLike.Expect ((StoppedValue X τ) 0) := by
  intro n
  induction n with
  | zero =>
    rfl
  | succ n ih =>
    calc
      ProbSpaceLike.Expect ((StoppedValue X τ) (Nat.succ n))
          = ProbSpaceLike.Expect ((StoppedValue X τ) n) := hM.2 n
      _ = ProbSpaceLike.Expect ((StoppedValue X τ) 0) := ih

theorem optional_stopping_iterated {Ω : Type u}
    (X : Nat → Ω → Nat) (τ σ : Nat)
    (hmin_assoc : ∀ n : Nat, Nat.min τ (Nat.min σ n) = Nat.min (Nat.min τ σ) n)
    (n : Nat) (ω : Ω) :
    StoppedValue (StoppedValue X τ) σ n ω = StoppedValue X (Nat.min τ σ) n ω := by
  have hmin : Nat.min τ (Nat.min σ n) = Nat.min (Nat.min τ σ) n := hmin_assoc n
  calc
    StoppedValue (StoppedValue X τ) σ n ω
        = X (Nat.min τ (Nat.min σ n)) ω := by
          rfl
    _ = X (Nat.min (Nat.min τ σ) n) ω := by
      rw [hmin]
    _ = StoppedValue X (Nat.min τ σ) n ω := by
      rfl

theorem uniform_integrable_extension {Ω : Type u} [ProbSpaceLike Ω]
    (X Y : Nat → Ω → Nat) (τ : Nat)
    (hlink : ∀ n : Nat,
      ProbSpaceLike.Expect (Y n) = ProbSpaceLike.Expect ((StoppedValue X τ) n))
    (hstable : ∀ n : Nat,
      ProbSpaceLike.Expect ((StoppedValue X τ) n) =
        ProbSpaceLike.Expect ((StoppedValue X τ) 0)) :
    ∀ n : Nat,
      ProbSpaceLike.Expect (Y n) = ProbSpaceLike.Expect ((StoppedValue X τ) 0) := by
  intro n
  have h1 : ProbSpaceLike.Expect (Y n) = ProbSpaceLike.Expect ((StoppedValue X τ) n) :=
    hlink n
  have h2 : ProbSpaceLike.Expect ((StoppedValue X τ) n) =
      ProbSpaceLike.Expect ((StoppedValue X τ) 0) :=
    hstable n
  calc
    ProbSpaceLike.Expect (Y n)
        = ProbSpaceLike.Expect ((StoppedValue X τ) n) := h1
    _ = ProbSpaceLike.Expect ((StoppedValue X τ) 0) := h2
