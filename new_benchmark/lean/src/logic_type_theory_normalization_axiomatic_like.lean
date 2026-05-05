/-
BENCHMARK_ID: TINY_MATHLIB_BATCH04_LOGIC_TYPE_THEORY_NORMALIZATION_AXIOMATIC_LIKE
PAIR_STEM: logic_type_theory_normalization_axiomatic_like
MATH_DOMAIN: Logic / Type Theory
SOURCE_MATHLIB: foundational formal systems patterns
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u v

class ContextLike (Ctx : Type u) where
  empty : Ctx
  extend : Ctx → Nat → Ctx
  depth : Ctx → Nat
  depth_empty : depth empty = 0
  depth_extend : ∀ Γ : Ctx, ∀ A : Nat, depth (extend Γ A) = depth Γ + 1

def TermLike {Ctx : Type u} [ContextLike Ctx] (Tm : Type v) : Prop :=
  ∃ shift : Tm → Tm, ∃ subst : Tm → Tm → Tm, True

def TypingLike {Ctx : Type u} [ContextLike Ctx] {Tm : Type v}
    (ty : Ctx → Tm → Nat → Prop) : Prop :=
  ∀ Γ : Ctx, ∀ t : Tm, ∀ A B : Nat, A = B → ty Γ t A → ty Γ t B

def ReductionLike {Tm : Type v} (step : Tm → Tm → Prop) : Prop :=
  ∀ t u v : Tm, step t u → step u v → step t v

def NeutralLike {Tm : Type v} (neutral : Tm → Prop) : Prop :=
  ∀ t u : Tm, neutral t → ¬ neutral u → t ≠ u

def NormalLike {Tm : Type v} (normal : Tm → Prop) (step : Tm → Tm → Prop) : Prop :=
  ∀ t u : Tm, normal t → step t u → False

theorem subject_reduction_like {Ctx : Type u} [ContextLike Ctx] {Tm : Type v}
    (ty : Ctx → Tm → Nat → Prop)
    (step : Tm → Tm → Prop)
    (hTyping : TypingLike ty)
    (hsubj : ∀ Γ : Ctx, ∀ t u : Tm, ∀ A : Nat, step t u → ty Γ t A → ty Γ u A) :
    ∀ Γ : Ctx, ∀ t u : Tm, ∀ A : Nat, step t u → ty Γ t A → ty Γ u A := by
  intro Γ t u A hstep hty
  have hkeep : ty Γ t A := hty
  have hcast : ty Γ t A := hTyping Γ t A A rfl hkeep
  have hred : ty Γ u A := hsubj Γ t u A hstep hcast
  exact hred

theorem progress_like {Ctx : Type u} [ContextLike Ctx] {Tm : Type v}
    (step : Tm → Tm → Prop)
    (normal : Tm → Prop)
    (hterm : TermLike (Ctx := Ctx) Tm)
    (hprogress : ∀ t : Tm, (∃ u : Tm, step t u) ∨ normal t) :
    ∀ t : Tm, (∃ u : Tm, step t u) ∨ normal t := by
  rcases hterm with ⟨shift, subst, htriv⟩
  intro t
  have hbase : (∃ u : Tm, step t u) ∨ normal t := hprogress t
  have _ : Tm → Tm := shift
  have _ : Tm → Tm → Tm := subst
  have _ : True := htriv
  exact hbase

theorem substitution_preserves_typing {Ctx : Type u} [ContextLike Ctx] {Tm : Type v}
    (ty : Ctx → Tm → Nat → Prop)
    (hTyping : TypingLike ty)
    (hsubst :
      ∀ Γ : Ctx, ∀ t s : Tm, ∀ A B : Nat,
        ty (ContextLike.extend Γ A) t B → ty Γ s A → ty Γ t B) :
    ∀ Γ : Ctx, ∀ t s : Tm, ∀ A B : Nat,
      ty (ContextLike.extend Γ A) t B → ty Γ s A → ty Γ t B := by
  intro Γ t s A B hty hs
  have htyped : ty (ContextLike.extend Γ A) t B := hty
  have hseed : ty Γ s A := hs
  have hcast : ty (ContextLike.extend Γ A) t B := hTyping (ContextLike.extend Γ A) t B B rfl htyped
  have hresult : ty Γ t B := hsubst Γ t s A B hcast hseed
  exact hresult

theorem reducibility_closure_step {Ctx : Type u} [ContextLike Ctx] {Tm : Type v}
    (step : Tm → Tm → Prop)
    (R : Tm → Prop)
    (hone : ∀ t u : Tm, step t u → R t → R u)
    (htrans : ReductionLike step) :
    ∀ t u v : Tm, step t u → step u v → R t → R v := by
  intro t u v htu huv hRt
  have hRu : R u := hone t u htu hRt
  have htv : step t v := htrans t u v htu huv
  have hRv : R v := hone u v huv hRu
  have _ : step t v := htv
  exact hRv

theorem strong_normalization_like {Ctx : Type u} [ContextLike Ctx] {Tm : Type v}
    (step : Tm → Tm → Prop)
    (SN normal : Tm → Prop)
    (hbase : ∀ t : Tm, normal t → SN t)
    (hstep : ∀ t u : Tm, step t u → SN u → SN t)
    (hseed : ∀ t u : Tm, step t u → SN u)
    (hprogress : ∀ t : Tm, normal t ∨ ∃ u : Tm, step t u) :
    ∀ t : Tm, SN t := by
  intro t
  have hcase : normal t ∨ ∃ u : Tm, step t u := hprogress t
  cases hcase with
  | inl hnorm =>
      exact hbase t hnorm
  | inr hred =>
      rcases hred with ⟨u, hu⟩
      have hSu : SN u := hseed t u hu
      have hSt : SN t := hstep t u hu hSu
      exact hSt

theorem normalization_by_evaluation_interface {Ctx : Type u} [ContextLike Ctx] {Tm : Type v}
    (ty : Ctx → Tm → Nat → Prop)
    (step : Tm → Tm → Prop)
    (normal SN nbe : Tm → Prop)
    (hTyping : TypingLike ty)
    (hsubj : ∀ Γ : Ctx, ∀ t u : Tm, ∀ A : Nat, step t u → ty Γ t A → ty Γ u A)
    (hbase : ∀ t : Tm, normal t → SN t)
    (hstep : ∀ t u : Tm, step t u → SN u → SN t)
    (hseed : ∀ t u : Tm, step t u → SN u)
    (hprogress : ∀ t : Tm, normal t ∨ ∃ u : Tm, step t u)
    (hnbe : ∀ t : Tm, SN t → nbe t) :
    ∀ Γ : Ctx, ∀ t : Tm, ∀ A : Nat, ty Γ t A → nbe t := by
  have hsr : ∀ Γ : Ctx, ∀ t u : Tm, ∀ A : Nat, step t u → ty Γ t A → ty Γ u A :=
    subject_reduction_like (Ctx := Ctx) (Tm := Tm) ty step hTyping hsubj
  have hsn : ∀ t : Tm, SN t :=
    strong_normalization_like (Ctx := Ctx) (Tm := Tm) step SN normal hbase hstep hseed hprogress
  intro Γ t A hty
  have htyped : ty Γ t A := hty
  have hsnt : SN t := hsn t
  have _ : ∀ Γ : Ctx, ∀ t u : Tm, ∀ A : Nat, step t u → ty Γ t A → ty Γ u A := hsr
  have _ : ty Γ t A := htyped
  exact hnbe t hsnt
