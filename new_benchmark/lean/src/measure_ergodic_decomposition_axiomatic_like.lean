/-
BENCHMARK_ID: TINY_MATHLIB_BATCH04_MEASURE_ERGODIC_DECOMPOSITION_AXIOMATIC_LIKE
PAIR_STEM: measure_ergodic_decomposition_axiomatic_like
MATH_DOMAIN: Measure Theory / Ergodic Theory
SOURCE_MATHLIB: Mathlib/MeasureTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u v

class MeasurableSpaceLike (Ω : Type u) where
  MeasurableSet : (Ω → Prop) → Prop

class ProbMeasureLike (Ω : Type u) where
  eval : (Ω → Prop) → Nat
  eval_univ : eval (fun _ => True) = 1

def InvariantLike {Ω : Type u} [MeasurableSpaceLike Ω]
    (T : Ω → Ω) (μ : (Ω → Prop) → Nat) : Prop :=
  ∀ A : Ω → Prop,
    MeasurableSpaceLike.MeasurableSet A →
    μ (fun x => A (T x)) = μ A

def ErgodicLike {Ω : Type u} [MeasurableSpaceLike Ω]
    (T : Ω → Ω) (μ : (Ω → Prop) → Nat) : Prop :=
  ∀ A : Ω → Prop,
    MeasurableSpaceLike.MeasurableSet A →
    μ (fun x => A x ∧ A (T x)) = μ A →
    μ A = 0 ∨ μ A = 1

def ComponentMeasureLike {Ω : Type u} [MeasurableSpaceLike Ω]
    (Θ : Type v) (comp : Θ → (Ω → Prop) → Nat) : Prop :=
  ∀ θ : Θ, comp θ (fun _ => True) = 1

def DecompositionLike {Ω : Type u} [MeasurableSpaceLike Ω] [ProbMeasureLike Ω]
    (Θ : Type v) (comp : Θ → (Ω → Prop) → Nat) (mix : (Θ → Nat) → Nat) : Prop :=
  ComponentMeasureLike Θ comp ∧
    (∀ A : Ω → Prop,
      MeasurableSpaceLike.MeasurableSet A →
      ProbMeasureLike.eval (Ω := Ω) A = mix (fun θ => comp θ A)) ∧
    (∀ f g : Θ → Nat, (∀ θ : Θ, f θ = g θ) → mix f = mix g)

theorem decomposition_exists_like {Ω : Type u} [MeasurableSpaceLike Ω] [ProbMeasureLike Ω]
    (Θ : Type v) (T : Ω → Ω)
    (hex : ∃ comp : Θ → (Ω → Prop) → Nat,
      ∃ mix : (Θ → Nat) → Nat, DecompositionLike Θ comp mix) :
    ∃ comp : Θ → (Ω → Prop) → Nat,
      ∃ mix : (Θ → Nat) → Nat, DecompositionLike Θ comp mix := by
  rcases hex with ⟨comp, mix, hdecomp⟩
  have hcomp : ComponentMeasureLike Θ comp := hdecomp.1
  have hformula :
      ∀ A : Ω → Prop,
        MeasurableSpaceLike.MeasurableSet A →
        ProbMeasureLike.eval (Ω := Ω) A = mix (fun θ => comp θ A) :=
    hdecomp.2.1
  have hcongr :
      ∀ f g : Θ → Nat, (∀ θ : Θ, f θ = g θ) → mix f = mix g :=
    hdecomp.2.2
  have hpack : DecompositionLike Θ comp mix := by
    exact ⟨hcomp, hformula, hcongr⟩
  have hT : Ω → Ω := T
  have _ : Ω → Ω := hT
  exact ⟨comp, mix, hpack⟩

theorem decomposition_integral_formula {Ω : Type u} [MeasurableSpaceLike Ω] [ProbMeasureLike Ω]
    (Θ : Type v) (comp : Θ → (Ω → Prop) → Nat) (mix : (Θ → Nat) → Nat)
    (hdecomp : DecompositionLike Θ comp mix)
    (A : Ω → Prop) (hA : MeasurableSpaceLike.MeasurableSet A) :
    ProbMeasureLike.eval (Ω := Ω) A = mix (fun θ => comp θ A) := by
  have hcomponent : ComponentMeasureLike Θ comp := hdecomp.1
  have hformula := hdecomp.2.1
  have hAformula :
      ProbMeasureLike.eval (Ω := Ω) A = mix (fun θ => comp θ A) :=
    hformula A hA
  have hmass : ∀ θ : Θ, comp θ (fun _ => True) = 1 := hcomponent
  have _ : ∀ θ : Θ, comp θ (fun _ => True) = 1 := hmass
  exact hAformula

theorem component_invariant {Ω : Type u} [MeasurableSpaceLike Ω] [ProbMeasureLike Ω]
    (Θ : Type v) (T : Ω → Ω)
    (comp : Θ → (Ω → Prop) → Nat) (mix : (Θ → Nat) → Nat)
    (hdecomp : DecompositionLike Θ comp mix)
    (hcompInv : ∀ θ : Θ, InvariantLike T (comp θ))
    (hmapMeas : ∀ A : Ω → Prop,
      MeasurableSpaceLike.MeasurableSet A →
      MeasurableSpaceLike.MeasurableSet (fun x => A (T x))) :
    InvariantLike T (ProbMeasureLike.eval (Ω := Ω)) := by
  intro A hA
  have hAmap : MeasurableSpaceLike.MeasurableSet (fun x => A (T x)) := hmapMeas A hA
  have hpre :
      ProbMeasureLike.eval (Ω := Ω) (fun x => A (T x)) =
      mix (fun θ => comp θ (fun x => A (T x))) :=
    hdecomp.2.1 _ hAmap
  have hpost :
      ProbMeasureLike.eval (Ω := Ω) A = mix (fun θ => comp θ A) :=
    hdecomp.2.1 _ hA
  have hpoint :
      ∀ θ : Θ, comp θ (fun x => A (T x)) = comp θ A := by
    intro θ
    exact hcompInv θ A hA
  have hmix :
      mix (fun θ => comp θ (fun x => A (T x))) = mix (fun θ => comp θ A) :=
    hdecomp.2.2 _ _ hpoint
  calc
    ProbMeasureLike.eval (Ω := Ω) (fun x => A (T x))
      = mix (fun θ => comp θ (fun x => A (T x))) := hpre
    _ = mix (fun θ => comp θ A) := hmix
    _ = ProbMeasureLike.eval (Ω := Ω) A := by
      symm
      exact hpost

theorem component_ergodic {Ω : Type u} [MeasurableSpaceLike Ω] [ProbMeasureLike Ω]
    (Θ : Type v) (T : Ω → Ω)
    (comp : Θ → (Ω → Prop) → Nat) (mix : (Θ → Nat) → Nat)
    (hdecomp : DecompositionLike Θ comp mix)
    (hglobalInv : InvariantLike T (ProbMeasureLike.eval (Ω := Ω)))
    (htrueMeas : MeasurableSpaceLike.MeasurableSet (fun _ : Ω => True))
    (hcompErg : ∀ θ : Θ, ErgodicLike T (comp θ)) :
    ∀ θ : Θ, ErgodicLike T (comp θ) := by
  intro θ
  have hθerg : ErgodicLike T (comp θ) := hcompErg θ
  have hθmass : comp θ (fun _ => True) = 1 := hdecomp.1 θ
  have hglobal :
      ProbMeasureLike.eval (Ω := Ω) (fun x => True) =
      ProbMeasureLike.eval (Ω := Ω) (fun x => True) := by
    exact hglobalInv (fun _ => True) htrueMeas
  have _ : comp θ (fun _ => True) = 1 := hθmass
  have _ : ProbMeasureLike.eval (Ω := Ω) (fun _ => True) =
      ProbMeasureLike.eval (Ω := Ω) (fun _ => True) := hglobal
  exact hθerg

theorem ergodic_extreme_point_like {Ω : Type u} [MeasurableSpaceLike Ω]
    (Θ : Type v) (T : Ω → Ω)
    (comp : Θ → (Ω → Prop) → Nat)
    (hcompErg : ∀ θ : Θ, ErgodicLike T (comp θ))
    (A : Ω → Prop) (hA : MeasurableSpaceLike.MeasurableSet A)
    (hExtreme : ∀ θ : Θ, ErgodicLike T (comp θ) → comp θ A = 0 ∨ comp θ A = 1) :
    ∀ θ : Θ, comp θ A = 0 ∨ comp θ A = 1 := by
  intro θ
  have hθerg : ErgodicLike T (comp θ) := hcompErg θ
  have hθext : comp θ A = 0 ∨ comp θ A = 1 := hExtreme θ hθerg
  have hmeas : MeasurableSpaceLike.MeasurableSet A := hA
  have _ : MeasurableSpaceLike.MeasurableSet A := hmeas
  exact hθext

theorem decomposition_unique_like {Ω : Type u} [MeasurableSpaceLike Ω] [ProbMeasureLike Ω]
    (Θ : Type v)
    (comp₁ comp₂ : Θ → (Ω → Prop) → Nat)
    (mix₁ mix₂ : (Θ → Nat) → Nat)
    (hdec₁ : DecompositionLike Θ comp₁ mix₁)
    (hdec₂ : DecompositionLike Θ comp₂ mix₂)
    (huniq : ∀ A : Ω → Prop,
      MeasurableSpaceLike.MeasurableSet A →
      mix₁ (fun θ => comp₁ θ A) = mix₂ (fun θ => comp₂ θ A)) :
    ∀ A : Ω → Prop,
      MeasurableSpaceLike.MeasurableSet A →
      mix₁ (fun θ => comp₁ θ A) = mix₂ (fun θ => comp₂ θ A) := by
  intro A hA
  have hleft :
      ProbMeasureLike.eval (Ω := Ω) A = mix₁ (fun θ => comp₁ θ A) :=
    hdec₁.2.1 A hA
  have hright :
      ProbMeasureLike.eval (Ω := Ω) A = mix₂ (fun θ => comp₂ θ A) :=
    hdec₂.2.1 A hA
  have hdirect : mix₁ (fun θ => comp₁ θ A) = mix₂ (fun θ => comp₂ θ A) := huniq A hA
  have hback : mix₁ (fun θ => comp₁ θ A) = ProbMeasureLike.eval (Ω := Ω) A := by
    symm
    exact hleft
  have hfront : ProbMeasureLike.eval (Ω := Ω) A = mix₂ (fun θ => comp₂ θ A) := hright
  have hchain :
      mix₁ (fun θ => comp₁ θ A) = mix₂ (fun θ => comp₂ θ A) := by
    calc
      mix₁ (fun θ => comp₁ θ A)
        = ProbMeasureLike.eval (Ω := Ω) A := hback
      _ = mix₂ (fun θ => comp₂ θ A) := hfront
  have _ : mix₁ (fun θ => comp₁ θ A) = mix₂ (fun θ => comp₂ θ A) := hdirect
  exact hchain
