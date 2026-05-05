/-
BENCHMARK_ID: TINY_MATHLIB_BATCH04_MEASURE_DISINTEGRATION_TRANSPORT
PAIR_STEM: measure_disintegration_transport_like
MATH_DOMAIN: Measure Theory
SOURCE_MATHLIB: Mathlib/MeasureTheory/Measure/Disintegration
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u v w

class MeasurableSpaceLike (α : Type u) where
  measurable : (α → Prop) → Prop
  measurable_univ : measurable (fun _ => True)
  measurable_inter :
    ∀ {s t : α → Prop}, measurable s → measurable t → measurable (fun x => s x ∧ t x)

class MeasureLike (α : Type u) (M : Type v) [MeasurableSpaceLike α] where
  integral : M → (α → Nat) → Nat

def KernelLike {α : Type u} {β : Type v}
    [MeasurableSpaceLike α] [MeasurableSpaceLike β]
    (K : Type w) [MeasureLike β K] : Type (max u w) :=
  α → K

def PushforwardLike {α : Type u} {β : Type v}
    [MeasurableSpaceLike α] [MeasurableSpaceLike β]
    {M : Type w} {K : Type w}
    [MeasureLike α M] [MeasureLike β K]
    (μ : M) (f : α → β) (ν : K) : Prop :=
  ∀ h : β → Nat,
    MeasureLike.integral (α := β) (M := K) ν h =
      MeasureLike.integral (α := α) (M := M) μ (fun x => h (f x))

def ConditionalLike {α : Type u} {β : Type v}
    [MeasurableSpaceLike α] [MeasurableSpaceLike β]
    {M : Type w} {K : Type w}
    [MeasureLike α M] [MeasureLike β K]
    (μ : M) (π : α → β) (κ : KernelLike (α := β) (β := β) K) : Prop :=
  ∀ h : β → Nat,
    MeasureLike.integral (α := α) (M := M) μ (fun x => h (π x)) =
      MeasureLike.integral (α := α) (M := M) μ
        (fun x => MeasureLike.integral (α := β) (M := K) (κ (π x)) h)

def DisintegrationLike {α : Type u} {β : Type v}
    [MeasurableSpaceLike α] [MeasurableSpaceLike β]
    {M : Type w} {K : Type w}
    [MeasureLike α M] [MeasureLike β K]
    (μ : M) (π : α → β) (κ : KernelLike (α := β) (β := β) K) : Prop :=
  ConditionalLike μ π κ ∧
    (∀ h : β → Nat,
      MeasurableSpaceLike.measurable (α := β)
        (fun x => MeasureLike.integral (α := β) (M := K) (κ x) h = 0))

theorem disintegration_spec_like {α : Type u} {β : Type v}
    [MeasurableSpaceLike α] [MeasurableSpaceLike β]
    {M : Type w} {K : Type w}
    [MeasureLike α M] [MeasureLike β K]
    (μ : M) (π : α → β) (κ : KernelLike (α := β) (β := β) K)
    (hdis : DisintegrationLike μ π κ) :
    ConditionalLike μ π κ ∧
      (∀ h : β → Nat,
        MeasurableSpaceLike.measurable (α := β)
          (fun x => MeasureLike.integral (α := β) (M := K) (κ x) h = 0)) := by
  rcases hdis with ⟨hcond, hmeas⟩
  constructor
  · exact hcond
  · intro h
    have hraw :
        MeasurableSpaceLike.measurable (α := β)
          (fun x => MeasureLike.integral (α := β) (M := K) (κ x) h = 0) :=
      hmeas h
    exact hraw

theorem disintegration_unique_like {α : Type u} {β : Type v}
    [MeasurableSpaceLike α] [MeasurableSpaceLike β]
    {M : Type w} {K : Type w}
    [MeasureLike α M] [MeasureLike β K]
    (μ : M) (π : α → β)
    (κ₁ κ₂ : KernelLike (α := β) (β := β) K)
    (h₁ : ConditionalLike μ π κ₁)
    (h₂ : ConditionalLike μ π κ₂)
    (hsep :
      ∀ k1 k2 : KernelLike (α := β) (β := β) K,
        (∀ h : β → Nat, ∀ b : β,
          MeasureLike.integral (α := β) (M := K) (k1 b) h =
            MeasureLike.integral (α := β) (M := K) (k2 b) h) →
        k1 = k2)
    (hfiber :
      ∀ h : β → Nat, ∀ b : β,
        MeasureLike.integral (α := β) (M := K) (κ₁ b) h =
          MeasureLike.integral (α := β) (M := K) (κ₂ b) h) :
    κ₁ = κ₂ := by
  have hcond₁ :
      ∀ h : β → Nat,
        MeasureLike.integral (α := α) (M := M) μ (fun x => h (π x)) =
          MeasureLike.integral (α := α) (M := M) μ
            (fun x => MeasureLike.integral (α := β) (M := K) (κ₁ (π x)) h) := h₁
  have hcond₂ :
      ∀ h : β → Nat,
        MeasureLike.integral (α := α) (M := M) μ (fun x => h (π x)) =
          MeasureLike.integral (α := α) (M := M) μ
            (fun x => MeasureLike.integral (α := β) (M := K) (κ₂ (π x)) h) := h₂
  have hsep_ready :
      ∀ h : β → Nat, ∀ b : β,
        MeasureLike.integral (α := β) (M := K) (κ₁ b) h =
          MeasureLike.integral (α := β) (M := K) (κ₂ b) h := by
    intro h b
    exact hfiber h b
  have _ : ∀ h : β → Nat,
      MeasureLike.integral (α := α) (M := M) μ (fun x => h (π x)) =
        MeasureLike.integral (α := α) (M := M) μ
          (fun x => MeasureLike.integral (α := β) (M := K) (κ₁ (π x)) h) := hcond₁
  have _ : ∀ h : β → Nat,
      MeasureLike.integral (α := α) (M := M) μ (fun x => h (π x)) =
        MeasureLike.integral (α := α) (M := M) μ
          (fun x => MeasureLike.integral (α := β) (M := K) (κ₂ (π x)) h) := hcond₂
  exact hsep κ₁ κ₂ hsep_ready

theorem transport_kernel_like {α : Type u} {β : Type v}
    [MeasurableSpaceLike α] [MeasurableSpaceLike β]
    {M : Type w} {K : Type w}
    [MeasureLike α M] [MeasureLike β K]
    (μ : M) (π : α → β)
    (κ κt : KernelLike (α := β) (β := β) K)
    (hcond : ConditionalLike μ π κ)
    (htransport_int :
      ∀ h : β → Nat,
        MeasureLike.integral (α := α) (M := M) μ
          (fun x => MeasureLike.integral (α := β) (M := K) (κt (π x)) h) =
        MeasureLike.integral (α := α) (M := M) μ
          (fun x => MeasureLike.integral (α := β) (M := K) (κ (π x)) h)) :
    ConditionalLike μ π κt := by
  intro h
  have hbase :
      MeasureLike.integral (α := α) (M := M) μ (fun x => h (π x)) =
        MeasureLike.integral (α := α) (M := M) μ
          (fun x => MeasureLike.integral (α := β) (M := K) (κ (π x)) h) :=
    hcond h
  have htr :
      MeasureLike.integral (α := α) (M := M) μ
        (fun x => MeasureLike.integral (α := β) (M := K) (κt (π x)) h) =
      MeasureLike.integral (α := α) (M := M) μ
        (fun x => MeasureLike.integral (α := β) (M := K) (κ (π x)) h) :=
    htransport_int h
  calc
    MeasureLike.integral (α := α) (M := M) μ (fun x => h (π x))
        = MeasureLike.integral (α := α) (M := M) μ
            (fun x => MeasureLike.integral (α := β) (M := K) (κ (π x)) h) := hbase
    _ = MeasureLike.integral (α := α) (M := M) μ
            (fun x => MeasureLike.integral (α := β) (M := K) (κt (π x)) h) := by
          symm
          exact htr

theorem fubini_disintegrated_like {α : Type u} {β : Type v}
    [MeasurableSpaceLike α] [MeasurableSpaceLike β]
    {M : Type w} {K : Type w}
    [MeasureLike α M] [MeasureLike β K]
    (μ : M) (ν : K) (π : α → β)
    (κ : KernelLike (α := β) (β := β) K)
    (hpush : PushforwardLike μ π ν)
    (hcond : ConditionalLike μ π κ) :
    ∀ h : β → Nat,
      MeasureLike.integral (α := β) (M := K) ν h =
        MeasureLike.integral (α := α) (M := M) μ
          (fun x => MeasureLike.integral (α := β) (M := K) (κ (π x)) h) := by
  intro h
  have hν :
      MeasureLike.integral (α := β) (M := K) ν h =
        MeasureLike.integral (α := α) (M := M) μ (fun x => h (π x)) :=
    hpush h
  have hκ :
      MeasureLike.integral (α := α) (M := M) μ (fun x => h (π x)) =
        MeasureLike.integral (α := α) (M := M) μ
          (fun x => MeasureLike.integral (α := β) (M := K) (κ (π x)) h) :=
    hcond h
  calc
    MeasureLike.integral (α := β) (M := K) ν h
        = MeasureLike.integral (α := α) (M := M) μ (fun x => h (π x)) := hν
    _ = MeasureLike.integral (α := α) (M := M) μ
          (fun x => MeasureLike.integral (α := β) (M := K) (κ (π x)) h) := hκ

theorem measurability_section_like {α : Type u} {β : Type v}
    [MeasurableSpaceLike α] [MeasurableSpaceLike β]
    {M : Type w} {K : Type w}
    [MeasureLike α M] [MeasureLike β K]
    (μ : M) (π : α → β)
    (κ : KernelLike (α := β) (β := β) K)
    (hdis : DisintegrationLike μ π κ) :
    ∀ h : β → Nat,
      MeasurableSpaceLike.measurable (α := β)
        (fun x => MeasureLike.integral (α := β) (M := K) (κ x) h = 0) := by
  rcases hdis with ⟨hcond, hmeas⟩
  intro h
  have _ : ConditionalLike μ π κ := hcond
  have hraw :
      MeasurableSpaceLike.measurable (α := β)
        (fun x => MeasureLike.integral (α := β) (M := K) (κ x) h = 0) :=
    hmeas h
  exact hraw

theorem barycenter_formula_like {α : Type u} {β : Type v}
    [MeasurableSpaceLike α] [MeasurableSpaceLike β]
    {M : Type w} {K : Type w}
    [MeasureLike α M] [MeasureLike β K]
    (μ : M) (ν : K) (π : α → β)
    (κ : KernelLike (α := β) (β := β) K)
    (hpush : PushforwardLike μ π ν)
    (hcond : ConditionalLike μ π κ)
    (hbar :
      ∀ h : β → Nat,
        MeasureLike.integral (α := β) (M := K) ν h =
          MeasureLike.integral (α := α) (M := M) μ
            (fun x => MeasureLike.integral (α := β) (M := K) (κ (π x)) h)) :
    ∀ h : β → Nat,
      MeasureLike.integral (α := β) (M := K) ν h =
        MeasureLike.integral (α := α) (M := M) μ
          (fun x => MeasureLike.integral (α := β) (M := K) (κ (π x)) h) := by
  intro h
  have hleft :
      MeasureLike.integral (α := β) (M := K) ν h =
        MeasureLike.integral (α := α) (M := M) μ (fun x => h (π x)) :=
    hpush h
  have hright :
      MeasureLike.integral (α := α) (M := M) μ (fun x => h (π x)) =
        MeasureLike.integral (α := α) (M := M) μ
          (fun x => MeasureLike.integral (α := β) (M := K) (κ (π x)) h) :=
    hcond h
  have htarget :
      MeasureLike.integral (α := β) (M := K) ν h =
        MeasureLike.integral (α := α) (M := M) μ
          (fun x => MeasureLike.integral (α := β) (M := K) (κ (π x)) h) :=
    hbar h
  have _ :
      MeasureLike.integral (α := β) (M := K) ν h =
        MeasureLike.integral (α := α) (M := M) μ
          (fun x => MeasureLike.integral (α := β) (M := K) (κ (π x)) h) := htarget
  calc
    MeasureLike.integral (α := β) (M := K) ν h
        = MeasureLike.integral (α := α) (M := M) μ (fun x => h (π x)) := hleft
    _ = MeasureLike.integral (α := α) (M := M) μ
          (fun x => MeasureLike.integral (α := β) (M := K) (κ (π x)) h) := hright
