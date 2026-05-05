/-
BENCHMARK_ID: TINY_MATHLIB_BATCH06_ANALYSIS_POINCARE_SOBOLEV_INEQUALITY_LIKE
PAIR_STEM: analysis_poincare_sobolev_inequality_like
MATH_DOMAIN: Analysis
SOURCE_MATHLIB: Mathlib/Analysis/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class SobolevStruct_poincare_inequality (E : Type u) where
  mean : E → Nat
  gradient : E → Nat
  oscillation : E → Nat
  energy : E → Nat
  localPatch : E → E
  renorm : E → E
  le_trans_nat : ∀ a b c : Nat, a ≤ b → b ≤ c → a ≤ c
  add_le_add_left_nat : ∀ a b c : Nat, a ≤ b → c + a ≤ c + b
  add_le_add_right_nat : ∀ a b c : Nat, a ≤ b → a + c ≤ b + c
  add_le_add_nat : ∀ a b c d : Nat, a ≤ b → c ≤ d → a + c ≤ b + d
  le_add_right_nat : ∀ a b : Nat, a ≤ a + b
  mean_localPatch_zero : ∀ x : E, mean (localPatch x) = 0
  gradient_localPatch_bound : ∀ x : E, gradient (localPatch x) ≤ gradient x + mean x
  oscillation_localPatch_bound : ∀ x : E, oscillation (localPatch x) ≤ oscillation x + mean x
  poincare_step : ∀ x : E, oscillation x ≤ gradient x + mean x
  holder_step : ∀ x : E, energy (renorm x) ≤ energy x + gradient x
  coercive_step : ∀ x : E, gradient (renorm x) ≤ gradient x + oscillation x
  compact_step : ∀ x : E, oscillation (renorm x) ≤ energy (renorm x)

structure EnergyData_poincare_sobolev_inequality (E : Type u) [h : SobolevStruct_poincare_inequality E] where
  state : E
  scale : Nat
  scale_pos : 0 < scale
  mean_le_scale : h.mean state ≤ scale
  gradient_le_scaled : h.gradient state ≤ scale + h.gradient (h.localPatch state)

def mean_operator_poincare_sobolev_inequality
    {E : Type u} [h : SobolevStruct_poincare_inequality E]
    (d : EnergyData_poincare_sobolev_inequality E) : Nat :=
  h.mean d.state

def gradient_norm_poincare_sobolev_inequality
    {E : Type u} [h : SobolevStruct_poincare_inequality E]
    (d : EnergyData_poincare_sobolev_inequality E) : Nat :=
  h.gradient d.state + d.scale

def oscillation_norm_poincare_sobolev_inequality
    {E : Type u} [h : SobolevStruct_poincare_inequality E]
    (d : EnergyData_poincare_sobolev_inequality E) : Nat :=
  h.oscillation d.state + h.mean d.state

theorem mean_zero_reduction_poincare_sobolev_inequality
    {E : Type u} [h : SobolevStruct_poincare_inequality E]
    (d : EnergyData_poincare_sobolev_inequality E) :
    h.mean (h.localPatch d.state) = 0 ∧
    h.gradient (h.localPatch d.state) ≤ gradient_norm_poincare_sobolev_inequality (E := E) d := by
  have hMeanZero : h.mean (h.localPatch d.state) = 0 :=
    h.mean_localPatch_zero d.state
  have hGradPatch :
      h.gradient (h.localPatch d.state) ≤ h.gradient d.state + h.mean d.state :=
    h.gradient_localPatch_bound d.state
  have hMeanLift :
      h.gradient d.state + h.mean d.state ≤ h.gradient d.state + d.scale :=
    h.add_le_add_left_nat (h.mean d.state) d.scale (h.gradient d.state) d.mean_le_scale
  have hGradScaled : h.gradient (h.localPatch d.state) ≤ h.gradient d.state + d.scale :=
    h.le_trans_nat _ _ _ hGradPatch hMeanLift
  have hGradFinal :
      h.gradient (h.localPatch d.state) ≤ gradient_norm_poincare_sobolev_inequality (E := E) d := by
    simpa [gradient_norm_poincare_sobolev_inequality] using hGradScaled
  exact And.intro hMeanZero hGradFinal

theorem local_patch_bound_poincare_sobolev_inequality
    {E : Type u} [h : SobolevStruct_poincare_inequality E]
    (d : EnergyData_poincare_sobolev_inequality E) :
    h.oscillation (h.localPatch d.state) ≤ oscillation_norm_poincare_sobolev_inequality (E := E) d ∧
    h.gradient (h.localPatch d.state) ≤ h.gradient d.state + d.scale := by
  have hOscPatch :
      h.oscillation (h.localPatch d.state) ≤ h.oscillation d.state + h.mean d.state :=
    h.oscillation_localPatch_bound d.state
  have hOscFinal :
      h.oscillation (h.localPatch d.state) ≤ oscillation_norm_poincare_sobolev_inequality (E := E) d := by
    simpa [oscillation_norm_poincare_sobolev_inequality] using hOscPatch
  have hGradPatch :
      h.gradient (h.localPatch d.state) ≤ h.gradient d.state + h.mean d.state :=
    h.gradient_localPatch_bound d.state
  have hMeanLift :
      h.gradient d.state + h.mean d.state ≤ h.gradient d.state + d.scale :=
    h.add_le_add_left_nat (h.mean d.state) d.scale (h.gradient d.state) d.mean_le_scale
  have hGradFinal : h.gradient (h.localPatch d.state) ≤ h.gradient d.state + d.scale :=
    h.le_trans_nat _ _ _ hGradPatch hMeanLift
  exact And.intro hOscFinal hGradFinal

theorem interpolation_step_poincare_sobolev_inequality
    {E : Type u} [h : SobolevStruct_poincare_inequality E]
    (x : E) :
    h.oscillation (h.localPatch x) ≤ h.gradient x + h.mean x + h.mean x := by
  have hPatch :
      h.oscillation (h.localPatch x) ≤ h.oscillation x + h.mean x :=
    h.oscillation_localPatch_bound x
  have hPoincare : h.oscillation x ≤ h.gradient x + h.mean x :=
    h.poincare_step x
  have hLift :
      h.oscillation x + h.mean x ≤ (h.gradient x + h.mean x) + h.mean x :=
    h.add_le_add_right_nat (h.oscillation x) (h.gradient x + h.mean x) (h.mean x) hPoincare
  have hChain :
      h.oscillation (h.localPatch x) ≤ (h.gradient x + h.mean x) + h.mean x :=
    h.le_trans_nat _ _ _ hPatch hLift
  simpa using hChain

theorem holder_chain_poincare_sobolev_inequality
    {E : Type u} [h : SobolevStruct_poincare_inequality E]
    (d : EnergyData_poincare_sobolev_inequality E) :
    h.energy (h.renorm d.state) ≤ h.energy d.state + gradient_norm_poincare_sobolev_inequality (E := E) d := by
  have hHolder :
      h.energy (h.renorm d.state) ≤ h.energy d.state + h.gradient d.state :=
    h.holder_step d.state
  have hGradLift : h.gradient d.state ≤ h.gradient d.state + d.scale :=
    h.le_add_right_nat (h.gradient d.state) d.scale
  have hEnergyLift :
      h.energy d.state + h.gradient d.state ≤ h.energy d.state + (h.gradient d.state + d.scale) :=
    h.add_le_add_left_nat (h.gradient d.state) (h.gradient d.state + d.scale) (h.energy d.state) hGradLift
  have hChain :
      h.energy (h.renorm d.state) ≤ h.energy d.state + (h.gradient d.state + d.scale) :=
    h.le_trans_nat _ _ _ hHolder hEnergyLift
  simpa [gradient_norm_poincare_sobolev_inequality] using hChain

theorem coercive_estimate_poincare_sobolev_inequality
    {E : Type u} [h : SobolevStruct_poincare_inequality E]
    (d : EnergyData_poincare_sobolev_inequality E) :
    h.gradient (h.renorm d.state) ≤
      gradient_norm_poincare_sobolev_inequality (E := E) d +
      oscillation_norm_poincare_sobolev_inequality (E := E) d ∧
    h.gradient d.state ≤ gradient_norm_poincare_sobolev_inequality (E := E) d := by
  have hCoercive :
      h.gradient (h.renorm d.state) ≤ h.gradient d.state + h.oscillation d.state :=
    h.coercive_step d.state
  have hGradLift : h.gradient d.state ≤ h.gradient d.state + d.scale :=
    h.le_add_right_nat (h.gradient d.state) d.scale
  have hOscLift : h.oscillation d.state ≤ h.oscillation d.state + h.mean d.state :=
    h.le_add_right_nat (h.oscillation d.state) (h.mean d.state)
  have hPairLift :
      h.gradient d.state + h.oscillation d.state ≤
        (h.gradient d.state + d.scale) + (h.oscillation d.state + h.mean d.state) :=
    h.add_le_add_nat
      (h.gradient d.state) (h.gradient d.state + d.scale)
      (h.oscillation d.state) (h.oscillation d.state + h.mean d.state)
      hGradLift hOscLift
  have hChain :
      h.gradient (h.renorm d.state) ≤
        (h.gradient d.state + d.scale) + (h.oscillation d.state + h.mean d.state) :=
    h.le_trans_nat _ _ _ hCoercive hPairLift
  have hGradBase : h.gradient d.state ≤ gradient_norm_poincare_sobolev_inequality (E := E) d := by
    simpa [gradient_norm_poincare_sobolev_inequality] using
      h.le_add_right_nat (h.gradient d.state) d.scale
  have hFirst :
      h.gradient (h.renorm d.state) ≤
        gradient_norm_poincare_sobolev_inequality (E := E) d +
          oscillation_norm_poincare_sobolev_inequality (E := E) d := by
    simpa [gradient_norm_poincare_sobolev_inequality, oscillation_norm_poincare_sobolev_inequality]
      using hChain
  exact And.intro hFirst hGradBase

theorem compact_embedding_step_poincare_sobolev_inequality
    {E : Type u} [h : SobolevStruct_poincare_inequality E]
    (d : EnergyData_poincare_sobolev_inequality E) :
    h.oscillation (h.renorm d.state) ≤ h.energy d.state + gradient_norm_poincare_sobolev_inequality (E := E) d := by
  have hCompact : h.oscillation (h.renorm d.state) ≤ h.energy (h.renorm d.state) :=
    h.compact_step d.state
  have hHolder :
      h.energy (h.renorm d.state) ≤ h.energy d.state + gradient_norm_poincare_sobolev_inequality (E := E) d :=
    holder_chain_poincare_sobolev_inequality (E := E) (h := h) d
  have hChain :
      h.oscillation (h.renorm d.state) ≤ h.energy d.state + gradient_norm_poincare_sobolev_inequality (E := E) d :=
    h.le_trans_nat _ _ _ hCompact hHolder
  exact hChain

theorem global_sobolev_bound_poincare_sobolev_inequality
    {E : Type u} [h : SobolevStruct_poincare_inequality E]
    (d : EnergyData_poincare_sobolev_inequality E) :
    h.oscillation (h.renorm d.state) ≤
      (h.energy d.state + gradient_norm_poincare_sobolev_inequality (E := E) d) +
        oscillation_norm_poincare_sobolev_inequality (E := E) d ∧
    h.gradient (h.localPatch d.state) ≤ gradient_norm_poincare_sobolev_inequality (E := E) d := by
  have hCompact :
      h.oscillation (h.renorm d.state) ≤ h.energy d.state + gradient_norm_poincare_sobolev_inequality (E := E) d :=
    compact_embedding_step_poincare_sobolev_inequality (E := E) (h := h) d
  have hAddOsc :
      h.energy d.state + gradient_norm_poincare_sobolev_inequality (E := E) d ≤
        (h.energy d.state + gradient_norm_poincare_sobolev_inequality (E := E) d) +
          oscillation_norm_poincare_sobolev_inequality (E := E) d :=
    h.le_add_right_nat
      (h.energy d.state + gradient_norm_poincare_sobolev_inequality (E := E) d)
      (oscillation_norm_poincare_sobolev_inequality (E := E) d)
  have hFirst :
      h.oscillation (h.renorm d.state) ≤
        (h.energy d.state + gradient_norm_poincare_sobolev_inequality (E := E) d) +
          oscillation_norm_poincare_sobolev_inequality (E := E) d :=
    h.le_trans_nat _ _ _ hCompact hAddOsc
  have hReduction :
      h.mean (h.localPatch d.state) = 0 ∧
      h.gradient (h.localPatch d.state) ≤ gradient_norm_poincare_sobolev_inequality (E := E) d :=
    mean_zero_reduction_poincare_sobolev_inequality (E := E) (h := h) d
  exact And.intro hFirst hReduction.right
