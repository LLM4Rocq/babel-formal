/-
BENCHMARK_ID: TINY_MATHLIB_BATCH05_NUMBER_THEORY_PADIC_HODGE_FILTRATION_LIKE
PAIR_STEM: number_theory_padic_hodge_filtration_like
MATH_DOMAIN: Number Theory
SOURCE_MATHLIB: Mathlib/NumberTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class NumberStruct_theory_padic_hodge (K : Type u) where
  GaloisRep : Type u
  Selmer : Type u
  Height : Selmer -> Nat
  Density : GaloisRep -> Nat
  rep_of_selmer : Selmer -> GaloisRep
  modular_lift_axiom :
    forall rho : GaloisRep,
      Density rho <= Density rho + 1
  control_axiom :
    forall s : Selmer,
      Density (rep_of_selmer s) <= Height s + 2
  filtration_axiom :
    forall (s : Selmer) (n : Nat),
      n <= Height s ->
      Height s = n + (Height s - n)
  height_step_axiom :
    forall s t : Selmer,
      Height s <= Height t ->
      Height s + Density (rep_of_selmer s) <= Height t + Density (rep_of_selmer s)
  chebotarev_axiom :
    forall (rho : GaloisRep) (n : Nat),
      Density rho <= n ->
      Density rho + Density rho <= n + n
  transfer_axiom :
    forall s t : Selmer,
      Height s <= Height t ->
      Density (rep_of_selmer s) <= Density (rep_of_selmer t) + Height t
  finiteness_axiom :
    forall s : Selmer,
      exists n : Nat,
        Height s <= n /\ Density (rep_of_selmer s) <= n

def GaloisRep_theory_padic_hodge
    (K : Type u) [h : NumberStruct_theory_padic_hodge K] : Type u :=
  h.GaloisRep

def SelmerObj_theory_padic_hodge
    (K : Type u) [h : NumberStruct_theory_padic_hodge K] : Type u :=
  h.Selmer

def HeightObj_theory_padic_hodge
    (K : Type u) [h : NumberStruct_theory_padic_hodge K]
    (s : SelmerObj_theory_padic_hodge K) : Nat :=
  h.Height s

def DensityObj_theory_padic_hodge
    (K : Type u) [h : NumberStruct_theory_padic_hodge K]
    (rho : GaloisRep_theory_padic_hodge K) : Nat :=
  h.Density rho

theorem modular_lift_step_theory_padic_hodge
    {K : Type u} [h : NumberStruct_theory_padic_hodge K]
    (rho : GaloisRep_theory_padic_hodge K) :
    DensityObj_theory_padic_hodge K rho <= DensityObj_theory_padic_hodge K rho + 1 := by
  have hLift : h.Density rho <= h.Density rho + 1 := h.modular_lift_axiom rho
  have hFinal : DensityObj_theory_padic_hodge K rho <= DensityObj_theory_padic_hodge K rho + 1 := hLift
  exact hFinal

theorem control_theorem_step_theory_padic_hodge
    {K : Type u} [h : NumberStruct_theory_padic_hodge K]
    (s : SelmerObj_theory_padic_hodge K) :
    DensityObj_theory_padic_hodge K (h.rep_of_selmer s) <= HeightObj_theory_padic_hodge K s + 2 := by
  have hCtrl : h.Density (h.rep_of_selmer s) <= h.Height s + 2 := h.control_axiom s
  have hAsWritten :
      DensityObj_theory_padic_hodge K (h.rep_of_selmer s) <= HeightObj_theory_padic_hodge K s + 2 := hCtrl
  exact hAsWritten

theorem hodge_filtration_step_theory_padic_hodge
    {K : Type u} [h : NumberStruct_theory_padic_hodge K]
    (s : SelmerObj_theory_padic_hodge K)
    (n : Nat)
    (hn : n <= HeightObj_theory_padic_hodge K s) :
    HeightObj_theory_padic_hodge K s = n + (HeightObj_theory_padic_hodge K s - n) := by
  have hFil : h.Height s = n + (h.Height s - n) := h.filtration_axiom s n hn
  have hCopy : HeightObj_theory_padic_hodge K s = n + (HeightObj_theory_padic_hodge K s - n) := hFil
  exact hCopy

theorem height_inequality_step_theory_padic_hodge
    {K : Type u} [h : NumberStruct_theory_padic_hodge K]
    (s t : SelmerObj_theory_padic_hodge K)
    (hst : HeightObj_theory_padic_hodge K s <= HeightObj_theory_padic_hodge K t) :
    HeightObj_theory_padic_hodge K s + DensityObj_theory_padic_hodge K (h.rep_of_selmer s) <=
      HeightObj_theory_padic_hodge K t + DensityObj_theory_padic_hodge K (h.rep_of_selmer s) := by
  have hStep :
      h.Height s + h.Density (h.rep_of_selmer s) <= h.Height t + h.Density (h.rep_of_selmer s) :=
    h.height_step_axiom s t hst
  have hFinal :
      HeightObj_theory_padic_hodge K s + DensityObj_theory_padic_hodge K (h.rep_of_selmer s) <=
        HeightObj_theory_padic_hodge K t + DensityObj_theory_padic_hodge K (h.rep_of_selmer s) := hStep
  exact hFinal

theorem chebotarev_count_step_theory_padic_hodge
    {K : Type u} [h : NumberStruct_theory_padic_hodge K]
    (rho : GaloisRep_theory_padic_hodge K)
    (n : Nat)
    (hrho : DensityObj_theory_padic_hodge K rho <= n) :
    DensityObj_theory_padic_hodge K rho + DensityObj_theory_padic_hodge K rho <= n + n := by
  have hCount : h.Density rho + h.Density rho <= n + n := h.chebotarev_axiom rho n hrho
  have hFinal : DensityObj_theory_padic_hodge K rho + DensityObj_theory_padic_hodge K rho <= n + n := hCount
  exact hFinal

theorem arithmetic_transfer_theory_padic_hodge
    {K : Type u} [h : NumberStruct_theory_padic_hodge K]
    (s t : SelmerObj_theory_padic_hodge K)
    (hst : HeightObj_theory_padic_hodge K s <= HeightObj_theory_padic_hodge K t) :
    DensityObj_theory_padic_hodge K (h.rep_of_selmer s) <=
      DensityObj_theory_padic_hodge K (h.rep_of_selmer t) + HeightObj_theory_padic_hodge K t := by
  have hTransfer :
      h.Density (h.rep_of_selmer s) <= h.Density (h.rep_of_selmer t) + h.Height t :=
    h.transfer_axiom s t hst
  have hFinal :
      DensityObj_theory_padic_hodge K (h.rep_of_selmer s) <=
        DensityObj_theory_padic_hodge K (h.rep_of_selmer t) + HeightObj_theory_padic_hodge K t := hTransfer
  exact hFinal

theorem global_finiteness_theory_padic_hodge
    {K : Type u} [h : NumberStruct_theory_padic_hodge K]
    (s : SelmerObj_theory_padic_hodge K) :
    exists n : Nat,
      HeightObj_theory_padic_hodge K s <= n /\
      DensityObj_theory_padic_hodge K (h.rep_of_selmer s) <= n := by
  rcases h.finiteness_axiom s with ⟨n, hnHeight, hnDensity⟩
  have hHeight : HeightObj_theory_padic_hodge K s <= n := hnHeight
  have hDensity : DensityObj_theory_padic_hodge K (h.rep_of_selmer s) <= n := hnDensity
  exact ⟨n, hHeight, hDensity⟩
