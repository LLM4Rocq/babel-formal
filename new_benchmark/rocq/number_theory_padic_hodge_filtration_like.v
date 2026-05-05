(*
BENCHMARK_ID: TINY_MATHLIB_BATCH05_NUMBER_THEORY_PADIC_HODGE_FILTRATION_LIKE
PAIR_STEM: number_theory_padic_hodge_filtration_like
MATH_DOMAIN: Number Theory
SOURCE_MATHLIB: Mathlib/NumberTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.

Class NumberStruct_theory_padic_hodge (K : Type) := {
  GaloisRep : Type;
  Selmer : Type;
  Height : Selmer -> nat;
  Density : GaloisRep -> nat;
  rep_of_selmer : Selmer -> GaloisRep;
  modular_lift_axiom :
    forall rho : GaloisRep,
      Density rho <= Density rho + 1;
  control_axiom :
    forall s : Selmer,
      Density (rep_of_selmer s) <= Height s + 2;
  filtration_axiom :
    forall (s : Selmer) (n : nat),
      n <= Height s ->
      Height s = n + (Height s - n);
  height_step_axiom :
    forall s t : Selmer,
      Height s <= Height t ->
      Height s + Density (rep_of_selmer s) <= Height t + Density (rep_of_selmer s);
  chebotarev_axiom :
    forall (rho : GaloisRep) (n : nat),
      Density rho <= n ->
      Density rho + Density rho <= n + n;
  transfer_axiom :
    forall s t : Selmer,
      Height s <= Height t ->
      Density (rep_of_selmer s) <= Density (rep_of_selmer t) + Height t;
  finiteness_axiom :
    forall s : Selmer,
      exists n : nat,
        Height s <= n /\ Density (rep_of_selmer s) <= n
}.

Definition GaloisRep_theory_padic_hodge
    (K : Type) `{NumberStruct_theory_padic_hodge K} : Type :=
  GaloisRep.

Definition SelmerObj_theory_padic_hodge
    (K : Type) `{NumberStruct_theory_padic_hodge K} : Type :=
  Selmer.

Definition HeightObj_theory_padic_hodge
    (K : Type) `{NumberStruct_theory_padic_hodge K}
    (s : SelmerObj_theory_padic_hodge K) : nat :=
  Height s.

Definition DensityObj_theory_padic_hodge
    (K : Type) `{NumberStruct_theory_padic_hodge K}
    (rho : GaloisRep_theory_padic_hodge K) : nat :=
  Density rho.

Lemma modular_lift_step_theory_padic_hodge
    {K : Type} `{NumberStruct_theory_padic_hodge K}
    (rho : GaloisRep_theory_padic_hodge K) :
    DensityObj_theory_padic_hodge K rho <= DensityObj_theory_padic_hodge K rho + 1.
Proof.
  assert (hLift : Density rho <= Density rho + 1).
  { apply modular_lift_axiom. }
  assert (hFinal : DensityObj_theory_padic_hodge K rho <= DensityObj_theory_padic_hodge K rho + 1).
  { exact hLift. }
  exact hFinal.
Qed.

Lemma control_theorem_step_theory_padic_hodge
    {K : Type} `{NumberStruct_theory_padic_hodge K}
    (s : SelmerObj_theory_padic_hodge K) :
    DensityObj_theory_padic_hodge K (rep_of_selmer s) <= HeightObj_theory_padic_hodge K s + 2.
Proof.
  assert (hCtrl : Density (rep_of_selmer s) <= Height s + 2).
  { apply control_axiom. }
  assert (hAsWritten : DensityObj_theory_padic_hodge K (rep_of_selmer s) <= HeightObj_theory_padic_hodge K s + 2).
  { exact hCtrl. }
  exact hAsWritten.
Qed.

Lemma hodge_filtration_step_theory_padic_hodge
    {K : Type} `{NumberStruct_theory_padic_hodge K}
    (s : SelmerObj_theory_padic_hodge K)
    (n : nat)
    (hn : n <= HeightObj_theory_padic_hodge K s) :
    HeightObj_theory_padic_hodge K s = n + (HeightObj_theory_padic_hodge K s - n).
Proof.
  assert (hFil : Height s = n + (Height s - n)).
  { apply filtration_axiom. exact hn. }
  assert (hCopy : HeightObj_theory_padic_hodge K s = n + (HeightObj_theory_padic_hodge K s - n)).
  { exact hFil. }
  exact hCopy.
Qed.

Lemma height_inequality_step_theory_padic_hodge
    {K : Type} `{NumberStruct_theory_padic_hodge K}
    (s t : SelmerObj_theory_padic_hodge K)
    (hst : HeightObj_theory_padic_hodge K s <= HeightObj_theory_padic_hodge K t) :
    HeightObj_theory_padic_hodge K s + DensityObj_theory_padic_hodge K (rep_of_selmer s) <=
      HeightObj_theory_padic_hodge K t + DensityObj_theory_padic_hodge K (rep_of_selmer s).
Proof.
  assert (hStep : Height s + Density (rep_of_selmer s) <= Height t + Density (rep_of_selmer s)).
  { apply height_step_axiom. exact hst. }
  assert (hFinal :
      HeightObj_theory_padic_hodge K s + DensityObj_theory_padic_hodge K (rep_of_selmer s) <=
        HeightObj_theory_padic_hodge K t + DensityObj_theory_padic_hodge K (rep_of_selmer s)).
  { exact hStep. }
  exact hFinal.
Qed.

Lemma chebotarev_count_step_theory_padic_hodge
    {K : Type} `{NumberStruct_theory_padic_hodge K}
    (rho : GaloisRep_theory_padic_hodge K)
    (n : nat)
    (hrho : DensityObj_theory_padic_hodge K rho <= n) :
    DensityObj_theory_padic_hodge K rho + DensityObj_theory_padic_hodge K rho <= n + n.
Proof.
  assert (hCount : Density rho + Density rho <= n + n).
  { apply chebotarev_axiom. exact hrho. }
  assert (hFinal : DensityObj_theory_padic_hodge K rho + DensityObj_theory_padic_hodge K rho <= n + n).
  { exact hCount. }
  exact hFinal.
Qed.

Lemma arithmetic_transfer_theory_padic_hodge
    {K : Type} `{NumberStruct_theory_padic_hodge K}
    (s t : SelmerObj_theory_padic_hodge K)
    (hst : HeightObj_theory_padic_hodge K s <= HeightObj_theory_padic_hodge K t) :
    DensityObj_theory_padic_hodge K (rep_of_selmer s) <=
      DensityObj_theory_padic_hodge K (rep_of_selmer t) + HeightObj_theory_padic_hodge K t.
Proof.
  assert (hTransfer : Density (rep_of_selmer s) <= Density (rep_of_selmer t) + Height t).
  { apply transfer_axiom. exact hst. }
  assert (hFinal : DensityObj_theory_padic_hodge K (rep_of_selmer s) <=
      DensityObj_theory_padic_hodge K (rep_of_selmer t) + HeightObj_theory_padic_hodge K t).
  { exact hTransfer. }
  exact hFinal.
Qed.

Lemma global_finiteness_theory_padic_hodge
    {K : Type} `{NumberStruct_theory_padic_hodge K}
    (s : SelmerObj_theory_padic_hodge K) :
    exists n : nat,
      HeightObj_theory_padic_hodge K s <= n /\
      DensityObj_theory_padic_hodge K (rep_of_selmer s) <= n.
Proof.
  destruct (finiteness_axiom s) as [n [hnHeight hnDensity]].
  assert (hHeight : HeightObj_theory_padic_hodge K s <= n).
  { exact hnHeight. }
  assert (hDensity : DensityObj_theory_padic_hodge K (rep_of_selmer s) <= n).
  { exact hnDensity. }
  exact (ex_intro _ n (conj hHeight hDensity)).
Qed.
