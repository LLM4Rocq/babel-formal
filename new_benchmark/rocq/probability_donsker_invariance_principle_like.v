(*
BENCHMARK_ID: TINY_MATHLIB_BATCH06_PROBABILITY_DONSKER_INVARIANCE_PRINCIPLE_LIKE
PAIR_STEM: probability_donsker_invariance_principle_like
MATH_DOMAIN: Probability
SOURCE_MATHLIB: Mathlib/Probability/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class InvarianceStruct_donsker_principle (Omega : Type) := {
  expect : (Omega -> nat) -> nat;
  finite_dimensional_axiom :
    forall X : nat -> Omega -> nat,
      forall L : Omega -> nat,
        forall B : nat -> nat,
          (forall n : nat, expect (X n) <= B n) ->
          (forall n : nat, B n <= expect L) ->
          forall n : nat, expect (X n) <= expect L;
  tightness_axiom :
    forall X : nat -> Omega -> nat,
      forall B : nat -> nat,
        (forall n : nat, expect (X n) <= B n) ->
        (forall n : nat, B n <= B (n + 1)) ->
        forall n : nat, expect (X n) <= B (n + 1);
  kolmogorov_step_axiom :
    forall X : nat -> Omega -> nat,
      forall B : nat -> nat,
        (forall n : nat, expect (X n) <= B (n + 1)) ->
        forall n : nat, expect (X n) <= B (n + 2);
  interpolation_axiom :
    forall X : nat -> Omega -> nat,
      forall I : nat -> Omega -> nat,
        forall B : nat -> nat,
          (forall n : nat, expect (I n) <= expect (X n)) ->
          (forall n : nat, expect (X n) <= B n) ->
          forall n : nat, expect (I n) <= B n;
  weak_limit_axiom :
    forall X : nat -> Omega -> nat,
      forall L : Omega -> nat,
        (forall n : nat, expect (X n) <= expect L) ->
        (forall n : nat, expect L <= expect (X n)) ->
        forall n : nat, expect (X n) = expect L;
  projection_axiom :
    forall X : nat -> Omega -> nat,
      forall P : nat -> Omega -> nat,
        (forall n : nat, expect (P n) <= expect (X n)) ->
        (forall n : nat, expect (X n) <= expect (P n)) ->
        forall n : nat, expect (P n) = expect (X n)
}.

Record ProcessData_donsker_invariance_principle
    {Omega : Type} `{InvarianceStruct_donsker_principle Omega} := {
  walk : nat -> Omega -> nat;
  bridge : nat -> Omega -> nat;
  limit : Omega -> nat;
  modulus : nat -> nat;
  walk_bound : forall n : nat, expect (walk n) <= modulus n;
  bridge_le_walk : forall n : nat, expect (bridge n) <= expect (walk n);
  modulus_mono : forall n : nat, modulus n <= modulus (n + 1)
}.

Definition rescaled_walk_donsker_invariance_principle
    {Omega : Type} `{InvarianceStruct_donsker_principle Omega}
    (P : ProcessData_donsker_invariance_principle) :
    nat -> Omega -> nat :=
  walk P.

Definition brownian_limit_donsker_invariance_principle
    {Omega : Type} `{InvarianceStruct_donsker_principle Omega}
    (P : ProcessData_donsker_invariance_principle) :
    Omega -> nat :=
  limit P.

Definition modulus_control_donsker_invariance_principle
    {Omega : Type} `{InvarianceStruct_donsker_principle Omega}
    (P : ProcessData_donsker_invariance_principle) :
    nat -> nat :=
  modulus P.

Lemma finite_dimensional_convergence_donsker_invariance_principle
    {Omega : Type} `{InvarianceStruct_donsker_principle Omega}
    (P : ProcessData_donsker_invariance_principle)
    (hCap :
      forall n : nat,
        modulus_control_donsker_invariance_principle P n <=
          expect (brownian_limit_donsker_invariance_principle P)) :
    forall n : nat,
      expect (rescaled_walk_donsker_invariance_principle P n) <=
        expect (brownian_limit_donsker_invariance_principle P).
Proof.
  assert (hWalkBound :
      forall n : nat,
        expect (rescaled_walk_donsker_invariance_principle P n) <=
          modulus_control_donsker_invariance_principle P n).
  {
    intro n.
    unfold rescaled_walk_donsker_invariance_principle.
    unfold modulus_control_donsker_invariance_principle.
    exact (walk_bound P n).
  }
  assert (hUpperBound :
      forall n : nat,
        modulus_control_donsker_invariance_principle P n <=
          expect (brownian_limit_donsker_invariance_principle P)).
  { exact hCap. }
  assert (hRaw : forall n : nat, expect (walk P n) <= expect (limit P)).
  {
    apply (finite_dimensional_axiom (walk P) (limit P) (modulus P)).
    - intro n. exact (walk_bound P n).
    - intro n. exact (hCap n).
  }
  assert (hKeep1 :
      forall n : nat,
        expect (rescaled_walk_donsker_invariance_principle P n) <=
          modulus_control_donsker_invariance_principle P n).
  { exact hWalkBound. }
  assert (hKeep2 :
      forall n : nat,
        modulus_control_donsker_invariance_principle P n <=
          expect (brownian_limit_donsker_invariance_principle P)).
  { exact hUpperBound. }
  intro n.
  unfold rescaled_walk_donsker_invariance_principle.
  unfold brownian_limit_donsker_invariance_principle.
  exact (hRaw n).
Qed.

Lemma tightness_criterion_donsker_invariance_principle
    {Omega : Type} `{InvarianceStruct_donsker_principle Omega}
    (P : ProcessData_donsker_invariance_principle) :
    forall n : nat,
      expect (rescaled_walk_donsker_invariance_principle P n) <=
        modulus_control_donsker_invariance_principle P (n + 1).
Proof.
  assert (hBoundRaw : forall n : nat, expect (walk P n) <= modulus P n).
  { exact (walk_bound P). }
  assert (hMonoRaw : forall n : nat, modulus P n <= modulus P (n + 1)).
  { exact (modulus_mono P). }
  assert (hTightRaw : forall n : nat, expect (walk P n) <= modulus P (n + 1)).
  { exact (tightness_axiom (walk P) (modulus P) hBoundRaw hMonoRaw). }
  assert (hRewrite :
      forall n : nat,
        expect (rescaled_walk_donsker_invariance_principle P n) <=
          modulus_control_donsker_invariance_principle P (n + 1)).
  {
    intro n.
    unfold rescaled_walk_donsker_invariance_principle.
    unfold modulus_control_donsker_invariance_principle.
    exact (hTightRaw n).
  }
  intro n.
  exact (hRewrite n).
Qed.

Lemma kolmogorov_bound_step_donsker_invariance_principle
    {Omega : Type} `{InvarianceStruct_donsker_principle Omega}
    (P : ProcessData_donsker_invariance_principle) :
    forall n : nat,
      expect (rescaled_walk_donsker_invariance_principle P n) <=
        modulus_control_donsker_invariance_principle P (n + 2).
Proof.
  assert (hTight :
      forall n : nat,
        expect (rescaled_walk_donsker_invariance_principle P n) <=
          modulus_control_donsker_invariance_principle P (n + 1)).
  { apply (tightness_criterion_donsker_invariance_principle P). }
  assert (hStepRaw : forall n : nat, expect (walk P n) <= modulus P (n + 2)).
  {
    apply (kolmogorov_step_axiom (walk P) (modulus P)).
    intro n.
    unfold rescaled_walk_donsker_invariance_principle in hTight.
    unfold modulus_control_donsker_invariance_principle in hTight.
    exact (hTight n).
  }
  assert (hStep :
      forall n : nat,
        expect (rescaled_walk_donsker_invariance_principle P n) <=
          modulus_control_donsker_invariance_principle P (n + 2)).
  {
    intro n.
    unfold rescaled_walk_donsker_invariance_principle.
    unfold modulus_control_donsker_invariance_principle.
    exact (hStepRaw n).
  }
  intro n.
  exact (hStep n).
Qed.

Lemma interpolation_error_donsker_invariance_principle
    {Omega : Type} `{InvarianceStruct_donsker_principle Omega}
    (P : ProcessData_donsker_invariance_principle) :
    forall n : nat,
      expect (bridge P n) <=
        modulus_control_donsker_invariance_principle P n.
Proof.
  assert (hBridgeLeWalk :
      forall n : nat,
        expect (bridge P n) <= expect (rescaled_walk_donsker_invariance_principle P n)).
  {
    intro n.
    unfold rescaled_walk_donsker_invariance_principle.
    exact (bridge_le_walk P n).
  }
  assert (hWalkLeBound :
      forall n : nat,
        expect (rescaled_walk_donsker_invariance_principle P n) <=
          modulus_control_donsker_invariance_principle P n).
  {
    intro n.
    unfold rescaled_walk_donsker_invariance_principle.
    unfold modulus_control_donsker_invariance_principle.
    exact (walk_bound P n).
  }
  assert (hRaw : forall n : nat, expect (bridge P n) <= modulus P n).
  {
    apply (interpolation_axiom (walk P) (bridge P) (modulus P)).
    - intro n. exact (bridge_le_walk P n).
    - intro n. exact (walk_bound P n).
  }
  assert (hKeep1 :
      forall n : nat,
        expect (bridge P n) <= expect (rescaled_walk_donsker_invariance_principle P n)).
  { exact hBridgeLeWalk. }
  assert (hKeep2 :
      forall n : nat,
        expect (rescaled_walk_donsker_invariance_principle P n) <=
          modulus_control_donsker_invariance_principle P n).
  { exact hWalkLeBound. }
  intro n.
  unfold modulus_control_donsker_invariance_principle.
  exact (hRaw n).
Qed.

Lemma weak_limit_identification_donsker_invariance_principle
    {Omega : Type} `{InvarianceStruct_donsker_principle Omega}
    (P : ProcessData_donsker_invariance_principle)
    (hCap :
      forall n : nat,
        modulus_control_donsker_invariance_principle P n <=
          expect (brownian_limit_donsker_invariance_principle P))
    (hLower :
      forall n : nat,
        expect (brownian_limit_donsker_invariance_principle P) <=
          expect (rescaled_walk_donsker_invariance_principle P n)) :
    forall n : nat,
      expect (rescaled_walk_donsker_invariance_principle P n) =
        expect (brownian_limit_donsker_invariance_principle P).
Proof.
  assert (hUpper :
      forall n : nat,
        expect (rescaled_walk_donsker_invariance_principle P n) <=
          expect (brownian_limit_donsker_invariance_principle P)).
  { apply (finite_dimensional_convergence_donsker_invariance_principle P hCap). }
  assert (hUpperRaw : forall n : nat, expect (walk P n) <= expect (limit P)).
  {
    intro n.
    unfold rescaled_walk_donsker_invariance_principle in hUpper.
    unfold brownian_limit_donsker_invariance_principle in hUpper.
    exact (hUpper n).
  }
  assert (hLowerRaw : forall n : nat, expect (limit P) <= expect (walk P n)).
  {
    intro n.
    unfold brownian_limit_donsker_invariance_principle in hLower.
    unfold rescaled_walk_donsker_invariance_principle in hLower.
    exact (hLower n).
  }
  assert (hEqRaw : forall n : nat, expect (walk P n) = expect (limit P)).
  { exact (weak_limit_axiom (walk P) (limit P) hUpperRaw hLowerRaw). }
  intro n.
  unfold rescaled_walk_donsker_invariance_principle.
  unfold brownian_limit_donsker_invariance_principle.
  exact (hEqRaw n).
Qed.

Lemma martingale_projection_step_donsker_invariance_principle
    {Omega : Type} `{InvarianceStruct_donsker_principle Omega}
    (P : ProcessData_donsker_invariance_principle)
    (Q : nat -> Omega -> nat)
    (hProjLe :
      forall n : nat,
        expect (Q n) <= expect (rescaled_walk_donsker_invariance_principle P n))
    (hProjGe :
      forall n : nat,
        expect (rescaled_walk_donsker_invariance_principle P n) <= expect (Q n)) :
    forall n : nat,
      expect (Q n) = expect (rescaled_walk_donsker_invariance_principle P n).
Proof.
  assert (hProjLeRaw : forall n : nat, expect (Q n) <= expect (walk P n)).
  {
    intro n.
    unfold rescaled_walk_donsker_invariance_principle in hProjLe.
    exact (hProjLe n).
  }
  assert (hProjGeRaw : forall n : nat, expect (walk P n) <= expect (Q n)).
  {
    intro n.
    unfold rescaled_walk_donsker_invariance_principle in hProjGe.
    exact (hProjGe n).
  }
  assert (hEqRaw : forall n : nat, expect (Q n) = expect (walk P n)).
  { exact (projection_axiom (walk P) Q hProjLeRaw hProjGeRaw). }
  assert (hPack :
      forall n : nat,
        expect (Q n) = expect (rescaled_walk_donsker_invariance_principle P n)).
  {
    intro n.
    unfold rescaled_walk_donsker_invariance_principle.
    exact (hEqRaw n).
  }
  intro n.
  exact (hPack n).
Qed.

Lemma invariance_principle_final_donsker_invariance_principle
    {Omega : Type} `{InvarianceStruct_donsker_principle Omega}
    (P : ProcessData_donsker_invariance_principle)
    (Q : nat -> Omega -> nat)
    (hCap :
      forall n : nat,
        modulus_control_donsker_invariance_principle P n <=
          expect (brownian_limit_donsker_invariance_principle P))
    (hLower :
      forall n : nat,
        expect (brownian_limit_donsker_invariance_principle P) <=
          expect (rescaled_walk_donsker_invariance_principle P n))
    (hProjLe :
      forall n : nat,
        expect (Q n) <= expect (rescaled_walk_donsker_invariance_principle P n))
    (hProjGe :
      forall n : nat,
        expect (rescaled_walk_donsker_invariance_principle P n) <= expect (Q n)) :
    forall n : nat,
      expect (Q n) = expect (brownian_limit_donsker_invariance_principle P).
Proof.
  assert (hWeak :
      forall n : nat,
        expect (rescaled_walk_donsker_invariance_principle P n) =
          expect (brownian_limit_donsker_invariance_principle P)).
  { apply (weak_limit_identification_donsker_invariance_principle P hCap hLower). }
  assert (hProj :
      forall n : nat,
        expect (Q n) = expect (rescaled_walk_donsker_invariance_principle P n)).
  { apply (martingale_projection_step_donsker_invariance_principle P Q hProjLe hProjGe). }
  intro n.
  assert (hEq1 : expect (Q n) = expect (rescaled_walk_donsker_invariance_principle P n)).
  { exact (hProj n). }
  assert (hEq2 :
      expect (rescaled_walk_donsker_invariance_principle P n) =
        expect (brownian_limit_donsker_invariance_principle P)).
  { exact (hWeak n). }
  rewrite hEq1.
  exact hEq2.
Qed.
