(*
BENCHMARK_ID: TINY_MATHLIB_BATCH04_ANALYSIS_SPECTRAL_THEOREM_SELFADJOINT_LIKE
PAIR_STEM: analysis_spectral_theorem_selfadjoint_like
MATH_DOMAIN: Functional Analysis
SOURCE_MATHLIB: Mathlib/Analysis/NormedSpace/Spectrum/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class HilbertSpaceLike (E : Type) := {
  inner : E -> E -> nat;
  inner_symm : forall x y : E, inner x y = inner y x;
  inner_pos : forall x : E, 0 <= inner x x
}.

Definition LinearOperatorLike {E : Type} `{HilbertSpaceLike E} (T : E -> E) : Prop :=
  forall x y : E, inner (T x) (T y) = inner (T y) (T x).

Definition SelfAdjointLike {E : Type} `{HilbertSpaceLike E} (T : E -> E) : Prop :=
  forall x y : E, inner (T x) y = inner x (T y).

Definition SpectralMeasureLike {E : Type} `{HilbertSpaceLike E} (T : E -> E) (mu : E -> E -> nat) : Prop :=
  (forall x y : E, mu x y = mu y x) /\ (forall x : E, mu x x = inner (T x) x).

Definition FunctionalCalculusLike {E : Type} `{HilbertSpaceLike E} (T : E -> E) (Phi : (E -> E) -> E -> E) : Prop :=
  (forall f g : E -> E, forall x : E, Phi f (Phi g x) = Phi g (Phi f x)) /\ (forall x : E, Phi T x = T x).

Definition ProjectionValuedLike {E : Type} `{HilbertSpaceLike E} (P : E -> E) : Prop :=
  (forall x : E, P (P x) = P x) /\ (forall x y : E, inner (P x) y = inner x (P y)).

Lemma spectral_resolution_exists {E : Type} `{HilbertSpaceLike E}
    (T : E -> E)
    (hself : SelfAdjointLike T)
    (hseed : exists mu : E -> E -> nat, (forall x y : E, mu x y = mu y x) /\ (forall x : E, mu x x = inner (T x) x)) :
    exists mu : E -> E -> nat, SpectralMeasureLike T mu.
Proof.
  destruct hseed as [mu [hsymm hdiag]].
  assert (hself_diag : forall x : E, inner (T x) x = inner x (T x)).
  { intro x. apply hself. }
  assert (hdiag_check : forall x : E, mu x x = inner (T x) x).
  { intro x. apply hdiag. }
  exists mu.
  split.
  - exact hsymm.
  - exact hdiag_check.
Qed.

Lemma spectral_resolution_unique {E : Type} `{HilbertSpaceLike E}
    (T : E -> E)
    (mu nu : E -> E -> nat)
    (hmu : SpectralMeasureLike T mu)
    (hnu : SpectralMeasureLike T nu)
    (huniq :
      forall mu' nu' : E -> E -> nat,
        SpectralMeasureLike T mu' ->
        SpectralMeasureLike T nu' ->
          (forall x : E, mu' x x = nu' x x) ->
            forall x y : E, mu' x y = nu' x y) :
    forall x y : E, mu x y = nu x y.
Proof.
  destruct hmu as [hmu_sym hmu_diag].
  destruct hnu as [hnu_sym hnu_diag].
  assert (hdiagEq : forall x : E, mu x x = nu x x).
  {
    intro x.
    rewrite hmu_diag.
    rewrite hnu_diag.
    reflexivity.
  }
  assert (hpoint : forall x y : E, mu x y = nu x y).
  {
    apply (huniq mu nu).
    - split; assumption.
    - split; assumption.
    - exact hdiagEq.
  }
  intros x y.
  apply hpoint.
Qed.

Lemma calculus_multiplicative {E : Type} `{HilbertSpaceLike E}
    (T : E -> E)
    (Phi : (E -> E) -> E -> E)
    (hcalc : FunctionalCalculusLike T Phi) :
    forall f g : E -> E, forall x : E, Phi f (Phi g x) = Phi g (Phi f x).
Proof.
  intros f g x.
  destruct hcalc as [hcomm hfix].
  assert (hfg : Phi f (Phi g x) = Phi g (Phi f x)).
  { apply hcomm. }
  assert (hgf : Phi g (Phi f x) = Phi f (Phi g x)).
  { apply hcomm. }
  assert (hroundtrip : Phi f (Phi g x) = Phi f (Phi g x)).
  { transitivity (Phi g (Phi f x)); assumption. }
  exact hfg.
Qed.

Lemma calculus_star_compatible {E : Type} `{HilbertSpaceLike E}
    (T : E -> E)
    (Phi : (E -> E) -> E -> E)
    (hself : SelfAdjointLike T)
    (hcalc : FunctionalCalculusLike T Phi)
    (hstar : forall f : E -> E, forall x : E, inner (Phi f x) x = inner x (Phi f x)) :
    forall x : E, inner (Phi T x) x = inner x (T x).
Proof.
  intro x.
  destruct hcalc as [hcomm hfix].
  assert (hstarT : inner (Phi T x) x = inner x (Phi T x)).
  { apply hstar. }
  assert (hright : inner x (Phi T x) = inner x (T x)).
  { rewrite hfix. reflexivity. }
  assert (hselfxx : inner (T x) x = inner x (T x)).
  { apply hself. }
  transitivity (inner x (Phi T x)).
  - exact hstarT.
  - exact hright.
Qed.

Lemma operator_reconstruction_like {E : Type} `{HilbertSpaceLike E}
    (T : E -> E)
    (mu : E -> E -> nat)
    (P : E -> E)
    (hmu : SpectralMeasureLike T mu)
    (hproj : ProjectionValuedLike P)
    (hreconstruct : forall x : E, inner (T (P x)) (P x) = inner (T x) x) :
    forall x : E, mu (P x) (P x) = inner (T x) x.
Proof.
  intro x.
  destruct hmu as [hsymm hdiag].
  destruct hproj as [hidem hselfP].
  assert (hbase : mu (P x) (P x) = inner (T (P x)) (P x)).
  { apply hdiag. }
  assert (hmove : inner (T (P x)) (P x) = inner (T x) x).
  { apply hreconstruct. }
  assert (hidemx : P (P x) = P x).
  { apply hidem. }
  transitivity (inner (T (P x)) (P x)).
  - exact hbase.
  - exact hmove.
Qed.

Lemma spectral_theorem_selfadjoint_like {E : Type} `{HilbertSpaceLike E}
    (T : E -> E)
    (Phi : (E -> E) -> E -> E)
    (P : E -> E)
    (hself : SelfAdjointLike T)
    (hseed : exists mu : E -> E -> nat, (forall x y : E, mu x y = mu y x) /\ (forall x : E, mu x x = inner (T x) x))
    (huniq :
      forall mu' nu' : E -> E -> nat,
        SpectralMeasureLike T mu' ->
        SpectralMeasureLike T nu' ->
          (forall x : E, mu' x x = nu' x x) ->
            forall x y : E, mu' x y = nu' x y)
    (hcalc : FunctionalCalculusLike T Phi)
    (hstar : forall f : E -> E, forall x : E, inner (Phi f x) x = inner x (Phi f x))
    (hproj : ProjectionValuedLike P)
    (hreconstruct : forall x : E, inner (T (P x)) (P x) = inner (T x) x) :
    exists mu : E -> E -> nat,
      SpectralMeasureLike T mu /\
      (forall x : E, mu (P x) (P x) = inner (T x) x) /\
      (forall x : E, inner (Phi T x) x = inner x (T x)).
Proof.
  destruct (spectral_resolution_exists (T := T) hself hseed) as [mu hmu].
  assert (hdiagEq : forall x : E, mu x x = mu x x).
  { intro x. reflexivity. }
  assert (hselfuniq : forall x y : E, mu x y = mu x y).
  { apply (huniq mu mu hmu hmu hdiagEq). }
  assert (hrecon : forall x : E, mu (P x) (P x) = inner (T x) x).
  { apply (operator_reconstruction_like (T := T) (mu := mu) (P := P) hmu hproj hreconstruct). }
  assert (hstarCompat : forall x : E, inner (Phi T x) x = inner x (T x)).
  { apply (calculus_star_compatible (T := T) (Phi := Phi) hself hcalc hstar). }
  exists mu.
  split.
  - exact hmu.
  - split.
    + exact hrecon.
    + exact hstarCompat.
Qed.
