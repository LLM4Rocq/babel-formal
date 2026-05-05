(*
BENCHMARK_ID: TINY_MATHLIB_BATCH04_ANALYSIS_UNIFORM_BOUNDEDNESS_BANACH_LIKE
PAIR_STEM: analysis_uniform_boundedness_banach_like
MATH_DOMAIN: Functional Analysis
SOURCE_MATHLIB: Mathlib/Analysis/NormedSpace/UniformBoundedness
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class BanachSpaceLike (E : Type) := {
  zero : E;
  add : E -> E -> E;
  smul : nat -> E -> E;
  norm : E -> nat;
  zero_add : forall x : E, add zero x = x;
  add_zero : forall x : E, add x zero = x;
  add_assoc : forall x y z : E, add (add x y) z = add x (add y z);
  smul_zero : forall a : nat, smul a zero = zero;
  norm_zero : norm zero = 0;
  norm_add_le : forall x y : E, norm (add x y) <= norm x + norm y
}.

Infix "+v" := add (at level 50, left associativity).
Notation "a •v x" := (smul a x) (at level 40, left associativity).

Definition LinearMapLike {E F : Type} `{BanachSpaceLike E} `{BanachSpaceLike F}
    (T : E -> F) : Prop :=
  (forall x y : E, T (x +v y) = T x +v T y) /\
    (forall a : nat, forall x : E, T (a •v x) = a •v T x) /\
    T (zero : E) = (zero : F).

Definition PointwiseBoundedLike {I E F : Type} `{BanachSpaceLike E} `{BanachSpaceLike F}
    (A : I -> E -> F) : Prop :=
  forall x : E, exists C : nat, forall i : I, norm (A i x) <= C.

Definition OperatorNormBoundedLike {I E F : Type} `{BanachSpaceLike E} `{BanachSpaceLike F}
    (A : I -> E -> F) : Prop :=
  exists C : nat, forall i : I, forall x : E, norm (A i x) <= C * norm x.

Definition DenseSetLike {E : Type} `{BanachSpaceLike E}
    (D : E -> Prop) : Prop :=
  forall x : E, exists y : E, D y /\ norm y <= norm x + 1.

Definition BallLike {E : Type} `{BanachSpaceLike E}
    (r : nat) (x : E) : Prop :=
  norm x <= r.

Lemma baire_cover_step {I E F : Type}
    `{BanachSpaceLike E} `{BanachSpaceLike F}
    (A : I -> E -> F)
    (hpt : PointwiseBoundedLike A)
    (hseed : exists n : nat,
      forall y : E, BallLike 1 y -> forall i : I, norm (A i y) <= n) :
    exists n : nat,
      forall y : E, BallLike 1 y -> forall i : I, norm (A i y) <= n.
Proof.
  destruct hseed as [n hn].
  assert (hpt0 : exists c0 : nat, forall i : I, norm (A i (zero : E)) <= c0).
  { apply hpt. }
  destruct hpt0 as [c0 hc0].
  assert (hzero_ball : BallLike 1 (zero : E)).
  {
    unfold BallLike.
    rewrite norm_zero.
    apply le_0_n.
  }
  assert (hzero_bound : forall i : I, norm (A i (zero : E)) <= n).
  {
    intro i.
    apply (hn (zero : E)); try exact hzero_ball.
  }
  exists n.
  exact hn.
Qed.

Lemma interior_nonempty_step {I E F : Type}
    `{BanachSpaceLike E} `{BanachSpaceLike F}
    (A : I -> E -> F)
    (D : E -> Prop)
    (hDense : DenseSetLike D)
    (hpt : PointwiseBoundedLike A) :
    exists y : E, D y /\ exists n : nat, forall i : I, norm (A i y) <= n.
Proof.
  destruct (hDense (zero : E)) as [y [hyD hyNorm]].
  destruct (hpt y) as [n hn].
  assert (hy0 : norm y <= norm (zero : E) + 1).
  { exact hyNorm. }
  assert (hz : norm (zero : E) = 0).
  { exact norm_zero. }
  assert (hy1 : norm y <= 1).
  {
    rewrite hz in hy0.
    exact hy0.
  }
  exists y.
  split.
  - exact hyD.
  - exists n.
    exact hn.
Qed.

Lemma local_uniform_bound_step {I E F : Type}
    `{BanachSpaceLike E} `{BanachSpaceLike F}
    (A : I -> E -> F)
    (r : nat)
    (hunit : exists n : nat,
      forall x : E, BallLike 1 x -> forall i : I, norm (A i x) <= n)
    (htransfer : forall m : nat,
      (exists n : nat,
        forall x : E, BallLike 1 x -> forall i : I, norm (A i x) <= n) ->
      exists n : nat,
        forall x : E, BallLike m x -> forall i : I, norm (A i x) <= n) :
    exists n : nat,
      forall x : E, BallLike r x -> forall i : I, norm (A i x) <= n.
Proof.
  assert (hr :
      exists n : nat,
        forall x : E, BallLike r x -> forall i : I, norm (A i x) <= n).
  { apply htransfer. exact hunit. }
  destruct hr as [n hn].
  exists n.
  exact hn.
Qed.

Lemma global_uniform_bound_step {I E F : Type}
    `{BanachSpaceLike E} `{BanachSpaceLike F}
    (A : I -> E -> F)
    (hlin : forall i : I, LinearMapLike (A i))
    (hlocal : exists n : nat,
      forall x : E, BallLike 1 x -> forall i : I, norm (A i x) <= n)
    (hglobalize : forall n : nat,
      (forall x : E, BallLike 1 x -> forall i : I, norm (A i x) <= n) ->
      forall x : E, forall i : I, norm (A i x) <= n * norm x) :
    OperatorNormBoundedLike A.
Proof.
  destruct hlocal as [n hn].
  assert (hlin0 : forall i : I, A i (zero : E) = (zero : F)).
  {
    intro i.
    destruct (hlin i) as [hadd [hsmul hzero]].
    exact hzero.
  }
  assert (hnorm : forall x : E, forall i : I, norm (A i x) <= n * norm x).
  { apply hglobalize. exact hn. }
  assert (hswap : forall i : I, forall x : E, norm (A i x) <= n * norm x).
  {
    intros i x.
    apply hnorm.
  }
  exists n.
  exact hswap.
Qed.

Lemma equicontinuity_corollary_like {I E F : Type}
    `{BanachSpaceLike E} `{BanachSpaceLike F}
    (A : I -> E -> F)
    (hOp : OperatorNormBoundedLike A) :
    exists C : nat, forall i : I, norm (A i (zero : E)) <= C.
Proof.
  destruct hOp as [C hC].
  exists (C * norm (zero : E)).
  intro i.
  assert (hz : norm (A i (zero : E)) <= C * norm (zero : E)).
  { apply hC. }
  assert (hnorm0 : norm (zero : E) = 0).
  { exact norm_zero. }
  assert (hkeep : C * norm (zero : E) = C * norm (zero : E)).
  { reflexivity. }
  exact hz.
Qed.

Lemma uniform_boundedness_theorem_like {I E F : Type}
    `{BanachSpaceLike E} `{BanachSpaceLike F}
    (A : I -> E -> F)
    (hpt : PointwiseBoundedLike A)
    (hseed : exists n : nat,
      forall y : E, BallLike 1 y -> forall i : I, norm (A i y) <= n)
    (hlin : forall i : I, LinearMapLike (A i))
    (htransfer : forall m : nat,
      (exists n : nat,
        forall x : E, BallLike 1 x -> forall i : I, norm (A i x) <= n) ->
      exists n : nat,
        forall x : E, BallLike m x -> forall i : I, norm (A i x) <= n)
    (hglobalize : forall n : nat,
      (forall x : E, BallLike 1 x -> forall i : I, norm (A i x) <= n) ->
      forall x : E, forall i : I, norm (A i x) <= n * norm x) :
    OperatorNormBoundedLike A /\
      (exists C : nat, forall i : I, norm (A i (zero : E)) <= C).
Proof.
  assert (hunit :
      exists n : nat,
        forall y : E, BallLike 1 y -> forall i : I, norm (A i y) <= n).
  { apply (baire_cover_step (A := A)); assumption. }
  assert (hlocal :
      exists n : nat,
        forall x : E, BallLike 1 x -> forall i : I, norm (A i x) <= n).
  { exact (local_uniform_bound_step A 1 hunit htransfer). }
  assert (hglobal : OperatorNormBoundedLike A).
  { exact (global_uniform_bound_step A hlin hlocal hglobalize). }
  assert (hequi : exists C : nat, forall i : I, norm (A i (zero : E)) <= C).
  { apply (equicontinuity_corollary_like (A := A)). exact hglobal. }
  split.
  - exact hglobal.
  - exact hequi.
Qed.
