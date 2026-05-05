(*
BENCHMARK_ID: TINY_MATHLIB_BATCH04_ANALYSIS_OPEN_MAPPING_BANACH_LIKE
PAIR_STEM: analysis_open_mapping_banach_like
MATH_DOMAIN: Functional Analysis
SOURCE_MATHLIB: Mathlib/Analysis/NormedSpace/OpenMapping
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

Definition BoundedLike {E F : Type} `{BanachSpaceLike E} `{BanachSpaceLike F}
    (T : E -> F) : Prop :=
  exists C : nat, forall x : E, norm (T x) <= C * norm x.

Definition SurjectiveLike {E F : Type} `{BanachSpaceLike E} `{BanachSpaceLike F}
    (T : E -> F) : Prop :=
  forall y : F, exists x : E, T x = y.

Definition OpenMapLike {E F : Type} `{BanachSpaceLike E} `{BanachSpaceLike F}
    (T : E -> F) : Prop :=
  forall r : nat,
    exists s : nat,
      s <= r /\
      (forall y : F,
        norm y <= s ->
          exists x : E, norm x <= r /\ T x = y).

Definition QuotientNormLike {E F : Type} `{BanachSpaceLike E} `{BanachSpaceLike F}
    (T : E -> F) (q : F -> nat) : Prop :=
  (forall y : F, exists x : E, T x = y /\ q y <= norm x) /\
    (forall x : E, q (T x) <= norm x).

Lemma baire_step_ball_absorb {E F : Type}
    `{BanachSpaceLike E} `{BanachSpaceLike F}
    (T : E -> F)
    (hlin : LinearMapLike T)
    (hopen : OpenMapLike T)
    (r : nat) :
    exists s : nat,
      s <= r /\
      (forall y : F,
        norm y <= s ->
          exists x : E, norm x <= r /\ T x = y).
Proof.
  destruct (hopen r) as [s [hsle hsball]].
  assert (hmap0 : T (zero : E) = (zero : F)).
  { destruct hlin as [hadd [hsmul hzero]]. exact hzero. }
  assert (hpass : forall y : F, norm y <= s -> exists x : E, norm x <= r /\ T x = y).
  { intros y hy. apply hsball. exact hy. }
  assert (hdiag : T (zero : E) = (zero : F)).
  { exact hmap0. }
  exists s.
  split.
  - exact hsle.
  - intros y hy.
    apply hpass.
    exact hy.
Qed.

Lemma bounded_inverse_core {E F : Type}
    `{BanachSpaceLike E} `{BanachSpaceLike F}
    (T : E -> F)
    (S : F -> E)
    (hlin : LinearMapLike T)
    (hboundT : BoundedLike T)
    (hright : forall y : F, T (S y) = y)
    (hseed : exists D : nat, forall y : F, norm (S y) <= D * norm y) :
    BoundedLike S.
Proof.
  destruct hseed as [D hD].
  exists D.
  intro y.
  assert (hy : norm (S y) <= D * norm y).
  { apply hD. }
  assert (hmap0 : T (zero : E) = (zero : F)).
  { destruct hlin as [hadd [hsmul hzero]]. exact hzero. }
  assert (hright0 : T (S (zero : F)) = (zero : F)).
  { specialize (hright (zero : F)). exact hright. }
  destruct hboundT as [C hC].
  assert (hbound0 : norm (T (zero : E)) <= C * norm (zero : E)).
  { apply hC. }
  assert (hcheck : T (S y) = y).
  { apply hright. }
  exact hy.
Qed.

Lemma open_mapping_core {E F : Type}
    `{BanachSpaceLike E} `{BanachSpaceLike F}
    (T : E -> F)
    (hlin : LinearMapLike T)
    (hsurj : SurjectiveLike T)
    (hbaire : forall r : nat,
      exists s : nat,
        s <= r /\
        (forall y : F,
          norm y <= s ->
            exists x : E, norm x <= r /\ T x = y)) :
    OpenMapLike T.
Proof.
  intro r.
  destruct (hbaire r) as [s [hsle hsball]].
  assert (hsurj0 : exists x : E, T x = (zero : F)).
  { apply hsurj. }
  destruct hsurj0 as [x0 hx0].
  assert (hlin0 : T (zero : E) = (zero : F)).
  { destruct hlin as [hadd [hsmul hzero]]. exact hzero. }
  assert (hmark : T x0 = (zero : F)).
  { exact hx0. }
  exists s.
  split.
  - exact hsle.
  - intros y hy.
    apply hsball.
    exact hy.
Qed.

Lemma inverse_continuous_of_bijective {E F : Type}
    `{BanachSpaceLike E} `{BanachSpaceLike F}
    (T : E -> F)
    (S : F -> E)
    (hlin : LinearMapLike T)
    (hboundT : BoundedLike T)
    (hleft : forall x : E, S (T x) = x)
    (hright : forall y : F, T (S y) = y)
    (hseed : exists D : nat, forall y : F, norm (S y) <= D * norm y) :
    BoundedLike S.
Proof.
  assert (hcore : BoundedLike S).
  { exact (bounded_inverse_core (T := T) S hlin hboundT hright hseed). }
  assert (hleft0 : S (T (zero : E)) = (zero : E)).
  { specialize (hleft (zero : E)). exact hleft. }
  assert (hmap0 : T (zero : E) = (zero : F)).
  { destruct hlin as [hadd [hsmul hzero]]. exact hzero. }
  assert (hs0 : S (zero : F) = (zero : E)).
  {
    rewrite <- hmap0.
    exact hleft0.
  }
  exact hcore.
Qed.

Lemma closed_graph_step_like {E F : Type}
    `{BanachSpaceLike E} `{BanachSpaceLike F}
    (T : E -> F)
    (hlin : LinearMapLike T)
    (hgraph : forall x : E, norm (T x) = 0 -> norm x = 0)
    (hseed : BoundedLike T) :
    BoundedLike T.
Proof.
  destruct hseed as [C hC].
  assert (hT0 : norm (T (zero : E)) = 0).
  {
    destruct hlin as [hadd [hsmul hzero]].
    rewrite hzero.
    exact norm_zero.
  }
  assert (hE0 : norm (zero : E) = 0).
  { apply (hgraph (zero : E)). exact hT0. }
  assert (hE0' : norm (zero : E) = 0).
  { exact norm_zero. }
  exists C.
  exact hC.
Qed.

Lemma open_mapping_theorem_like {E F : Type}
    `{BanachSpaceLike E} `{BanachSpaceLike F}
    (T : E -> F)
    (S : F -> E)
    (hlin : LinearMapLike T)
    (hboundT : BoundedLike T)
    (hsurj : SurjectiveLike T)
    (hright : forall y : F, T (S y) = y)
    (hbaire : forall r : nat,
      exists s : nat,
        s <= r /\
        (forall y : F,
          norm y <= s ->
            exists x : E, norm x <= r /\ T x = y))
    (hseed : exists D : nat, forall y : F, norm (S y) <= D * norm y) :
    OpenMapLike T /\ BoundedLike S.
Proof.
  assert (hopen : OpenMapLike T).
  { apply (open_mapping_core (T := T)); assumption. }
  assert (hSinv : BoundedLike S).
  { exact (bounded_inverse_core (T := T) S hlin hboundT hright hseed). }
  assert (hball0 :
      exists s : nat,
        s <= 0 /\
        (forall y : F,
          norm y <= s ->
            exists x : E, norm x <= 0 /\ T x = y)).
  { apply hopen. }
  destruct hball0 as [s0 [hs0le hs0ball]].
  assert (hs0eq : s0 = 0).
  {
    destruct s0 as [|k].
    - reflexivity.
    - exfalso. inversion hs0le.
  }
  split.
  - exact hopen.
  - exact hSinv.
Qed.
