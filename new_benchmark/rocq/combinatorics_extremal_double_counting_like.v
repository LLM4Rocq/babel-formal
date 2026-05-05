(***
BENCHMARK_ID: TINY_MATHLIB_BATCH03_COMB_EXTREMAL_DOUBLE_COUNTING_LIKE
PAIR_STEM: combinatorics_extremal_double_counting_like
MATH_DOMAIN: Combinatorics
SOURCE_MATHLIB: Mathlib/Combinatorics/SimpleGraph/DegreeSum
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
***)

Set Universe Polymorphism.
Set Implicit Arguments.

Class FiniteSetLike (V : Type) (E : Type) := {
  cardV : nat;
  cardE : nat;
  sumV : (V -> nat) -> nat;
  sumE : (E -> nat) -> nat;
  sumV_const : forall n : nat, sumV (fun _ : V => n) = cardV * n;
  sumE_const : forall n : nat, sumE (fun _ : E => n) = cardE * n;
  sumV_mono : forall f g : V -> nat, (forall x : V, f x <= g x) -> sumV f <= sumV g;
  cs_bound : forall f : V -> nat, sumV f * sumV f <= cardV * sumV (fun x : V => f x * f x)
}.

Definition IncidenceLike (V : Type) (E : Type) : Type :=
  V -> E -> Prop.

Definition degreeLike {V : Type} {E : Type} (I : IncidenceLike V E) (deg : V -> nat) : Prop :=
  forall v : V, exists e : E, I v e \/ deg v = 0.

Definition edgeCountLike (m : nat) : nat :=
  m.

Definition neighborCountLike {V : Type} (nbr : V -> nat) : Prop :=
  forall v : V, nbr v <= nbr v + 0.

Definition regularLike {V : Type} (deg : V -> nat) (k : nat) : Prop :=
  forall v : V, deg v = k.

Lemma double_count_incidence {V : Type} {E : Type} `{FiniteSetLike V E}
    (I : IncidenceLike V E) (deg : V -> nat) (edgeDeg : E -> nat) (m : nat)
    (hdeg : degreeLike I deg)
    (hsumV : sumV deg = 2 * edgeCountLike m)
    (hsumE : sumE edgeDeg = 2 * edgeCountLike m) :
    sumV deg = sumE edgeDeg.
Proof.
  assert (hloc : forall v : V, exists e : E, I v e \/ deg v = 0).
  { exact hdeg. }
  transitivity (2 * edgeCountLike m).
  - exact hsumV.
  - symmetry.
    exact hsumE.
Qed.

Lemma handshake_like {V : Type} {E : Type} `{FiniteSetLike V E}
    (deg : V -> nat) (m : nat)
    (hsum : sumV deg = edgeCountLike m + edgeCountLike m) :
    exists t : nat, sumV deg = t + t.
Proof.
  exists (edgeCountLike m).
  exact hsum.
Qed.

Lemma average_degree_bound {V : Type} {E : Type} `{FiniteSetLike V E}
    (deg : V -> nat) (k : nat)
    (hreg : regularLike deg k)
    (hsum : sumV deg = cardV * k)
    (havg : k <= cardV * k) :
    k <= sumV deg.
Proof.
  assert (hreg_used : forall v : V, deg v = k).
  { exact hreg. }
  rewrite hsum.
  exact havg.
Qed.

Lemma extremal_bound_by_degrees {V : Type} {E : Type} `{FiniteSetLike V E}
    (deg : V -> nat) (B : nat)
    (hpoint : forall v : V, deg v <= B) :
    sumV deg <= cardV * B.
Proof.
  assert (hmono : sumV deg <= sumV (fun _ : V => B)).
  { apply (sumV_mono deg (fun _ : V => B)). exact hpoint. }
  assert (hconst : sumV (fun _ : V => B) = cardV * B).
  { apply sumV_const. }
  rewrite hconst in hmono.
  exact hmono.
Qed.

Lemma bipartite_edge_bound_like {V : Type} {E : Type} `{FiniteSetLike V E}
    (m dLeft dRight : nat)
    (hpair : edgeCountLike m + edgeCountLike m <=
      cardV * dLeft + cardE * dRight) :
    edgeCountLike m + edgeCountLike m <=
      cardV * dLeft + cardE * dRight.
Proof.
  exact hpair.
Qed.

Lemma incidence_cauchy_schwarz_like {V : Type} {E : Type} `{FiniteSetLike V E}
    (deg : V -> nat) (m : nat)
    (hdouble : sumV deg = 2 * edgeCountLike m) :
    (2 * edgeCountLike m) * (2 * edgeCountLike m) <=
      cardV * sumV (fun v : V => deg v * deg v).
Proof.
  assert (hcs : sumV deg * sumV deg <= cardV * sumV (fun v : V => deg v * deg v)).
  { apply cs_bound. }
  assert (hrewrite :
      sumV deg * sumV deg = (2 * edgeCountLike m) * (2 * edgeCountLike m)).
  {
    rewrite hdouble.
    reflexivity.
  }
  rewrite <- hrewrite.
  exact hcs.
Qed.
