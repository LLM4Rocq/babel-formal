(*
BENCHMARK_ID: TINY_MATHLIB_BATCH04_COMBINATORICS_POLYNOMIAL_METHOD_INCIDENCE_LIKE
PAIR_STEM: combinatorics_polynomial_method_incidence_like
MATH_DOMAIN: Combinatorics / Algebra
SOURCE_MATHLIB: Mathlib/Combinatorics
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class FiniteFieldLike (F : Type) := {
  zero : F;
  one : F;
  add : F -> F -> F;
  mul : F -> F -> F;
  card : nat;
  card_pos : 0 < card;
  add_assoc : forall x y z : F, add (add x y) z = add x (add y z);
  add_zero : forall x : F, add x zero = x;
  zero_add : forall x : F, add zero x = x;
  mul_one : forall x : F, mul x one = x;
  one_mul : forall x : F, mul one x = x
}.

Infix "+x" := add (at level 50, left associativity).
Infix "*x" := mul (at level 40, left associativity).

Definition PolynomialLike {F : Type} `{FiniteFieldLike F} (P : Type) : Prop :=
  exists eval : P -> F -> F, exists deg : P -> nat, True.

Definition VanishingSetLike {F : Type} `{FiniteFieldLike F} {P : Type}
    (eval : P -> F -> F) (p : P) (X : F -> Prop) : Prop :=
  forall x : F, X x -> eval p x = zero.

Definition DegreeLike {F : Type} `{FiniteFieldLike F} {P : Type}
    (deg : P -> nat) (p : P) (d : nat) : Prop :=
  deg p = d.

Definition MultiplicityLike {F : Type} `{FiniteFieldLike F} {P : Type}
    (eval : P -> F -> F) (p : P) (x : F) (m : nat) : Prop :=
  forall k : nat, k < m -> eval p x = zero.

Definition IncidenceSetLike {F : Type} `{FiniteFieldLike F} (Pts Lines : Type)
    (inc : Pts -> Lines -> nat) : Prop :=
  forall p : Pts, forall l : Lines, inc p l <= 1.

Lemma interpolation_bound_like {F : Type} `{FiniteFieldLike F} {P : Type}
    (hpoly : @PolynomialLike F _ P)
    (deg : P -> nat) (p : P) (d n : nat)
    (hdeg : @DegreeLike F _ P deg p d)
    (hpoints : n <= d + 1)
    (hdegree_card : d + 1 <= card)
    (htrans : forall a b c : nat, a <= b -> b <= c -> a <= c) :
    n <= card.
Proof.
  destruct hpoly as [eval0 [deg0 htriv]].
  assert (hstep1 : n <= d + 1).
  { exact hpoints. }
  assert (hstep2 : d + 1 <= card).
  { exact hdegree_card. }
  assert (hbound : n <= card).
  { apply (htrans n (d + 1) card hstep1 hstep2). }
  assert (hcheck : deg p = d).
  { exact hdeg. }
  assert (htrue : True).
  { exact htriv. }
  exact hbound.
Qed.

Lemma vanishing_multiplicity_step {F : Type} `{FiniteFieldLike F} {P : Type}
    (eval : P -> F -> F) (p : P) (x : F) (m n : nat)
    (hmult : MultiplicityLike eval p x m)
    (hstep : forall k : nat, k < n -> k < m) :
    MultiplicityLike eval p x n.
Proof.
  intros k hk.
  assert (hkm : k < m).
  { apply hstep. exact hk. }
  assert (hz : eval p x = zero).
  { exact (hmult k hkm). }
  exact hz.
Qed.

Lemma polynomial_partition_step {F : Type} `{FiniteFieldLike F} {Pts Lines : Type}
    (inc : Pts -> Lines -> nat) (cut : Pts -> nat) (cross : Lines -> nat)
    (c d : nat)
    (hpartition : forall p : Pts, forall l : Lines, inc p l <= cut p + cross l)
    (hcompress : forall p : Pts, forall l : Lines, cut p + cross l <= c + d)
    (htrans : forall a b e : nat, a <= b -> b <= e -> a <= e) :
    forall p : Pts, forall l : Lines, inc p l <= c + d.
Proof.
  intros p l.
  assert (hlocal : inc p l <= cut p + cross l).
  { apply hpartition. }
  assert (hfold : cut p + cross l <= c + d).
  { apply hcompress. }
  exact (htrans (inc p l) (cut p + cross l) (c + d) hlocal hfold).
Qed.

Lemma incidence_bound_core {F : Type} `{FiniteFieldLike F}
    (pts lines I d e : nat)
    (hinc : I <= pts * lines)
    (hgeom : pts * lines <= d * e)
    (htrans : forall a b c : nat, a <= b -> b <= c -> a <= c) :
    I <= d * e.
Proof.
  assert (hstep : I <= pts * lines).
  { exact hinc. }
  assert (hmul : pts * lines <= d * e).
  { exact hgeom. }
  exact (htrans I (pts * lines) (d * e) hstep hmul).
Qed.

Lemma sum_product_interface_like {F : Type} `{FiniteFieldLike F}
    (a b c d e : nat)
    (hsplit : a <= b + c)
    (htransfer : b + c <= d + e)
    (hbudget : d + e <= card)
    (htrans : forall x y z : nat, x <= y -> y <= z -> x <= z) :
    a <= card.
Proof.
  assert (h1 : a <= b + c).
  { exact hsplit. }
  assert (h2 : b + c <= d + e).
  { exact htransfer. }
  assert (h3 : d + e <= card).
  { exact hbudget. }
  assert (h4 : a <= d + e).
  { exact (htrans a (b + c) (d + e) h1 h2). }
  exact (htrans a (d + e) card h4 h3).
Qed.

Lemma polynomial_method_incidence_theorem_like {F : Type} `{FiniteFieldLike F}
    {Pts Lines : Type}
    (inc : Pts -> Lines -> nat) (cut : Pts -> nat) (cross : Lines -> nat)
    (pts lines I c d : nat)
    (hpartition : forall p : Pts, forall l : Lines, inc p l <= cut p + cross l)
    (hcompress : forall p : Pts, forall l : Lines, cut p + cross l <= c + d)
    (hcount : I <= pts * lines)
    (hgeom : pts * lines <= c * d)
    (hlift : c * d <= c * d + (c + d))
    (hbudget : c * d + (c + d) <= card)
    (htrans : forall x y z : nat, x <= y -> y <= z -> x <= z) :
    I <= card.
Proof.
  assert (hlocal : forall p : Pts, forall l : Lines, inc p l <= c + d).
  { apply (@polynomial_partition_step F _ Pts Lines inc cut cross c d hpartition hcompress htrans). }
  assert (hcore : I <= c * d).
  { apply (@incidence_bound_core F _ pts lines I c d hcount hgeom htrans). }
  assert (hstage : I <= c * d + (c + d)).
  { exact (htrans I (c * d) (c * d + (c + d)) hcore hlift). }
  assert (hfinal : I <= card).
  { exact (htrans I (c * d + (c + d)) card hstage hbudget). }
  exact hfinal.
Qed.
