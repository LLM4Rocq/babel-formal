(***
BENCHMARK_ID: TINY_MATHLIB_BATCH03_ALG_LOCALIZATION_UNIVERSAL_LIKE
PAIR_STEM: algebra_localization_universal_like
MATH_DOMAIN: Commutative Algebra
SOURCE_MATHLIB: Mathlib/RingTheory/Localization/Basic
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
***)

Set Universe Polymorphism.
Set Implicit Arguments.

Class CommMonoidLike (A : Type) := {
  c_one : A;
  c_mul : A -> A -> A;
  c_mul_assoc : forall a b c : A, c_mul (c_mul a b) c = c_mul a (c_mul b c);
  c_one_mul : forall a : A, c_mul c_one a = a;
  c_mul_one : forall a : A, c_mul a c_one = a;
  c_mul_comm : forall a b : A, c_mul a b = c_mul b a
}.

Infix "*" := c_mul (at level 40, left associativity).

Definition IsSubmonoidLike {A : Type} `{CommMonoidLike A} (S : A -> Prop) : Prop :=
  S c_one /\ forall a b : A, S a -> S b -> S (a * b).

Record LocalizationLike (A : Type) `{CommMonoidLike A} (S : A -> Prop) := {
  L : Type;
  oneL : L;
  mulL : L -> L -> L;
  of : A -> L;
  l_one_mul : forall x : L, mulL oneL x = x;
  l_mul_one : forall x : L, mulL x oneL = x;
  l_mul_assoc : forall x y z : L, mulL (mulL x y) z = mulL x (mulL y z);
  of_one : of c_one = oneL;
  of_mul : forall a b : A, of (a * b) = mulL (of a) (of b);
  denom_unit : forall s : A, S s -> exists invs : L, mulL (of s) invs = oneL /\ mulL invs (of s) = oneL;
  lift_data : forall (B : Type) (oneB : B) (mulB : B -> B -> B) (f : A -> B),
      f c_one = oneB ->
      (forall a b : A, f (a * b) = mulB (f a) (f b)) ->
      (forall s : A, S s -> exists i : B, mulB (f s) i = oneB /\ mulB i (f s) = oneB) ->
      L -> B;
  lift_one : forall (B : Type) (oneB : B) (mulB : B -> B -> B) (f : A -> B)
      (h1 : f c_one = oneB)
      (hmul : forall a b : A, f (a * b) = mulB (f a) (f b))
      (hS : forall s : A, S s -> exists i : B, mulB (f s) i = oneB /\ mulB i (f s) = oneB),
      lift_data (B := B) (oneB := oneB) mulB f h1 hmul hS oneL = oneB;
  lift_mul : forall (B : Type) (oneB : B) (mulB : B -> B -> B) (f : A -> B)
      (h1 : f c_one = oneB)
      (hmul : forall a b : A, f (a * b) = mulB (f a) (f b))
      (hS : forall s : A, S s -> exists i : B, mulB (f s) i = oneB /\ mulB i (f s) = oneB)
      (x y : L),
      lift_data (B := B) (oneB := oneB) mulB f h1 hmul hS (mulL x y) =
        mulB (lift_data (B := B) (oneB := oneB) mulB f h1 hmul hS x)
             (lift_data (B := B) (oneB := oneB) mulB f h1 hmul hS y);
  lift_of : forall (B : Type) (oneB : B) (mulB : B -> B -> B) (f : A -> B)
      (h1 : f c_one = oneB)
      (hmul : forall a b : A, f (a * b) = mulB (f a) (f b))
      (hS : forall s : A, S s -> exists i : B, mulB (f s) i = oneB /\ mulB i (f s) = oneB)
      (a : A),
      lift_data (B := B) (oneB := oneB) mulB f h1 hmul hS (of a) = f a;
  lift_unique_axiom : forall (B : Type) (oneB : B) (mulB : B -> B -> B) (f : A -> B)
      (h1 : f c_one = oneB)
      (hmul : forall a b : A, f (a * b) = mulB (f a) (f b))
      (hS : forall s : A, S s -> exists i : B, mulB (f s) i = oneB /\ mulB i (f s) = oneB)
      (g : L -> B),
      g oneL = oneB ->
      (forall x y : L, g (mulL x y) = mulB (g x) (g y)) ->
      (forall a : A, g (of a) = f a) ->
      g = lift_data (B := B) (oneB := oneB) mulB f h1 hmul hS
}.

Definition ofMap {A : Type} `{CommMonoidLike A} {S : A -> Prop}
    (Loc : LocalizationLike (A := A) S) : A -> L Loc :=
  of Loc.

Definition lift {A : Type} `{CommMonoidLike A} {S : A -> Prop}
    (Loc : LocalizationLike (A := A) S)
    (B : Type)
    (oneB : B)
    (mulB : B -> B -> B)
    (f : A -> B)
    (h1 : f c_one = oneB)
    (hmul : forall a b : A, f (a * b) = mulB (f a) (f b))
    (hS : forall s : A, S s -> exists i : B, mulB (f s) i = oneB /\ mulB i (f s) = oneB) :
    L Loc -> B :=
  lift_data (B := B) (oneB := oneB) Loc mulB f h1 hmul hS.

Definition IsUnitLike {B : Type} (oneB : B) (mulB : B -> B -> B) (x : B) : Prop :=
  exists y : B, mulB x y = oneB /\ mulB y x = oneB.

Lemma of_mem_units {A : Type} `{CommMonoidLike A} {S : A -> Prop}
    (Loc : LocalizationLike (A := A) S) (s : A) (hs : S s) :
    IsUnitLike (oneL Loc) (mulL Loc) ((ofMap Loc) s).
Proof.
  unfold IsUnitLike.
  destruct (denom_unit Loc s hs) as [u [hul hur]].
  exists u.
  split.
  - exact hul.
  - exact hur.
Qed.

Lemma lift_comp_of {A : Type} `{CommMonoidLike A} {S : A -> Prop}
    (Loc : LocalizationLike (A := A) S)
    (B : Type)
    (oneB : B)
    (mulB : B -> B -> B)
    (f : A -> B)
    (h1 : f c_one = oneB)
    (hmul : forall a b : A, f (a * b) = mulB (f a) (f b))
    (hS : forall s : A, S s -> exists i : B, mulB (f s) i = oneB /\ mulB i (f s) = oneB) :
    forall a : A, lift (B := B) (oneB := oneB) Loc mulB f h1 hmul hS ((ofMap Loc) a) = f a.
Proof.
  intro a.
  unfold lift.
  unfold ofMap.
  exact (lift_of (B := B) (oneB := oneB) Loc mulB f h1 hmul hS a).
Qed.

Lemma lift_unique {A : Type} `{CommMonoidLike A} {S : A -> Prop}
    (Loc : LocalizationLike (A := A) S)
    (B : Type)
    (oneB : B)
    (mulB : B -> B -> B)
    (f : A -> B)
    (h1 : f c_one = oneB)
    (hmul : forall a b : A, f (a * b) = mulB (f a) (f b))
    (hS : forall s : A, S s -> exists i : B, mulB (f s) i = oneB /\ mulB i (f s) = oneB)
    (g : L Loc -> B)
    (hg_one : g (oneL Loc) = oneB)
    (hg_mul : forall x y : L Loc, g (mulL Loc x y) = mulB (g x) (g y))
    (hg_of : forall a : A, g ((ofMap Loc) a) = f a) :
    g = lift (B := B) (oneB := oneB) Loc mulB f h1 hmul hS.
Proof.
  unfold lift.
  apply (lift_unique_axiom (B := B) (oneB := oneB) Loc mulB f h1 hmul hS g).
  - exact hg_one.
  - exact hg_mul.
  - intro a.
    unfold ofMap.
    exact (hg_of a).
Qed.

Lemma localization_induction {A : Type} `{CommMonoidLike A} {S : A -> Prop}
    (Loc : LocalizationLike (A := A) S)
    (P : L Loc -> Prop)
    (h_one : P (oneL Loc))
    (h_mul : forall x y : L Loc, P x -> P y -> P (mulL Loc x y))
    (h_of : forall a : A, P ((ofMap Loc) a))
    (h_cancel : forall x : L Loc, forall s : A, S s -> P (mulL Loc x ((ofMap Loc) s)) -> P x)
    (x : L Loc)
    (hx : exists a s : A, S s /\ mulL Loc x ((ofMap Loc) s) = (ofMap Loc) a) :
    P x.
Proof.
  destruct hx as [a [s [hs hrepr]]].
  assert (hPa : P ((ofMap Loc) a)).
  { apply h_of. }
  assert (hPone : P (oneL Loc)).
  { exact h_one. }
  assert (hPa_mul_one : P (mulL Loc ((ofMap Loc) a) (oneL Loc))).
  { apply h_mul; assumption. }
  assert (hPa' : P ((ofMap Loc) a)).
  {
    rewrite (l_mul_one Loc ((ofMap Loc) a)) in hPa_mul_one.
    exact hPa_mul_one.
  }
  assert (hPxs : P (mulL Loc x ((ofMap Loc) s))).
  {
    rewrite hrepr.
    exact hPa'.
  }
  exact (h_cancel x s hs hPxs).
Qed.

Lemma eq_of_cross_multiply {A : Type} `{CommMonoidLike A} {S : A -> Prop}
    (Loc : LocalizationLike (A := A) S)
    (x y : L Loc) (s : A) (hs : S s)
    (hxy : mulL Loc x ((ofMap Loc) s) = mulL Loc y ((ofMap Loc) s)) :
    x = y.
Proof.
  destruct (denom_unit Loc s hs) as [u [hsu hus]].
  assert (hmul : mulL Loc (mulL Loc x ((ofMap Loc) s)) u =
                 mulL Loc (mulL Loc y ((ofMap Loc) s)) u).
  {
    rewrite hxy.
    reflexivity.
  }
  assert (hx_assoc : mulL Loc x (mulL Loc ((ofMap Loc) s) u) =
                     mulL Loc y (mulL Loc ((ofMap Loc) s) u)).
  {
    transitivity (mulL Loc (mulL Loc x ((ofMap Loc) s)) u).
    - symmetry.
      apply l_mul_assoc.
    - transitivity (mulL Loc (mulL Loc y ((ofMap Loc) s)) u).
      + exact hmul.
      + apply l_mul_assoc.
  }
  assert (hreduce : mulL Loc x (oneL Loc) = mulL Loc y (oneL Loc)).
  {
    assert (hsu' : mulL Loc ((ofMap Loc) s) u = oneL Loc).
    { unfold ofMap. exact hsu. }
    rewrite hsu' in hx_assoc.
    exact hx_assoc.
  }
  transitivity (mulL Loc x (oneL Loc)).
  - symmetry.
    apply l_mul_one.
  - transitivity (mulL Loc y (oneL Loc)).
    + exact hreduce.
    + apply l_mul_one.
Qed.

Lemma localization_universal {A : Type} `{CommMonoidLike A} {S : A -> Prop}
    (Loc : LocalizationLike (A := A) S)
    (B : Type)
    (oneB : B)
    (mulB : B -> B -> B)
    (f : A -> B)
    (h1 : f c_one = oneB)
    (hmul : forall a b : A, f (a * b) = mulB (f a) (f b))
    (hS : forall s : A, S s -> exists i : B, mulB (f s) i = oneB /\ mulB i (f s) = oneB) :
    exists g : L Loc -> B,
      g (oneL Loc) = oneB /\
      (forall x y : L Loc, g (mulL Loc x y) = mulB (g x) (g y)) /\
      (forall a : A, g ((ofMap Loc) a) = f a) /\
      (forall g' : L Loc -> B,
        g' (oneL Loc) = oneB ->
        (forall x y : L Loc, g' (mulL Loc x y) = mulB (g' x) (g' y)) ->
        (forall a : A, g' ((ofMap Loc) a) = f a) ->
        g' = g).
Proof.
  exists (lift (B := B) (oneB := oneB) Loc mulB f h1 hmul hS).
  split.
  - unfold lift.
    apply (lift_one (B := B) (oneB := oneB) Loc mulB f h1 hmul hS).
  - split.
    + intros x y.
      unfold lift.
      apply (lift_mul (B := B) (oneB := oneB) Loc mulB f h1 hmul hS x y).
    + split.
      * intro a.
        apply (lift_comp_of (B := B) (oneB := oneB) Loc mulB f h1 hmul hS a).
      * intros g' hg'_one hg'_mul hg'_of.
        apply (lift_unique (B := B) (oneB := oneB) Loc mulB f h1 hmul hS g').
        -- exact hg'_one.
        -- exact hg'_mul.
        -- exact hg'_of.
Qed.
