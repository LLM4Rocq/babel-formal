(*
BENCHMARK_ID: TINY_MATHLIB_BATCH04_ORDER_DOMAIN_SCOTT_CONTINUITY
PAIR_STEM: order_domain_scott_continuity_like
MATH_DOMAIN: Domain Theory
SOURCE_MATHLIB: Mathlib/Order/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class DcpoLike (A : Type) := {
  leq : A -> A -> Prop;
  leq_refl : forall x : A, leq x x;
  leq_trans : forall x y z : A, leq x y -> leq y z -> leq x z;
  sup : (A -> Prop) -> A;
  leq_sup : forall (S : A -> Prop) (x : A), S x -> leq x (sup S);
  sup_leq : forall (S : A -> Prop) (x : A), (forall y : A, S y -> leq y x) -> leq (sup S) x
}.

Infix "<=" := leq (at level 70).

Definition DirectedLike {A : Type} `{DcpoLike A} (S : A -> Prop) : Prop :=
  forall x y : A, S x -> S y -> exists z : A, S z /\ x <= z /\ y <= z.

Definition SupLike {A : Type} `{DcpoLike A} (S : A -> Prop) (x : A) : Prop :=
  x = sup S.

Definition ScottContinuousLike {A : Type} `{DcpoLike A} (f : A -> A) : Prop :=
  (forall x y : A, x <= y -> f x <= f y) /\
  (forall S : A -> Prop,
    DirectedLike S ->
    f (sup S) <= sup (fun y : A => exists x : A, S x /\ y = f x)).

Definition WayBelowLike {A : Type} `{DcpoLike A} (x y : A) : Prop :=
  forall S : A -> Prop,
    DirectedLike S ->
    y <= sup S ->
    exists z : A, S z /\ x <= z.

Definition AlgebraicLike {A : Type} `{DcpoLike A} : Prop :=
  forall y : A,
    exists S : A -> Prop,
      DirectedLike S /\
      y <= sup S /\
      (forall z : A, S z -> WayBelowLike z y).

Lemma scott_mono {A : Type} `{DcpoLike A}
    (f : A -> A) (hsc : ScottContinuousLike f) :
    forall x y : A, x <= y -> f x <= f y.
Proof.
  intros x y hxy.
  destruct hsc as [hmono hsup].
  assert (hstep : f x <= f y).
  {
    apply hmono.
    exact hxy.
  }
  exact hstep.
Qed.

Lemma scott_preserves_sup {A : Type} `{DcpoLike A}
    (f : A -> A) (hsc : ScottContinuousLike f)
    (S : A -> Prop) (hdir : DirectedLike S) :
    f (sup S) <= sup (fun y : A => exists x : A, S x /\ y = f x).
Proof.
  destruct hsc as [hmono hsup].
  assert (hresult :
      f (sup S) <= sup (fun y : A => exists x : A, S x /\ y = f x)).
  {
    apply hsup.
    exact hdir.
  }
  exact hresult.
Qed.

Lemma waybelow_interpolation {A : Type} `{DcpoLike A}
    (x y : A) (hxy : WayBelowLike x y)
    (S : A -> Prop) (hdir : DirectedLike S)
    (hyS : y <= sup S) :
    exists z : A, S z /\ x <= z.
Proof.
  assert (hwitness : exists z : A, S z /\ x <= z).
  {
    apply hxy.
    - exact hdir.
    - exact hyS.
  }
  destruct hwitness as [z hz].
  exists z.
  exact hz.
Qed.

Lemma compact_basis_expand {A : Type} `{DcpoLike A}
    (hAlg : AlgebraicLike) (y : A) :
    exists S : A -> Prop,
      DirectedLike S /\
      y <= sup S /\
      (forall z : A, S z -> WayBelowLike z y).
Proof.
  assert (hy :
      exists S : A -> Prop,
        DirectedLike S /\ y <= sup S /\ (forall z : A, S z -> WayBelowLike z y)).
  {
    apply hAlg.
  }
  destruct hy as [S [hdir [hsup hwb]]].
  exists S.
  split.
  - exact hdir.
  - split.
    + exact hsup.
    + exact hwb.
Qed.

Lemma fixedpoint_chain_limit {A : Type} `{DcpoLike A}
    (f : A -> A) (hsc : ScottContinuousLike f)
    (S : A -> Prop) (hdir : DirectedLike S)
    (hstep : forall x : A, S x -> x <= f x) :
    sup S <= f (sup S).
Proof.
  assert (hmono : forall x y : A, x <= y -> f x <= f y).
  {
    apply scott_mono.
    exact hsc.
  }
  assert (hupper : forall x : A, S x -> x <= f (sup S)).
  {
    intros x hx.
    assert (hxfx : x <= f x).
    {
      apply hstep.
      exact hx.
    }
    assert (hxsup : x <= sup S).
    {
      apply leq_sup.
      exact hx.
    }
    assert (hfxsup : f x <= f (sup S)).
    {
      apply hmono.
      exact hxsup.
    }
    exact (leq_trans _ _ _ hxfx hfxsup).
  }
  apply sup_leq.
  exact hupper.
Qed.

Lemma least_fixedpoint_scott {A : Type} `{DcpoLike A}
    (f : A -> A) (hsc : ScottContinuousLike f)
    (S : A -> Prop) (hdir : DirectedLike S)
    (hpref : forall x : A, S x -> f x <= x) :
    f (sup S) <= sup S.
Proof.
  assert (hpres :
      f (sup S) <= sup (fun y : A => exists x : A, S x /\ y = f x)).
  {
    apply scott_preserves_sup.
    - exact hsc.
    - exact hdir.
  }
  assert (himage_le :
      sup (fun y : A => exists x : A, S x /\ y = f x) <= sup S).
  {
    apply sup_leq.
    intros y hy.
    destruct hy as [x [hxS hyEq]].
    assert (hfx_le_x : f x <= x).
    {
      apply hpref.
      exact hxS.
    }
    assert (hx_le_sup : x <= sup S).
    {
      apply leq_sup.
      exact hxS.
    }
    assert (hfx_le_sup : f x <= sup S).
    {
      exact (leq_trans _ _ _ hfx_le_x hx_le_sup).
    }
    rewrite hyEq.
    exact hfx_le_sup.
  }
  exact (leq_trans _ _ _ hpres himage_le).
Qed.
