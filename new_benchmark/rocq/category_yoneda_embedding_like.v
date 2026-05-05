(*
BENCHMARK_ID: TINY_MATHLIB_BATCH03_CATEGORY_YONEDA_EMBEDDING_LIKE
PAIR_STEM: category_yoneda_embedding_like
MATH_DOMAIN: Category Theory
SOURCE_MATHLIB: Mathlib/CategoryTheory/Yoneda
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class CategoryLike (Obj : Type) := {
  Hom : Obj -> Obj -> Type;
  id : forall {X : Obj}, Hom X X;
  comp : forall {X Y Z : Obj}, Hom X Y -> Hom Y Z -> Hom X Z;
  comp_assoc :
    forall {W X Y Z : Obj} (f : Hom W X) (g : Hom X Y) (h : Hom Y Z),
      comp (comp f g) h = comp f (comp g h);
  id_comp : forall {X Y : Obj} (f : Hom X Y), comp id f = f;
  comp_id : forall {X Y : Obj} (f : Hom X Y), comp f id = f
}.

Arguments Hom {Obj} _ _ _.
Arguments id {Obj} _ {X}.
Arguments comp {Obj} _ {X Y Z} _ _.

Definition HomFunctorLike {C : Type} (CC : CategoryLike C) (A : C) : C -> Type :=
  fun X => Hom CC A X.

Definition YonedaObjLike {C : Type} (CC : CategoryLike C) (A : C) : C -> Type :=
  HomFunctorLike CC A.

Definition YonedaMapLike {C : Type} (CC : CategoryLike C) {A B : C}
    (f : Hom CC A B) :
    forall X : C, HomFunctorLike CC B X -> HomFunctorLike CC A X :=
  fun X u => comp CC f u.

Lemma yoneda_map_id {C : Type} (CC : CategoryLike C) (A : C) :
    forall X : C, forall u : HomFunctorLike CC A X,
      YonedaMapLike CC (id CC (X := A)) X u = u.
Proof.
  intros X u.
  assert (hDef : YonedaMapLike CC (id CC (X := A)) X u = comp CC (id CC (X := A)) u).
  { reflexivity. }
  assert (hId : comp CC (id CC (X := A)) u = u).
  { apply id_comp. }
  transitivity (comp CC (id CC (X := A)) u).
  - exact hDef.
  - exact hId.
Qed.

Lemma yoneda_map_comp {C : Type} (CC : CategoryLike C)
    {A B D : C} (f : Hom CC A B) (g : Hom CC B D) :
    forall X : C, forall u : HomFunctorLike CC D X,
      YonedaMapLike CC (comp CC f g) X u = YonedaMapLike CC f X (YonedaMapLike CC g X u).
Proof.
  intros X u.
  assert (hLeft : YonedaMapLike CC (comp CC f g) X u = comp CC (comp CC f g) u).
  { reflexivity. }
  assert (hInner : YonedaMapLike CC g X u = comp CC g u).
  { reflexivity. }
  assert (hRight : YonedaMapLike CC f X (YonedaMapLike CC g X u) = comp CC f (YonedaMapLike CC g X u)).
  { reflexivity. }
  assert (hAssoc : comp CC (comp CC f g) u = comp CC f (comp CC g u)).
  { apply comp_assoc. }
  transitivity (comp CC (comp CC f g) u).
  - exact hLeft.
  - transitivity (comp CC f (comp CC g u)).
    + exact hAssoc.
    + transitivity (comp CC f (YonedaMapLike CC g X u)).
      * rewrite hInner.
        reflexivity.
      * symmetry.
        exact hRight.
Qed.

Definition YonedaFaithfulLike {C : Type} (CC : CategoryLike C) : Prop :=
  forall (A B : C) (f g : Hom CC A B),
    (forall X : C, YonedaMapLike CC f X = YonedaMapLike CC g X) -> f = g.

Lemma yoneda_faithful_cancel {C : Type} (CC : CategoryLike C)
    (hFaithful : YonedaFaithfulLike CC) {A B : C}
    (f g : Hom CC A B)
    (hEq : forall X : C, YonedaMapLike CC f X = YonedaMapLike CC g X) :
    f = g.
Proof.
  assert (hPointwise : forall X : C, YonedaMapLike CC f X = YonedaMapLike CC g X).
  {
    intro X.
    apply hEq.
  }
  assert (hAtB : YonedaMapLike CC f B = YonedaMapLike CC g B).
  { apply hPointwise. }
  assert (hAtId : YonedaMapLike CC f B (id CC (X := B)) = YonedaMapLike CC g B (id CC (X := B))).
  {
    apply (f_equal (fun q => q (id CC (X := B)))).
    exact hAtB.
  }
  assert (hComp : comp CC f (id CC (X := B)) = comp CC g (id CC (X := B))).
  {
    unfold YonedaMapLike in hAtId.
    exact hAtId.
  }
  assert (hById : f = g).
  {
    transitivity (comp CC f (id CC (X := B))).
    - symmetry.
      apply comp_id.
    - transitivity (comp CC g (id CC (X := B))).
      + exact hComp.
      + apply comp_id.
  }
  assert (hByFaithful : f = g).
  {
    apply (hFaithful A B f g).
    exact hPointwise.
  }
  transitivity g.
  - exact hByFaithful.
  - transitivity f.
    + symmetry.
      exact hById.
    + exact hById.
Qed.

Lemma yoneda_ext {C : Type} (CC : CategoryLike C)
    {A B : C} (f g : Hom CC A B)
    (hEq : forall X : C, forall u : HomFunctorLike CC B X,
      YonedaMapLike CC f X u = YonedaMapLike CC g X u) :
    f = g.
Proof.
  assert (hEval : YonedaMapLike CC f B (id CC (X := B)) = YonedaMapLike CC g B (id CC (X := B))).
  {
    exact (hEq B (id CC (X := B))).
  }
  assert (hAtId : comp CC f (id CC (X := B)) = comp CC g (id CC (X := B))).
  {
    unfold YonedaMapLike in hEval.
    exact hEval.
  }
  assert (hLeftId : f = comp CC f (id CC (X := B))).
  {
    symmetry.
    apply comp_id.
  }
  assert (hRightId : comp CC g (id CC (X := B)) = g).
  { apply comp_id. }
  transitivity (comp CC f (id CC (X := B))).
  - exact hLeftId.
  - transitivity (comp CC g (id CC (X := B))).
    + exact hAtId.
    + exact hRightId.
Qed.

Lemma yoneda_full_lift {C : Type} (CC : CategoryLike C)
    {A B : C}
    (tau : forall X : C, HomFunctorLike CC B X -> HomFunctorLike CC A X)
    (hNat : forall {X Y : C} (k : Hom CC X Y) (u : HomFunctorLike CC B X),
      tau Y (comp CC u k) = comp CC (tau X u) k) :
    exists f : Hom CC A B, f = tau B (id CC (X := B)).
Proof.
  assert (hNatId :
      tau B (comp CC (id CC (X := B)) (id CC (X := B)))
        = comp CC (tau B (id CC (X := B))) (id CC (X := B))).
  { apply hNat. }
  assert (hLeft :
      tau B (comp CC (id CC (X := B)) (id CC (X := B)))
        = tau B (id CC (X := B))).
  { rewrite id_comp. reflexivity. }
  assert (hRight :
      comp CC (tau B (id CC (X := B))) (id CC (X := B))
        = tau B (id CC (X := B))).
  { apply comp_id. }
  assert (hStable : tau B (id CC (X := B)) = tau B (id CC (X := B))).
  {
    transitivity (tau B (comp CC (id CC (X := B)) (id CC (X := B)))).
    - symmetry.
      exact hLeft.
    - transitivity (comp CC (tau B (id CC (X := B))) (id CC (X := B))).
      + exact hNatId.
      + exact hRight.
  }
  refine (ex_intro _ (tau B (id CC (X := B))) _).
  exact hStable.
Qed.

Lemma yoneda_full_spec {C : Type} (CC : CategoryLike C)
    {A B : C}
    (tau : forall X : C, HomFunctorLike CC B X -> HomFunctorLike CC A X)
    (hNat : forall {X Y : C} (k : Hom CC X Y) (u : HomFunctorLike CC B X),
      tau Y (comp CC u k) = comp CC (tau X u) k)
    (f : Hom CC A B)
    (hf : f = tau B (id CC (X := B))) :
    forall X : C, forall u : HomFunctorLike CC B X,
      tau X u = YonedaMapLike CC f X u.
Proof.
  intros X u.
  assert (hNatStep : tau X (comp CC (id CC (X := B)) u)
      = comp CC (tau B (id CC (X := B))) u).
  { apply hNat. }
  assert (hId : comp CC (id CC (X := B)) u = u).
  { apply id_comp. }
  assert (hf' : tau B (id CC (X := B)) = f).
  { symmetry. exact hf. }
  transitivity (tau X (comp CC (id CC (X := B)) u)).
  - rewrite hId.
    reflexivity.
  - rewrite hNatStep.
    rewrite hf'.
    unfold YonedaMapLike.
    reflexivity.
Qed.

Lemma yoneda_full_unique {C : Type} (CC : CategoryLike C)
    {A B : C}
    (tau : forall X : C, HomFunctorLike CC B X -> HomFunctorLike CC A X)
    (hNat : forall {X Y : C} (k : Hom CC X Y) (u : HomFunctorLike CC B X),
      tau Y (comp CC u k) = comp CC (tau X u) k)
    (f g : Hom CC A B)
    (hF : forall X : C, forall u : HomFunctorLike CC B X,
      tau X u = YonedaMapLike CC f X u)
    (hG : forall X : C, forall u : HomFunctorLike CC B X,
      tau X u = YonedaMapLike CC g X u) :
    f = g.
Proof.
  assert (hNatId :
      tau B (comp CC (id CC (X := B)) (id CC (X := B)))
        = comp CC (tau B (id CC (X := B))) (id CC (X := B))).
  { apply hNat. }
  assert (hNatConsistency : tau B (id CC (X := B)) = tau B (id CC (X := B))).
  {
    transitivity (tau B (comp CC (id CC (X := B)) (id CC (X := B)))).
    - symmetry.
      rewrite id_comp.
      reflexivity.
    - transitivity (comp CC (tau B (id CC (X := B))) (id CC (X := B))).
      + exact hNatId.
      + apply comp_id.
  }
  assert (hFId : tau B (id CC (X := B)) = comp CC f (id CC (X := B))).
  {
    unfold YonedaMapLike.
    exact (hF B (id CC (X := B))).
  }
  assert (hGId : tau B (id CC (X := B)) = comp CC g (id CC (X := B))).
  {
    unfold YonedaMapLike.
    exact (hG B (id CC (X := B))).
  }
  assert (hComp : comp CC f (id CC (X := B)) = comp CC g (id CC (X := B))).
  {
    transitivity (tau B (id CC (X := B))).
    - symmetry.
      exact hFId.
    - transitivity (tau B (id CC (X := B))).
      + exact hNatConsistency.
      + exact hGId.
  }
  transitivity (comp CC f (id CC (X := B))).
  - symmetry. apply comp_id.
  - transitivity (comp CC g (id CC (X := B))).
    + exact hComp.
    + apply comp_id.
Qed.
