(*
BENCHMARK_ID: TINY_MATHLIB_BATCH06_CATEGORY_GROTHENDIECK_FIBRATION_CLEAVAGE_LIKE
PAIR_STEM: category_grothendieck_fibration_cleavage_like
MATH_DOMAIN: Category Theory
SOURCE_MATHLIB: Mathlib/CategoryTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class FibStruct_grothendieck_fibration_cleavage (Obj : Type) := {
  Hom : Obj -> Obj -> Type;
  id : forall {X : Obj}, Hom X X;
  comp : forall {X Y Z : Obj}, Hom X Y -> Hom Y Z -> Hom X Z;
  comp_assoc :
    forall {W X Y Z : Obj} (f : Hom W X) (g : Hom X Y) (h : Hom Y Z),
      comp (comp f g) h = comp f (comp g h);
  id_comp : forall {X Y : Obj} (f : Hom X Y), comp id f = f;
  comp_id : forall {X Y : Obj} (f : Hom X Y), comp f id = f;
  cancel_right :
    forall {X Y Z : Obj} (f g : Hom X Y) (h : Hom Y Z),
      comp f h = comp g h -> f = g
}.

Arguments Hom {Obj} _ _ _.
Arguments id {Obj} _ {X}.
Arguments comp {Obj} _ {X Y Z} _ _.
Arguments comp_assoc {Obj} _ {W X Y Z} _ _ _.
Arguments id_comp {Obj} _ {X Y} _.
Arguments comp_id {Obj} _ {X Y} _.
Arguments cancel_right {Obj} _ {X Y Z} _ _ _ _.

Record CleavageData_grothendieck_fibration_cleavage
    (Obj : Type)
    (F : FibStruct_grothendieck_fibration_cleavage Obj) := {
  pullObj : forall {X Y : Obj}, Hom F X Y -> Obj -> Obj;
  lift : forall {X Y : Obj} (f : Hom F X Y) (e : Obj), Hom F (pullObj f e) e;
  compare : forall {X Y Z : Obj}
    (f : Hom F X Y) (g : Hom F Y Z) (e : Obj),
      Hom F (pullObj (comp F f g) e) (pullObj f (pullObj g e));
  factor :
    forall {X Y : Obj} (f : Hom F X Y) (e z : Obj) (h : Hom F z e),
      exists u : Hom F z (pullObj f e), comp F u (lift f e) = h;
  unique :
    forall {X Y : Obj} (f : Hom F X Y) (e z : Obj) (h : Hom F z e)
      (u1 u2 : Hom F z (pullObj f e)),
      comp F u1 (lift f e) = h -> comp F u2 (lift f e) = h -> u1 = u2;
  compare_spec :
    forall {X Y Z : Obj} (f : Hom F X Y) (g : Hom F Y Z) (e : Obj),
      comp F (comp F (compare f g e) (lift f (pullObj g e))) (lift g e)
      = lift (comp F f g) e;
  reindex_comp :
    forall {X Y Z : Obj} (f : Hom F X Y) (g : Hom F Y Z) (e z : Obj)
      (u : Hom F z (pullObj (comp F f g) e)),
      comp F u (lift (comp F f g) e) =
        comp F (comp F (comp F u (compare f g e)) (lift f (pullObj g e))) (lift g e)
}.

Definition cartesian_lift_grothendieck_fibration_cleavage
    {Obj : Type} {F : FibStruct_grothendieck_fibration_cleavage Obj}
    (clv : @CleavageData_grothendieck_fibration_cleavage Obj F)
    {X Y : Obj} (f : Hom F X Y) (e : Obj) :
    Hom F (pullObj clv f e) e :=
  lift clv f e.

Definition pullback_obj_grothendieck_fibration_cleavage
    {Obj : Type} {F : FibStruct_grothendieck_fibration_cleavage Obj}
    (clv : @CleavageData_grothendieck_fibration_cleavage Obj F)
    {X Y : Obj} (f : Hom F X Y) (e : Obj) : Obj :=
  pullObj clv f e.

Definition reindex_morphism_grothendieck_fibration_cleavage
    {Obj : Type} {F : FibStruct_grothendieck_fibration_cleavage Obj}
    (clv : @CleavageData_grothendieck_fibration_cleavage Obj F)
    {X Y : Obj}
    (f : Hom F X Y) (e z : Obj)
    (u : Hom F z (pullback_obj_grothendieck_fibration_cleavage clv f e)) : Hom F z e :=
  comp F u (cartesian_lift_grothendieck_fibration_cleavage clv f e).

Lemma cartesian_lift_exists_grothendieck_fibration_cleavage
    {Obj : Type} {F : FibStruct_grothendieck_fibration_cleavage Obj}
    (clv : @CleavageData_grothendieck_fibration_cleavage Obj F)
    {X Y : Obj}
    (f : Hom F X Y) (e z : Obj) (h : Hom F z e) :
    exists u : Hom F z (pullback_obj_grothendieck_fibration_cleavage clv f e),
      reindex_morphism_grothendieck_fibration_cleavage clv f e z u = h.
Proof.
  pose proof (factor clv f e z h) as hfac.
  destruct hfac as [u hu].
  assert (hu' : comp F u (cartesian_lift_grothendieck_fibration_cleavage clv f e) = h).
  { exact hu. }
  exists u.
  change (comp F u (cartesian_lift_grothendieck_fibration_cleavage clv f e) = h).
  exact hu'.
Qed.

Lemma cartesian_lift_unique_grothendieck_fibration_cleavage
    {Obj : Type} {F : FibStruct_grothendieck_fibration_cleavage Obj}
    (clv : @CleavageData_grothendieck_fibration_cleavage Obj F)
    {X Y : Obj}
    (f : Hom F X Y) (e z : Obj) (h : Hom F z e)
    (u1 u2 : Hom F z (pullback_obj_grothendieck_fibration_cleavage clv f e))
    (hu1 : reindex_morphism_grothendieck_fibration_cleavage clv f e z u1 = h)
    (hu2 : reindex_morphism_grothendieck_fibration_cleavage clv f e z u2 = h) :
    u1 = u2.
Proof.
  assert (hu1' : comp F u1 (lift clv f e) = h).
  { exact hu1. }
  assert (hu2' : comp F u2 (lift clv f e) = h).
  { exact hu2. }
  assert (hcomp : comp F u1 (lift clv f e) = comp F u2 (lift clv f e)).
  {
    rewrite hu1'. rewrite hu2'. reflexivity.
  }
  exact (cancel_right F u1 u2 (lift clv f e) hcomp).
Qed.

Lemma reindex_identity_grothendieck_fibration_cleavage
    {Obj : Type} {F : FibStruct_grothendieck_fibration_cleavage Obj}
    (clv : @CleavageData_grothendieck_fibration_cleavage Obj F)
    {X Y : Obj} (f : Hom F X Y) (e : Obj) :
    reindex_morphism_grothendieck_fibration_cleavage
      clv (comp F (id F (X := X)) f) e
      (pullback_obj_grothendieck_fibration_cleavage clv (comp F (id F (X := X)) f) e)
      (id F (X := pullback_obj_grothendieck_fibration_cleavage clv (comp F (id F (X := X)) f) e))
      = cartesian_lift_grothendieck_fibration_cleavage clv (comp F (id F (X := X)) f) e.
Proof.
  change (comp F (id F) (cartesian_lift_grothendieck_fibration_cleavage clv (comp F (id F (X := X)) f) e) =
    cartesian_lift_grothendieck_fibration_cleavage clv (comp F (id F (X := X)) f) e).
  apply (id_comp F).
Qed.

Lemma reindex_composition_grothendieck_fibration_cleavage
    {Obj : Type} {F : FibStruct_grothendieck_fibration_cleavage Obj}
    (clv : @CleavageData_grothendieck_fibration_cleavage Obj F)
    {X Y Z : Obj} (f : Hom F X Y) (g : Hom F Y Z)
    (e z : Obj)
    (u : Hom F z (pullback_obj_grothendieck_fibration_cleavage clv (comp F f g) e)) :
    reindex_morphism_grothendieck_fibration_cleavage clv (comp F f g) e z u =
      reindex_morphism_grothendieck_fibration_cleavage clv g e z
        (reindex_morphism_grothendieck_fibration_cleavage
          clv f (pullback_obj_grothendieck_fibration_cleavage clv g e) z
          (comp F u (compare clv f g e))).
Proof.
  cbn [reindex_morphism_grothendieck_fibration_cleavage
    cartesian_lift_grothendieck_fibration_cleavage
    pullback_obj_grothendieck_fibration_cleavage].
  change (comp F u (lift clv (comp F f g) e) =
    comp F (comp F (comp F u (compare clv f g e))
      (lift clv f (pullObj clv g e))) (lift clv g e)).
  pose proof (reindex_comp clv f g e z u) as hcore.
  exact hcore.
Qed.

Lemma cartesian_factorization_grothendieck_fibration_cleavage
    {Obj : Type} {F : FibStruct_grothendieck_fibration_cleavage Obj}
    (clv : @CleavageData_grothendieck_fibration_cleavage Obj F)
    {X Y : Obj}
    (f : Hom F X Y) (e z : Obj) (h : Hom F z e) :
    exists u : Hom F z (pullback_obj_grothendieck_fibration_cleavage clv f e),
      reindex_morphism_grothendieck_fibration_cleavage clv f e z u = h /\
      (forall v : Hom F z (pullback_obj_grothendieck_fibration_cleavage clv f e),
        reindex_morphism_grothendieck_fibration_cleavage clv f e z v = h -> v = u).
Proof.
  pose proof (cartesian_lift_exists_grothendieck_fibration_cleavage clv f e z h) as hex.
  destruct hex as [u hu].
  exists u.
  split.
  - exact hu.
  - intros v hv.
    exact (unique clv f e z v u hv hu).
Qed.

Lemma cleavage_coherence_grothendieck_fibration_cleavage
    {Obj : Type} {F : FibStruct_grothendieck_fibration_cleavage Obj}
    (clv : @CleavageData_grothendieck_fibration_cleavage Obj F)
    {X Y Z : Obj} (f : Hom F X Y) (g : Hom F Y Z)
    (e z : Obj)
    (u : Hom F z (pullback_obj_grothendieck_fibration_cleavage clv (comp F f g) e)) :
    reindex_morphism_grothendieck_fibration_cleavage clv (comp F f g) e z u =
      reindex_morphism_grothendieck_fibration_cleavage clv g e z
        (comp F (comp F u (compare clv f g e))
          (cartesian_lift_grothendieck_fibration_cleavage
            clv f (pullback_obj_grothendieck_fibration_cleavage clv g e))).
Proof.
  pose proof (reindex_composition_grothendieck_fibration_cleavage clv f g e z u) as hcomp.
  assert (hunfold :
      reindex_morphism_grothendieck_fibration_cleavage
        clv f (pullback_obj_grothendieck_fibration_cleavage clv g e) z
        (comp F u (compare clv f g e))
      = comp F (comp F u (compare clv f g e))
          (cartesian_lift_grothendieck_fibration_cleavage
            clv f (pullback_obj_grothendieck_fibration_cleavage clv g e))).
  { reflexivity. }
  rewrite hcomp.
  rewrite hunfold.
  reflexivity.
Qed.

Lemma fiber_equivalence_grothendieck_fibration_cleavage
    {Obj : Type} {F : FibStruct_grothendieck_fibration_cleavage Obj}
    (clv : @CleavageData_grothendieck_fibration_cleavage Obj F)
    {X Y : Obj} (f : Hom F X Y) (e z : Obj)
    (u1 u2 : Hom F z (pullback_obj_grothendieck_fibration_cleavage clv f e)) :
    reindex_morphism_grothendieck_fibration_cleavage clv f e z u1 =
      reindex_morphism_grothendieck_fibration_cleavage clv f e z u2 <->
      u1 = u2.
Proof.
  split.
  - intro hreindex.
    assert (hu1 : reindex_morphism_grothendieck_fibration_cleavage clv f e z u1 =
      reindex_morphism_grothendieck_fibration_cleavage clv f e z u1).
    { reflexivity. }
    assert (hu2 : reindex_morphism_grothendieck_fibration_cleavage clv f e z u2 =
      reindex_morphism_grothendieck_fibration_cleavage clv f e z u1).
    { symmetry. exact hreindex. }
    assert (huniq : u2 = u1).
    {
      exact (unique clv f e z u2 u1 hu2 hu1).
    }
    symmetry.
    exact huniq.
  - intro hu.
    rewrite hu.
    reflexivity.
Qed.
