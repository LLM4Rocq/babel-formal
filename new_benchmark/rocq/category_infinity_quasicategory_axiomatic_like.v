(*
BENCHMARK_ID: TINY_MATHLIB_BATCH05_CATEGORY_INFINITY_QUASICATEGORY_AXIOMATIC_LIKE
PAIR_STEM: category_infinity_quasicategory_axiomatic_like
MATH_DOMAIN: Category Theory
SOURCE_MATHLIB: Mathlib/CategoryTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class CatStruct_infinity_quasicategory (Obj : Type) := {
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

Record Functor_infinity_quasicategory {Obj : Type}
    (C : CatStruct_infinity_quasicategory Obj) := {
  obj : Obj -> Obj;
  map : forall {X Y : Obj}, Hom C X Y -> Hom C X Y;
  map_id : forall X : Obj, map (id C (X := X)) = id C;
  map_comp : forall {X Y Z : Obj} (f : Hom C X Y) (g : Hom C Y Z),
      map (comp C f g) = comp C (map f) (map g)
}.

Record NatIso_infinity_quasicategory {Obj : Type}
    (C : CatStruct_infinity_quasicategory Obj)
    (F G : Functor_infinity_quasicategory C) := {
  hom : forall X : Obj, Hom C X X;
  inv : forall X : Obj, Hom C X X;
  left_inv : forall X : Obj,
      comp C (hom X) (inv X) = id C;
  right_inv : forall X : Obj,
      comp C (inv X) (hom X) = id C;
  naturality : forall {X Y : Obj} (f : Hom C X Y),
      comp C (map F f) (hom Y) = comp C (hom X) (map G f)
}.

Definition whisker_infinity_quasicategory {Obj : Type}
    {C : CatStruct_infinity_quasicategory Obj}
    (F G H : Functor_infinity_quasicategory C)
    (eta : @NatIso_infinity_quasicategory Obj C F G) :
    @NatIso_infinity_quasicategory Obj C F G.
Proof.
  refine {| hom := hom eta;
            inv := inv eta |}.
  - intro X.
    exact (left_inv eta X).
  - intro X.
    exact (right_inv eta X).
  - intros X Y f.
    assert (hNat : comp C (map F f) (hom eta Y) = comp C (hom eta X) (map G f)).
    { apply naturality. }
    exact hNat.
Defined.

Arguments whisker_infinity_quasicategory {Obj} {C} _ _ _ _.

Definition compose_infinity_quasicategory {Obj : Type}
    {C : CatStruct_infinity_quasicategory Obj}
    (F G : Functor_infinity_quasicategory C) :
    Functor_infinity_quasicategory C.
Proof.
  refine {| obj := fun X => obj G (obj F X);
            map := fun X Y f => map G (map F f) |}.
  - intro X.
    assert (hF : map F (id C (X := X)) = id C).
    { apply map_id. }
    assert (hG : map G (id C (X := X)) = id C).
    { apply map_id. }
    rewrite hF.
    exact hG.
  - intros X Y Z f g.
    assert (hF : map F (comp C f g) = comp C (map F f) (map F g)).
    { apply map_comp. }
    assert (hG : map G (comp C (map F f) (map F g)) =
      comp C (map G (map F f)) (map G (map F g))).
    { apply map_comp. }
    rewrite hF.
    exact hG.
Defined.

Lemma whisker_assoc_infinity_quasicategory {Obj : Type}
    {C : CatStruct_infinity_quasicategory Obj}
    (F G H K : Functor_infinity_quasicategory C)
    (eta : @NatIso_infinity_quasicategory Obj C F G) :
    forall X : Obj,
      hom (whisker_infinity_quasicategory F G K
        (whisker_infinity_quasicategory F G H eta)) X =
      hom (whisker_infinity_quasicategory F G
        (compose_infinity_quasicategory H K) eta) X.
Proof.
  intro X.
  assert (hLeft :
      hom (whisker_infinity_quasicategory F G K
        (whisker_infinity_quasicategory F G H eta)) X = hom eta X).
  { reflexivity. }
  assert (hRight :
      hom (whisker_infinity_quasicategory F G
        (compose_infinity_quasicategory H K) eta) X = hom eta X).
  { reflexivity. }
  rewrite hLeft.
  symmetry.
  exact hRight.
Qed.

Lemma unit_whisker_infinity_quasicategory {Obj : Type}
    {C : CatStruct_infinity_quasicategory Obj}
    (F G : Functor_infinity_quasicategory C)
    (eta : @NatIso_infinity_quasicategory Obj C F G) :
    forall X : Obj,
      hom (whisker_infinity_quasicategory F G
        ({| obj := fun Z => Z;
            map := fun X Y f => f;
            map_id := fun X => eq_refl;
            map_comp := fun X Y Z f g => eq_refl |})
        eta) X = hom eta X.
Proof.
  intro X.
  assert (hDef :
      hom (whisker_infinity_quasicategory F G
        ({| obj := fun Z => Z;
            map := fun X Y f => f;
            map_id := fun X => eq_refl;
            map_comp := fun X Y Z f g => eq_refl |}) eta) X = hom eta X).
  { reflexivity. }
  exact hDef.
Qed.

Lemma counit_whisker_infinity_quasicategory {Obj : Type}
    {C : CatStruct_infinity_quasicategory Obj}
    (F G H : Functor_infinity_quasicategory C)
    (eta : @NatIso_infinity_quasicategory Obj C F G) :
    forall X : Obj,
      comp C (hom (whisker_infinity_quasicategory F G H eta) X)
        (inv (whisker_infinity_quasicategory F G H eta) X) = id C.
Proof.
  intro X.
  assert (hLeft :
      comp C (hom (whisker_infinity_quasicategory F G H eta) X)
        (inv (whisker_infinity_quasicategory F G H eta) X) = id C).
  {
    exact (left_inv (whisker_infinity_quasicategory F G H eta) X).
  }
  exact hLeft.
Qed.

Lemma pasting_coherence_infinity_quasicategory {Obj : Type}
    {C : CatStruct_infinity_quasicategory Obj}
    (F G H : Functor_infinity_quasicategory C)
    (eta : @NatIso_infinity_quasicategory Obj C F G)
    {X Y Z : Obj} (f : Hom C X Y) (g : Hom C Y Z) :
    comp C (map F (comp C f g))
      (hom (whisker_infinity_quasicategory F G H eta) Z) =
    comp C (hom (whisker_infinity_quasicategory F G H eta) X)
      (map G (comp C f g)).
Proof.
  assert (hNat :
      comp C (map F (comp C f g)) (hom eta Z) =
      comp C (hom eta X) (map G (comp C f g))).
  { apply naturality. }
  assert (hLeft : hom (whisker_infinity_quasicategory F G H eta) Z = hom eta Z).
  { reflexivity. }
  assert (hRight : hom (whisker_infinity_quasicategory F G H eta) X = hom eta X).
  { reflexivity. }
  rewrite hLeft.
  rewrite hRight.
  exact hNat.
Qed.

Lemma comparison_full_infinity_quasicategory {Obj : Type}
    {C : CatStruct_infinity_quasicategory Obj}
    (H G : Functor_infinity_quasicategory C)
    (hfull : forall {X Y : Obj} (f : Hom C X Y),
      exists g : Hom C X Y, map H g = f)
    {X Y : Obj} (f : Hom C X Y) :
    exists g : Hom C X Y,
      map (compose_infinity_quasicategory H G) g = map G f.
Proof.
  destruct (hfull _ _ f) as [g hg].
  exists g.
  change (map G (map H g) = map G f).
  rewrite hg.
  reflexivity.
Qed.

Lemma comparison_faithful_infinity_quasicategory {Obj : Type}
    {C : CatStruct_infinity_quasicategory Obj}
    (H G : Functor_infinity_quasicategory C)
    (hfaith_H : forall {X Y : Obj} {f g : Hom C X Y}, map H f = map H g -> f = g)
    (hfaith_G : forall {X Y : Obj} {f g : Hom C X Y}, map G f = map G g -> f = g)
    {X Y : Obj} {f g : Hom C X Y}
    (hcomp : map (compose_infinity_quasicategory H G) f =
      map (compose_infinity_quasicategory H G) g) :
    f = g.
Proof.
  change (map G (map H f) = map G (map H g)) in hcomp.
  assert (hInner : map H f = map H g).
  { apply hfaith_G. exact hcomp. }
  apply hfaith_H.
  exact hInner.
Qed.

Lemma equivalence_core_infinity_quasicategory {Obj : Type}
    {C : CatStruct_infinity_quasicategory Obj}
    (H G : Functor_infinity_quasicategory C)
    (hfull : forall {X Y : Obj} (f : Hom C X Y),
      exists g : Hom C X Y, map H g = f)
    (hfaith_H : forall {X Y : Obj} {f g : Hom C X Y}, map H f = map H g -> f = g)
    (hfaith_G : forall {X Y : Obj} {f g : Hom C X Y}, map G f = map G g -> f = g) :
    (forall {X Y : Obj} (f : Hom C X Y),
      exists g : Hom C X Y, map (compose_infinity_quasicategory H G) g = map G f) /\
    (forall {X Y : Obj} {f g : Hom C X Y},
      map (compose_infinity_quasicategory H G) f =
        map (compose_infinity_quasicategory H G) g -> f = g).
Proof.
  split.
  - intros X Y f.
    exact (comparison_full_infinity_quasicategory H G hfull f).
  - intros X Y f g hEq.
    exact (comparison_faithful_infinity_quasicategory H G hfaith_H hfaith_G hEq).
Qed.
