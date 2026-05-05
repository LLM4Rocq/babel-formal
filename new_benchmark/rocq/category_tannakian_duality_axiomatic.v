(*
BENCHMARK_ID: TINY_MATHLIB_BATCH04_CATEGORY_TANNAKIAN_DUALITY_AXIOMATIC
PAIR_STEM: category_tannakian_duality_axiomatic
MATH_DOMAIN: Category Theory / Representation Theory
SOURCE_MATHLIB: Mathlib/CategoryTheory/Monoidal
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class TensorCategoryLike (Obj : Type) := {
  Hom : Obj -> Obj -> Type;
  id : forall {X : Obj}, Hom X X;
  comp : forall {X Y Z : Obj}, Hom X Y -> Hom Y Z -> Hom X Z;
  comp_assoc :
    forall {W X Y Z : Obj} (f : Hom W X) (g : Hom X Y) (h : Hom Y Z),
      comp (comp f g) h = comp f (comp g h);
  id_comp : forall {X Y : Obj} (f : Hom X Y), comp id f = f;
  comp_id : forall {X Y : Obj} (f : Hom X Y), comp f id = f;
  tensorObj : Obj -> Obj -> Obj;
  tensorHom :
    forall {X1 X2 Y1 Y2 : Obj},
      Hom X1 Y1 -> Hom X2 Y2 -> Hom (tensorObj X1 X2) (tensorObj Y1 Y2);
  tensor_id : forall X Y : Obj, tensorHom (id (X := X)) (id (X := Y)) = id;
  tensor_comp :
    forall {A1 A2 B1 B2 C1 C2 : Obj}
      (f1 : Hom A1 B1) (g1 : Hom B1 C1)
      (f2 : Hom A2 B2) (g2 : Hom B2 C2),
      tensorHom (comp f1 g1) (comp f2 g2) = comp (tensorHom f1 f2) (tensorHom g1 g2)
}.

Arguments Hom {Obj} _ _ _.
Arguments id {Obj} _ {X}.
Arguments comp {Obj} _ {X Y Z} _ _.
Arguments comp_assoc {Obj} _ {W X Y Z} _ _ _.
Arguments id_comp {Obj} _ {X Y} _.
Arguments comp_id {Obj} _ {X Y} _.
Arguments tensorObj {Obj} _ _ _.
Arguments tensorHom {Obj} _ {X1 X2 Y1 Y2} _ _.

Record FiberFunctorLike (C : Type) (TC : TensorCategoryLike C) := {
  FObj : C -> Type;
  map : forall {X Y : C}, Hom TC X Y -> FObj X -> FObj Y;
  map_id : forall (X : C) (x : FObj X), map (id TC (X := X)) x = x;
  map_comp :
    forall {X Y Z : C} (f : Hom TC X Y) (g : Hom TC Y Z) (x : FObj X),
      map (comp TC f g) x = map g (map f x);
  reflects_eq :
    forall {X Y : C} (f g : Hom TC X Y), (forall x : FObj X, map f x = map g x) -> f = g;
  realizes :
    forall {X Y : C} (t : FObj X -> FObj Y),
      exists f : Hom TC X Y, forall x : FObj X, map f x = t x
}.

Arguments FObj {C TC} _ _.
Arguments map {C TC} _ {X Y} _ _.
Arguments map_id {C TC} _ X x.
Arguments map_comp {C TC} _ {X Y Z} _ _ _.
Arguments reflects_eq {C TC} _ {X Y} _ _ _.
Arguments realizes {C TC} _ {X Y} _.

Definition EndCoalgebraLike {C : Type} {TC : TensorCategoryLike C}
    (omega : @FiberFunctorLike C TC) : Prop :=
  forall (X Y : C) (f g : Hom TC X Y),
    (forall x : FObj omega X, map omega f x = map omega g x) -> f = g.

Definition RepresentationLike {C : Type} {TC : TensorCategoryLike C}
    (omega : @FiberFunctorLike C TC) : Prop :=
  forall (X Y : C) (t : FObj omega X -> FObj omega Y),
    exists f : Hom TC X Y, forall x : FObj omega X, map omega f x = t x.

Definition ReconstructionLike {C : Type} {TC : TensorCategoryLike C}
    (omega : @FiberFunctorLike C TC) : Prop :=
  EndCoalgebraLike omega /\ RepresentationLike omega /\
    (forall X : C, exists x : FObj omega X, True).

Definition ComparisonLike {C : Type} {TC : TensorCategoryLike C}
    (omega : @FiberFunctorLike C TC) : Prop :=
  ReconstructionLike omega /\
    (forall (X Y : C) (f g : Hom TC X Y),
      (forall x : FObj omega X, map omega f x = map omega g x) <-> f = g).

Lemma fiber_reflects_iso {C : Type} {TC : TensorCategoryLike C}
    (omega : @FiberFunctorLike C TC) (hCmp : ComparisonLike omega)
    {X Y : C} (f g : Hom TC X Y)
    (hpt : forall x : FObj omega X, map omega f x = map omega g x) :
    f = g.
Proof.
  assert (hiff : (forall x : FObj omega X, map omega f x = map omega g x) <-> f = g).
  { exact (proj2 hCmp X Y f g). }
  assert (hForward : (forall x : FObj omega X, map omega f x = map omega g x) -> f = g).
  { exact (proj1 hiff). }
  assert (hEq : f = g).
  { exact (hForward hpt). }
  assert (hRecFaithful : EndCoalgebraLike omega).
  { exact (proj1 (proj1 hCmp)). }
  assert (hEq' : f = g).
  { exact (hRecFaithful X Y f g hpt). }
  assert (hKeep' : f = g).
  { exact hEq'. }
  exact hEq.
Qed.

Lemma reconstruction_faithful {C : Type} {TC : TensorCategoryLike C}
    (omega : @FiberFunctorLike C TC)
    (hRec : ReconstructionLike omega) :
    EndCoalgebraLike omega.
Proof.
  assert (hFaith : EndCoalgebraLike omega).
  { exact (proj1 hRec). }
  assert (hRep : RepresentationLike omega).
  { exact (proj1 (proj2 hRec)). }
  assert (hNonempty : forall X : C, exists x : FObj omega X, True).
  { exact (proj2 (proj2 hRec)). }
  assert (hKeepRep : RepresentationLike omega).
  { exact hRep. }
  assert (hKeepNonempty : forall X : C, exists x : FObj omega X, True).
  { exact hNonempty. }
  exact hFaith.
Qed.

Lemma reconstruction_full {C : Type} {TC : TensorCategoryLike C}
    (omega : @FiberFunctorLike C TC)
    (hRec : ReconstructionLike omega)
    {X Y : C} (t : FObj omega X -> FObj omega Y) :
    exists f : Hom TC X Y, forall x : FObj omega X, map omega f x = t x.
Proof.
  assert (hRep : RepresentationLike omega).
  { exact (proj1 (proj2 hRec)). }
  assert (hWitness : exists f : Hom TC X Y, forall x : FObj omega X, map omega f x = t x).
  { exact (hRep X Y t). }
  destruct hWitness as [f hf].
  assert (hKeep : forall x : FObj omega X, map omega f x = t x).
  { exact hf. }
  exists f.
  exact hKeep.
Qed.

Lemma tannaka_unit_like {C : Type} {TC : TensorCategoryLike C}
    (omega : @FiberFunctorLike C TC)
    (hCmp : ComparisonLike omega)
    {X Y : C} (f : Hom TC X Y) :
    exists u : Hom TC X Y,
      (forall x : FObj omega X, map omega u x = map omega f x) /\ u = f.
Proof.
  assert (hRec : ReconstructionLike omega).
  { exact (proj1 hCmp). }
  assert (hFull : exists u : Hom TC X Y, forall x : FObj omega X, map omega u x = map omega f x).
  { exact (@reconstruction_full C TC omega hRec X Y (fun x => map omega f x)). }
  destruct hFull as [u hu].
  assert (hiff : (forall x : FObj omega X, map omega u x = map omega f x) <-> u = f).
  { exact (proj2 hCmp X Y u f). }
  assert (huEq : u = f).
  { exact (proj1 hiff hu). }
  assert (huMap : forall x : FObj omega X, map omega u x = map omega f x).
  { exact hu. }
  exists u.
  split.
  - exact huMap.
  - exact huEq.
Qed.

Lemma tannaka_counit_like {C : Type} {TC : TensorCategoryLike C}
    (omega : @FiberFunctorLike C TC)
    (hCmp : ComparisonLike omega)
    {X Y : C} (f : Hom TC X Y) :
    exists v : Hom TC X Y,
      v = f /\ (forall x : FObj omega X, map omega f x = map omega v x).
Proof.
  destruct (@tannaka_unit_like C TC omega hCmp X Y f) as [u [huMap huEq]].
  exists u.
  split.
  - exact huEq.
  - intro x.
    specialize (huMap x) as hstep.
    symmetry.
    exact hstep.
Qed.

Lemma tannaka_equivalence_like {C : Type} {TC : TensorCategoryLike C}
    (omega : @FiberFunctorLike C TC)
    (hCmp : ComparisonLike omega) :
    (forall {X Y : C} (f : Hom TC X Y), exists v : Hom TC X Y, v = f) /\
      (forall {X Y : C} (f g : Hom TC X Y),
        (forall x : FObj omega X, map omega f x = map omega g x) <-> f = g).
Proof.
  split.
  - intros X Y f.
    destruct (@tannaka_counit_like C TC omega hCmp X Y f) as [v [hvEq hvMap]].
    assert (hKeep : forall x : FObj omega X, map omega f x = map omega v x).
    { exact hvMap. }
    exists v.
    exact hvEq.
  - intros X Y f g.
    assert (hiff : (forall x : FObj omega X, map omega f x = map omega g x) <-> f = g).
    { exact (proj2 hCmp X Y f g). }
    exact hiff.
Qed.
