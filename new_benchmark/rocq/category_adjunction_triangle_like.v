(*
BENCHMARK_ID: TINY_MATHLIB_BATCH03_CATEGORY_ADJUNCTION_TRIANGLE_LIKE
PAIR_STEM: category_adjunction_triangle_like
MATH_DOMAIN: Category Theory
SOURCE_MATHLIB: Mathlib/CategoryTheory/Adjunction
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

Record FunctorLike (C D : Type) (CC : CategoryLike C) (DD : CategoryLike D) := {
  obj : C -> D;
  map : forall {X Y : C}, Hom CC X Y -> Hom DD (obj X) (obj Y);
  map_id : forall X : C, map (id CC (X := X)) = id DD;
  map_comp : forall {X Y Z : C} (f : Hom CC X Y) (g : Hom CC Y Z),
      map (comp CC f g) = comp DD (map f) (map g)
}.

Record NatTransLike {C D : Type} {CC : CategoryLike C} {DD : CategoryLike D}
    (F G : @FunctorLike C D CC DD) := {
  app : forall X : C, Hom DD (obj F X) (obj G X);
  naturality : forall {X Y : C} (f : Hom CC X Y),
      comp DD (app X) (map G f) = comp DD (map F f) (app Y)
}.

Record AdjunctionLike {C D : Type} {CC : CategoryLike C} {DD : CategoryLike D}
    (F : @FunctorLike C D CC DD) (G : @FunctorLike D C DD CC) := {
  unit : forall X : C, Hom CC X (obj G (obj F X));
  counit : forall Y : D, Hom DD (obj F (obj G Y)) Y;
  unit_naturality_axiom :
    forall {X X' : C} (f : Hom CC X X'),
      comp CC f (unit X') = comp CC (unit X) (map G (map F f));
  counit_naturality_axiom :
    forall {Y Y' : D} (g : Hom DD Y Y'),
      comp DD (map F (map G g)) (counit Y') = comp DD (counit Y) g;
  triangle_left_axiom :
    forall X : C,
      comp DD (map F (unit X)) (counit (obj F X)) = id DD;
  triangle_right_axiom :
    forall Y : D,
      comp CC (unit (obj G Y)) (map G (counit Y)) = id CC;
  hom_equiv_to :
    forall {X : C} {Y : D}, Hom DD (obj F X) Y -> Hom CC X (obj G Y);
  hom_equiv_from :
    forall {X : C} {Y : D}, Hom CC X (obj G Y) -> Hom DD (obj F X) Y;
  hom_equiv_natural_left_axiom :
    forall {X X' : C} {Y : D} (f : Hom CC X X') (k : Hom DD (obj F X') Y),
      hom_equiv_to (comp DD (map F f) k) = comp CC f (hom_equiv_to k);
  hom_equiv_natural_right_axiom :
    forall {X : C} {Y Y' : D} (k : Hom DD (obj F X) Y) (g : Hom DD Y Y'),
      hom_equiv_to (comp DD k g) = comp CC (hom_equiv_to k) (map G g)
}.

Definition leftWhisker {C D : Type} {CC : CategoryLike C} {DD : CategoryLike D}
    (F : @FunctorLike C D CC DD) {X Y : C} (f : Hom CC X Y) :
    Hom DD (obj F X) (obj F Y) :=
  map F f.

Definition rightWhisker {C D : Type} {CC : CategoryLike C} {DD : CategoryLike D}
    (F : @FunctorLike C D CC DD) {X Y : C} (f : Hom CC X Y) :
    Hom DD (obj F X) (obj F Y) :=
  map F f.

Lemma unit_naturality {C D : Type} {CC : CategoryLike C} {DD : CategoryLike D}
    (F : @FunctorLike C D CC DD) (G : @FunctorLike D C DD CC)
    (A : @AdjunctionLike C D CC DD F G) {X X' : C} (f : Hom CC X X') :
    comp CC f (unit A X') = comp CC (unit A X) (rightWhisker G (leftWhisker F f)).
Proof.
  assert (hNat : comp CC f (unit A X') = comp CC (unit A X) (map G (map F f))).
  { exact (unit_naturality_axiom A f). }
  assert (hLeft : leftWhisker F f = map F f).
  { reflexivity. }
  assert (hRightStep : rightWhisker G (leftWhisker F f) = map G (leftWhisker F f)).
  { reflexivity. }
  assert (hRight : rightWhisker G (leftWhisker F f) = map G (map F f)).
  {
    rewrite hRightStep.
    rewrite hLeft.
    reflexivity.
  }
  assert (hPost : comp CC (unit A X) (map G (map F f))
      = comp CC (unit A X) (rightWhisker G (leftWhisker F f))).
  {
    rewrite hRight.
    reflexivity.
  }
  rewrite hPost in hNat.
  exact hNat.
Qed.

Lemma counit_naturality {C D : Type} {CC : CategoryLike C} {DD : CategoryLike D}
    (F : @FunctorLike C D CC DD) (G : @FunctorLike D C DD CC)
    (A : @AdjunctionLike C D CC DD F G) {Y Y' : D} (g : Hom DD Y Y') :
    comp DD (leftWhisker F (rightWhisker G g)) (counit A Y') = comp DD (counit A Y) g.
Proof.
  assert (hNat : comp DD (map F (map G g)) (counit A Y') = comp DD (counit A Y) g).
  { exact (counit_naturality_axiom A g). }
  assert (hRight : rightWhisker G g = map G g).
  { reflexivity. }
  assert (hLeftStep : leftWhisker F (rightWhisker G g) = map F (rightWhisker G g)).
  { reflexivity. }
  assert (hLeft : leftWhisker F (rightWhisker G g) = map F (map G g)).
  {
    rewrite hLeftStep.
    rewrite hRight.
    reflexivity.
  }
  assert (hPre : comp DD (leftWhisker F (rightWhisker G g)) (counit A Y')
      = comp DD (map F (map G g)) (counit A Y')).
  {
    rewrite hLeft.
    reflexivity.
  }
  rewrite hPre.
  exact hNat.
Qed.

Lemma triangle_left {C D : Type} {CC : CategoryLike C} {DD : CategoryLike D}
    (F : @FunctorLike C D CC DD) (G : @FunctorLike D C DD CC)
    (A : @AdjunctionLike C D CC DD F G) (X : C) :
    comp DD (leftWhisker F (unit A X)) (counit A (obj F X)) = id DD.
Proof.
  assert (hTri : comp DD (map F (unit A X)) (counit A (obj F X)) = id DD).
  { exact (triangle_left_axiom A X). }
  assert (hLeft : leftWhisker F (unit A X) = map F (unit A X)).
  { reflexivity. }
  assert (hPre : comp DD (leftWhisker F (unit A X)) (counit A (obj F X))
      = comp DD (map F (unit A X)) (counit A (obj F X))).
  {
    rewrite hLeft.
    reflexivity.
  }
  rewrite hPre.
  exact hTri.
Qed.

Lemma triangle_right {C D : Type} {CC : CategoryLike C} {DD : CategoryLike D}
    (F : @FunctorLike C D CC DD) (G : @FunctorLike D C DD CC)
    (A : @AdjunctionLike C D CC DD F G) (Y : D) :
    comp CC (unit A (obj G Y)) (rightWhisker G (counit A Y)) = id CC.
Proof.
  assert (hTri : comp CC (unit A (obj G Y)) (map G (counit A Y)) = id CC).
  { exact (triangle_right_axiom A Y). }
  assert (hRight : rightWhisker G (counit A Y) = map G (counit A Y)).
  { reflexivity. }
  assert (hPre : comp CC (unit A (obj G Y)) (rightWhisker G (counit A Y))
      = comp CC (unit A (obj G Y)) (map G (counit A Y))).
  {
    rewrite hRight.
    reflexivity.
  }
  rewrite hPre.
  exact hTri.
Qed.

Lemma hom_equiv_natural_left {C D : Type} {CC : CategoryLike C} {DD : CategoryLike D}
    (F : @FunctorLike C D CC DD) (G : @FunctorLike D C DD CC)
    (A : @AdjunctionLike C D CC DD F G) {X X' : C} {Y : D}
    (f : Hom CC X X') (k : Hom DD (obj F X') Y) :
    hom_equiv_to A (comp DD (leftWhisker F f) k) = comp CC f (hom_equiv_to A k).
Proof.
  assert (hNat : hom_equiv_to A (comp DD (map F f) k) = comp CC f (hom_equiv_to A k)).
  { exact (hom_equiv_natural_left_axiom A f k). }
  assert (hLeft : leftWhisker F f = map F f).
  { reflexivity. }
  assert (hComp : comp DD (leftWhisker F f) k = comp DD (map F f) k).
  {
    rewrite hLeft.
    reflexivity.
  }
  rewrite hComp.
  exact hNat.
Qed.

Lemma hom_equiv_natural_right {C D : Type} {CC : CategoryLike C} {DD : CategoryLike D}
    (F : @FunctorLike C D CC DD) (G : @FunctorLike D C DD CC)
    (A : @AdjunctionLike C D CC DD F G) {X : C} {Y Y' : D}
    (k : Hom DD (obj F X) Y) (g : Hom DD Y Y') :
    hom_equiv_to A (comp DD k g) = comp CC (hom_equiv_to A k) (rightWhisker G g).
Proof.
  assert (hNat : hom_equiv_to A (comp DD k g) = comp CC (hom_equiv_to A k) (map G g)).
  { exact (hom_equiv_natural_right_axiom A k g). }
  assert (hRight : rightWhisker G g = map G g).
  { reflexivity. }
  assert (hPost : comp CC (hom_equiv_to A k) (map G g)
      = comp CC (hom_equiv_to A k) (rightWhisker G g)).
  {
    rewrite hRight.
    reflexivity.
  }
  rewrite hPost in hNat.
  exact hNat.
Qed.
