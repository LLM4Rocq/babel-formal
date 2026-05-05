(*
BENCHMARK_ID: TINY_MATHLIB_BATCH05_CATEGORY_PRO_ARROW_EQUIPMENT_LIKE
PAIR_STEM: category_pro_arrow_equipment_like
MATH_DOMAIN: Category Theory
SOURCE_MATHLIB: Mathlib/CategoryTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class CatStruct_pro_arrow_equipment (Obj : Type) := {
  Hom : Obj -> Obj -> Type;
  id : forall {X : Obj}, Hom X X;
  comp : forall {X Y Z : Obj}, Hom X Y -> Hom Y Z -> Hom X Z;
  comp_assoc :
    forall {W X Y Z : Obj} (f : Hom W X) (g : Hom X Y) (h : Hom Y Z),
      comp (comp f g) h = comp f (comp g h);
  id_comp : forall {X Y : Obj} (f : Hom X Y), comp id f = f;
  comp_id : forall {X Y : Obj} (f : Hom X Y), comp f id = f;
  ProArr : Obj -> Obj -> Type;
  whiskerL : forall {A B C : Obj}, Hom A B -> ProArr B C -> ProArr A C;
  whiskerR : forall {A B C : Obj}, ProArr A B -> Hom B C -> ProArr A C;
  whisker_assoc_axiom :
    forall {A B C D : Obj} (f : Hom A B) (p : ProArr B C) (g : Hom C D),
      whiskerR (whiskerL f p) g = whiskerL f (whiskerR p g);
  unit_whisker_axiom : forall {A B : Obj} (p : ProArr A B), whiskerL id p = p;
  counit_whisker_axiom : forall {A B : Obj} (p : ProArr A B), whiskerR p id = p;
  comparison_full_axiom :
    forall {A B : Obj} (p q : ProArr A B),
      whiskerR p id = whiskerR q id -> p = q;
  comparison_faithful_axiom :
    forall {A B : Obj} (p q : ProArr A B),
      p = q -> whiskerL id p = whiskerL id q
}.

Arguments Hom {Obj} _ _ _.
Arguments id {Obj} _ {X}.
Arguments comp {Obj} _ {X Y Z} _ _.
Arguments ProArr {Obj} _ _ _.
Arguments whiskerL {Obj} _ {A B C} _ _.
Arguments whiskerR {Obj} _ {A B C} _ _. 

Infix "~>" := Hom (at level 90, right associativity).
Infix "~~>" := ProArr (at level 90, right associativity).
Infix ">>>" := comp (at level 50, left associativity).

Record Functor_pro_arrow_equipment
    (C D : Type)
    (CC : CatStruct_pro_arrow_equipment C)
    (DD : CatStruct_pro_arrow_equipment D) := {
  obj : C -> D;
  map_hom : forall {X Y : C}, Hom CC X Y -> Hom DD (obj X) (obj Y);
  map_pro : forall {X Y : C}, ProArr CC X Y -> ProArr DD (obj X) (obj Y);
  map_id : forall X : C, map_hom (id CC (X := X)) = id DD;
  map_comp :
    forall {X Y Z : C} (f : Hom CC X Y) (g : Hom CC Y Z),
      map_hom (comp CC f g) = comp DD (map_hom f) (map_hom g)
}.

Record NatIso_pro_arrow_equipment
    {C : Type} {CC : CatStruct_pro_arrow_equipment C}
    (F G : @Functor_pro_arrow_equipment C C CC CC) := {
  hom : forall X : C, Hom CC (obj F X) (obj G X);
  inv : forall X : C, Hom CC (obj G X) (obj F X);
  hom_inv_id : forall X : C, comp CC (hom X) (inv X) = id CC;
  inv_hom_id : forall X : C, comp CC (inv X) (hom X) = id CC
}.

Definition whisker_pro_arrow_equipment
    {C : Type} {CC : CatStruct_pro_arrow_equipment C}
    {A B D : C} (f : Hom CC A B) (p : ProArr CC B D) : ProArr CC A D :=
  whiskerL CC f p.

Definition compose_pro_arrow_equipment
    {C : Type} {CC : CatStruct_pro_arrow_equipment C}
    {A B D : C} (f : Hom CC A B) (g : Hom CC B D) : Hom CC A D :=
  comp CC f g.

Lemma whisker_assoc_pro_arrow_equipment
    {C : Type} {CC : CatStruct_pro_arrow_equipment C}
    {A B D E : C} (f : Hom CC A B) (p : ProArr CC B D) (g : Hom CC D E) :
    whiskerR CC (whisker_pro_arrow_equipment f p) g =
      whisker_pro_arrow_equipment f (whiskerR CC p g).
Proof.
  assert (hAssoc :
      whiskerR CC (whiskerL CC f p) g = whiskerL CC f (whiskerR CC p g)).
  { exact (@whisker_assoc_axiom C CC A B D E f p g). }
  assert (hLeft : whisker_pro_arrow_equipment f p = whiskerL CC f p).
  { reflexivity. }
  assert (hRight :
      whisker_pro_arrow_equipment f (whiskerR CC p g) = whiskerL CC f (whiskerR CC p g)).
  { reflexivity. }
  rewrite hLeft.
  rewrite hRight.
  exact hAssoc.
Qed.

Lemma unit_whisker_pro_arrow_equipment
    {C : Type} {CC : CatStruct_pro_arrow_equipment C}
    {A B : C} (p : ProArr CC A B) :
    whisker_pro_arrow_equipment (id CC (X := A)) p = p.
Proof.
  assert (hUnit : whiskerL CC (id CC (X := A)) p = p).
  { exact (@unit_whisker_axiom C CC A B p). }
  assert (hDef :
      whisker_pro_arrow_equipment (id CC (X := A)) p = whiskerL CC (id CC (X := A)) p).
  { reflexivity. }
  rewrite hDef.
  exact hUnit.
Qed.

Lemma counit_whisker_pro_arrow_equipment
    {C : Type} {CC : CatStruct_pro_arrow_equipment C}
    {A B : C} (p : ProArr CC A B) :
    whiskerR CC p (id CC (X := B)) = p.
Proof.
  assert (hCounit : whiskerR CC p (id CC (X := B)) = p).
  { exact (@counit_whisker_axiom C CC A B p). }
  exact hCounit.
Qed.

Lemma pasting_coherence_pro_arrow_equipment
    {C : Type} {CC : CatStruct_pro_arrow_equipment C}
    {A B : C} (p : ProArr CC A B) :
    whiskerR CC
      (whisker_pro_arrow_equipment (id CC (X := A)) p)
      (id CC (X := B)) = p.
Proof.
  assert (hAssoc :
      whiskerR CC
        (whisker_pro_arrow_equipment (id CC (X := A)) p)
        (id CC (X := B)) =
      whisker_pro_arrow_equipment (id CC (X := A))
        (whiskerR CC p (id CC (X := B)))).
  {
    exact (whisker_assoc_pro_arrow_equipment (id CC (X := A)) p (id CC (X := B))).
  }
  assert (hCounit : whiskerR CC p (id CC (X := B)) = p).
  { exact (counit_whisker_pro_arrow_equipment p). }
  assert (hUnit : whisker_pro_arrow_equipment (id CC (X := A)) p = p).
  { exact (unit_whisker_pro_arrow_equipment p). }
  rewrite hAssoc.
  rewrite hCounit.
  exact hUnit.
Qed.

Lemma comparison_full_pro_arrow_equipment
    {C : Type} {CC : CatStruct_pro_arrow_equipment C}
    {A B : C} (p q : ProArr CC A B)
    (hComp : whiskerR CC p (id CC (X := B)) = whiskerR CC q (id CC (X := B))) :
    p = q.
Proof.
  assert (hFull : whiskerR CC p (id CC (X := B)) = whiskerR CC q (id CC (X := B))).
  { exact hComp. }
  exact (@comparison_full_axiom C CC A B p q hFull).
Qed.

Lemma comparison_faithful_pro_arrow_equipment
    {C : Type} {CC : CatStruct_pro_arrow_equipment C}
    {A B : C} (p q : ProArr CC A B) (hpq : p = q) :
    whisker_pro_arrow_equipment (id CC (X := A)) p =
      whisker_pro_arrow_equipment (id CC (X := A)) q.
Proof.
  assert (hFaith :
      whiskerL CC (id CC (X := A)) p = whiskerL CC (id CC (X := A)) q).
  { exact (@comparison_faithful_axiom C CC A B p q hpq). }
  assert (hLeft :
      whisker_pro_arrow_equipment (id CC (X := A)) p = whiskerL CC (id CC (X := A)) p).
  { reflexivity. }
  assert (hRight :
      whisker_pro_arrow_equipment (id CC (X := A)) q = whiskerL CC (id CC (X := A)) q).
  { reflexivity. }
  rewrite hLeft.
  rewrite hRight.
  exact hFaith.
Qed.

Lemma equivalence_core_pro_arrow_equipment
    {C : Type} {CC : CatStruct_pro_arrow_equipment C}
    {A B : C} (p q : ProArr CC A B) :
    p = q <->
      whiskerR CC p (id CC (X := B)) = whiskerR CC q (id CC (X := B)).
Proof.
  split.
  - intro hpq.
    rewrite hpq.
    reflexivity.
  - intro hWhisk.
    exact (comparison_full_pro_arrow_equipment p q hWhisk).
Qed.
