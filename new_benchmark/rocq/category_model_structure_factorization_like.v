(*
BENCHMARK_ID: TINY_MATHLIB_BATCH05_CATEGORY_MODEL_STRUCTURE_FACTORIZATION_LIKE
PAIR_STEM: category_model_structure_factorization_like
MATH_DOMAIN: Category Theory
SOURCE_MATHLIB: Mathlib/CategoryTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class CatStruct_model_structure_factorization (C : Type) := {
  Hom : C -> C -> Type;
  id : forall {X : C}, Hom X X;
  comp : forall {X Y Z : C}, Hom X Y -> Hom Y Z -> Hom X Z;
  comp_assoc :
    forall {W X Y Z : C} (f : Hom W X) (g : Hom X Y) (h : Hom Y Z),
      comp (comp f g) h = comp f (comp g h);
  id_comp : forall {X Y : C} (f : Hom X Y), comp id f = f;
  comp_id : forall {X Y : C} (f : Hom X Y), comp f id = f
}.

Arguments Hom {C} _ _ _.
Arguments id {C} _ {X}.
Arguments comp {C} _ {X Y Z} _ _.
Arguments comp_assoc {C} _ {W X Y Z} _ _ _. 
Arguments id_comp {C} _ {X Y} _. 
Arguments comp_id {C} _ {X Y} _. 

Infix "⟶" := Hom (at level 90, right associativity).
Infix "≫" := comp (at level 40, left associativity).

Record Functor_model_structure_factorization
    (C D : Type)
    (CC : CatStruct_model_structure_factorization C)
    (DD : CatStruct_model_structure_factorization D) := {
  obj : C -> D;
  map : forall {X Y : C}, Hom CC X Y -> Hom DD (obj X) (obj Y);
  map_id : forall X : C, map (id CC (X := X)) = id DD;
  map_comp : forall {X Y Z : C} (f : Hom CC X Y) (g : Hom CC Y Z),
      map (comp CC f g) = comp DD (map f) (map g)
}.

Arguments obj {C D CC DD} _ _. 
Arguments map {C D CC DD} _ {X Y} _. 

Record NatIso_model_structure_factorization
    {C D : Type}
    {CC : CatStruct_model_structure_factorization C}
    {DD : CatStruct_model_structure_factorization D}
    (F G : @Functor_model_structure_factorization C D CC DD) := {
  hom : forall X : C, Hom DD (obj F X) (obj G X);
  inv : forall X : C, Hom DD (obj G X) (obj F X);
  left_inv : forall X : C, comp DD (hom X) (inv X) = id DD;
  right_inv : forall X : C, comp DD (inv X) (hom X) = id DD
}.

Arguments hom {C D CC DD F G} _ _.
Arguments inv {C D CC DD F G} _ _.
Arguments left_inv {C D CC DD F G} _ _.
Arguments right_inv {C D CC DD F G} _ _.

Definition whisker_model_structure_factorization
    {C D : Type}
    {CC : CatStruct_model_structure_factorization C}
    {DD : CatStruct_model_structure_factorization D}
    {F G H : @Functor_model_structure_factorization C D CC DD}
    (alpha : NatIso_model_structure_factorization F G)
    (beta : NatIso_model_structure_factorization G H) :
    NatIso_model_structure_factorization F H.
Proof.
  refine {| hom := fun X => comp DD (hom alpha X) (hom beta X);
            inv := fun X => comp DD (inv beta X) (inv alpha X) |}.
  - intro X.
    rewrite (comp_assoc DD (hom alpha X) (hom beta X) (comp DD (inv beta X) (inv alpha X))).
    rewrite <- (comp_assoc DD (hom beta X) (inv beta X) (inv alpha X)).
    rewrite (left_inv beta X).
    rewrite (id_comp DD (inv alpha X)).
    apply (left_inv alpha X).
  - intro X.
    rewrite (comp_assoc DD (inv beta X) (inv alpha X) (comp DD (hom alpha X) (hom beta X))).
    rewrite <- (comp_assoc DD (inv alpha X) (hom alpha X) (hom beta X)).
    rewrite (right_inv alpha X).
    rewrite (id_comp DD (hom beta X)).
    apply (right_inv beta X).
Defined.

Definition compose_model_structure_factorization
    {C D E : Type}
    {CC : CatStruct_model_structure_factorization C}
    {CD : CatStruct_model_structure_factorization D}
    {CE : CatStruct_model_structure_factorization E}
    (F : @Functor_model_structure_factorization C D CC CD)
    (G : @Functor_model_structure_factorization D E CD CE) :
    @Functor_model_structure_factorization C E CC CE.
Proof.
  refine {| obj := fun X => obj G (obj F X);
            map := fun X Y f => map G (map F f) |}.
  - intro X.
    pose proof (map_id F X) as hF.
    pose proof (map_id G (obj F X)) as hG.
    rewrite hF.
    exact hG.
  - intros X Y Z f g.
    pose proof (map_comp F f g) as hF.
    pose proof (map_comp G (map F f) (map F g)) as hG.
    rewrite hF.
    exact hG.
Defined.

Lemma whisker_assoc_model_structure_factorization
    {C D : Type}
    {CC : CatStruct_model_structure_factorization C}
    {DD : CatStruct_model_structure_factorization D}
    {F G H K : @Functor_model_structure_factorization C D CC DD}
    (alpha : NatIso_model_structure_factorization F G)
    (beta : NatIso_model_structure_factorization G H)
    (gamma : NatIso_model_structure_factorization H K)
    (X : C) :
    hom (whisker_model_structure_factorization
      (whisker_model_structure_factorization alpha beta) gamma) X =
      comp DD (hom alpha X) (comp DD (hom beta X) (hom gamma X)).
Proof.
  change (comp DD (comp DD (hom alpha X) (hom beta X)) (hom gamma X) =
          comp DD (hom alpha X) (comp DD (hom beta X) (hom gamma X))).
  apply (comp_assoc DD (hom alpha X) (hom beta X) (hom gamma X)).
Qed.

Lemma unit_whisker_model_structure_factorization
    {C D : Type}
    {CC : CatStruct_model_structure_factorization C}
    {DD : CatStruct_model_structure_factorization D}
    {F G : @Functor_model_structure_factorization C D CC DD}
    (iota : NatIso_model_structure_factorization F F)
    (alpha : NatIso_model_structure_factorization F G)
    (hiota : forall X : C, hom iota X = id DD)
    (X : C) :
    hom (whisker_model_structure_factorization iota alpha) X = hom alpha X.
Proof.
  change (comp DD (hom iota X) (hom alpha X) = hom alpha X).
  rewrite (hiota X).
  apply (id_comp DD (hom alpha X)).
Qed.

Lemma counit_whisker_model_structure_factorization
    {C D : Type}
    {CC : CatStruct_model_structure_factorization C}
    {DD : CatStruct_model_structure_factorization D}
    {F G : @Functor_model_structure_factorization C D CC DD}
    (alpha : NatIso_model_structure_factorization F G)
    (iota : NatIso_model_structure_factorization G G)
    (hiota : forall X : C, hom iota X = id DD)
    (X : C) :
    hom (whisker_model_structure_factorization alpha iota) X = hom alpha X.
Proof.
  change (comp DD (hom alpha X) (hom iota X) = hom alpha X).
  rewrite (hiota X).
  apply (comp_id DD (hom alpha X)).
Qed.

Lemma pasting_coherence_model_structure_factorization
    {C D : Type}
    {CC : CatStruct_model_structure_factorization C}
    {DD : CatStruct_model_structure_factorization D}
    {F G H K L : @Functor_model_structure_factorization C D CC DD}
    (alpha : NatIso_model_structure_factorization F G)
    (beta : NatIso_model_structure_factorization G H)
    (gamma : NatIso_model_structure_factorization H K)
    (delta : NatIso_model_structure_factorization K L)
    (X : C) :
    hom (whisker_model_structure_factorization
      (whisker_model_structure_factorization
        (whisker_model_structure_factorization alpha beta) gamma) delta) X =
      comp DD (hom alpha X) (comp DD (hom beta X) (comp DD (hom gamma X) (hom delta X))).
Proof.
  change (comp DD (comp DD (comp DD (hom alpha X) (hom beta X)) (hom gamma X)) (hom delta X) =
          comp DD (hom alpha X) (comp DD (hom beta X) (comp DD (hom gamma X) (hom delta X)))).
  rewrite (comp_assoc DD (comp DD (hom alpha X) (hom beta X)) (hom gamma X) (hom delta X)).
  rewrite (comp_assoc DD (hom alpha X) (hom beta X) (comp DD (hom gamma X) (hom delta X))).
  reflexivity.
Qed.

Lemma comparison_full_model_structure_factorization
    {C D : Type}
    {CC : CatStruct_model_structure_factorization C}
    {DD : CatStruct_model_structure_factorization D}
    {F G : @Functor_model_structure_factorization C D CC DD}
    (alpha beta : NatIso_model_structure_factorization F G)
    (hhom : forall X : C, hom alpha X = hom beta X)
    (hinv : forall X : C, inv alpha X = inv beta X) :
    forall X : C,
      (comp DD (hom alpha X) (inv alpha X) = comp DD (hom beta X) (inv beta X)) /\
      (comp DD (inv alpha X) (hom alpha X) = comp DD (inv beta X) (hom beta X)).
Proof.
  intro X.
  split.
  - rewrite (hhom X).
    rewrite (hinv X).
    reflexivity.
  - rewrite (hinv X).
    rewrite (hhom X).
    reflexivity.
Qed.

Lemma comparison_faithful_model_structure_factorization
    {C D : Type}
    {CC : CatStruct_model_structure_factorization C}
    {DD : CatStruct_model_structure_factorization D}
    {F G H : @Functor_model_structure_factorization C D CC DD}
    (alpha beta : NatIso_model_structure_factorization F G)
    (gamma : NatIso_model_structure_factorization G H)
    (hwhisk : forall X : C,
      hom (whisker_model_structure_factorization alpha gamma) X =
      hom (whisker_model_structure_factorization beta gamma) X)
    (hcancel : forall (X : C) (f g : Hom DD (obj F X) (obj G X)),
      comp DD f (hom gamma X) = comp DD g (hom gamma X) -> f = g) :
    forall X : C, hom alpha X = hom beta X.
Proof.
  intro X.
  assert (hraw : comp DD (hom alpha X) (hom gamma X) = comp DD (hom beta X) (hom gamma X)).
  { exact (hwhisk X). }
  apply (hcancel X (hom alpha X) (hom beta X)).
  exact hraw.
Qed.

Lemma equivalence_core_model_structure_factorization
    {C D : Type}
    {CC : CatStruct_model_structure_factorization C}
    {DD : CatStruct_model_structure_factorization D}
    {F G H : @Functor_model_structure_factorization C D CC DD}
    (alpha beta : NatIso_model_structure_factorization F G)
    (gamma : NatIso_model_structure_factorization G H)
    (hwhisk : forall X : C,
      hom (whisker_model_structure_factorization alpha gamma) X =
      hom (whisker_model_structure_factorization beta gamma) X)
    (hcancel : forall (X : C) (f g : Hom DD (obj F X) (obj G X)),
      comp DD f (hom gamma X) = comp DD g (hom gamma X) -> f = g)
    (hinv : forall X : C, inv alpha X = inv beta X) :
    forall X : C, comp DD (hom alpha X) (inv alpha X) = comp DD (hom beta X) (inv beta X).
Proof.
  assert (hhom : forall X : C, hom alpha X = hom beta X).
  { intro X. apply (comparison_faithful_model_structure_factorization alpha beta gamma hwhisk hcancel X). }
  pose proof (comparison_full_model_structure_factorization alpha beta hhom hinv) as hcmp.
  intro X.
  exact (proj1 (hcmp X)).
Qed.
