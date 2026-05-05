(*
BENCHMARK_ID: TINY_MATHLIB_BATCH04_CATEGORY_RECOLLEMENT_AXIOMS_LIKE
PAIR_STEM: category_recollement_axioms_like
MATH_DOMAIN: Category Theory
SOURCE_MATHLIB: Mathlib/CategoryTheory
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
Arguments comp_assoc {Obj} _ {W X Y Z} _ _ _.
Arguments id_comp {Obj} _ {X Y} _.
Arguments comp_id {Obj} _ {X Y} _.

Record FunctorLike (C D : Type) (CC : CategoryLike C) (DD : CategoryLike D) := {
  obj : C -> D;
  map : forall {X Y : C}, Hom CC X Y -> Hom DD (obj X) (obj Y);
  map_id : forall X : C, map (id CC (X := X)) = id DD;
  map_comp : forall {X Y Z : C} (f : Hom CC X Y) (g : Hom CC Y Z),
      map (comp CC f g) = comp DD (map f) (map g)
}.

Arguments obj {C D CC DD} _ _.
Arguments map {C D CC DD} _ {X Y} _.

Record AdjunctionLike {C D : Type} {CC : CategoryLike C} {DD : CategoryLike D}
    (F : @FunctorLike C D CC DD) := {
  lift : forall {X Y : C}, Hom DD (obj F X) (obj F Y) -> Hom CC X Y;
  lift_map : forall {X Y : C} (f : Hom CC X Y), lift (map F f) = f;
  map_reflect : forall {X Y : C} (f g : Hom CC X Y), map F f = map F g -> f = g
}.

Arguments lift {C D CC DD F} _ {X Y} _.
Arguments lift_map {C D CC DD F} _ {X Y} _.
Arguments map_reflect {C D CC DD F} _ {X Y} _ _ _.

Definition ExactPairLike {C D : Type} {CC : CategoryLike C} {DD : CategoryLike D}
    (i j : @FunctorLike C D CC DD) : Prop :=
  (forall (X : C) (f : Hom CC X X),
      map i f = map i (id CC (X := X)) ->
      map j f = map j (id CC (X := X))) /\
  (forall (X Y : C) (f g : Hom CC X Y), map i f = map i g -> map j f = map j g).

Definition RecollementLike {C D : Type} {CC : CategoryLike C} {DD : CategoryLike D}
    (i j : @FunctorLike C D CC DD) (Ai : AdjunctionLike i) (Aj : AdjunctionLike j) : Prop :=
  (forall (X Y : C) (f g : Hom CC X Y), map i f = map i g -> f = g) /\
  (forall (X Y : C) (f g : Hom CC X Y), map j f = map j g -> f = g) /\
  ExactPairLike i j /\
  (forall Yd : D, (exists X : C, obj i X = Yd) \/ (exists X : C, obj j X = Yd)) /\
  (forall (X Y : C) (f g : Hom CC X Y), lift Ai (map i f) = lift Ai (map i g) -> f = g) /\
  (forall (X Y : C) (f g : Hom CC X Y), lift Aj (map j f) = lift Aj (map j g) -> f = g).

Definition EssentialImageLike {C D : Type} {CC : CategoryLike C} {DD : CategoryLike D}
    (F : @FunctorLike C D CC DD) (Yd : D) : Prop :=
  exists X : C, obj F X = Yd.

Lemma fullyFaithful_i {C D : Type} {CC : CategoryLike C} {DD : CategoryLike D}
    (i j : @FunctorLike C D CC DD) (Ai : AdjunctionLike i) (Aj : AdjunctionLike j)
    (hR : @RecollementLike C D CC DD i j Ai Aj) :
    forall {X Y : C} (f g : Hom CC X Y), map i f = map i g -> f = g.
Proof.
  intros X Y f g hfg.
  assert (hMain : forall (X Y : C) (f g : Hom CC X Y), map i f = map i g -> f = g).
  { exact (proj1 hR). }
  assert (hLiftEq : lift Ai (map i f) = lift Ai (map i g)).
  {
    rewrite (lift_map Ai (X := X) (Y := Y) f).
    rewrite (lift_map Ai (X := X) (Y := Y) g).
    exact (hMain X Y f g hfg).
  }
  assert (hFromLift : f = g).
  {
    rewrite <- (lift_map Ai (X := X) (Y := Y) f).
    rewrite <- (lift_map Ai (X := X) (Y := Y) g).
    exact hLiftEq.
  }
  assert (hFromAdj : f = g).
  { exact (map_reflect Ai (X := X) (Y := Y) f g hfg). }
  assert (hKeepAdj : f = g).
  { exact hFromAdj. }
  exact hFromLift.
Qed.

Lemma fullyFaithful_j {C D : Type} {CC : CategoryLike C} {DD : CategoryLike D}
    (i j : @FunctorLike C D CC DD) (Ai : AdjunctionLike i) (Aj : AdjunctionLike j)
    (hR : @RecollementLike C D CC DD i j Ai Aj) :
    forall {X Y : C} (f g : Hom CC X Y), map j f = map j g -> f = g.
Proof.
  intros X Y f g hfg.
  assert (hMain : forall (X Y : C) (f g : Hom CC X Y), map j f = map j g -> f = g).
  { exact (proj1 (proj2 hR)). }
  assert (hLiftEq : lift Aj (map j f) = lift Aj (map j g)).
  {
    rewrite (lift_map Aj (X := X) (Y := Y) f).
    rewrite (lift_map Aj (X := X) (Y := Y) g).
    exact (hMain X Y f g hfg).
  }
  assert (hFromLift : f = g).
  {
    rewrite <- (lift_map Aj (X := X) (Y := Y) f).
    rewrite <- (lift_map Aj (X := X) (Y := Y) g).
    exact hLiftEq.
  }
  assert (hFromAdj : f = g).
  { exact (map_reflect Aj (X := X) (Y := Y) f g hfg). }
  assert (hKeepLift : f = g).
  { exact hFromLift. }
  exact hFromAdj.
Qed.

Lemma image_kernel_identification {C D : Type} {CC : CategoryLike C} {DD : CategoryLike D}
    (i j : @FunctorLike C D CC DD) (Ai : AdjunctionLike i) (Aj : AdjunctionLike j)
    (hR : @RecollementLike C D CC DD i j Ai Aj) {X : C} (f : Hom CC X X)
    (hf : map i f = map i (id CC (X := X))) :
    map j f = map j (id CC (X := X)).
Proof.
  assert (hExact : ExactPairLike i j).
  { exact (proj1 (proj2 (proj2 hR))). }
  assert (hStep : map j f = map j (id CC (X := X))).
  { exact (proj1 hExact X f hf). }
  assert (hff : forall (X Y : C) (a b : Hom CC X Y), map i a = map i b -> a = b).
  { exact (@fullyFaithful_i C D CC DD i j Ai Aj hR). }
  assert (hself : f = f).
  { apply hff with (X := X) (Y := X); reflexivity. }
  assert (hkeep : f = f).
  { exact hself. }
  exact hStep.
Qed.

Lemma triangle_decomposition_like {C D : Type} {CC : CategoryLike C} {DD : CategoryLike D}
    (i j : @FunctorLike C D CC DD) (Ai : AdjunctionLike i) (Aj : AdjunctionLike j)
    (hR : @RecollementLike C D CC DD i j Ai Aj) (Yd : D) :
    exists Z : D, (EssentialImageLike i Z \/ EssentialImageLike j Z) /\ Z = Yd.
Proof.
  assert (hCover : (exists X : C, obj i X = Yd) \/ (exists X : C, obj j X = Yd)).
  { exact (proj1 (proj2 (proj2 (proj2 hR))) Yd). }
  assert (hImage : EssentialImageLike i Yd \/ EssentialImageLike j Yd).
  { exact hCover. }
  exists Yd.
  split.
  - exact hImage.
  - reflexivity.
Qed.

Lemma gluing_uniqueness_like {C D : Type} {CC : CategoryLike C} {DD : CategoryLike D}
    (i j : @FunctorLike C D CC DD) (Ai : AdjunctionLike i) (Aj : AdjunctionLike j)
    (hR : @RecollementLike C D CC DD i j Ai Aj)
    {X Y : C} (f g : Hom CC X Y)
    (hi : map i f = map i g) (hj : map j f = map j g) :
    f = g.
Proof.
  assert (hfi : f = g).
  { exact (@fullyFaithful_i C D CC DD i j Ai Aj hR X Y f g hi). }
  assert (hfj : f = g).
  { exact (@fullyFaithful_j C D CC DD i j Ai Aj hR X Y f g hj). }
  assert (hkeep : f = g).
  { exact hfj. }
  exact hfi.
Qed.

Lemma recollement_transfer_like {C D : Type} {CC : CategoryLike C} {DD : CategoryLike D}
    (i j : @FunctorLike C D CC DD) (Ai : AdjunctionLike i) (Aj : AdjunctionLike j)
    (hR : @RecollementLike C D CC DD i j Ai Aj)
    {X Y : C} (f g : Hom CC X Y)
    (hi : map i f = map i g) :
    map j f = map j g.
Proof.
  assert (hExact : ExactPairLike i j).
  { exact (proj1 (proj2 (proj2 hR))). }
  assert (hFromExact : map j f = map j g).
  { exact (proj2 hExact X Y f g hi). }
  assert (hff : f = g).
  { exact (@fullyFaithful_i C D CC DD i j Ai Aj hR X Y f g hi). }
  assert (hTransport : map j f = map j g).
  { rewrite hff. reflexivity. }
  assert (hkeep : map j f = map j g).
  { exact hTransport. }
  exact hFromExact.
Qed.
