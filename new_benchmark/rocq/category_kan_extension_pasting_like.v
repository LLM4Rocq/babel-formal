(*
BENCHMARK_ID: TINY_MATHLIB_BATCH04_CATEGORY_KAN_EXTENSION_PASTING_LIKE
PAIR_STEM: category_kan_extension_pasting_like
MATH_DOMAIN: Category Theory
SOURCE_MATHLIB: Mathlib/CategoryTheory/KanExtension
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

Record NatTransLike {C D : Type} {CC : CategoryLike C} {DD : CategoryLike D}
    (F G : @FunctorLike C D CC DD) := {
  app : forall X : C, Hom DD (obj F X) (obj G X);
  naturality : forall {X Y : C} (f : Hom CC X Y),
      comp DD (app X) (map G f) = comp DD (map F f) (app Y)
}.

Arguments obj {C D CC DD} _ _.
Arguments map {C D CC DD} _ {X Y} _.
Arguments app {C D CC DD F G} _ _.
Arguments naturality {C D CC DD F G} _ {X Y} _.

Definition LanLike {C D : Type} {CC : CategoryLike C} {DD : CategoryLike D}
    (J F L : @FunctorLike C D CC DD) (eta : @NatTransLike C D CC DD F L) : Prop :=
  forall (X : @FunctorLike C D CC DD) (tau : @NatTransLike C D CC DD F X),
    exists sigma : @NatTransLike C D CC DD L X,
      (forall Z : C, comp DD (app eta Z) (app sigma Z) = app tau Z) /\
      (forall psi : @NatTransLike C D CC DD L X,
        (forall Z : C, comp DD (app eta Z) (app psi Z) = app tau Z) -> psi = sigma).

Definition RanLike {C D : Type} {CC : CategoryLike C} {DD : CategoryLike D}
    (J F R : @FunctorLike C D CC DD) (rho : @NatTransLike C D CC DD R F) : Prop :=
  forall (X : @FunctorLike C D CC DD) (tau : @NatTransLike C D CC DD X F),
    exists sigma : @NatTransLike C D CC DD X R,
      (forall Z : C, comp DD (app sigma Z) (app rho Z) = app tau Z) /\
      (forall psi : @NatTransLike C D CC DD X R,
        (forall Z : C, comp DD (app psi Z) (app rho Z) = app tau Z) -> psi = sigma).

Definition WhiskerLike {C D : Type} {CC : CategoryLike C} {DD : CategoryLike D}
    {F G H : @FunctorLike C D CC DD}
    (alpha : @NatTransLike C D CC DD F G) (beta : @NatTransLike C D CC DD G H) :
    @NatTransLike C D CC DD F H.
Proof.
  refine {| app := fun X => comp DD (app alpha X) (app beta X) |}.
  intros X Y f.
  rewrite (comp_assoc DD (app alpha X) (app beta X) (map H f)).
  rewrite (naturality beta f).
  rewrite <- (comp_assoc DD (app alpha X) (map G f) (app beta Y)).
  rewrite (naturality alpha f).
  rewrite (comp_assoc DD (map F f) (app alpha Y) (app beta Y)).
  reflexivity.
Defined.

Lemma lan_universal_factor {C D : Type} {CC : CategoryLike C} {DD : CategoryLike D}
    (J F L : @FunctorLike C D CC DD) (eta : @NatTransLike C D CC DD F L)
    (hLan : @LanLike C D CC DD J F L eta)
    (X : @FunctorLike C D CC DD) (tau : @NatTransLike C D CC DD F X) :
    exists sigma : @NatTransLike C D CC DD L X,
      forall Z : C, comp DD (app eta Z) (app sigma Z) = app tau Z.
Proof.
  pose proof (hLan X tau) as hSpec.
  destruct hSpec as [sigma [hsigma huniq]].
  assert (forall Z : C, comp DD (app eta Z) (app sigma Z) = app tau Z) as hKeep.
  { exact hsigma. }
  assert (forall psi : @NatTransLike C D CC DD L X,
      (forall Z : C, comp DD (app eta Z) (app psi Z) = app tau Z) -> psi = sigma) as hUniq.
  { exact huniq. }
  exists sigma.
  exact hKeep.
Qed.

Lemma lan_universal_unique {C D : Type} {CC : CategoryLike C} {DD : CategoryLike D}
    (J F L : @FunctorLike C D CC DD) (eta : @NatTransLike C D CC DD F L)
    (hLan : @LanLike C D CC DD J F L eta)
    (X : @FunctorLike C D CC DD) (tau : @NatTransLike C D CC DD F X)
    (sigma1 sigma2 : @NatTransLike C D CC DD L X)
    (hsigma1 : forall Z : C, comp DD (app eta Z) (app sigma1 Z) = app tau Z)
    (hsigma2 : forall Z : C, comp DD (app eta Z) (app sigma2 Z) = app tau Z) :
    sigma1 = sigma2.
Proof.
  pose proof (hLan X tau) as hSpec.
  destruct hSpec as [sigma [hsigma huniq]].
  assert (sigma1 = sigma) as hs1.
  { apply huniq. exact hsigma1. }
  assert (sigma2 = sigma) as hs2.
  { apply huniq. exact hsigma2. }
  rewrite hs1.
  symmetry.
  exact hs2.
Qed.

Lemma ran_universal_factor {C D : Type} {CC : CategoryLike C} {DD : CategoryLike D}
    (J F R : @FunctorLike C D CC DD) (rho : @NatTransLike C D CC DD R F)
    (hRan : @RanLike C D CC DD J F R rho)
    (X : @FunctorLike C D CC DD) (tau : @NatTransLike C D CC DD X F) :
    exists sigma : @NatTransLike C D CC DD X R,
      forall Z : C, comp DD (app sigma Z) (app rho Z) = app tau Z.
Proof.
  pose proof (hRan X tau) as hSpec.
  destruct hSpec as [sigma [hsigma huniq]].
  assert (forall Z : C, comp DD (app sigma Z) (app rho Z) = app tau Z) as hKeep.
  { exact hsigma. }
  assert (forall psi : @NatTransLike C D CC DD X R,
      (forall Z : C, comp DD (app psi Z) (app rho Z) = app tau Z) -> psi = sigma) as hUniq.
  { exact huniq. }
  exists sigma.
  exact hKeep.
Qed.

Lemma ran_universal_unique {C D : Type} {CC : CategoryLike C} {DD : CategoryLike D}
    (J F R : @FunctorLike C D CC DD) (rho : @NatTransLike C D CC DD R F)
    (hRan : @RanLike C D CC DD J F R rho)
    (X : @FunctorLike C D CC DD) (tau : @NatTransLike C D CC DD X F)
    (sigma1 sigma2 : @NatTransLike C D CC DD X R)
    (hsigma1 : forall Z : C, comp DD (app sigma1 Z) (app rho Z) = app tau Z)
    (hsigma2 : forall Z : C, comp DD (app sigma2 Z) (app rho Z) = app tau Z) :
    sigma1 = sigma2.
Proof.
  pose proof (hRan X tau) as hSpec.
  destruct hSpec as [sigma [hsigma huniq]].
  assert (sigma1 = sigma) as hs1.
  { apply huniq. exact hsigma1. }
  assert (sigma2 = sigma) as hs2.
  { apply huniq. exact hsigma2. }
  rewrite hs1.
  symmetry.
  exact hs2.
Qed.

Lemma lan_pasting_like {C D : Type} {CC : CategoryLike C} {DD : CategoryLike D}
    (J F L M : @FunctorLike C D CC DD)
    (eta : @NatTransLike C D CC DD F L) (theta : @NatTransLike C D CC DD L M)
    (hLan1 : @LanLike C D CC DD J F L eta) (hLan2 : @LanLike C D CC DD J L M theta)
    (X : @FunctorLike C D CC DD) (tau : @NatTransLike C D CC DD F X) :
    exists sigma : @NatTransLike C D CC DD M X,
      forall Z : C, comp DD (app (WhiskerLike eta theta) Z) (app sigma Z) = app tau Z.
Proof.
  pose proof (hLan1 X tau) as h1.
  destruct h1 as [mu [hmu hmuuniq]].
  pose proof (hLan2 X mu) as h2.
  destruct h2 as [sigma [hsigma hsigmauniq]].
  assert (forall Z : C,
      comp DD (app (WhiskerLike eta theta) Z) (app sigma Z) = app tau Z) as hcompose.
  {
    intro Z.
    change (comp DD (comp DD (app eta Z) (app theta Z)) (app sigma Z) = app tau Z).
    rewrite (comp_assoc DD (app eta Z) (app theta Z) (app sigma Z)).
    rewrite hsigma.
    exact (hmu Z).
  }
  assert (forall Z : C, comp DD (app theta Z) (app sigma Z) = app mu Z) as hKeep1.
  { exact hsigma. }
  assert (forall nu : @NatTransLike C D CC DD M X,
      (forall Z : C, comp DD (app theta Z) (app nu Z) = app mu Z) -> nu = sigma) as hKeep2.
  { exact hsigmauniq. }
  assert (forall nu : @NatTransLike C D CC DD L X,
      (forall Z : C, comp DD (app eta Z) (app nu Z) = app tau Z) -> nu = mu) as hKeep3.
  { exact hmuuniq. }
  exists sigma.
  exact hcompose.
Qed.

Lemma ran_pasting_like {C D : Type} {CC : CategoryLike C} {DD : CategoryLike D}
    (J F R S : @FunctorLike C D CC DD)
    (rho : @NatTransLike C D CC DD R F) (theta : @NatTransLike C D CC DD S R)
    (hRan1 : @RanLike C D CC DD J F R rho) (hRan2 : @RanLike C D CC DD J R S theta)
    (X : @FunctorLike C D CC DD) (tau : @NatTransLike C D CC DD X F) :
    exists sigma : @NatTransLike C D CC DD X S,
      forall Z : C, comp DD (app sigma Z) (app (WhiskerLike theta rho) Z) = app tau Z.
Proof.
  pose proof (hRan1 X tau) as h1.
  destruct h1 as [mu [hmu hmuuniq]].
  pose proof (hRan2 X mu) as h2.
  destruct h2 as [sigma [hsigma hsigmauniq]].
  assert (forall Z : C,
      comp DD (app sigma Z) (app (WhiskerLike theta rho) Z) = app tau Z) as hcompose.
  {
    intro Z.
    change (comp DD (app sigma Z) (comp DD (app theta Z) (app rho Z)) = app tau Z).
    rewrite <- (comp_assoc DD (app sigma Z) (app theta Z) (app rho Z)).
    rewrite hsigma.
    exact (hmu Z).
  }
  assert (forall Z : C, comp DD (app sigma Z) (app theta Z) = app mu Z) as hKeep1.
  { exact hsigma. }
  assert (forall nu : @NatTransLike C D CC DD X S,
      (forall Z : C, comp DD (app nu Z) (app theta Z) = app mu Z) -> nu = sigma) as hKeep2.
  { exact hsigmauniq. }
  assert (forall nu : @NatTransLike C D CC DD X R,
      (forall Z : C, comp DD (app nu Z) (app rho Z) = app tau Z) -> nu = mu) as hKeep3.
  { exact hmuuniq. }
  exists sigma.
  exact hcompose.
Qed.
