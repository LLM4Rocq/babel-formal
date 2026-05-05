/-
BENCHMARK_ID: TINY_MATHLIB_BATCH06_CATEGORY_GROTHENDIECK_FIBRATION_CLEAVAGE_LIKE
PAIR_STEM: category_grothendieck_fibration_cleavage_like
MATH_DOMAIN: Category Theory
SOURCE_MATHLIB: Mathlib/CategoryTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u v

class FibStruct_grothendieck_fibration_cleavage (Obj : Type u) where
  Hom : Obj -> Obj -> Type v
  id : {X : Obj} -> Hom X X
  comp : {X Y Z : Obj} -> Hom X Y -> Hom Y Z -> Hom X Z
  comp_assoc :
    forall {W X Y Z : Obj} (f : Hom W X) (g : Hom X Y) (h : Hom Y Z),
      comp (comp f g) h = comp f (comp g h)
  id_comp : forall {X Y : Obj} (f : Hom X Y), comp id f = f
  comp_id : forall {X Y : Obj} (f : Hom X Y), comp f id = f
  cancel_right :
    forall {X Y Z : Obj} (f g : Hom X Y) (h : Hom Y Z),
      comp f h = comp g h -> f = g

infixr:10 " ⟶ " => FibStruct_grothendieck_fibration_cleavage.Hom
infixl:80 " ≫ " => FibStruct_grothendieck_fibration_cleavage.comp

structure CleavageData_grothendieck_fibration_cleavage
    (Obj : Type u) [FibStruct_grothendieck_fibration_cleavage Obj] where
  pullObj : {X Y : Obj} -> (X ⟶ Y) -> Obj -> Obj
  lift : {X Y : Obj} -> (f : X ⟶ Y) -> (e : Obj) -> (pullObj f e ⟶ e)
  compare : {X Y Z : Obj} ->
    (f : X ⟶ Y) -> (g : Y ⟶ Z) -> (e : Obj) ->
      (pullObj (f ≫ g) e ⟶ pullObj f (pullObj g e))
  factor :
    forall {X Y : Obj} (f : X ⟶ Y) (e z : Obj) (h : z ⟶ e),
      Exists fun u : z ⟶ pullObj f e => u ≫ lift f e = h
  unique :
    forall {X Y : Obj} (f : X ⟶ Y) (e z : Obj) (h : z ⟶ e)
      (u1 u2 : z ⟶ pullObj f e),
      u1 ≫ lift f e = h -> u2 ≫ lift f e = h -> u1 = u2
  compare_spec :
    forall {X Y Z : Obj} (f : X ⟶ Y) (g : Y ⟶ Z) (e : Obj),
      compare f g e ≫ lift f (pullObj g e) ≫ lift g e = lift (f ≫ g) e
  reindex_comp :
    forall {X Y Z : Obj} (f : X ⟶ Y) (g : Y ⟶ Z) (e z : Obj)
      (u : z ⟶ pullObj (f ≫ g) e),
      u ≫ lift (f ≫ g) e = ((u ≫ compare f g e) ≫ lift f (pullObj g e)) ≫ lift g e

def cartesian_lift_grothendieck_fibration_cleavage
    {Obj : Type u} [FibStruct_grothendieck_fibration_cleavage Obj]
    (clv : CleavageData_grothendieck_fibration_cleavage Obj) {X Y : Obj}
    (f : X ⟶ Y) (e : Obj) :
    clv.pullObj f e ⟶ e :=
  clv.lift f e

def pullback_obj_grothendieck_fibration_cleavage
    {Obj : Type u} [FibStruct_grothendieck_fibration_cleavage Obj]
    (clv : CleavageData_grothendieck_fibration_cleavage Obj) {X Y : Obj}
    (f : X ⟶ Y) (e : Obj) : Obj :=
  clv.pullObj f e

def reindex_morphism_grothendieck_fibration_cleavage
    {Obj : Type u} [FibStruct_grothendieck_fibration_cleavage Obj]
    (clv : CleavageData_grothendieck_fibration_cleavage Obj) {X Y : Obj}
    (f : X ⟶ Y) (e z : Obj)
    (u : z ⟶ pullback_obj_grothendieck_fibration_cleavage clv f e) : z ⟶ e :=
  u ≫ cartesian_lift_grothendieck_fibration_cleavage clv f e

theorem cartesian_lift_exists_grothendieck_fibration_cleavage
    {Obj : Type u} [FibStruct_grothendieck_fibration_cleavage Obj]
    (clv : CleavageData_grothendieck_fibration_cleavage Obj) {X Y : Obj}
    (f : X ⟶ Y) (e z : Obj) (h : z ⟶ e) :
    Exists fun u : z ⟶ pullback_obj_grothendieck_fibration_cleavage clv f e =>
      reindex_morphism_grothendieck_fibration_cleavage clv f e z u = h := by
  rcases clv.factor f e z h with ⟨u, hu⟩
  have hu' : u ≫ cartesian_lift_grothendieck_fibration_cleavage clv f e = h := by
    simpa [cartesian_lift_grothendieck_fibration_cleavage] using hu
  refine Exists.intro u ?_
  calc
    reindex_morphism_grothendieck_fibration_cleavage clv f e z u
        = u ≫ cartesian_lift_grothendieck_fibration_cleavage clv f e := by
          rfl
    _ = h := hu'

theorem cartesian_lift_unique_grothendieck_fibration_cleavage
    {Obj : Type u} [FibStruct_grothendieck_fibration_cleavage Obj]
    (clv : CleavageData_grothendieck_fibration_cleavage Obj) {X Y : Obj}
    (f : X ⟶ Y) (e z : Obj) (h : z ⟶ e)
    (u1 u2 : z ⟶ pullback_obj_grothendieck_fibration_cleavage clv f e)
    (hu1 : reindex_morphism_grothendieck_fibration_cleavage clv f e z u1 = h)
    (hu2 : reindex_morphism_grothendieck_fibration_cleavage clv f e z u2 = h) :
    u1 = u2 := by
  have hu1' : u1 ≫ clv.lift f e = h := by
    simpa [reindex_morphism_grothendieck_fibration_cleavage,
      cartesian_lift_grothendieck_fibration_cleavage] using hu1
  have hu2' : u2 ≫ clv.lift f e = h := by
    simpa [reindex_morphism_grothendieck_fibration_cleavage,
      cartesian_lift_grothendieck_fibration_cleavage] using hu2
  have hcore : u1 = u2 := clv.unique f e z h u1 u2 hu1' hu2'
  exact hcore

theorem reindex_identity_grothendieck_fibration_cleavage
    {Obj : Type u} [FibStruct_grothendieck_fibration_cleavage Obj]
    (clv : CleavageData_grothendieck_fibration_cleavage Obj)
    {X Y : Obj} (f : X ⟶ Y) (e : Obj) :
    reindex_morphism_grothendieck_fibration_cleavage
      clv (FibStruct_grothendieck_fibration_cleavage.id (X := X) ≫ f) e
      (pullback_obj_grothendieck_fibration_cleavage
        clv (FibStruct_grothendieck_fibration_cleavage.id (X := X) ≫ f) e)
      (FibStruct_grothendieck_fibration_cleavage.id
        (X := pullback_obj_grothendieck_fibration_cleavage
          clv (FibStruct_grothendieck_fibration_cleavage.id (X := X) ≫ f) e))
      = cartesian_lift_grothendieck_fibration_cleavage
          clv (FibStruct_grothendieck_fibration_cleavage.id (X := X) ≫ f) e := by
  change
    FibStruct_grothendieck_fibration_cleavage.id ≫
      cartesian_lift_grothendieck_fibration_cleavage
        clv (FibStruct_grothendieck_fibration_cleavage.id (X := X) ≫ f) e =
      cartesian_lift_grothendieck_fibration_cleavage
        clv (FibStruct_grothendieck_fibration_cleavage.id (X := X) ≫ f) e
  exact FibStruct_grothendieck_fibration_cleavage.id_comp _

theorem reindex_composition_grothendieck_fibration_cleavage
    {Obj : Type u} [FibStruct_grothendieck_fibration_cleavage Obj]
    (clv : CleavageData_grothendieck_fibration_cleavage Obj)
    {X Y Z : Obj} (f : X ⟶ Y) (g : Y ⟶ Z)
    (e z : Obj)
    (u : z ⟶ pullback_obj_grothendieck_fibration_cleavage clv (f ≫ g) e) :
    reindex_morphism_grothendieck_fibration_cleavage clv (f ≫ g) e z u =
      reindex_morphism_grothendieck_fibration_cleavage clv g e z
        (reindex_morphism_grothendieck_fibration_cleavage
          clv f (pullback_obj_grothendieck_fibration_cleavage clv g e) z
          (u ≫ clv.compare f g e)) := by
  have hcompare := clv.compare_spec f g e
  have hstep :
      u ≫ cartesian_lift_grothendieck_fibration_cleavage clv (f ≫ g) e =
        ((u ≫ clv.compare f g e) ≫
            cartesian_lift_grothendieck_fibration_cleavage
              clv f (pullback_obj_grothendieck_fibration_cleavage clv g e)) ≫
          cartesian_lift_grothendieck_fibration_cleavage clv g e := by
    calc
      u ≫ cartesian_lift_grothendieck_fibration_cleavage clv (f ≫ g) e
          = u ≫
              (clv.compare f g e ≫
                cartesian_lift_grothendieck_fibration_cleavage
                  clv f (pullback_obj_grothendieck_fibration_cleavage clv g e) ≫
                cartesian_lift_grothendieck_fibration_cleavage clv g e) := by
            simpa [cartesian_lift_grothendieck_fibration_cleavage] using
              congrArg (fun t => u ≫ t) (Eq.symm hcompare)
      _ = u ≫ clv.compare f g e ≫
            cartesian_lift_grothendieck_fibration_cleavage
              clv f (pullback_obj_grothendieck_fibration_cleavage clv g e) ≫
            cartesian_lift_grothendieck_fibration_cleavage clv g e := by
            calc
              u ≫
                  (clv.compare f g e ≫
                    cartesian_lift_grothendieck_fibration_cleavage
                      clv f (pullback_obj_grothendieck_fibration_cleavage clv g e) ≫
                    cartesian_lift_grothendieck_fibration_cleavage clv g e)
                  = u ≫
                    (clv.compare f g e ≫
                      (cartesian_lift_grothendieck_fibration_cleavage
                        clv f (pullback_obj_grothendieck_fibration_cleavage clv g e) ≫
                        cartesian_lift_grothendieck_fibration_cleavage clv g e)) := by
                    have hinner :
                        clv.compare f g e ≫
                            cartesian_lift_grothendieck_fibration_cleavage
                              clv f (pullback_obj_grothendieck_fibration_cleavage clv g e) ≫
                          cartesian_lift_grothendieck_fibration_cleavage clv g e
                          =
                        clv.compare f g e ≫
                          (cartesian_lift_grothendieck_fibration_cleavage
                            clv f (pullback_obj_grothendieck_fibration_cleavage clv g e) ≫
                            cartesian_lift_grothendieck_fibration_cleavage clv g e) :=
                      FibStruct_grothendieck_fibration_cleavage.comp_assoc
                        (f := clv.compare f g e)
                        (g := cartesian_lift_grothendieck_fibration_cleavage
                          clv f (pullback_obj_grothendieck_fibration_cleavage clv g e))
                        (h := cartesian_lift_grothendieck_fibration_cleavage clv g e)
                    exact congrArg (fun t => u ≫ t) hinner
              _ = (u ≫ clv.compare f g e) ≫
                    (cartesian_lift_grothendieck_fibration_cleavage
                      clv f (pullback_obj_grothendieck_fibration_cleavage clv g e) ≫
                      cartesian_lift_grothendieck_fibration_cleavage clv g e) := by
                    symm
                    exact FibStruct_grothendieck_fibration_cleavage.comp_assoc
                      (f := u)
                      (g := clv.compare f g e)
                      (h := cartesian_lift_grothendieck_fibration_cleavage
                        clv f (pullback_obj_grothendieck_fibration_cleavage clv g e) ≫
                        cartesian_lift_grothendieck_fibration_cleavage clv g e)
              _ = ((u ≫ clv.compare f g e) ≫
                    cartesian_lift_grothendieck_fibration_cleavage
                      clv f (pullback_obj_grothendieck_fibration_cleavage clv g e)) ≫
                    cartesian_lift_grothendieck_fibration_cleavage clv g e := by
                    symm
                    exact FibStruct_grothendieck_fibration_cleavage.comp_assoc
                      (f := u ≫ clv.compare f g e)
                      (g := cartesian_lift_grothendieck_fibration_cleavage
                        clv f (pullback_obj_grothendieck_fibration_cleavage clv g e))
                      (h := cartesian_lift_grothendieck_fibration_cleavage clv g e)
      _ = ((u ≫ clv.compare f g e) ≫
            cartesian_lift_grothendieck_fibration_cleavage
              clv f (pullback_obj_grothendieck_fibration_cleavage clv g e)) ≫
            cartesian_lift_grothendieck_fibration_cleavage clv g e := by
            rfl
  calc
    reindex_morphism_grothendieck_fibration_cleavage clv (f ≫ g) e z u
        = u ≫ cartesian_lift_grothendieck_fibration_cleavage clv (f ≫ g) e := by
          rfl
    _ = ((u ≫ clv.compare f g e) ≫
          cartesian_lift_grothendieck_fibration_cleavage
            clv f (pullback_obj_grothendieck_fibration_cleavage clv g e)) ≫
          cartesian_lift_grothendieck_fibration_cleavage clv g e := hstep
    _ = reindex_morphism_grothendieck_fibration_cleavage clv g e z
          (reindex_morphism_grothendieck_fibration_cleavage
            clv f (pullback_obj_grothendieck_fibration_cleavage clv g e) z
            (u ≫ clv.compare f g e)) := by
          rfl

theorem cartesian_factorization_grothendieck_fibration_cleavage
    {Obj : Type u} [FibStruct_grothendieck_fibration_cleavage Obj]
    (clv : CleavageData_grothendieck_fibration_cleavage Obj) {X Y : Obj}
    (f : X ⟶ Y) (e z : Obj) (h : z ⟶ e) :
    Exists fun u : z ⟶ pullback_obj_grothendieck_fibration_cleavage clv f e =>
      reindex_morphism_grothendieck_fibration_cleavage clv f e z u = h ∧
      (forall v : z ⟶ pullback_obj_grothendieck_fibration_cleavage clv f e,
        reindex_morphism_grothendieck_fibration_cleavage clv f e z v = h -> v = u) := by
  rcases cartesian_lift_exists_grothendieck_fibration_cleavage clv f e z h with ⟨u, hu⟩
  refine Exists.intro u ?_
  refine And.intro hu ?_
  intro v hv
  have huniq :=
    cartesian_lift_unique_grothendieck_fibration_cleavage clv f e z h v u hv hu
  exact huniq

theorem cleavage_coherence_grothendieck_fibration_cleavage
    {Obj : Type u} [FibStruct_grothendieck_fibration_cleavage Obj]
    (clv : CleavageData_grothendieck_fibration_cleavage Obj)
    {X Y Z : Obj} (f : X ⟶ Y) (g : Y ⟶ Z)
    (e z : Obj)
    (u : z ⟶ pullback_obj_grothendieck_fibration_cleavage clv (f ≫ g) e) :
    reindex_morphism_grothendieck_fibration_cleavage clv (f ≫ g) e z u =
      reindex_morphism_grothendieck_fibration_cleavage clv g e z
        ((u ≫ clv.compare f g e) ≫
          cartesian_lift_grothendieck_fibration_cleavage
            clv f (pullback_obj_grothendieck_fibration_cleavage clv g e)) := by
  have hcomp :=
    reindex_composition_grothendieck_fibration_cleavage clv f g e z u
  have hunfold :
      reindex_morphism_grothendieck_fibration_cleavage
        clv f (pullback_obj_grothendieck_fibration_cleavage clv g e) z
        (u ≫ clv.compare f g e)
      = (u ≫ clv.compare f g e) ≫
          cartesian_lift_grothendieck_fibration_cleavage
            clv f (pullback_obj_grothendieck_fibration_cleavage clv g e) := by
    rfl
  calc
    reindex_morphism_grothendieck_fibration_cleavage clv (f ≫ g) e z u
        = reindex_morphism_grothendieck_fibration_cleavage clv g e z
            (reindex_morphism_grothendieck_fibration_cleavage
              clv f (pullback_obj_grothendieck_fibration_cleavage clv g e) z
              (u ≫ clv.compare f g e)) := hcomp
    _ = reindex_morphism_grothendieck_fibration_cleavage clv g e z
          ((u ≫ clv.compare f g e) ≫
            cartesian_lift_grothendieck_fibration_cleavage
              clv f (pullback_obj_grothendieck_fibration_cleavage clv g e)) := by
          rw [hunfold]

theorem fiber_equivalence_grothendieck_fibration_cleavage
    {Obj : Type u} [FibStruct_grothendieck_fibration_cleavage Obj]
    (clv : CleavageData_grothendieck_fibration_cleavage Obj)
    {X Y : Obj} (f : X ⟶ Y) (e z : Obj)
    (u1 u2 : z ⟶ pullback_obj_grothendieck_fibration_cleavage clv f e) :
    reindex_morphism_grothendieck_fibration_cleavage clv f e z u1 =
      reindex_morphism_grothendieck_fibration_cleavage clv f e z u2 ↔
      u1 = u2 := by
  constructor
  · intro hreindex
    have hu1 :
        reindex_morphism_grothendieck_fibration_cleavage clv f e z u1 =
          reindex_morphism_grothendieck_fibration_cleavage clv f e z u1 := by
      rfl
    have hu2 :
        reindex_morphism_grothendieck_fibration_cleavage clv f e z u2 =
          reindex_morphism_grothendieck_fibration_cleavage clv f e z u1 := by
      exact Eq.trans (Eq.symm hreindex) hu1
    have huniq : u2 = u1 :=
      cartesian_lift_unique_grothendieck_fibration_cleavage
      clv f e z
      (reindex_morphism_grothendieck_fibration_cleavage clv f e z u1)
      u2 u1 hu2 hu1
    exact Eq.symm huniq
  · intro hu
    calc
      reindex_morphism_grothendieck_fibration_cleavage clv f e z u1
          = reindex_morphism_grothendieck_fibration_cleavage clv f e z u2 := by
            rw [hu]
