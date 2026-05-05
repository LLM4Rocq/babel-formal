/-
BENCHMARK_ID: TINY_MATHLIB_BATCH05_CATEGORY_INFINITY_QUASICATEGORY_AXIOMATIC_LIKE
PAIR_STEM: category_infinity_quasicategory_axiomatic_like
MATH_DOMAIN: Category Theory
SOURCE_MATHLIB: Mathlib/CategoryTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u v

class CatStruct_infinity_quasicategory (Obj : Type u) where
  Hom : Obj -> Obj -> Type v
  id : {X : Obj} -> Hom X X
  comp : {X Y Z : Obj} -> Hom X Y -> Hom Y Z -> Hom X Z
  comp_assoc :
    forall {W X Y Z : Obj} (f : Hom W X) (g : Hom X Y) (h : Hom Y Z),
      comp (comp f g) h = comp f (comp g h)
  id_comp : forall {X Y : Obj} (f : Hom X Y), comp id f = f
  comp_id : forall {X Y : Obj} (f : Hom X Y), comp f id = f

infixr:10 " ~>iq " => CatStruct_infinity_quasicategory.Hom
infixr:80 " >>>iq " => CatStruct_infinity_quasicategory.comp

structure Functor_infinity_quasicategory (Obj : Type u) [CatStruct_infinity_quasicategory Obj] where
  obj : Obj -> Obj
  map : {X Y : Obj} -> (X ~>iq Y) -> (X ~>iq Y)
  map_id : forall X : Obj,
      map (CatStruct_infinity_quasicategory.id (X := X)) =
        CatStruct_infinity_quasicategory.id
  map_comp : forall {X Y Z : Obj} (f : X ~>iq Y) (g : Y ~>iq Z),
      map (f >>>iq g) = map f >>>iq map g

structure NatIso_infinity_quasicategory (Obj : Type u) [CatStruct_infinity_quasicategory Obj]
    (F G : Functor_infinity_quasicategory Obj) where
  hom : forall X : Obj, X ~>iq X
  inv : forall X : Obj, X ~>iq X
  left_inv : forall X : Obj,
      hom X >>>iq inv X = CatStruct_infinity_quasicategory.id
  right_inv : forall X : Obj,
      inv X >>>iq hom X = CatStruct_infinity_quasicategory.id
  naturality : forall {X Y : Obj} (f : X ~>iq Y),
      F.map f >>>iq hom Y = hom X >>>iq G.map f

def whisker_infinity_quasicategory {Obj : Type u}
    [CatStruct_infinity_quasicategory Obj]
    (F G H : Functor_infinity_quasicategory Obj)
    (eta : NatIso_infinity_quasicategory Obj F G) :
    NatIso_infinity_quasicategory Obj F G :=
  { hom := eta.hom
    inv := eta.inv
    left_inv := by
      intro X
      exact eta.left_inv X
    right_inv := by
      intro X
      exact eta.right_inv X
    naturality := by
      intro X Y f
      have hNat : F.map f >>>iq eta.hom Y = eta.hom X >>>iq G.map f :=
        eta.naturality f
      exact hNat }

def compose_infinity_quasicategory {Obj : Type u}
    [CatStruct_infinity_quasicategory Obj]
    (F G : Functor_infinity_quasicategory Obj) :
    Functor_infinity_quasicategory Obj :=
  { obj := fun X => G.obj (F.obj X)
    map := fun {X Y} f => G.map (F.map f)
    map_id := by
      intro X
      have hF : F.map CatStruct_infinity_quasicategory.id = CatStruct_infinity_quasicategory.id :=
        F.map_id X
      have hG : G.map CatStruct_infinity_quasicategory.id = CatStruct_infinity_quasicategory.id :=
        G.map_id X
      calc
        G.map (F.map CatStruct_infinity_quasicategory.id)
            = G.map CatStruct_infinity_quasicategory.id := by
                rw [hF]
        _ = CatStruct_infinity_quasicategory.id := hG
    map_comp := by
      intro X Y Z f g
      have hF : F.map (f >>>iq g) = F.map f >>>iq F.map g := F.map_comp f g
      have hG : G.map (F.map f >>>iq F.map g) = G.map (F.map f) >>>iq G.map (F.map g) :=
        G.map_comp (F.map f) (F.map g)
      calc
        G.map (F.map (f >>>iq g))
            = G.map (F.map f >>>iq F.map g) := by
                rw [hF]
        _ = G.map (F.map f) >>>iq G.map (F.map g) := hG }

theorem whisker_assoc_infinity_quasicategory {Obj : Type u}
    [CatStruct_infinity_quasicategory Obj]
    (F G H K : Functor_infinity_quasicategory Obj)
    (eta : NatIso_infinity_quasicategory Obj F G) :
    forall X : Obj,
      (whisker_infinity_quasicategory F G K
        (whisker_infinity_quasicategory F G H eta)).hom X =
      (whisker_infinity_quasicategory F G
        (compose_infinity_quasicategory H K) eta).hom X := by
  intro X
  have hLeft :
      (whisker_infinity_quasicategory F G K
        (whisker_infinity_quasicategory F G H eta)).hom X = eta.hom X := by
    rfl
  have hRight :
      (whisker_infinity_quasicategory F G
        (compose_infinity_quasicategory H K) eta).hom X = eta.hom X := by
    rfl
  calc
    (whisker_infinity_quasicategory F G K
      (whisker_infinity_quasicategory F G H eta)).hom X = eta.hom X := hLeft
    _ = (whisker_infinity_quasicategory F G
          (compose_infinity_quasicategory H K) eta).hom X := by
          exact Eq.symm hRight

theorem unit_whisker_infinity_quasicategory {Obj : Type u}
    [CatStruct_infinity_quasicategory Obj]
    (F G : Functor_infinity_quasicategory Obj)
    (eta : NatIso_infinity_quasicategory Obj F G) :
    forall X : Obj,
      (whisker_infinity_quasicategory F G
        ({ obj := fun Z => Z
           map := fun {X Y} f => f
           map_id := by intro X; rfl
           map_comp := by intro X Y Z f g; rfl } : Functor_infinity_quasicategory Obj)
        eta).hom X = eta.hom X := by
  intro X
  have hDef :
      (whisker_infinity_quasicategory F G
        ({ obj := fun Z => Z
           map := fun {X Y} f => f
           map_id := by intro X; rfl
           map_comp := by intro X Y Z f g; rfl } : Functor_infinity_quasicategory Obj)
        eta).hom X = eta.hom X := by
    rfl
  exact hDef

theorem counit_whisker_infinity_quasicategory {Obj : Type u}
    [CatStruct_infinity_quasicategory Obj]
    (F G H : Functor_infinity_quasicategory Obj)
    (eta : NatIso_infinity_quasicategory Obj F G) :
    forall X : Obj,
      (whisker_infinity_quasicategory F G H eta).hom X >>>iq
        (whisker_infinity_quasicategory F G H eta).inv X =
          CatStruct_infinity_quasicategory.id := by
  intro X
  have hLeft :
      (whisker_infinity_quasicategory F G H eta).hom X >>>iq
        (whisker_infinity_quasicategory F G H eta).inv X =
          CatStruct_infinity_quasicategory.id :=
    (whisker_infinity_quasicategory F G H eta).left_inv X
  exact hLeft

theorem pasting_coherence_infinity_quasicategory {Obj : Type u}
    [CatStruct_infinity_quasicategory Obj]
    (F G H : Functor_infinity_quasicategory Obj)
    (eta : NatIso_infinity_quasicategory Obj F G)
    {X Y Z : Obj} (f : X ~>iq Y) (g : Y ~>iq Z) :
    F.map (f >>>iq g) >>>iq
      (whisker_infinity_quasicategory F G H eta).hom Z =
    (whisker_infinity_quasicategory F G H eta).hom X >>>iq
      G.map (f >>>iq g) := by
  have hNat :
      F.map (f >>>iq g) >>>iq eta.hom Z =
        eta.hom X >>>iq G.map (f >>>iq g) :=
    eta.naturality (f >>>iq g)
  have hLeft : (whisker_infinity_quasicategory F G H eta).hom Z = eta.hom Z := by
    rfl
  have hRight : (whisker_infinity_quasicategory F G H eta).hom X = eta.hom X := by
    rfl
  calc
    F.map (f >>>iq g) >>>iq (whisker_infinity_quasicategory F G H eta).hom Z
        = F.map (f >>>iq g) >>>iq eta.hom Z := by
            rw [hLeft]
    _ = eta.hom X >>>iq G.map (f >>>iq g) := hNat
    _ = (whisker_infinity_quasicategory F G H eta).hom X >>>iq G.map (f >>>iq g) := by
          rw [hRight]

theorem comparison_full_infinity_quasicategory {Obj : Type u}
    [CatStruct_infinity_quasicategory Obj]
    (H G : Functor_infinity_quasicategory Obj)
    (hfull : forall {X Y : Obj} (f : X ~>iq Y),
      exists g : X ~>iq Y, H.map g = f)
    {X Y : Obj} (f : X ~>iq Y) :
    exists g : X ~>iq Y,
      (compose_infinity_quasicategory H G).map g = G.map f := by
  rcases hfull f with ⟨g, hg⟩
  have hComp : (compose_infinity_quasicategory H G).map g = G.map (H.map g) := by
    rfl
  have hMapped : G.map (H.map g) = G.map f := by
    rw [hg]
  refine ⟨g, ?_⟩
  calc
    (compose_infinity_quasicategory H G).map g = G.map (H.map g) := hComp
    _ = G.map f := hMapped

theorem comparison_faithful_infinity_quasicategory {Obj : Type u}
    [CatStruct_infinity_quasicategory Obj]
    (H G : Functor_infinity_quasicategory Obj)
    (hfaith_H : forall {X Y : Obj} {f g : X ~>iq Y}, H.map f = H.map g -> f = g)
    (hfaith_G : forall {X Y : Obj} {f g : X ~>iq Y}, G.map f = G.map g -> f = g)
    {X Y : Obj} {f g : X ~>iq Y}
    (hcomp : (compose_infinity_quasicategory H G).map f =
      (compose_infinity_quasicategory H G).map g) :
    f = g := by
  have hExpand : G.map (H.map f) = G.map (H.map g) := by
    exact hcomp
  have hInner : H.map f = H.map g := hfaith_G hExpand
  have hOut : f = g := hfaith_H hInner
  exact hOut

theorem equivalence_core_infinity_quasicategory {Obj : Type u}
    [CatStruct_infinity_quasicategory Obj]
    (H G : Functor_infinity_quasicategory Obj)
    (hfull : forall {X Y : Obj} (f : X ~>iq Y),
      exists g : X ~>iq Y, H.map g = f)
    (hfaith_H : forall {X Y : Obj} {f g : X ~>iq Y}, H.map f = H.map g -> f = g)
    (hfaith_G : forall {X Y : Obj} {f g : X ~>iq Y}, G.map f = G.map g -> f = g) :
    (forall {X Y : Obj} (f : X ~>iq Y),
      exists g : X ~>iq Y, (compose_infinity_quasicategory H G).map g = G.map f) ∧
    (forall {X Y : Obj} {f g : X ~>iq Y},
      (compose_infinity_quasicategory H G).map f =
        (compose_infinity_quasicategory H G).map g -> f = g) := by
  constructor
  · intro X Y f
    exact comparison_full_infinity_quasicategory H G hfull f
  · intro X Y f g hEq
    exact comparison_faithful_infinity_quasicategory H G hfaith_H hfaith_G hEq
