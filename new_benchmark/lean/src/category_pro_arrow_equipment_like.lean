/-
BENCHMARK_ID: TINY_MATHLIB_BATCH05_CATEGORY_PRO_ARROW_EQUIPMENT_LIKE
PAIR_STEM: category_pro_arrow_equipment_like
MATH_DOMAIN: Category Theory
SOURCE_MATHLIB: Mathlib/CategoryTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u v w

class CatStruct_pro_arrow_equipment (Obj : Type u) where
  Hom : Obj → Obj → Type v
  id : {X : Obj} → Hom X X
  comp : {X Y Z : Obj} → Hom X Y → Hom Y Z → Hom X Z
  comp_assoc :
    ∀ {W X Y Z : Obj} (f : Hom W X) (g : Hom X Y) (h : Hom Y Z),
      comp (comp f g) h = comp f (comp g h)
  id_comp : ∀ {X Y : Obj} (f : Hom X Y), comp id f = f
  comp_id : ∀ {X Y : Obj} (f : Hom X Y), comp f id = f
  ProArr : Obj → Obj → Type w
  whiskerL : ∀ {A B C : Obj}, Hom A B → ProArr B C → ProArr A C
  whiskerR : ∀ {A B C : Obj}, ProArr A B → Hom B C → ProArr A C
  whisker_assoc_axiom :
    ∀ {A B C D : Obj} (f : Hom A B) (p : ProArr B C) (g : Hom C D),
      whiskerR (whiskerL f p) g = whiskerL f (whiskerR p g)
  unit_whisker_axiom : ∀ {A B : Obj} (p : ProArr A B), whiskerL id p = p
  counit_whisker_axiom : ∀ {A B : Obj} (p : ProArr A B), whiskerR p id = p
  comparison_full_axiom :
    ∀ {A B : Obj} (p q : ProArr A B),
      whiskerR p id = whiskerR q id → p = q
  comparison_faithful_axiom :
    ∀ {A B : Obj} (p q : ProArr A B),
      p = q → whiskerL id p = whiskerL id q

infixr:10 " ~> " => CatStruct_pro_arrow_equipment.Hom
infixr:10 " ~~> " => CatStruct_pro_arrow_equipment.ProArr
infixr:80 " >>> " => CatStruct_pro_arrow_equipment.comp

structure Functor_pro_arrow_equipment
    (C : Type u) (D : Type u)
    [CatStruct_pro_arrow_equipment C] [CatStruct_pro_arrow_equipment D] where
  obj : C → D
  map_hom : ∀ {X Y : C}, (X ~> Y) → (obj X ~> obj Y)
  map_pro : ∀ {X Y : C}, (X ~~> Y) → (obj X ~~> obj Y)
  map_id : ∀ X : C, map_hom (CatStruct_pro_arrow_equipment.id (X := X)) = CatStruct_pro_arrow_equipment.id
  map_comp :
    ∀ {X Y Z : C} (f : X ~> Y) (g : Y ~> Z),
      map_hom (f >>> g) = (map_hom f) >>> (map_hom g)

structure NatIso_pro_arrow_equipment
    {C : Type u} [CatStruct_pro_arrow_equipment C]
    (F G : Functor_pro_arrow_equipment C C) where
  hom : ∀ X : C, F.obj X ~> G.obj X
  inv : ∀ X : C, G.obj X ~> F.obj X
  hom_inv_id : ∀ X : C, (hom X) >>> (inv X) = CatStruct_pro_arrow_equipment.id
  inv_hom_id : ∀ X : C, (inv X) >>> (hom X) = CatStruct_pro_arrow_equipment.id

def whisker_pro_arrow_equipment
    {C : Type u} [CatStruct_pro_arrow_equipment C]
    {A B D : C} (f : A ~> B) (p : B ~~> D) : A ~~> D :=
  CatStruct_pro_arrow_equipment.whiskerL f p

def compose_pro_arrow_equipment
    {C : Type u} [CatStruct_pro_arrow_equipment C]
    {A B D : C} (f : A ~> B) (g : B ~> D) : A ~> D :=
  f >>> g

theorem whisker_assoc_pro_arrow_equipment
    {C : Type u} [CatStruct_pro_arrow_equipment C]
    {A B D E : C} (f : A ~> B) (p : B ~~> D) (g : D ~> E) :
    CatStruct_pro_arrow_equipment.whiskerR (whisker_pro_arrow_equipment f p) g =
      whisker_pro_arrow_equipment f (CatStruct_pro_arrow_equipment.whiskerR p g) := by
  have hAssoc :
      CatStruct_pro_arrow_equipment.whiskerR
          (CatStruct_pro_arrow_equipment.whiskerL f p) g =
        CatStruct_pro_arrow_equipment.whiskerL f (CatStruct_pro_arrow_equipment.whiskerR p g) :=
    CatStruct_pro_arrow_equipment.whisker_assoc_axiom f p g
  have hLeft : whisker_pro_arrow_equipment f p = CatStruct_pro_arrow_equipment.whiskerL f p := by
    rfl
  have hRight :
      whisker_pro_arrow_equipment f (CatStruct_pro_arrow_equipment.whiskerR p g) =
        CatStruct_pro_arrow_equipment.whiskerL f (CatStruct_pro_arrow_equipment.whiskerR p g) := by
    rfl
  calc
    CatStruct_pro_arrow_equipment.whiskerR (whisker_pro_arrow_equipment f p) g
        = CatStruct_pro_arrow_equipment.whiskerR (CatStruct_pro_arrow_equipment.whiskerL f p) g := by
            rw [hLeft]
    _ = CatStruct_pro_arrow_equipment.whiskerL f (CatStruct_pro_arrow_equipment.whiskerR p g) := hAssoc
    _ = whisker_pro_arrow_equipment f (CatStruct_pro_arrow_equipment.whiskerR p g) := by
          rw [hRight]

theorem unit_whisker_pro_arrow_equipment
    {C : Type u} [CatStruct_pro_arrow_equipment C]
    {A B : C} (p : A ~~> B) :
    whisker_pro_arrow_equipment (CatStruct_pro_arrow_equipment.id (X := A)) p = p := by
  have hUnit :
      CatStruct_pro_arrow_equipment.whiskerL (CatStruct_pro_arrow_equipment.id (X := A)) p = p :=
    CatStruct_pro_arrow_equipment.unit_whisker_axiom p
  have hDef :
      whisker_pro_arrow_equipment (CatStruct_pro_arrow_equipment.id (X := A)) p =
        CatStruct_pro_arrow_equipment.whiskerL (CatStruct_pro_arrow_equipment.id (X := A)) p := by
    rfl
  calc
    whisker_pro_arrow_equipment (CatStruct_pro_arrow_equipment.id (X := A)) p
        = CatStruct_pro_arrow_equipment.whiskerL (CatStruct_pro_arrow_equipment.id (X := A)) p := hDef
    _ = p := hUnit

theorem counit_whisker_pro_arrow_equipment
    {C : Type u} [CatStruct_pro_arrow_equipment C]
    {A B : C} (p : A ~~> B) :
    CatStruct_pro_arrow_equipment.whiskerR p (CatStruct_pro_arrow_equipment.id (X := B)) = p := by
  have hCounit :
      CatStruct_pro_arrow_equipment.whiskerR p (CatStruct_pro_arrow_equipment.id (X := B)) = p :=
    CatStruct_pro_arrow_equipment.counit_whisker_axiom p
  exact hCounit

theorem pasting_coherence_pro_arrow_equipment
    {C : Type u} [CatStruct_pro_arrow_equipment C]
    {A B : C} (p : A ~~> B) :
    CatStruct_pro_arrow_equipment.whiskerR
      (whisker_pro_arrow_equipment (CatStruct_pro_arrow_equipment.id (X := A)) p)
      (CatStruct_pro_arrow_equipment.id (X := B)) = p := by
  have hAssoc :
      CatStruct_pro_arrow_equipment.whiskerR
        (whisker_pro_arrow_equipment (CatStruct_pro_arrow_equipment.id (X := A)) p)
        (CatStruct_pro_arrow_equipment.id (X := B)) =
      whisker_pro_arrow_equipment (CatStruct_pro_arrow_equipment.id (X := A))
        (CatStruct_pro_arrow_equipment.whiskerR p (CatStruct_pro_arrow_equipment.id (X := B))) :=
    whisker_assoc_pro_arrow_equipment
      (CatStruct_pro_arrow_equipment.id (X := A)) p (CatStruct_pro_arrow_equipment.id (X := B))
  have hCounit :
      CatStruct_pro_arrow_equipment.whiskerR p (CatStruct_pro_arrow_equipment.id (X := B)) = p :=
    counit_whisker_pro_arrow_equipment p
  have hUnit :
      whisker_pro_arrow_equipment (CatStruct_pro_arrow_equipment.id (X := A)) p = p :=
    unit_whisker_pro_arrow_equipment p
  calc
    CatStruct_pro_arrow_equipment.whiskerR
        (whisker_pro_arrow_equipment (CatStruct_pro_arrow_equipment.id (X := A)) p)
        (CatStruct_pro_arrow_equipment.id (X := B))
        = whisker_pro_arrow_equipment (CatStruct_pro_arrow_equipment.id (X := A))
            (CatStruct_pro_arrow_equipment.whiskerR p (CatStruct_pro_arrow_equipment.id (X := B))) := hAssoc
    _ = whisker_pro_arrow_equipment (CatStruct_pro_arrow_equipment.id (X := A)) p := by
          rw [hCounit]
    _ = p := hUnit

theorem comparison_full_pro_arrow_equipment
    {C : Type u} [CatStruct_pro_arrow_equipment C]
    {A B : C} (p q : A ~~> B)
    (hComp : CatStruct_pro_arrow_equipment.whiskerR p (CatStruct_pro_arrow_equipment.id (X := B)) =
      CatStruct_pro_arrow_equipment.whiskerR q (CatStruct_pro_arrow_equipment.id (X := B))) :
    p = q := by
  have hFull :
      CatStruct_pro_arrow_equipment.whiskerR p (CatStruct_pro_arrow_equipment.id (X := B)) =
        CatStruct_pro_arrow_equipment.whiskerR q (CatStruct_pro_arrow_equipment.id (X := B)) :=
    hComp
  exact CatStruct_pro_arrow_equipment.comparison_full_axiom p q hFull

theorem comparison_faithful_pro_arrow_equipment
    {C : Type u} [CatStruct_pro_arrow_equipment C]
    {A B : C} (p q : A ~~> B) (hpq : p = q) :
    whisker_pro_arrow_equipment (CatStruct_pro_arrow_equipment.id (X := A)) p =
      whisker_pro_arrow_equipment (CatStruct_pro_arrow_equipment.id (X := A)) q := by
  have hFaith :
      CatStruct_pro_arrow_equipment.whiskerL (CatStruct_pro_arrow_equipment.id (X := A)) p =
        CatStruct_pro_arrow_equipment.whiskerL (CatStruct_pro_arrow_equipment.id (X := A)) q :=
    CatStruct_pro_arrow_equipment.comparison_faithful_axiom p q hpq
  have hLeft :
      whisker_pro_arrow_equipment (CatStruct_pro_arrow_equipment.id (X := A)) p =
        CatStruct_pro_arrow_equipment.whiskerL (CatStruct_pro_arrow_equipment.id (X := A)) p := by
    rfl
  have hRight :
      whisker_pro_arrow_equipment (CatStruct_pro_arrow_equipment.id (X := A)) q =
        CatStruct_pro_arrow_equipment.whiskerL (CatStruct_pro_arrow_equipment.id (X := A)) q := by
    rfl
  calc
    whisker_pro_arrow_equipment (CatStruct_pro_arrow_equipment.id (X := A)) p
        = CatStruct_pro_arrow_equipment.whiskerL (CatStruct_pro_arrow_equipment.id (X := A)) p := hLeft
    _ = CatStruct_pro_arrow_equipment.whiskerL (CatStruct_pro_arrow_equipment.id (X := A)) q := hFaith
    _ = whisker_pro_arrow_equipment (CatStruct_pro_arrow_equipment.id (X := A)) q := by
          rw [hRight]

theorem equivalence_core_pro_arrow_equipment
    {C : Type u} [CatStruct_pro_arrow_equipment C]
    {A B : C} (p q : A ~~> B) :
    p = q ↔
      CatStruct_pro_arrow_equipment.whiskerR p (CatStruct_pro_arrow_equipment.id (X := B)) =
        CatStruct_pro_arrow_equipment.whiskerR q (CatStruct_pro_arrow_equipment.id (X := B)) := by
  constructor
  · intro hpq
    have hCong :
        CatStruct_pro_arrow_equipment.whiskerR p (CatStruct_pro_arrow_equipment.id (X := B)) =
          CatStruct_pro_arrow_equipment.whiskerR q (CatStruct_pro_arrow_equipment.id (X := B)) := by
      simpa [hpq]
    exact hCong
  · intro hWhisk
    exact comparison_full_pro_arrow_equipment p q hWhisk
