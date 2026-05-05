/-
BENCHMARK_ID: TINY_MATHLIB_BATCH06_TOPOLOGY_ANR_HOMOTOPY_EXTENSION_LIKE
PAIR_STEM: topology_anr_homotopy_extension_like
MATH_DOMAIN: Topology
SOURCE_MATHLIB: Mathlib/Topology/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u v w

class ANRStruct_homotopy_extension (A : Type u) (X : Type v) (Y : Type w) where
  cofibration : A → X
  extend : (A → Y) → X → Y
  extend_on_cofibration :
    ∀ (f : A → Y) (a : A),
      extend f (cofibration a) = f a
  extend_natural :
    ∀ (f : A → Y) (u : X → X),
      (∀ a : A, u (cofibration a) = cofibration a) →
      ∀ x : X,
        extend f (u x) = extend f x
  homotopy_fill : (A → Nat → Y) → (X → Y) → X → Nat → Y
  homotopy_boundary :
    ∀ (F : A → Nat → Y) (g : X → Y) (a : A) (n : Nat),
      homotopy_fill F g (cofibration a) n = F a n
  homotopy_start :
    ∀ (F : A → Nat → Y) (g : X → Y) (x : X),
      homotopy_fill F g x 0 = g x
  glue_fill :
    ∀ (H1 H2 : X → Nat → Y),
      (∀ x : X, H1 x 0 = H2 x 0) →
      (∀ x : X, H1 x 1 = H2 x 1) →
      ∃ H : X → Nat → Y,
        (∀ x : X, H x 0 = H1 x 0) ∧
        (∀ x : X, H x 1 = H2 x 1)
  endpoint_stable :
    ∀ (H : X → Nat → Y),
      (∀ a : A, H (cofibration a) 0 = H (cofibration a) 1) →
      ∀ a : A,
        ∀ n : Nat,
          H (cofibration a) n = H (cofibration a) 0

structure CylData_anr_homotopy_extension (X : Type v) (Y : Type w) where
  left : X → Y
  right : X → Y
  bridge : X → Nat → Y
  bridge_left : ∀ x : X, bridge x 0 = left x
  bridge_right : ∀ x : X, bridge x 1 = right x

def cofibration_map_anr_homotopy_extension
    {A : Type u} {X : Type v} {Y : Type w}
    [h : ANRStruct_homotopy_extension A X Y] :
    A → X :=
  h.cofibration

def extension_operator_anr_homotopy_extension
    {A : Type u} {X : Type v} {Y : Type w}
    [h : ANRStruct_homotopy_extension A X Y]
    (f : A → Y) :
    X → Y :=
  h.extend f

def terminal_homotopy_anr_homotopy_extension
    {A : Type u} {X : Type v} {Y : Type w}
    [h : ANRStruct_homotopy_extension A X Y]
    (F : A → Nat → Y)
    (g : X → Y) :
    X → Y :=
  fun x : X => h.homotopy_fill F g x 1

theorem cofibration_lift_exists_anr_homotopy_extension
    {A : Type u} {X : Type v} {Y : Type w}
    [h : ANRStruct_homotopy_extension A X Y]
    (f : A → Y) :
    ∃ g : X → Y,
      (∀ a : A,
        g (cofibration_map_anr_homotopy_extension (A := A) (X := X) (Y := Y) a) = f a) ∧
      (∃ Hcyl : X → Nat → Y,
        (∀ x : X, Hcyl x 0 = g x) ∧
        (∀ a : A,
          Hcyl (cofibration_map_anr_homotopy_extension (A := A) (X := X) (Y := Y) a) 0 = f a)) := by
  refine ⟨extension_operator_anr_homotopy_extension (A := A) (X := X) (Y := Y) f, ?_, ?_⟩
  · intro a
    have hAgree : h.extend f (h.cofibration a) = f a := h.extend_on_cofibration f a
    exact hAgree
  · refine ⟨h.homotopy_fill (fun a : A => fun _n : Nat => f a) (h.extend f), ?_, ?_⟩
    · intro x
      exact h.homotopy_start (fun a : A => fun _n : Nat => f a) (h.extend f) x
    · intro a
      have hBoundary0 :
          h.homotopy_fill (fun a : A => fun _n : Nat => f a) (h.extend f) (h.cofibration a) 0 =
            (fun a' : A => fun _n : Nat => f a') a 0 :=
        h.homotopy_boundary (fun a' : A => fun _n : Nat => f a') (h.extend f) a 0
      calc
        h.homotopy_fill (fun a' : A => fun _n : Nat => f a') (h.extend f) (h.cofibration a) 0
            = (fun a' : A => fun _n : Nat => f a') a 0 := hBoundary0
        _ = f a := rfl

theorem homotopy_extension_step_anr_homotopy_extension
    {A : Type u} {X : Type v} {Y : Type w}
    [h : ANRStruct_homotopy_extension A X Y]
    (F : A → Nat → Y)
    (g : X → Y) :
    (∀ a : A,
      terminal_homotopy_anr_homotopy_extension (A := A) (X := X) (Y := Y) F g
        (cofibration_map_anr_homotopy_extension (A := A) (X := X) (Y := Y) a) = F a 1) ∧
    (∀ a : A,
      h.homotopy_fill F g (cofibration_map_anr_homotopy_extension (A := A) (X := X) (Y := Y) a) 0 = F a 0) ∧
    (∀ x : X, h.homotopy_fill F g x 0 = g x) := by
  refine ⟨?_, ?_, ?_⟩
  · intro a
    have hBoundary : h.homotopy_fill F g (h.cofibration a) 1 = F a 1 := h.homotopy_boundary F g a 1
    exact hBoundary
  · intro a
    have hBoundary0 : h.homotopy_fill F g (h.cofibration a) 0 = F a 0 := h.homotopy_boundary F g a 0
    exact hBoundary0
  · intro x
    have hStart : h.homotopy_fill F g x 0 = g x := h.homotopy_start F g x
    exact hStart

theorem relative_retraction_anr_homotopy_extension
    {A : Type u} {X : Type v} {Y : Type w}
    [h : ANRStruct_homotopy_extension A X Y]
    (f : A → Y)
    (u : X → X)
    (hu : ∀ a : A,
      u (cofibration_map_anr_homotopy_extension (A := A) (X := X) (Y := Y) a) =
        cofibration_map_anr_homotopy_extension (A := A) (X := X) (Y := Y) a) :
    (∀ a : A,
      extension_operator_anr_homotopy_extension (A := A) (X := X) (Y := Y) f
        (u (cofibration_map_anr_homotopy_extension (A := A) (X := X) (Y := Y) a)) = f a) ∧
    (∀ x : X,
      extension_operator_anr_homotopy_extension (A := A) (X := X) (Y := Y) f (u x) =
        extension_operator_anr_homotopy_extension (A := A) (X := X) (Y := Y) f x) := by
  have hNat : ∀ x : X, h.extend f (u x) = h.extend f x := h.extend_natural f u hu
  refine ⟨?_, ?_⟩
  · intro a
    have hStep1 : h.extend f (u (h.cofibration a)) = h.extend f (h.cofibration a) := hNat (h.cofibration a)
    have hStep2 : h.extend f (h.cofibration a) = f a := h.extend_on_cofibration f a
    calc
      extension_operator_anr_homotopy_extension (A := A) (X := X) (Y := Y) f
          (u (cofibration_map_anr_homotopy_extension (A := A) (X := X) (Y := Y) a))
          = h.extend f (u (h.cofibration a)) := rfl
      _ = h.extend f (h.cofibration a) := hStep1
      _ = f a := hStep2
  · intro x
    exact hNat x

theorem glued_homotopy_continuous_anr_homotopy_extension
    {A : Type u} {X : Type v} {Y : Type w}
    [h : ANRStruct_homotopy_extension A X Y]
    (c : CylData_anr_homotopy_extension X Y)
    (hEq : ∀ x : X, c.left x = c.right x) :
    ∃ H : X → Nat → Y,
      (∀ x : X, H x 0 = c.left x) ∧
      (∀ x : X, H x 1 = c.right x) ∧
      (∀ x : X, H x 0 = H x 1) := by
  have hSame0 : ∀ x : X, c.bridge x 0 = c.bridge x 0 := by
    intro x
    rfl
  have hSame1 : ∀ x : X, c.bridge x 1 = c.bridge x 1 := by
    intro x
    rfl
  rcases h.glue_fill c.bridge c.bridge hSame0 hSame1 with ⟨H, hH0, hH1⟩
  refine ⟨H, ?_, ?_, ?_⟩
  · intro x
    calc
      H x 0 = c.bridge x 0 := hH0 x
      _ = c.left x := c.bridge_left x
  · intro x
    calc
      H x 1 = c.bridge x 1 := hH1 x
      _ = c.right x := c.bridge_right x
  · intro x
    calc
      H x 0 = c.bridge x 0 := hH0 x
      _ = c.left x := c.bridge_left x
      _ = c.right x := hEq x
      _ = c.bridge x 1 := by
        symm
        exact c.bridge_right x
      _ = H x 1 := by
        symm
        exact hH1 x

theorem endpoint_control_anr_homotopy_extension
    {A : Type u} {X : Type v} {Y : Type w}
    [h : ANRStruct_homotopy_extension A X Y]
    (H : X → Nat → Y)
    (hRel : ∀ a : A,
      H (cofibration_map_anr_homotopy_extension (A := A) (X := X) (Y := Y) a) 0 =
        H (cofibration_map_anr_homotopy_extension (A := A) (X := X) (Y := Y) a) 1) :
    (∀ a : A,
      H (cofibration_map_anr_homotopy_extension (A := A) (X := X) (Y := Y) a) 1 =
        H (cofibration_map_anr_homotopy_extension (A := A) (X := X) (Y := Y) a) 0) ∧
    (∀ a : A,
      H (cofibration_map_anr_homotopy_extension (A := A) (X := X) (Y := Y) a) 2 =
        H (cofibration_map_anr_homotopy_extension (A := A) (X := X) (Y := Y) a) 1) := by
  have hStable : ∀ a : A, ∀ n : Nat, H (h.cofibration a) n = H (h.cofibration a) 0 :=
    h.endpoint_stable H hRel
  refine ⟨?_, ?_⟩
  · intro a
    exact hStable a 1
  · intro a
    have hTwo : H (h.cofibration a) 2 = H (h.cofibration a) 0 := hStable a 2
    have hOne : H (h.cofibration a) 1 = H (h.cofibration a) 0 := hStable a 1
    calc
      H (cofibration_map_anr_homotopy_extension (A := A) (X := X) (Y := Y) a) 2
          = H (h.cofibration a) 0 := hTwo
      _ = H (h.cofibration a) 1 := by
        symm
        exact hOne

theorem extension_naturality_anr_homotopy_extension
    {A : Type u} {X : Type v} {Y : Type w}
    [h : ANRStruct_homotopy_extension A X Y]
    (f : A → Y)
    (u v : X → X)
    (hu : ∀ a : A,
      u (cofibration_map_anr_homotopy_extension (A := A) (X := X) (Y := Y) a) =
        cofibration_map_anr_homotopy_extension (A := A) (X := X) (Y := Y) a)
    (hv : ∀ a : A,
      v (cofibration_map_anr_homotopy_extension (A := A) (X := X) (Y := Y) a) =
        cofibration_map_anr_homotopy_extension (A := A) (X := X) (Y := Y) a) :
    (∀ x : X,
      extension_operator_anr_homotopy_extension (A := A) (X := X) (Y := Y) f (u (v x)) =
        extension_operator_anr_homotopy_extension (A := A) (X := X) (Y := Y) f x) ∧
    (∀ x : X,
      extension_operator_anr_homotopy_extension (A := A) (X := X) (Y := Y) f (v (u x)) =
        extension_operator_anr_homotopy_extension (A := A) (X := X) (Y := Y) f x) := by
  have hNatU : ∀ x : X, h.extend f (u x) = h.extend f x := h.extend_natural f u hu
  have hNatV : ∀ x : X, h.extend f (v x) = h.extend f x := h.extend_natural f v hv
  refine ⟨?_, ?_⟩
  · intro x
    calc
      extension_operator_anr_homotopy_extension (A := A) (X := X) (Y := Y) f (u (v x))
          = h.extend f (u (v x)) := rfl
      _ = h.extend f (v x) := hNatU (v x)
      _ = h.extend f x := hNatV x
      _ = extension_operator_anr_homotopy_extension (A := A) (X := X) (Y := Y) f x := rfl
  · intro x
    calc
      extension_operator_anr_homotopy_extension (A := A) (X := X) (Y := Y) f (v (u x))
          = h.extend f (v (u x)) := rfl
      _ = h.extend f (u x) := hNatV (u x)
      _ = h.extend f x := hNatU x
      _ = extension_operator_anr_homotopy_extension (A := A) (X := X) (Y := Y) f x := rfl

theorem homotopy_extension_universal_anr_homotopy_extension
    {A : Type u} {X : Type v} {Y : Type w}
    [h : ANRStruct_homotopy_extension A X Y]
    (F : A → Nat → Y)
    (g0 : X → Y)
    (h0 : ∀ a : A,
      g0 (cofibration_map_anr_homotopy_extension (A := A) (X := X) (Y := Y) a) = F a 0) :
    ∃ H : X → Nat → Y,
      (∀ x : X, H x 0 = g0 x) ∧
      (∀ a : A,
        H (cofibration_map_anr_homotopy_extension (A := A) (X := X) (Y := Y) a) 1 = F a 1) ∧
      (∀ a : A,
        H (cofibration_map_anr_homotopy_extension (A := A) (X := X) (Y := Y) a) 0 =
          g0 (cofibration_map_anr_homotopy_extension (A := A) (X := X) (Y := Y) a)) := by
  refine ⟨h.homotopy_fill F g0, ?_, ?_, ?_⟩
  · intro x
    exact h.homotopy_start F g0 x
  · intro a
    exact h.homotopy_boundary F g0 a 1
  · intro a
    have hBdry0 : h.homotopy_fill F g0 (h.cofibration a) 0 = F a 0 := h.homotopy_boundary F g0 a 0
    calc
      h.homotopy_fill F g0 (h.cofibration a) 0 = F a 0 := hBdry0
      _ = g0 (h.cofibration a) := by
        symm
        exact h0 a
