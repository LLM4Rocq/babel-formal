/-
BENCHMARK_ID: TINY_MATHLIB_BATCH04_TOPOLOGY_BAIRE_CATEGORY_ADVANCED
PAIR_STEM: topology_baire_category_advanced_like
MATH_DOMAIN: Topology / Analysis
SOURCE_MATHLIB: Mathlib/Topology/Baire/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u

class TopologicalSpaceLike (α : Type u) where
  IsOpen : (α → Prop) → Prop
  open_univ : IsOpen (fun _ => True)
  open_inter :
    ∀ {s t : α → Prop}, IsOpen s → IsOpen t → IsOpen (fun x => s x ∧ t x)
  inhabited_space : ∃ x : α, True

def DenseLike {α : Type u} [TopologicalSpaceLike α] (s : α → Prop) : Prop :=
  ∀ U : α → Prop,
    TopologicalSpaceLike.IsOpen U →
      (∃ x : α, U x) →
        ∃ x : α, s x ∧ U x

def OpenLike {α : Type u} [TopologicalSpaceLike α] (s : α → Prop) : Prop :=
  TopologicalSpaceLike.IsOpen s

def MeagreLike {α : Type u} [TopologicalSpaceLike α] (s : α → Prop) : Prop :=
  ∃ F : Nat → α → Prop,
    (∀ n : Nat, DenseLike (fun x => ¬ F n x)) ∧
      (∀ x : α, s x → ∃ n : Nat, F n x)

def ResidualLike {α : Type u} [TopologicalSpaceLike α] (s : α → Prop) : Prop :=
  ∃ t : α → Prop,
    MeagreLike t ∧
      (∀ x : α, s x ↔ ¬ t x)

def GdeltaLike {α : Type u} [TopologicalSpaceLike α] (s : α → Prop) : Prop :=
  ∃ U : Nat → α → Prop,
    (∀ n : Nat, OpenLike (U n)) ∧
      (∀ x : α, s x ↔ ∀ n : Nat, U n x)

theorem residual_dense {α : Type u} [TopologicalSpaceLike α]
    (R : α → Prop)
    (hR : ResidualLike R)
    (hDenseComp : ∀ t : α → Prop, MeagreLike t → DenseLike (fun x => ¬ t x)) :
    DenseLike R := by
  intro U hU hNonempty
  rcases hR with ⟨t, htMeagre, htChar⟩
  have hDenseNotT : DenseLike (fun x => ¬ t x) := hDenseComp t htMeagre
  have hMeet : ∃ x : α, (¬ t x) ∧ U x := hDenseNotT U hU hNonempty
  rcases hMeet with ⟨x, hNotTx, hUx⟩
  have hRx : R x := (htChar x).2 hNotTx
  exact ⟨x, hRx, hUx⟩

theorem meagre_union {α : Type u} [TopologicalSpaceLike α]
    (A B : α → Prop)
    (hA : MeagreLike A)
    (hB : MeagreLike B)
    (hUnionClosure :
      ∀ s t : α → Prop,
        MeagreLike s → MeagreLike t → MeagreLike (fun x => s x ∨ t x)) :
    MeagreLike (fun x => A x ∨ B x) := by
  have hAB : MeagreLike (fun x => A x ∨ B x) := hUnionClosure A B hA hB
  have hKeep : MeagreLike (fun x => A x ∨ B x) := hAB
  exact hKeep

theorem baire_intersection_dense {α : Type u} [TopologicalSpaceLike α]
    (U : Nat → α → Prop)
    (hOpen : ∀ n : Nat, OpenLike (U n))
    (hDenseEach : ∀ n : Nat, DenseLike (U n))
    (hBaire :
      ∀ V : α → Prop,
        OpenLike V →
          (∃ x : α, V x) →
            ∃ x : α, (∀ n : Nat, U n x) ∧ V x) :
    DenseLike (fun x => ∀ n : Nat, U n x) := by
  intro V hV hNonempty
  have hMeetAll : ∃ x : α, (∀ n : Nat, U n x) ∧ V x := hBaire V hV hNonempty
  rcases hMeetAll with ⟨x, hxAll, hxV⟩
  have hKeepDense : ∀ n : Nat, DenseLike (U n) := hDenseEach
  have _ : ∀ n : Nat, OpenLike (U n) := hOpen
  exact ⟨x, hxAll, hxV⟩

theorem generic_point_exists_like {α : Type u} [TopologicalSpaceLike α]
    (R : α → Prop)
    (hDenseR : DenseLike R) :
    ∃ x : α, R x := by
  have hUnivOpen : OpenLike (fun _ : α => True) := TopologicalSpaceLike.open_univ
  rcases (TopologicalSpaceLike.inhabited_space (α := α)) with ⟨x0, hx0⟩
  have hNonemptyUniv : ∃ x : α, (fun _ : α => True) x := by
    refine ⟨x0, ?_⟩
    trivial
  have hMeet : ∃ x : α, R x ∧ (fun _ : α => True) x :=
    hDenseR (fun _ : α => True) hUnivOpen hNonemptyUniv
  rcases hMeet with ⟨x, hRx, hTrue⟩
  have _ : True := hTrue
  exact ⟨x, hRx⟩

theorem open_mapping_baire_step {α : Type u} [TopologicalSpaceLike α]
    (f : α → α) (A B : α → Prop)
    (hAOpen : OpenLike A)
    (hADense : DenseLike A)
    (hImageOpen :
      ∀ s : α → Prop,
        OpenLike s → OpenLike (fun y => ∃ x : α, s x ∧ y = f x))
    (hImageDense :
      ∀ s : α → Prop,
        DenseLike s → DenseLike (fun y => ∃ x : α, s x ∧ y = f x))
    (hAB : ∀ x : α, A x → B (f x)) :
    DenseLike B := by
  intro U hU hNonempty
  have hImgOpenA : OpenLike (fun y => ∃ x : α, A x ∧ y = f x) := hImageOpen A hAOpen
  have hImgDenseA : DenseLike (fun y : α => ∃ x : α, A x ∧ y = f x) :=
    hImageDense A hADense
  have hMeet : ∃ y : α, (∃ x : α, A x ∧ y = f x) ∧ U y :=
    hImgDenseA U hU hNonempty
  rcases hMeet with ⟨y, hyImg, hyU⟩
  rcases hyImg with ⟨x, hAx, hyEq⟩
  have hBx : B (f x) := hAB x hAx
  have _ : OpenLike (fun y => ∃ x : α, A x ∧ y = f x) := hImgOpenA
  refine ⟨f x, hBx, ?_⟩
  simpa [hyEq] using hyU

theorem baire_category_transfer {α : Type u} [TopologicalSpaceLike α]
    (f : α → α) (R : α → Prop)
    (hResidual : ResidualLike R)
    (hPreResidual :
      ∀ s : α → Prop,
        ResidualLike s → ResidualLike (fun x => s (f x)))
    (hDenseOfResidual :
      ∀ s : α → Prop,
        ResidualLike s → DenseLike s) :
    DenseLike (fun x => R (f x)) := by
  have hPre : ResidualLike (fun x => R (f x)) := hPreResidual R hResidual
  have hDensePre : DenseLike (fun x => R (f x)) :=
    hDenseOfResidual (fun x => R (f x)) hPre
  have hKeep : DenseLike (fun x => R (f x)) := hDensePre
  exact hKeep
