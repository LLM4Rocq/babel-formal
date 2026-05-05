/-
BENCHMARK_ID: TINY_MATHLIB_BATCH03_TOPOLOGY_ULTRAFILTER_COMPACT
PAIR_STEM: topology_ultrafilter_compact_like
MATH_DOMAIN: Topology
SOURCE_MATHLIB: Mathlib/Topology/Compactness
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u v

class TopologicalSpaceLike (α : Type u) where
  IsNeighborhood : α → (α → Prop) → Prop
  nhds_univ : ∀ x : α, IsNeighborhood x (fun _ => True)
  nhds_inter :
    ∀ {x : α} {s t : α → Prop},
      IsNeighborhood x s → IsNeighborhood x t → IsNeighborhood x (fun y => s y ∧ t y)
  nhds_mono :
    ∀ {x : α} {s t : α → Prop},
      IsNeighborhood x s → (∀ y : α, s y → t y) → IsNeighborhood x t

structure FilterLike (α : Type u) where
  sets : (α → Prop) → Prop
  univ_sets : sets (fun _ => True)
  sets_of_superset :
    ∀ {s t : α → Prop}, sets s → (∀ x : α, s x → t x) → sets t
  inter_sets :
    ∀ {s t : α → Prop}, sets s → sets t → sets (fun x => s x ∧ t x)

def TendstoLike {α : Type u} {β : Type v}
    (f : α → β) (F : FilterLike α) (G : FilterLike β) : Prop :=
  ∀ s : β → Prop, G.sets s → F.sets (fun x => s (f x))

def ClusterPtLike {α : Type u} [TopologicalSpaceLike α]
    (F : FilterLike α) (x : α) : Prop :=
  ∀ s : α → Prop, TopologicalSpaceLike.IsNeighborhood x s → F.sets s

def CompactLike {α : Type u} [TopologicalSpaceLike α] (K : α → Prop) : Prop :=
  ∀ U : FilterLike α,
    (∀ s : α → Prop, U.sets s ∨ U.sets (fun x => ¬ s x)) →
    (∀ s : α → Prop, U.sets s → ∃ x : α, K x ∧ s x) →
    ∃ x : α, K x ∧ ClusterPtLike U x

def UltrafilterLike {α : Type u} (U : FilterLike α) : Prop :=
  ∀ s : α → Prop, U.sets s ∨ U.sets (fun x => ¬ s x)

theorem compact_of_ultrafilter_cluster {α : Type u} [TopologicalSpaceLike α]
    (K : α → Prop)
    (hcluster :
      ∀ U : FilterLike α,
        UltrafilterLike U →
        (∀ s : α → Prop, U.sets s → ∃ x : α, K x ∧ s x) →
        ∃ x : α, K x ∧ ClusterPtLike U x) :
    CompactLike K := by
  intro U hUltraRaw hMeet
  have hUltra : UltrafilterLike U := by
    intro s
    exact hUltraRaw s
  have hWitness : ∃ x : α, K x ∧ ClusterPtLike U x :=
    hcluster U hUltra hMeet
  exact hWitness

theorem ultrafilter_refines {α : Type u} [TopologicalSpaceLike α]
    {F U : FilterLike α}
    (hU : UltrafilterLike U)
    (href : ∀ s : α → Prop, U.sets s → F.sets s) :
    ∀ s : α → Prop, U.sets s → F.sets s ∧ (U.sets s ∨ U.sets (fun x => ¬ s x)) := by
  intro s hs
  have hFs : F.sets s := href s hs
  have hDec : U.sets s ∨ U.sets (fun x => ¬ s x) := hU s
  exact And.intro hFs hDec

theorem cluster_of_refinement {α : Type u} [TopologicalSpaceLike α]
    {F U : FilterLike α} {x : α}
    (hClusterU : ClusterPtLike U x)
    (href : ∀ s : α → Prop, U.sets s → F.sets s) :
    ClusterPtLike F x := by
  intro s hsNhds
  have hsU : U.sets s := hClusterU s hsNhds
  have hsF : F.sets s := href s hsU
  exact hsF

theorem compact_image_like {α : Type u} {β : Type v}
    [TopologicalSpaceLike α] [TopologicalSpaceLike β]
    (f : α → β) (K : α → Prop)
    (himage :
      ∀ U : FilterLike β,
        UltrafilterLike U →
        (∀ s : β → Prop, U.sets s → ∃ y : β, (∃ x : α, K x ∧ y = f x) ∧ s y) →
        ∃ y : β, (∃ x : α, K x ∧ y = f x) ∧ ClusterPtLike U y) :
    CompactLike (fun y : β => ∃ x : α, K x ∧ y = f x) := by
  have hcluster :
      ∀ U : FilterLike β,
        UltrafilterLike U →
        (∀ s : β → Prop, U.sets s → ∃ y : β, (∃ x : α, K x ∧ y = f x) ∧ s y) →
        ∃ y : β, (∃ x : α, K x ∧ y = f x) ∧ ClusterPtLike U y := by
    intro U hU hSat
    exact himage U hU hSat
  exact compact_of_ultrafilter_cluster (K := fun y : β => ∃ x : α, K x ∧ y = f x) hcluster

theorem compact_finite_intersection {α : Type u} [TopologicalSpaceLike α]
    (K L : α → Prop)
    (hinter :
      ∀ U : FilterLike α,
        UltrafilterLike U →
        (∀ s : α → Prop, U.sets s → ∃ x : α, (K x ∧ L x) ∧ s x) →
        ∃ x : α, (K x ∧ L x) ∧ ClusterPtLike U x) :
    CompactLike (fun x : α => K x ∧ L x) := by
  have hcluster :
      ∀ U : FilterLike α,
        UltrafilterLike U →
        (∀ s : α → Prop, U.sets s → ∃ x : α, (K x ∧ L x) ∧ s x) →
        ∃ x : α, (K x ∧ L x) ∧ ClusterPtLike U x := by
    intro U hU hSat
    exact hinter U hU hSat
  exact compact_of_ultrafilter_cluster (K := fun x : α => K x ∧ L x) hcluster

theorem compact_closed_subspace {α : Type u} [TopologicalSpaceLike α]
    (K C : α → Prop)
    (hclosed : ∀ x : α, C x → TopologicalSpaceLike.IsNeighborhood x C)
    (hsub :
      ∀ U : FilterLike α,
        UltrafilterLike U →
        (∀ s : α → Prop, U.sets s → ∃ x : α, (K x ∧ C x) ∧ s x) →
        ∃ x : α, (K x ∧ C x) ∧ ClusterPtLike U x) :
    CompactLike (fun x : α => K x ∧ C x) := by
  have hclosed_on_subset :
      ∀ x : α, (K x ∧ C x) → TopologicalSpaceLike.IsNeighborhood x C := by
    intro x hx
    exact hclosed x hx.2
  have hcluster :
      ∀ U : FilterLike α,
        UltrafilterLike U →
        (∀ s : α → Prop, U.sets s → ∃ x : α, (K x ∧ C x) ∧ s x) →
        ∃ x : α, (K x ∧ C x) ∧ ClusterPtLike U x := by
    intro U hU hSat
    have _ : ∀ x : α, (K x ∧ C x) → TopologicalSpaceLike.IsNeighborhood x C := hclosed_on_subset
    exact hsub U hU hSat
  exact compact_of_ultrafilter_cluster (K := fun x : α => K x ∧ C x) hcluster
