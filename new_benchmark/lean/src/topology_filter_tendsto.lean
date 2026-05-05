/-
BENCHMARK_ID: TINY_MATHLIB_BATCH04_TOPO_FILTER_TENDSTO
PAIR_STEM: topology_filter_tendsto
MATH_DOMAIN: Topology
SOURCE_MATHLIB: Mathlib/Order/Filter
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 14
-/

universe u v w

namespace TopologyFilterTendsto

structure Filter (α : Type u) where
  sets : (α → Prop) → Prop
  univ_sets : sets (fun _ => True)
  sets_of_superset :
    ∀ {s t : α → Prop}, sets s → (∀ x : α, s x → t x) → sets t
  inter_sets :
    ∀ {s t : α → Prop}, sets s → sets t → sets (fun x => s x ∧ t x)

def preimage {α : Type u} {β : Type v} (f : α → β) (s : β → Prop) : α → Prop :=
  fun x => s (f x)

def map {α : Type u} {β : Type v} (f : α → β) (F : Filter α) : Filter β :=
  { sets := fun s => F.sets (preimage f s)
    univ_sets := by
      have hpre : preimage f (fun _ : β => True) = (fun _ : α => True) := by
        rfl
      rw [hpre]
      exact F.univ_sets
    sets_of_superset := by
      intro s t hs hst
      apply F.sets_of_superset hs
      intro x hx
      exact hst (f x) hx
    inter_sets := by
      intro s t hs ht
      have hinter : F.sets (fun x => preimage f s x ∧ preimage f t x) :=
        F.inter_sets hs ht
      have hpre :
          preimage f (fun y => s y ∧ t y) = (fun x => preimage f s x ∧ preimage f t x) := by
        rfl
      rw [hpre]
      exact hinter }

def Tendsto {α : Type u} {β : Type v} (f : α → β) (F : Filter α) (G : Filter β) : Prop :=
  ∀ s : β → Prop, G.sets s → F.sets (preimage f s)

theorem tendsto_id {α : Type u} (F : Filter α) :
    Tendsto (fun x : α => x) F F := by
  intro s hs
  have hpre : preimage (fun x : α => x) s = s := by
    rfl
  rw [hpre]
  exact hs

theorem tendsto_comp {α : Type u} {β : Type v} {γ : Type w}
    {f : α → β} {g : β → γ} {F : Filter α} {G : Filter β} {H : Filter γ}
    (hg : Tendsto g G H) (hf : Tendsto f F G) :
    Tendsto (fun x => g (f x)) F H := by
  intro s hs
  have hgs : G.sets (preimage g s) := hg s hs
  have hfs : F.sets (preimage f (preimage g s)) := hf (preimage g s) hgs
  have hpre :
      preimage (fun x => g (f x)) s = preimage f (preimage g s) := by
    rfl
  rw [hpre]
  exact hfs

theorem tendsto_const {α : Type u} {β : Type v}
    (F : Filter α) (G : Filter β) (c : β)
    (hGc : ∀ s : β → Prop, G.sets s → s c) :
    Tendsto (fun _ : α => c) F G := by
  intro s hs
  have hsc : s c := hGc s hs
  have huniv : F.sets (fun _ : α => True) := F.univ_sets
  have hsub : ∀ x : α, True → preimage (fun _ : α => c) s x := by
    intro x hx
    exact hsc
  have hpre : F.sets (preimage (fun _ : α => c) s) := F.sets_of_superset huniv hsub
  exact hpre

theorem tendsto_mono {α : Type u} {β : Type v}
    {f : α → β} {F : Filter α} {G H : Filter β}
    (hFG : Tendsto f F G)
    (hGH : ∀ s : β → Prop, H.sets s → G.sets s) :
    Tendsto f F H := by
  intro s hsH
  have hsG : G.sets s := hGH s hsH
  have hsF : F.sets (preimage f s) := hFG s hsG
  exact hsF

theorem map_id {α : Type u} (F : Filter α) :
    ∀ s : α → Prop, (map (fun x : α => x) F).sets s ↔ F.sets s := by
  intro s
  constructor
  · intro hs
    have hdef : (map (fun x : α => x) F).sets s = F.sets (preimage (fun x : α => x) s) := by
      rfl
    have hpre : preimage (fun x : α => x) s = s := by
      rfl
    rw [hdef] at hs
    rw [hpre] at hs
    exact hs
  · intro hs
    have hpre : preimage (fun x : α => x) s = s := by
      rfl
    have hdef : (map (fun x : α => x) F).sets s = F.sets (preimage (fun x : α => x) s) := by
      rfl
    rw [hdef]
    rw [hpre]
    exact hs

theorem map_comp {α : Type u} {β : Type v} {γ : Type w}
    (g : β → γ) (f : α → β) (F : Filter α) :
    ∀ s : γ → Prop, (map g (map f F)).sets s ↔ (map (fun x => g (f x)) F).sets s := by
  intro s
  constructor
  · intro hs
    have hleft1 : (map g (map f F)).sets s = (map f F).sets (preimage g s) := by
      rfl
    have hleft2 : (map f F).sets (preimage g s) = F.sets (preimage f (preimage g s)) := by
      rfl
    have hpre : preimage f (preimage g s) = preimage (fun x => g (f x)) s := by
      rfl
    have hright : (map (fun x => g (f x)) F).sets s = F.sets (preimage (fun x => g (f x)) s) := by
      rfl
    rw [hleft1] at hs
    rw [hleft2] at hs
    rw [hpre] at hs
    rw [hright]
    exact hs
  · intro hs
    have hleft1 : (map g (map f F)).sets s = (map f F).sets (preimage g s) := by
      rfl
    have hleft2 : (map f F).sets (preimage g s) = F.sets (preimage f (preimage g s)) := by
      rfl
    have hpre : preimage f (preimage g s) = preimage (fun x => g (f x)) s := by
      rfl
    have hright : (map (fun x => g (f x)) F).sets s = F.sets (preimage (fun x => g (f x)) s) := by
      rfl
    rw [hright] at hs
    rw [← hpre] at hs
    rw [← hleft2] at hs
    rw [hleft1]
    exact hs

theorem tendsto_map {α : Type u} {β : Type v}
    (f : α → β) (F : Filter α) :
    Tendsto f F (map f F) := by
  intro s hs
  have hs' : F.sets (preimage f s) := hs
  exact hs'

theorem tendsto_of_eq {α : Type u} {β : Type v}
    {f g : α → β} {F : Filter α} {G : Filter β}
    (hfg : Tendsto f F G) (heq : f = g) :
    Tendsto g F G := by
  intro s hs
  have hpre_f : F.sets (preimage f s) := hfg s hs
  have hpre_eq : preimage g s = preimage f s := by
    rw [← heq]
  rw [hpre_eq]
  exact hpre_f

theorem tendsto_inter {α : Type u} {β : Type v}
    {f : α → β} {F : Filter α} {G : Filter β}
    (hFG : Tendsto f F G) (s t : β → Prop)
    (hs : G.sets s) (ht : G.sets t) :
    F.sets (preimage f (fun y => s y ∧ t y)) := by
  have hsF : F.sets (preimage f s) := hFG s hs
  have htF : F.sets (preimage f t) := hFG t ht
  have hinter : F.sets (fun x => preimage f s x ∧ preimage f t x) :=
    F.inter_sets hsF htF
  have hpre :
      preimage f (fun y => s y ∧ t y) = (fun x => preimage f s x ∧ preimage f t x) := by
    rfl
  rw [hpre]
  exact hinter

theorem tendsto_principal_like {α : Type u} {β : Type v}
    {f : α → β} {F : Filter α} {G : Filter β} (c : β)
    (hGc : ∀ s : β → Prop, G.sets s ↔ s c) :
    Tendsto f F G ↔ ∀ s : β → Prop, s c → F.sets (preimage f s) := by
  constructor
  · intro hT s hs
    have hsG : G.sets s := (hGc s).2 hs
    have hsF : F.sets (preimage f s) := hT s hsG
    exact hsF
  · intro hpc
    intro s hsG
    have hsc : s c := (hGc s).1 hsG
    have hsF : F.sets (preimage f s) := hpc s hsc
    exact hsF

end TopologyFilterTendsto
