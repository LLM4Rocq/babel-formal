/-
BENCHMARK_ID: TINY_MATHLIB_BATCH03_COMB_EXTREMAL_DOUBLE_COUNTING_LIKE
PAIR_STEM: combinatorics_extremal_double_counting_like
MATH_DOMAIN: Combinatorics
SOURCE_MATHLIB: Mathlib/Combinatorics/SimpleGraph/DegreeSum
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

universe u v

class FiniteSetLike (V : Type u) (E : Type v) where
  cardV : Nat
  cardE : Nat
  sumV : (V → Nat) → Nat
  sumE : (E → Nat) → Nat
  sumV_const : ∀ n : Nat, sumV (fun _ : V => n) = cardV * n
  sumE_const : ∀ n : Nat, sumE (fun _ : E => n) = cardE * n
  sumV_mono : ∀ f g : V → Nat, (∀ x : V, f x ≤ g x) → sumV f ≤ sumV g
  cs_bound : ∀ f : V → Nat, sumV f * sumV f ≤ cardV * sumV (fun x : V => f x * f x)

def IncidenceLike (V : Type u) (E : Type v) : Type (max u v) :=
  V → E → Prop

def degreeLike {V : Type u} {E : Type v} (I : IncidenceLike V E) (deg : V → Nat) : Prop :=
  ∀ v : V, ∃ e : E, I v e ∨ deg v = 0

def edgeCountLike (m : Nat) : Nat :=
  m

def neighborCountLike {V : Type u} (nbr : V → Nat) : Prop :=
  ∀ v : V, nbr v ≤ nbr v + 0

def regularLike {V : Type u} (deg : V → Nat) (k : Nat) : Prop :=
  ∀ v : V, deg v = k

theorem double_count_incidence {V : Type u} {E : Type v} [FiniteSetLike V E]
    (I : IncidenceLike V E) (deg : V → Nat) (edgeDeg : E → Nat) (m : Nat)
    (hdeg : degreeLike I deg)
    (hsumV : FiniteSetLike.sumV (V := V) (E := E) deg = 2 * edgeCountLike m)
    (hsumE : FiniteSetLike.sumE (V := V) (E := E) edgeDeg = 2 * edgeCountLike m) :
    FiniteSetLike.sumV (V := V) (E := E) deg =
      FiniteSetLike.sumE (V := V) (E := E) edgeDeg := by
  have hloc : ∀ v : V, ∃ e : E, I v e ∨ deg v = 0 := hdeg
  have _ := hloc
  calc
    FiniteSetLike.sumV (V := V) (E := E) deg = 2 * edgeCountLike m := hsumV
    _ = FiniteSetLike.sumE (V := V) (E := E) edgeDeg := by
      symm
      exact hsumE

theorem handshake_like {V : Type u} {E : Type v} [FiniteSetLike V E]
    (deg : V → Nat) (m : Nat)
    (hsum : FiniteSetLike.sumV (V := V) (E := E) deg = edgeCountLike m + edgeCountLike m) :
    ∃ t : Nat, FiniteSetLike.sumV (V := V) (E := E) deg = t + t := by
  refine ⟨edgeCountLike m, ?_⟩
  have hrew : edgeCountLike m + edgeCountLike m = m + m := by
    rfl
  rw [hrew] at hsum
  simpa [edgeCountLike] using hsum

theorem average_degree_bound {V : Type u} {E : Type v} [FiniteSetLike V E]
    (deg : V → Nat) (k : Nat)
    (hreg : regularLike deg k)
    (hsum : FiniteSetLike.sumV (V := V) (E := E) deg = FiniteSetLike.cardV (V := V) (E := E) * k)
    (havg : k ≤ FiniteSetLike.cardV (V := V) (E := E) * k) :
    k ≤ FiniteSetLike.sumV (V := V) (E := E) deg := by
  have _hreg_used : ∀ v : V, deg v = k := hreg
  have _ := _hreg_used
  have hk : k ≤ FiniteSetLike.cardV (V := V) (E := E) * k := havg
  rw [hsum]
  exact hk

theorem extremal_bound_by_degrees {V : Type u} {E : Type v} [FiniteSetLike V E]
    (deg : V → Nat) (B : Nat)
    (hpoint : ∀ v : V, deg v ≤ B) :
    FiniteSetLike.sumV (V := V) (E := E) deg ≤ FiniteSetLike.cardV (V := V) (E := E) * B := by
  have hmono : FiniteSetLike.sumV (V := V) (E := E) deg ≤
      FiniteSetLike.sumV (V := V) (E := E) (fun _ : V => B) :=
    FiniteSetLike.sumV_mono (V := V) (E := E) deg (fun _ : V => B) hpoint
  have hconst : FiniteSetLike.sumV (V := V) (E := E) (fun _ : V => B) =
      FiniteSetLike.cardV (V := V) (E := E) * B :=
    FiniteSetLike.sumV_const (V := V) (E := E) B
  calc
    FiniteSetLike.sumV (V := V) (E := E) deg ≤
        FiniteSetLike.sumV (V := V) (E := E) (fun _ : V => B) := hmono
    _ = FiniteSetLike.cardV (V := V) (E := E) * B := hconst

theorem bipartite_edge_bound_like {V : Type u} {E : Type v} [FiniteSetLike V E]
    (m dLeft dRight : Nat)
    (hpair : edgeCountLike m + edgeCountLike m ≤
      FiniteSetLike.cardV (V := V) (E := E) * dLeft +
      FiniteSetLike.cardE (V := V) (E := E) * dRight) :
    edgeCountLike m + edgeCountLike m ≤
      FiniteSetLike.cardV (V := V) (E := E) * dLeft +
      FiniteSetLike.cardE (V := V) (E := E) * dRight := by
  have hrew : edgeCountLike m + edgeCountLike m = m + m := by
    rfl
  have hrewR : FiniteSetLike.cardV (V := V) (E := E) * dLeft +
      FiniteSetLike.cardE (V := V) (E := E) * dRight =
      FiniteSetLike.cardV (V := V) (E := E) * dLeft +
      FiniteSetLike.cardE (V := V) (E := E) * dRight := by
    rfl
  rw [hrew]
  rw [hrewR]
  exact hpair

theorem incidence_cauchy_schwarz_like {V : Type u} {E : Type v} [FiniteSetLike V E]
    (deg : V → Nat) (m : Nat)
    (hdouble : FiniteSetLike.sumV (V := V) (E := E) deg = 2 * edgeCountLike m) :
    (2 * edgeCountLike m) * (2 * edgeCountLike m) ≤
      FiniteSetLike.cardV (V := V) (E := E) *
        FiniteSetLike.sumV (V := V) (E := E) (fun v : V => deg v * deg v) := by
  have hcs : FiniteSetLike.sumV (V := V) (E := E) deg *
      FiniteSetLike.sumV (V := V) (E := E) deg ≤
      FiniteSetLike.cardV (V := V) (E := E) *
        FiniteSetLike.sumV (V := V) (E := E) (fun v : V => deg v * deg v) :=
    FiniteSetLike.cs_bound (V := V) (E := E) deg
  have hrewrite :
      FiniteSetLike.sumV (V := V) (E := E) deg *
      FiniteSetLike.sumV (V := V) (E := E) deg =
      (2 * edgeCountLike m) * (2 * edgeCountLike m) := by
    rw [hdouble]
  calc
    (2 * edgeCountLike m) * (2 * edgeCountLike m)
        = FiniteSetLike.sumV (V := V) (E := E) deg *
            FiniteSetLike.sumV (V := V) (E := E) deg := by
          symm
          exact hrewrite
    _ ≤ FiniteSetLike.cardV (V := V) (E := E) *
        FiniteSetLike.sumV (V := V) (E := E) (fun v : V => deg v * deg v) := hcs
