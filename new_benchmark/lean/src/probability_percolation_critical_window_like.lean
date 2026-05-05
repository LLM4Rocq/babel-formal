/-
BENCHMARK_ID: TINY_MATHLIB_BATCH07_PROBABILITY_PERCOLATION_CRITICAL_WINDOW_LIKE
PAIR_STEM: probability_percolation_critical_window_like
MATH_DOMAIN: Probability
SOURCE_MATHLIB: Mathlib/Probability/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
-/

class FrameworkStruct_probability_percolation_critical_window where
  open_event : Nat -> Prop
  close_event : Nat -> Prop
  window : Nat -> Nat
  complement : forall n : Nat, open_event n -> close_event (window n) -> False
  window_idem : forall n : Nat, window (window n) = window n
  seed_open : open_event 0
  propagate_open : forall n : Nat, open_event n -> open_event (window n)

structure ContextData_probability_percolation_critical_window
    [FrameworkStruct_probability_percolation_critical_window] where
  t : Nat
  u : Nat
  htu : FrameworkStruct_probability_percolation_critical_window.window t = u
  ht : FrameworkStruct_probability_percolation_critical_window.open_event t

def primary_map_probability_percolation_critical_window
    [FrameworkStruct_probability_percolation_critical_window]
    (ctx : ContextData_probability_percolation_critical_window) : Prop :=
  FrameworkStruct_probability_percolation_critical_window.open_event ctx.t

def secondary_map_probability_percolation_critical_window
    [FrameworkStruct_probability_percolation_critical_window]
    (ctx : ContextData_probability_percolation_critical_window) : Prop :=
  FrameworkStruct_probability_percolation_critical_window.open_event ctx.u

def tertiary_map_probability_percolation_critical_window
    [FrameworkStruct_probability_percolation_critical_window]
    (ctx : ContextData_probability_percolation_critical_window) : Prop :=
  FrameworkStruct_probability_percolation_critical_window.open_event
    (FrameworkStruct_probability_percolation_critical_window.window ctx.u)

theorem stability_step_probability_percolation_critical_window
    [FrameworkStruct_probability_percolation_critical_window]
    (ctx : ContextData_probability_percolation_critical_window)
    (hneg : primary_map_probability_percolation_critical_window ctx -> False) :
    False := by
  have hprim : primary_map_probability_percolation_critical_window ctx := by
    unfold primary_map_probability_percolation_critical_window
    exact ctx.ht
  exact hneg hprim

theorem factorization_step_probability_percolation_critical_window
    [FrameworkStruct_probability_percolation_critical_window]
    (ctx : ContextData_probability_percolation_critical_window)
    (hprim : primary_map_probability_percolation_critical_window ctx) :
    secondary_map_probability_percolation_critical_window ctx := by
  unfold secondary_map_probability_percolation_critical_window
  have hwin :
      FrameworkStruct_probability_percolation_critical_window.open_event
        (FrameworkStruct_probability_percolation_critical_window.window ctx.t) :=
    FrameworkStruct_probability_percolation_critical_window.propagate_open _ hprim
  rw [ctx.htu] at hwin
  exact hwin

theorem comparison_step_probability_percolation_critical_window
    [FrameworkStruct_probability_percolation_critical_window]
    (ctx : ContextData_probability_percolation_critical_window) :
    primary_map_probability_percolation_critical_window ctx ->
      secondary_map_probability_percolation_critical_window ctx /\
      exists k : Nat,
        FrameworkStruct_probability_percolation_critical_window.open_event k := by
  intro hprim
  have hsec :
      secondary_map_probability_percolation_critical_window ctx :=
    factorization_step_probability_percolation_critical_window ctx hprim
  refine And.intro hsec ?_
  refine Exists.intro ctx.u ?_
  unfold secondary_map_probability_percolation_critical_window at hsec
  exact hsec

theorem transport_step_probability_percolation_critical_window
    [FrameworkStruct_probability_percolation_critical_window]
    (ctx : ContextData_probability_percolation_critical_window)
    (hclose : FrameworkStruct_probability_percolation_critical_window.close_event
      (FrameworkStruct_probability_percolation_critical_window.window ctx.u))
    (hsec : secondary_map_probability_percolation_critical_window ctx) :
    False := by
  unfold secondary_map_probability_percolation_critical_window at hsec
  exact FrameworkStruct_probability_percolation_critical_window.complement ctx.u hsec hclose

theorem coherence_step_probability_percolation_critical_window
    [FrameworkStruct_probability_percolation_critical_window]
    (ctx : ContextData_probability_percolation_critical_window)
    (hprim : primary_map_probability_percolation_critical_window ctx) :
    tertiary_map_probability_percolation_critical_window ctx := by
  have hsec :
      secondary_map_probability_percolation_critical_window ctx :=
    factorization_step_probability_percolation_critical_window ctx hprim
  unfold tertiary_map_probability_percolation_critical_window
  unfold secondary_map_probability_percolation_critical_window at hsec
  exact FrameworkStruct_probability_percolation_critical_window.propagate_open _ hsec

theorem iteration_step_probability_percolation_critical_window
    [FrameworkStruct_probability_percolation_critical_window]
    (ctx : ContextData_probability_percolation_critical_window) :
    exists k : Nat,
      FrameworkStruct_probability_percolation_critical_window.window k = k /\
      FrameworkStruct_probability_percolation_critical_window.open_event k := by
  refine Exists.intro
    (FrameworkStruct_probability_percolation_critical_window.window ctx.u) ?_
  refine And.intro ?hfixed ?hopen
  · exact FrameworkStruct_probability_percolation_critical_window.window_idem _
  · have hsec :
        secondary_map_probability_percolation_critical_window ctx :=
      factorization_step_probability_percolation_critical_window ctx ctx.ht
    unfold secondary_map_probability_percolation_critical_window at hsec
    exact FrameworkStruct_probability_percolation_critical_window.propagate_open _ hsec

theorem main_result_probability_percolation_critical_window
    [FrameworkStruct_probability_percolation_critical_window]
    (ctx : ContextData_probability_percolation_critical_window) :
    primary_map_probability_percolation_critical_window ctx ->
    exists k : Nat,
      secondary_map_probability_percolation_critical_window ctx /\
      (FrameworkStruct_probability_percolation_critical_window.open_event k /\
        tertiary_map_probability_percolation_critical_window ctx) := by
  intro hprim
  have hsec :
      secondary_map_probability_percolation_critical_window ctx :=
    factorization_step_probability_percolation_critical_window ctx hprim
  have hter :
      tertiary_map_probability_percolation_critical_window ctx :=
    coherence_step_probability_percolation_critical_window ctx hprim
  refine Exists.intro
    (FrameworkStruct_probability_percolation_critical_window.window ctx.u) ?_
  refine And.intro hsec ?_
  refine And.intro ?hopen hter
  unfold secondary_map_probability_percolation_critical_window at hsec
  exact FrameworkStruct_probability_percolation_critical_window.propagate_open _ hsec
