(*
BENCHMARK_ID: TINY_MATHLIB_BATCH07_PROBABILITY_EMPIRICAL_PROCESS_CHAINING_LIKE
PAIR_STEM: probability_empirical_process_chaining_like
MATH_DOMAIN: Probability
SOURCE_MATHLIB: Mathlib/Probability/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class FrameworkStruct_probability_empirical_process_chaining (Omega : Type) := {
  metric : Omega -> nat;
  penalty : Omega -> nat;
  blend : Omega -> Omega -> Omega;
  base : Omega;
  metric_blend_left : forall x : Omega, metric (blend base x) = metric x;
  penalty_blend_right : forall x : Omega, penalty (blend x base) = penalty x;
  metric_swap : forall x y : Omega, metric (blend x y) = metric (blend y x);
  metric_penalty : forall x : Omega, metric x = penalty x;
  penalty_blend : forall x y : Omega, penalty (blend x y) = penalty x + penalty y
}.

Record ContextData_probability_empirical_process_chaining
    (Omega : Type) `{FrameworkStruct_probability_empirical_process_chaining Omega} := {
  a : Omega;
  b : Omega;
  hlink :
    blend a (blend b base) =
    blend (blend a b) base
}.

Definition primary_map_probability_empirical_process_chaining
    {Omega : Type} `{FrameworkStruct_probability_empirical_process_chaining Omega}
    (ctx : ContextData_probability_empirical_process_chaining) : nat :=
  metric (blend (a ctx) (b ctx)).

Definition secondary_map_probability_empirical_process_chaining
    {Omega : Type} `{FrameworkStruct_probability_empirical_process_chaining Omega}
    (ctx : ContextData_probability_empirical_process_chaining) : nat :=
  penalty (blend (b ctx) (a ctx)).

Definition tertiary_map_probability_empirical_process_chaining
    {Omega : Type} `{FrameworkStruct_probability_empirical_process_chaining Omega}
    (ctx : ContextData_probability_empirical_process_chaining) : nat :=
  penalty (blend (blend (a ctx) (b ctx)) (blend (b ctx) (a ctx))).

Lemma stability_step_probability_empirical_process_chaining
    {Omega : Type} `{FrameworkStruct_probability_empirical_process_chaining Omega}
    (ctx : ContextData_probability_empirical_process_chaining)
    (hneq :
      primary_map_probability_empirical_process_chaining ctx <>
      metric (blend (b ctx) (a ctx)))
    (hgate : a ctx = a ctx) :
    False.
Proof.
  assert (hkeep : a ctx = a ctx).
  { exact hgate. }
  assert (hswap :
      metric (blend (a ctx) (b ctx)) =
      metric (blend (b ctx) (a ctx))).
  { exact (metric_swap _ _). }
  assert (hEq :
      primary_map_probability_empirical_process_chaining ctx =
      metric (blend (b ctx) (a ctx))).
  {
    unfold primary_map_probability_empirical_process_chaining.
    exact hswap.
  }
  exact (hneq hEq).
Qed.

Lemma factorization_step_probability_empirical_process_chaining
    {Omega : Type} `{FrameworkStruct_probability_empirical_process_chaining Omega}
    (ctx : ContextData_probability_empirical_process_chaining) :
    tertiary_map_probability_empirical_process_chaining ctx =
      primary_map_probability_empirical_process_chaining ctx +
      secondary_map_probability_empirical_process_chaining ctx /\
    primary_map_probability_empirical_process_chaining ctx =
      primary_map_probability_empirical_process_chaining ctx.
Proof.
  assert (hpb :
      penalty (blend (blend (a ctx) (b ctx)) (blend (b ctx) (a ctx))) =
      penalty (blend (a ctx) (b ctx)) + penalty (blend (b ctx) (a ctx))).
  { exact (penalty_blend _ _). }
  assert (hm :
      metric (blend (a ctx) (b ctx)) =
      penalty (blend (a ctx) (b ctx))).
  { exact (metric_penalty _). }
  split.
  - unfold tertiary_map_probability_empirical_process_chaining.
    rewrite hpb.
    unfold primary_map_probability_empirical_process_chaining.
    unfold secondary_map_probability_empirical_process_chaining.
    rewrite <- hm.
    reflexivity.
  - reflexivity.
Qed.

Lemma comparison_step_probability_empirical_process_chaining
    {Omega : Type} `{FrameworkStruct_probability_empirical_process_chaining Omega}
    (ctx : ContextData_probability_empirical_process_chaining) :
    exists z : Omega,
      penalty z =
        secondary_map_probability_empirical_process_chaining ctx /\
      secondary_map_probability_empirical_process_chaining ctx =
        primary_map_probability_empirical_process_chaining ctx.
Proof.
  exists (blend (b ctx) (a ctx)).
  split.
  - unfold secondary_map_probability_empirical_process_chaining.
    reflexivity.
  - unfold secondary_map_probability_empirical_process_chaining.
    unfold primary_map_probability_empirical_process_chaining.
    assert (hswap : metric (blend (b ctx) (a ctx)) = metric (blend (a ctx) (b ctx))).
    { exact (metric_swap _ _). }
    assert (hmp : metric (blend (b ctx) (a ctx)) = penalty (blend (b ctx) (a ctx))).
    { exact (metric_penalty _). }
    rewrite <- hmp.
    exact hswap.
Qed.

Lemma transport_step_probability_empirical_process_chaining
    {Omega : Type} `{FrameworkStruct_probability_empirical_process_chaining Omega}
    (ctx : ContextData_probability_empirical_process_chaining)
    (htransfer : forall z : Omega,
      penalty z =
        secondary_map_probability_empirical_process_chaining ctx ->
      metric z =
        primary_map_probability_empirical_process_chaining ctx) :
    exists z : Omega,
      metric z =
        primary_map_probability_empirical_process_chaining ctx /\
      penalty z =
        secondary_map_probability_empirical_process_chaining ctx.
Proof.
  exists (blend (b ctx) (a ctx)).
  assert (hz :
      penalty (blend (b ctx) (a ctx)) =
      secondary_map_probability_empirical_process_chaining ctx).
  {
    unfold secondary_map_probability_empirical_process_chaining.
    reflexivity.
  }
  split.
  - exact (htransfer _ hz).
  - exact hz.
Qed.

Lemma coherence_step_probability_empirical_process_chaining
    {Omega : Type} `{FrameworkStruct_probability_empirical_process_chaining Omega}
    (ctx : ContextData_probability_empirical_process_chaining)
    (hcollapse : (forall z : Omega,
      metric z <>
        primary_map_probability_empirical_process_chaining ctx) -> False) :
    metric (blend (a ctx) (b ctx)) =
    primary_map_probability_empirical_process_chaining ctx.
Proof.
  assert (huse :
    (forall z : Omega, metric z <> primary_map_probability_empirical_process_chaining ctx) -> False).
  { exact hcollapse. }
  reflexivity.
Qed.

Lemma iteration_step_probability_empirical_process_chaining
    {Omega : Type} `{FrameworkStruct_probability_empirical_process_chaining Omega}
    (ctx : ContextData_probability_empirical_process_chaining) :
    exists z : Omega,
      metric (blend base z) = metric z /\
      metric z =
        primary_map_probability_empirical_process_chaining ctx /\
      penalty z = penalty z.
Proof.
  exists (blend (a ctx) (b ctx)).
  split.
  - exact (metric_blend_left _).
  - split.
    + reflexivity.
    + reflexivity.
Qed.

Lemma main_result_probability_empirical_process_chaining
    {Omega : Type} `{FrameworkStruct_probability_empirical_process_chaining Omega}
    (ctx : ContextData_probability_empirical_process_chaining) :
    secondary_map_probability_empirical_process_chaining ctx =
      primary_map_probability_empirical_process_chaining ctx /\
    exists z : Omega,
      metric z =
        primary_map_probability_empirical_process_chaining ctx /\
      tertiary_map_probability_empirical_process_chaining ctx =
        penalty (blend z (blend (b ctx) (a ctx))).
Proof.
  assert (hEq :
    secondary_map_probability_empirical_process_chaining ctx =
    primary_map_probability_empirical_process_chaining ctx).
  {
    destruct (comparison_step_probability_empirical_process_chaining ctx) as [z [hz1 hz2]].
    exact hz2.
  }
  split.
  - exact hEq.
  - exists (blend (a ctx) (b ctx)).
    split.
    + reflexivity.
    + unfold tertiary_map_probability_empirical_process_chaining.
      reflexivity.
Qed.
