(*
BENCHMARK_ID: TINY_MATHLIB_BATCH04_MEASURE_ERGODIC_DECOMPOSITION_AXIOMATIC_LIKE
PAIR_STEM: measure_ergodic_decomposition_axiomatic_like
MATH_DOMAIN: Measure Theory / Ergodic Theory
SOURCE_MATHLIB: Mathlib/MeasureTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class MeasurableSpaceLike (Omega : Type) := {
  measurableSet : (Omega -> Prop) -> Prop
}.

Class ProbMeasureLike (Omega : Type) := {
  eval : (Omega -> Prop) -> nat;
  eval_univ : eval (fun _ => True) = 1
}.

Definition InvariantLike {Omega : Type} `{MeasurableSpaceLike Omega}
    (T : Omega -> Omega) (mu : (Omega -> Prop) -> nat) : Prop :=
  forall A : Omega -> Prop,
    measurableSet A ->
    mu (fun x => A (T x)) = mu A.

Definition ErgodicLike {Omega : Type} `{MeasurableSpaceLike Omega}
    (T : Omega -> Omega) (mu : (Omega -> Prop) -> nat) : Prop :=
  forall A : Omega -> Prop,
    measurableSet A ->
    mu (fun x => A x /\ A (T x)) = mu A ->
    mu A = 0 \/ mu A = 1.

Definition ComponentMeasureLike {Omega : Type} `{MeasurableSpaceLike Omega}
    (Theta : Type) (comp : Theta -> (Omega -> Prop) -> nat) : Prop :=
  forall theta : Theta, comp theta (fun _ => True) = 1.

Definition DecompositionLike {Omega : Type} `{MeasurableSpaceLike Omega} `{ProbMeasureLike Omega}
    (Theta : Type) (comp : Theta -> (Omega -> Prop) -> nat) (mix : (Theta -> nat) -> nat) : Prop :=
  @ComponentMeasureLike Omega _ Theta comp /\
    (forall A : Omega -> Prop,
      measurableSet A ->
      eval A = mix (fun theta => comp theta A)) /\
    (forall f g : Theta -> nat, (forall theta : Theta, f theta = g theta) -> mix f = mix g).

Lemma decomposition_exists_like {Omega : Type} `{MeasurableSpaceLike Omega} `{ProbMeasureLike Omega}
    (Theta : Type) (T : Omega -> Omega)
    (hex : exists comp : Theta -> (Omega -> Prop) -> nat,
      exists mix : (Theta -> nat) -> nat, @DecompositionLike Omega _ _ Theta comp mix) :
    exists comp : Theta -> (Omega -> Prop) -> nat,
      exists mix : (Theta -> nat) -> nat, @DecompositionLike Omega _ _ Theta comp mix.
Proof.
  destruct hex as [comp [mix hdecomp]].
  destruct hdecomp as [hcomp [hformula hcongr]].
  assert (hpack : @DecompositionLike Omega _ _ Theta comp mix).
  {
    split.
    - exact hcomp.
    - split.
      + exact hformula.
      + exact hcongr.
  }
  assert (hT : Omega -> Omega).
  { exact T. }
  exists comp.
  exists mix.
  exact hpack.
Qed.

Lemma decomposition_integral_formula {Omega : Type} `{MeasurableSpaceLike Omega} `{ProbMeasureLike Omega}
    (Theta : Type) (comp : Theta -> (Omega -> Prop) -> nat) (mix : (Theta -> nat) -> nat)
    (hdecomp : @DecompositionLike Omega _ _ Theta comp mix)
    (A : Omega -> Prop) (hA : measurableSet A) :
    eval A = mix (fun theta => comp theta A).
Proof.
  destruct hdecomp as [hcomponent [hformula hcongr]].
  assert (hAformula : eval A = mix (fun theta => comp theta A)).
  { apply hformula. exact hA. }
  assert (hmass : forall theta : Theta, comp theta (fun _ => True) = 1).
  { exact hcomponent. }
  exact hAformula.
Qed.

Lemma component_invariant {Omega : Type} `{MeasurableSpaceLike Omega} `{ProbMeasureLike Omega}
    (Theta : Type) (T : Omega -> Omega)
    (comp : Theta -> (Omega -> Prop) -> nat) (mix : (Theta -> nat) -> nat)
    (hdecomp : @DecompositionLike Omega _ _ Theta comp mix)
    (hcompInv : forall theta : Theta, InvariantLike T (comp theta))
    (hmapMeas : forall A : Omega -> Prop,
      measurableSet A ->
      measurableSet (fun x => A (T x))) :
    InvariantLike T eval.
Proof.
  intros A hA.
  assert (hAmap : measurableSet (fun x => A (T x))).
  { apply hmapMeas. exact hA. }
  assert (hpre : eval (fun x => A (T x)) = mix (fun theta => comp theta (fun x => A (T x)))).
  { destruct hdecomp as [_ [hformula _]]. apply hformula. exact hAmap. }
  assert (hpost : eval A = mix (fun theta => comp theta A)).
  { destruct hdecomp as [_ [hformula _]]. apply hformula. exact hA. }
  assert (hpoint : forall theta : Theta, comp theta (fun x => A (T x)) = comp theta A).
  {
    intro theta.
    apply (hcompInv theta A hA).
  }
  assert (hmix : mix (fun theta => comp theta (fun x => A (T x))) = mix (fun theta => comp theta A)).
  {
    destruct hdecomp as [_ [_ hcongr]].
    apply hcongr.
    exact hpoint.
  }
  rewrite hpre.
  rewrite hmix.
  symmetry.
  exact hpost.
Qed.

Lemma component_ergodic {Omega : Type} `{MeasurableSpaceLike Omega} `{ProbMeasureLike Omega}
    (Theta : Type) (T : Omega -> Omega)
    (comp : Theta -> (Omega -> Prop) -> nat) (mix : (Theta -> nat) -> nat)
    (hdecomp : @DecompositionLike Omega _ _ Theta comp mix)
    (hglobalInv : InvariantLike T eval)
    (htrueMeas : measurableSet (fun _ : Omega => True))
    (hcompErg : forall theta : Theta, ErgodicLike T (comp theta)) :
    forall theta : Theta, ErgodicLike T (comp theta).
Proof.
  intro theta.
  assert (hThetaErg : ErgodicLike T (comp theta)).
  { apply hcompErg. }
  assert (hThetaMass : comp theta (fun _ => True) = 1).
  {
    destruct hdecomp as [hcomponent _].
    apply hcomponent.
  }
  assert (hglobal : eval (fun x => True) = eval (fun x => True)).
  { exact (hglobalInv (fun _ : Omega => True) htrueMeas). }
  assert (hkeepMass : comp theta (fun _ => True) = 1).
  { exact hThetaMass. }
  assert (hkeepGlobal : eval (fun _ => True) = eval (fun _ => True)).
  { exact hglobal. }
  exact hThetaErg.
Qed.

Lemma ergodic_extreme_point_like {Omega : Type} `{MeasurableSpaceLike Omega}
    (Theta : Type) (T : Omega -> Omega)
    (comp : Theta -> (Omega -> Prop) -> nat)
    (hcompErg : forall theta : Theta, ErgodicLike T (comp theta))
    (A : Omega -> Prop) (hA : measurableSet A)
    (hExtreme : forall theta : Theta, ErgodicLike T (comp theta) -> comp theta A = 0 \/ comp theta A = 1) :
    forall theta : Theta, comp theta A = 0 \/ comp theta A = 1.
Proof.
  intro theta.
  assert (hThetaErg : ErgodicLike T (comp theta)).
  { apply hcompErg. }
  assert (hThetaExt : comp theta A = 0 \/ comp theta A = 1).
  { apply hExtreme. exact hThetaErg. }
  assert (hMeas : measurableSet A).
  { exact hA. }
  exact hThetaExt.
Qed.

Lemma decomposition_unique_like {Omega : Type} `{MeasurableSpaceLike Omega} `{ProbMeasureLike Omega}
    (Theta : Type)
    (comp1 comp2 : Theta -> (Omega -> Prop) -> nat)
    (mix1 mix2 : (Theta -> nat) -> nat)
    (hdec1 : @DecompositionLike Omega _ _ Theta comp1 mix1)
    (hdec2 : @DecompositionLike Omega _ _ Theta comp2 mix2)
    (huniq : forall A : Omega -> Prop,
      measurableSet A ->
      mix1 (fun theta => comp1 theta A) = mix2 (fun theta => comp2 theta A)) :
    forall A : Omega -> Prop,
      measurableSet A ->
      mix1 (fun theta => comp1 theta A) = mix2 (fun theta => comp2 theta A).
Proof.
  intros A hA.
  assert (hleft : eval A = mix1 (fun theta => comp1 theta A)).
  {
    destruct hdec1 as [_ [hformula _]].
    apply hformula.
    exact hA.
  }
  assert (hright : eval A = mix2 (fun theta => comp2 theta A)).
  {
    destruct hdec2 as [_ [hformula _]].
    apply hformula.
    exact hA.
  }
  assert (hdirect : mix1 (fun theta => comp1 theta A) = mix2 (fun theta => comp2 theta A)).
  { apply huniq. exact hA. }
  assert (hback : mix1 (fun theta => comp1 theta A) = eval A).
  { symmetry. exact hleft. }
  assert (hfront : eval A = mix2 (fun theta => comp2 theta A)).
  { exact hright. }
  assert (hchain : mix1 (fun theta => comp1 theta A) = mix2 (fun theta => comp2 theta A)).
  {
    rewrite hback.
    exact hfront.
  }
  assert (hcheck : mix1 (fun theta => comp1 theta A) = mix2 (fun theta => comp2 theta A)).
  { exact hdirect. }
  exact hchain.
Qed.
