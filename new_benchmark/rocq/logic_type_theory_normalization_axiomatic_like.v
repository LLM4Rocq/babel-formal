(*
BENCHMARK_ID: TINY_MATHLIB_BATCH04_LOGIC_TYPE_THEORY_NORMALIZATION_AXIOMATIC_LIKE
PAIR_STEM: logic_type_theory_normalization_axiomatic_like
MATH_DOMAIN: Logic / Type Theory
SOURCE_MATHLIB: foundational formal systems patterns
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class ContextLike (Ctx : Type) := {
  empty : Ctx;
  extend : Ctx -> nat -> Ctx;
  depth : Ctx -> nat;
  depth_empty : depth empty = 0;
  depth_extend : forall Gamma : Ctx, forall A : nat, depth (extend Gamma A) = depth Gamma + 1
}.

Definition TermLike {Ctx : Type} `{ContextLike Ctx} (Tm : Type) : Prop :=
  exists shift : Tm -> Tm, exists subst : Tm -> Tm -> Tm, True.

Definition TypingLike {Ctx : Type} `{ContextLike Ctx} {Tm : Type}
    (ty : Ctx -> Tm -> nat -> Prop) : Prop :=
  forall Gamma : Ctx, forall t : Tm, forall A B : nat, A = B -> ty Gamma t A -> ty Gamma t B.

Definition ReductionLike {Tm : Type} (step : Tm -> Tm -> Prop) : Prop :=
  forall t u v : Tm, step t u -> step u v -> step t v.

Definition NeutralLike {Tm : Type} (neutral : Tm -> Prop) : Prop :=
  forall t u : Tm, neutral t -> ~ neutral u -> t <> u.

Definition NormalLike {Tm : Type} (normal : Tm -> Prop) (step : Tm -> Tm -> Prop) : Prop :=
  forall t u : Tm, normal t -> step t u -> False.

Lemma subject_reduction_like {Ctx : Type} `{ContextLike Ctx} {Tm : Type}
    (ty : Ctx -> Tm -> nat -> Prop)
    (step : Tm -> Tm -> Prop)
    (hTyping : TypingLike ty)
    (hsubj : forall Gamma : Ctx, forall t u : Tm, forall A : nat, step t u -> ty Gamma t A -> ty Gamma u A) :
    forall Gamma : Ctx, forall t u : Tm, forall A : nat, step t u -> ty Gamma t A -> ty Gamma u A.
Proof.
  intros Gamma t u A hstep hty.
  assert (hkeep : ty Gamma t A).
  { exact hty. }
  assert (hcast : ty Gamma t A).
  { apply (hTyping Gamma t A A eq_refl). exact hkeep. }
  assert (hred : ty Gamma u A).
  { apply (hsubj Gamma t u A hstep hcast). }
  exact hred.
Qed.

Lemma progress_like {Ctx : Type} `{ContextLike Ctx} {Tm : Type}
    (step : Tm -> Tm -> Prop)
    (normal : Tm -> Prop)
    (hterm : TermLike Tm)
    (hprogress : forall t : Tm, (exists u : Tm, step t u) \/ normal t) :
    forall t : Tm, (exists u : Tm, step t u) \/ normal t.
Proof.
  destruct hterm as [shift [subst htriv]].
  intro t.
  assert (hbase : (exists u : Tm, step t u) \/ normal t).
  { apply hprogress. }
  assert (hshift : Tm -> Tm).
  { exact shift. }
  assert (hsubst : Tm -> Tm -> Tm).
  { exact subst. }
  assert (htrue : True).
  { exact htriv. }
  exact hbase.
Qed.

Lemma substitution_preserves_typing {Ctx : Type} `{ContextLike Ctx} {Tm : Type}
    (ty : Ctx -> Tm -> nat -> Prop)
    (hTyping : TypingLike ty)
    (hsubst :
      forall Gamma : Ctx, forall t s : Tm, forall A B : nat,
        ty (extend Gamma A) t B -> ty Gamma s A -> ty Gamma t B) :
    forall Gamma : Ctx, forall t s : Tm, forall A B : nat,
      ty (extend Gamma A) t B -> ty Gamma s A -> ty Gamma t B.
Proof.
  intros Gamma t s A B hty hs.
  assert (htyped : ty (extend Gamma A) t B).
  { exact hty. }
  assert (hseed : ty Gamma s A).
  { exact hs. }
  assert (hcast : ty (extend Gamma A) t B).
  { apply (hTyping (extend Gamma A) t B B eq_refl). exact htyped. }
  assert (hresult : ty Gamma t B).
  { apply (hsubst Gamma t s A B hcast hseed). }
  exact hresult.
Qed.

Lemma reducibility_closure_step {Ctx : Type} `{ContextLike Ctx} {Tm : Type}
    (step : Tm -> Tm -> Prop)
    (R : Tm -> Prop)
    (hone : forall t u : Tm, step t u -> R t -> R u)
    (htrans : ReductionLike step) :
    forall t u v : Tm, step t u -> step u v -> R t -> R v.
Proof.
  intros t u v htu huv hRt.
  assert (hRu : R u).
  { apply (hone t u htu hRt). }
  assert (htv : step t v).
  { apply (htrans t u v htu huv). }
  assert (hRv : R v).
  { apply (hone u v huv hRu). }
  assert (hkeep : step t v).
  { exact htv. }
  exact hRv.
Qed.

Lemma strong_normalization_like {Ctx : Type} `{ContextLike Ctx} {Tm : Type}
    (step : Tm -> Tm -> Prop)
    (SN normal : Tm -> Prop)
    (hbase : forall t : Tm, normal t -> SN t)
    (hstep : forall t u : Tm, step t u -> SN u -> SN t)
    (hseed : forall t u : Tm, step t u -> SN u)
    (hprogress : forall t : Tm, normal t \/ exists u : Tm, step t u) :
    forall t : Tm, SN t.
Proof.
  intro t.
  assert (hcase : normal t \/ exists u : Tm, step t u).
  { apply hprogress. }
  destruct hcase as [hnorm | hred].
  - apply hbase.
    exact hnorm.
  - destruct hred as [u hu].
    assert (hSu : SN u).
    { apply (hseed t u hu). }
    assert (hSt : SN t).
    { apply (hstep t u hu hSu). }
    exact hSt.
Qed.

Lemma normalization_by_evaluation_interface {Ctx : Type} `{ContextLike Ctx} {Tm : Type}
    (ty : Ctx -> Tm -> nat -> Prop)
    (step : Tm -> Tm -> Prop)
    (normal SN nbe : Tm -> Prop)
    (hTyping : TypingLike ty)
    (hsubj : forall Gamma : Ctx, forall t u : Tm, forall A : nat, step t u -> ty Gamma t A -> ty Gamma u A)
    (hbase : forall t : Tm, normal t -> SN t)
    (hstep : forall t u : Tm, step t u -> SN u -> SN t)
    (hseed : forall t u : Tm, step t u -> SN u)
    (hprogress : forall t : Tm, normal t \/ exists u : Tm, step t u)
    (hnbe : forall t : Tm, SN t -> nbe t) :
    forall Gamma : Ctx, forall t : Tm, forall A : nat, ty Gamma t A -> nbe t.
Proof.
  assert (hsr : forall Gamma : Ctx, forall t u : Tm, forall A : nat, step t u -> ty Gamma t A -> ty Gamma u A).
  { apply (@subject_reduction_like Ctx _ Tm ty step hTyping hsubj). }
  assert (hsn : forall t : Tm, SN t).
  { apply (@strong_normalization_like Ctx _ Tm step SN normal hbase hstep hseed hprogress). }
  intros Gamma t A hty.
  assert (htyped : ty Gamma t A).
  { exact hty. }
  assert (hsnt : SN t).
  { apply hsn. }
  assert (hkeep : forall Gamma : Ctx, forall t u : Tm, forall A : nat, step t u -> ty Gamma t A -> ty Gamma u A).
  { exact hsr. }
  apply hnbe.
  exact hsnt.
Qed.
