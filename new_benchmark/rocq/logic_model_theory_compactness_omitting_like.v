(*
BENCHMARK_ID: TINY_MATHLIB_BATCH04_LOGIC_MODEL_THEORY_COMPACTNESS_OMITTING_LIKE
PAIR_STEM: logic_model_theory_compactness_omitting_like
MATH_DOMAIN: Model Theory
SOURCE_MATHLIB: Mathlib/ModelTheory
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class LanguageLike (L : Type) := {
  Formula : Type;
  Model : Type;
  Sat : Model -> Formula -> Prop
}.

Definition TheoryLike {L : Type} `{LanguageLike L} : Type :=
  Formula -> Prop.

Definition TypeLike {L : Type} `{LanguageLike L} : Type :=
  Formula -> Prop.

Definition SatisfiableLike {L : Type} `{LanguageLike L} (T : TheoryLike) : Prop :=
  exists M : Model,
    forall phi : Formula, T phi -> Sat M phi.

Definition OmitsLike {L : Type} `{LanguageLike L}
    (M : Model) (p : TypeLike) : Prop :=
  forall phi : Formula,
    p phi -> ~ Sat M phi.

Definition ElementaryChainLike {L : Type} `{LanguageLike L}
    (Ms : nat -> Model) (T : TheoryLike) : Prop :=
  forall n : nat,
    forall phi : Formula,
      T phi -> Sat (Ms n) phi -> Sat (Ms (n + 1)) phi.

Lemma finite_satisfiable_compact {L : Type} `{LanguageLike L}
    (T : TheoryLike)
    (S : nat -> Formula -> Prop)
    (hcompact :
      (forall n : nat,
        exists M : Model,
          forall phi : Formula, T phi -> S n phi -> Sat M phi) ->
      SatisfiableLike T)
    (hfin :
      forall n : nat,
        exists M : Model,
          forall phi : Formula, T phi -> S n phi -> Sat M phi) :
    SatisfiableLike T.
Proof.
  assert (hseed :
      forall n : nat,
        exists M : Model,
          forall phi : Formula, T phi -> S n phi -> Sat M phi).
  { exact hfin. }
  assert (hsat : SatisfiableLike T).
  { apply hcompact. exact hseed. }
  exact hsat.
Qed.

Lemma henkin_extension_step {L : Type} `{LanguageLike L}
    (T T' : TheoryLike)
    (hincl : forall phi : Formula, T phi -> T' phi)
    (hsat : SatisfiableLike T)
    (hhenkin :
      forall M : Model,
        (forall phi : Formula, T phi -> Sat M phi) ->
        forall psi : Formula, T' psi -> Sat M psi) :
    SatisfiableLike T'.
Proof.
  destruct hsat as [M hM].
  assert (hbase : forall phi : Formula, T phi -> Sat M phi).
  { exact hM. }
  assert (hext : forall psi : Formula, T' psi -> Sat M psi).
  { apply (hhenkin M hbase). }
  assert (hinc : forall phi : Formula, T phi -> T' phi).
  { exact hincl. }
  exists M.
  exact hext.
Qed.

Lemma chain_model_union_like {L : Type} `{LanguageLike L}
    (Ms : nat -> Model)
    (T : TheoryLike)
    (hchain : ElementaryChainLike Ms T)
    (hroot : forall phi : Formula, T phi -> Sat (Ms 0) phi)
    (hunion :
      forall phi : Formula,
        T phi -> (exists n : nat, Sat (Ms n) phi) -> Sat (Ms 0) phi) :
    SatisfiableLike T.
Proof.
  assert (hwitness : forall phi : Formula, T phi -> exists n : nat, Sat (Ms n) phi).
  {
    intros phi hphi.
    exists 0.
    apply hroot.
    exact hphi.
  }
  assert (hmodel : forall phi : Formula, T phi -> Sat (Ms 0) phi).
  {
    intros phi hphi.
    apply (hunion phi hphi).
    apply hwitness.
    exact hphi.
  }
  assert (hchain_keep : ElementaryChainLike Ms T).
  { exact hchain. }
  exists (Ms 0).
  exact hmodel.
Qed.

Lemma omitting_types_step {L : Type} `{LanguageLike L}
    (T : TheoryLike)
    (p : TypeLike)
    (hsat : SatisfiableLike T)
    (homit :
      forall M : Model,
        (forall phi : Formula, T phi -> Sat M phi) ->
        forall phi : Formula, p phi -> ~ Sat M phi) :
    exists M : Model,
      (forall phi : Formula, T phi -> Sat M phi) /\
      OmitsLike M p.
Proof.
  destruct hsat as [M hM].
  assert (hT : forall phi : Formula, T phi -> Sat M phi).
  { exact hM. }
  assert (hO : OmitsLike M p).
  { exact (homit M hT). }
  exists M.
  split.
  - exact hT.
  - exact hO.
Qed.

Lemma complete_theory_model_exists {L : Type} `{LanguageLike L}
    (T T' : TheoryLike)
    (hincl : forall phi : Formula, T phi -> T' phi)
    (hsat' : SatisfiableLike T') :
    SatisfiableLike T.
Proof.
  destruct hsat' as [M hM'].
  assert (hM : forall phi : Formula, T phi -> Sat M phi).
  {
    intros phi hphi.
    assert (hphi' : T' phi).
    { apply hincl. exact hphi. }
    apply hM'.
    exact hphi'.
  }
  exists M.
  exact hM.
Qed.

Lemma compactness_omitting_types_like {L : Type} `{LanguageLike L}
    (T T' : TheoryLike)
    (p : TypeLike)
    (S : nat -> Formula -> Prop)
    (hcompact :
      (forall n : nat,
        exists M : Model,
          forall phi : Formula, T' phi -> S n phi -> Sat M phi) ->
      SatisfiableLike T')
    (hfin :
      forall n : nat,
        exists M : Model,
          forall phi : Formula, T' phi -> S n phi -> Sat M phi)
    (hincl : forall phi : Formula, T phi -> T' phi)
    (homit :
      forall M : Model,
        (forall phi : Formula, T' phi -> Sat M phi) ->
        forall phi : Formula, p phi -> ~ Sat M phi) :
    exists M : Model,
      (forall phi : Formula, T phi -> Sat M phi) /\
      OmitsLike M p.
Proof.
  assert (hsat' : SatisfiableLike T').
  { apply (@finite_satisfiable_compact L _ T' S hcompact hfin). }
  assert (hsatT : SatisfiableLike T).
  { apply (@complete_theory_model_exists L _ T T' hincl hsat'). }
  assert (homitModel :
      exists M : Model,
        (forall phi : Formula, T' phi -> Sat M phi) /\
        OmitsLike M p).
  { apply (@omitting_types_step L _ T' p hsat' homit). }
  destruct homitModel as [M [hM' hOmits]].
  assert (hM : forall phi : Formula, T phi -> Sat M phi).
  {
    intros phi hphi.
    apply hM'.
    apply hincl.
    exact hphi.
  }
  assert (hkeep : SatisfiableLike T).
  { exact hsatT. }
  exists M.
  split.
  - exact hM.
  - exact hOmits.
Qed.
