(*
BENCHMARK_ID: TINY_MATHLIB_BATCH04_TOPOLOGY_SHEAF_DESCENT_CECH
PAIR_STEM: topology_sheaf_descent_cech_like
MATH_DOMAIN: Topology / Sheaf Theory
SOURCE_MATHLIB: Mathlib/Topology/Sheaves/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class SiteLike (X : Type) := {
  leq : X -> X -> Prop;
  leq_refl : forall U : X, leq U U;
  leq_trans : forall U V W : X, leq W V -> leq V U -> leq W U;
  Cover : X -> (X -> Prop) -> Prop;
  cover_mem : forall (U : X) (I : X -> Prop), Cover U I -> forall V : X, I V -> leq V U;
  cover_refl : forall U : X, Cover U (fun V : X => V = U);
  cover_refine :
    forall (U : X) (I J : X -> Prop),
      Cover U I ->
      (forall V : X, I V -> Cover V J) ->
      Cover U (fun W : X => exists V : X, I V /\ J W)
}.

Record PresheafLike (X : Type) `{SiteLike X} := {
  Sect : Type;
  res : X -> X -> Sect -> Sect;
  res_id : forall U : X, forall s : Sect, res U U s = s;
  res_comp :
    forall U V W : X,
      forall s : Sect,
      leq W V -> leq V U ->
      res W V (res V U s) = res W U s
}.

Arguments Sect {X _} _.
Arguments res {X _} _ _ _ _. 

Definition CompatibleFamilyLike {X : Type} `{SiteLike X}
    (F : PresheafLike)
    (I : X -> Prop)
    (sigma : X -> Sect F) : Prop :=
  forall V W : X,
    I V -> I W ->
    forall T : X,
      leq T V -> leq T W ->
      res F T V (sigma V) = res F T W (sigma W).

Definition MatchingObjectLike {X : Type} `{SiteLike X}
    (F : PresheafLike)
    (I : X -> Prop) : Prop :=
  exists sigma : X -> Sect F, CompatibleFamilyLike F I sigma.

Definition DescentDataLike {X : Type} `{SiteLike X}
    (F : PresheafLike)
    (U : X) (I : X -> Prop) : Prop :=
  forall sigma : X -> Sect F,
    CompatibleFamilyLike F I sigma ->
    exists s : Sect F, forall V : X, I V -> res F V U s = sigma V.

Definition CechComplexLike {X : Type} `{SiteLike X}
    (F : PresheafLike)
    (U : X) (I : X -> Prop) : Prop :=
  DescentDataLike F U I /\
  (forall s t : Sect F, (forall V : X, I V -> res F V U s = res F V U t) -> s = t).

Lemma sheaf_condition_matching {X : Type} `{SiteLike X}
    (F : PresheafLike)
    (U : X) (I : X -> Prop)
    (hdesc : DescentDataLike F U I) :
    forall sigma : X -> Sect F,
      CompatibleFamilyLike F I sigma ->
      exists s : Sect F, forall V : X, I V -> res F V U s = sigma V.
Proof.
  intros sigma hsigma.
  assert (hglue : exists s : Sect F, forall V : X, I V -> res F V U s = sigma V).
  {
    apply hdesc.
    exact hsigma.
  }
  destruct hglue as [s hs].
  exists s.
  exact hs.
Qed.

Lemma descent_gluing_exists {X : Type} `{SiteLike X}
    (F : PresheafLike)
    (U : X) (I : X -> Prop)
    (hdesc : DescentDataLike F U I)
    (hmatch : MatchingObjectLike F I) :
    exists s : Sect F,
      exists sigma : X -> Sect F,
        CompatibleFamilyLike F I sigma /\
        (forall V : X, I V -> res F V U s = sigma V).
Proof.
  destruct hmatch as [sigma hsigma].
  assert (hglue : exists s : Sect F, forall V : X, I V -> res F V U s = sigma V).
  {
    apply hdesc.
    exact hsigma.
  }
  destruct hglue as [s hs].
  exists s.
  exists sigma.
  split.
  - exact hsigma.
  - exact hs.
Qed.

Lemma descent_gluing_unique {X : Type} `{SiteLike X}
    (F : PresheafLike)
    (U : X) (I : X -> Prop)
    (hcech : CechComplexLike F U I)
    (s t : Sect F)
    (hEq : forall V : X, I V -> res F V U s = res F V U t) :
    s = t.
Proof.
  destruct hcech as [hdesc huniq].
  assert (hresult : s = t).
  {
    apply huniq.
    exact hEq.
  }
  exact hresult.
Qed.

Lemma cech_exactness_degree1 {X : Type} `{SiteLike X}
    (F : PresheafLike)
    (U : X) (I : X -> Prop)
    (hcech : CechComplexLike F U I) :
    forall sigma : X -> Sect F,
      CompatibleFamilyLike F I sigma ->
      exists s : Sect F,
        (forall V : X, I V -> res F V U s = sigma V) /\
        (forall t : Sect F,
          (forall V : X, I V -> res F V U t = sigma V) ->
          t = s).
Proof.
  intros sigma hsigma.
  destruct hcech as [hdesc huniq].
  assert (hglue : exists s : Sect F, forall V : X, I V -> res F V U s = sigma V).
  {
    apply hdesc.
    exact hsigma.
  }
  destruct hglue as [s hs].
  exists s.
  split.
  - exact hs.
  - intros t ht.
    assert (hts : t = s).
    {
      apply huniq.
      intros V hV.
      transitivity (sigma V).
      + apply ht.
        exact hV.
      + symmetry.
        apply hs.
        exact hV.
    }
    exact hts.
Qed.

Lemma cech_descent_equivalence {X : Type} `{SiteLike X}
    (F : PresheafLike)
    (U : X) (I : X -> Prop)
    (hcech : CechComplexLike F U I) :
    DescentDataLike F U I /\
    (forall sigma : X -> Sect F,
      CompatibleFamilyLike F I sigma ->
      exists s : Sect F,
        (forall V : X, I V -> res F V U s = sigma V) /\
        (forall t : Sect F,
          (forall V : X, I V -> res F V U t = sigma V) ->
          t = s)).
Proof.
  split.
  - exact (proj1 hcech).
  - intros sigma hsigma.
    destruct hcech as [hdesc huniq].
    assert (hglue : exists s : Sect F, forall V : X, I V -> res F V U s = sigma V).
    {
      apply hdesc.
      exact hsigma.
    }
    destruct hglue as [s hs].
    exists s.
    split.
    + exact hs.
    + intros t ht.
      assert (hts : t = s).
      {
        apply huniq.
        intros V hV.
        transitivity (sigma V).
        * apply ht.
          exact hV.
        * symmetry.
          apply hs.
          exact hV.
      }
      exact hts.
Qed.

Lemma hypercover_refinement_transfer {X : Type} `{SiteLike X}
    (F : PresheafLike)
    (U : X)
    (I J : X -> Prop)
    (hsub : forall V : X, J V -> I V)
    (hlift :
      forall sigmaJ : X -> Sect F,
        CompatibleFamilyLike F J sigmaJ ->
        exists sigmaI : X -> Sect F,
          CompatibleFamilyLike F I sigmaI /\
          (forall V : X, J V -> sigmaI V = sigmaJ V))
    (hdescI : DescentDataLike F U I) :
    DescentDataLike F U J.
Proof.
  intros sigmaJ hsigmaJ.
  destruct (hlift sigmaJ hsigmaJ) as [sigmaI [hsigmaI hagree]].
  destruct (hdescI sigmaI hsigmaI) as [s hsI].
  exists s.
  intros V hV.
  assert (hIV : I V).
  {
    apply hsub.
    exact hV.
  }
  assert (hresI : res F V U s = sigmaI V).
  {
    apply hsI.
    exact hIV.
  }
  assert (hEq : sigmaI V = sigmaJ V).
  {
    apply hagree.
    exact hV.
  }
  rewrite hEq in hresI.
  exact hresI.
Qed.
