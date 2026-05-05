(*
BENCHMARK_ID: TINY_MATHLIB_BATCH06_CATEGORY_VAN_KAMPEN_COLIMIT_LIKE
PAIR_STEM: category_van_kampen_colimit_like
MATH_DOMAIN: Category Theory
SOURCE_MATHLIB: Mathlib/CategoryTheory/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class VanKampenStruct_colimit (Obj : Type) := {
  Arrow : Type;
  pull : Arrow -> Obj -> Obj;
  compare : Arrow -> Obj -> Obj;
  IsColim : Obj -> Prop;
  PullbackStable : Obj -> Prop;
  DescentData : Obj -> Prop;
  Reflective : Obj -> Prop;
  colim_pull : forall (a : Arrow) (c : Obj), IsColim c -> IsColim (pull a c);
  compare_sound : forall (a : Arrow) (c : Obj), IsColim (pull a c) -> PullbackStable (compare a c);
  compare_complete :
    forall (a : Arrow) (c : Obj), PullbackStable (compare a c) -> IsColim (pull a c);
  descent_intro : forall (c : Obj), PullbackStable c -> DescentData c;
  descent_elim : forall (c : Obj), DescentData c -> PullbackStable c;
  reflection_intro : forall (c : Obj), DescentData c -> Reflective c;
  reflection_elim : forall (c : Obj), Reflective c -> PullbackStable c;
  universal_square_rule : forall (c : Obj), Reflective c -> PullbackStable c -> IsColim c;
  colim_implies_stable : forall (c : Obj), IsColim c -> PullbackStable c;
  pull_idempotent : forall (a : Arrow) (c : Obj), pull a (pull a c) = pull a c;
  compare_transport : forall (a : Arrow) (c : Obj), compare a (pull a c) = compare a c
}.

Arguments Arrow {Obj} _.
Arguments pull {Obj} _ _ _.
Arguments compare {Obj} _ _ _. 
Arguments IsColim {Obj} _ _. 
Arguments PullbackStable {Obj} _ _. 
Arguments DescentData {Obj} _ _. 
Arguments Reflective {Obj} _ _. 
Arguments colim_pull {Obj} _ _ _ _. 
Arguments compare_sound {Obj} _ _ _ _. 
Arguments compare_complete {Obj} _ _ _ _. 
Arguments descent_intro {Obj} _ _ _. 
Arguments descent_elim {Obj} _ _ _. 
Arguments reflection_intro {Obj} _ _ _. 
Arguments reflection_elim {Obj} _ _ _. 
Arguments universal_square_rule {Obj} _ _ _ _. 
Arguments colim_implies_stable {Obj} _ _ _. 

Record CoconeData_van_kampen_colimit
    (Obj : Type)
    (V : VanKampenStruct_colimit Obj) := {
  apex : Obj;
  has_colimit : IsColim V apex;
  has_stability : PullbackStable V apex;
  has_reflection : Reflective V apex
}.

Definition pullback_cocone_van_kampen_colimit
    {Obj : Type} {V : VanKampenStruct_colimit Obj}
    (a : Arrow V)
    (K : @CoconeData_van_kampen_colimit Obj V) : Obj :=
  pull V a (apex K).

Definition comparison_functor_van_kampen_colimit
    {Obj : Type} {V : VanKampenStruct_colimit Obj}
    (a : Arrow V)
    (K : @CoconeData_van_kampen_colimit Obj V) : Obj :=
  compare V a (apex K).

Definition descent_kernel_van_kampen_colimit
    {Obj : Type} {V : VanKampenStruct_colimit Obj}
    (a : Arrow V)
    (K : @CoconeData_van_kampen_colimit Obj V) : Prop :=
  DescentData V (comparison_functor_van_kampen_colimit a K).

Lemma pullback_preserves_colimit_van_kampen_colimit
    {Obj : Type} {V : VanKampenStruct_colimit Obj}
    (a : Arrow V)
    (K : @CoconeData_van_kampen_colimit Obj V) :
    IsColim V (pullback_cocone_van_kampen_colimit a K).
Proof.
  pose proof (has_colimit K) as hcolim_apex.
  pose proof (colim_pull V a (apex K) hcolim_apex) as hpull.
  change (IsColim V (pull V a (apex K))).
  exact hpull.
Qed.

Lemma comparison_faithful_van_kampen_colimit
    {Obj : Type} {V : VanKampenStruct_colimit Obj}
    (a : Arrow V)
    (K : @CoconeData_van_kampen_colimit Obj V) :
    PullbackStable V (comparison_functor_van_kampen_colimit a K).
Proof.
  pose proof (pullback_preserves_colimit_van_kampen_colimit a K) as hpull.
  assert (hstable_cmp : PullbackStable V (compare V a (apex K))).
  {
    apply (compare_sound V a (apex K)).
    exact hpull.
  }
  change (PullbackStable V (compare V a (apex K))).
  exact hstable_cmp.
Qed.

Lemma comparison_full_van_kampen_colimit
    {Obj : Type} {V : VanKampenStruct_colimit Obj}
    (a : Arrow V)
    (K : @CoconeData_van_kampen_colimit Obj V)
    (hstable_cmp : PullbackStable V (comparison_functor_van_kampen_colimit a K)) :
    IsColim V (pullback_cocone_van_kampen_colimit a K).
Proof.
  assert (hstable_raw : PullbackStable V (compare V a (apex K))).
  { exact hstable_cmp. }
  pose proof (compare_complete V a (apex K) hstable_raw) as hcolim_raw.
  change (IsColim V (pull V a (apex K))).
  exact hcolim_raw.
Qed.

Lemma descent_effective_van_kampen_colimit
    {Obj : Type} {V : VanKampenStruct_colimit Obj}
    (a : Arrow V)
    (K : @CoconeData_van_kampen_colimit Obj V) :
    descent_kernel_van_kampen_colimit a K.
Proof.
  pose proof (comparison_faithful_van_kampen_colimit a K) as hstable_cmp.
  pose proof (descent_intro V (comparison_functor_van_kampen_colimit a K) hstable_cmp) as hdescent_cmp.
  change (DescentData V (comparison_functor_van_kampen_colimit a K)).
  exact hdescent_cmp.
Qed.

Lemma cocone_reflection_van_kampen_colimit
    {Obj : Type} {V : VanKampenStruct_colimit Obj}
    (a : Arrow V)
    (K : @CoconeData_van_kampen_colimit Obj V) :
    Reflective V (comparison_functor_van_kampen_colimit a K).
Proof.
  pose proof (descent_effective_van_kampen_colimit a K) as hdescent.
  pose proof (reflection_intro V (comparison_functor_van_kampen_colimit a K) hdescent) as hreflect.
  exact hreflect.
Qed.

Lemma universal_square_van_kampen_colimit
    {Obj : Type} {V : VanKampenStruct_colimit Obj}
    (a : Arrow V)
    (K : @CoconeData_van_kampen_colimit Obj V) :
    IsColim V (comparison_functor_van_kampen_colimit a K).
Proof.
  pose proof (cocone_reflection_van_kampen_colimit a K) as hreflect.
  pose proof (reflection_elim V (comparison_functor_van_kampen_colimit a K) hreflect)
    as hstable_from_reflection.
  pose proof (descent_effective_van_kampen_colimit a K) as hdescent.
  pose proof (descent_elim V (comparison_functor_van_kampen_colimit a K) hdescent)
    as hstable_from_descent.
  pose proof (universal_square_rule V (comparison_functor_van_kampen_colimit a K)
    hreflect hstable_from_descent) as hcolim_cmp.
  assert (PullbackStable V (comparison_functor_van_kampen_colimit a K)) as hkeep.
  { exact hstable_from_reflection. }
  exact hcolim_cmp.
Qed.

Lemma van_kampen_characterization_colimit
    {Obj : Type} {V : VanKampenStruct_colimit Obj}
    (a : Arrow V)
    (K : @CoconeData_van_kampen_colimit Obj V) :
    IsColim V (pullback_cocone_van_kampen_colimit a K) <->
      IsColim V (comparison_functor_van_kampen_colimit a K).
Proof.
  split.
  - intro hpull.
    assert (hstable_cmp : PullbackStable V (comparison_functor_van_kampen_colimit a K)).
    {
      apply (compare_sound V a (apex K)).
      exact hpull.
    }
    assert (hdescent_cmp : descent_kernel_van_kampen_colimit a K).
    {
      unfold descent_kernel_van_kampen_colimit.
      apply (descent_intro V (comparison_functor_van_kampen_colimit a K)).
      exact hstable_cmp.
    }
    assert (hreflect_cmp : Reflective V (comparison_functor_van_kampen_colimit a K)).
    {
      apply (reflection_intro V (comparison_functor_van_kampen_colimit a K)).
      unfold descent_kernel_van_kampen_colimit in hdescent_cmp.
      exact hdescent_cmp.
    }
    assert (hstable_again : PullbackStable V (comparison_functor_van_kampen_colimit a K)).
    {
      apply (reflection_elim V (comparison_functor_van_kampen_colimit a K)).
      exact hreflect_cmp.
    }
    apply (universal_square_rule V (comparison_functor_van_kampen_colimit a K)).
    + exact hreflect_cmp.
    + exact hstable_again.
  - intro hcmp.
    assert (hstable_cmp : PullbackStable V (comparison_functor_van_kampen_colimit a K)).
    {
      apply (colim_implies_stable V (comparison_functor_van_kampen_colimit a K)).
      exact hcmp.
    }
    pose proof (comparison_full_van_kampen_colimit a K hstable_cmp) as hpull.
    exact hpull.
Qed.
