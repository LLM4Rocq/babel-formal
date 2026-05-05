(*
BENCHMARK_ID: TINY_MATHLIB_BATCH05_COMBINATORICS_FLAG_ALGEBRA_DENSITY_LIKE
PAIR_STEM: combinatorics_flag_algebra_density_like
MATH_DOMAIN: Combinatorics
SOURCE_MATHLIB: Mathlib/Combinatorics/*
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class CombStruct_flag_algebra_density (X : Type) := {
  le : X -> X -> Prop;
  le_refl : forall x : X, le x x;
  le_trans : forall {x y z : X}, le x y -> le y z -> le x z;
  add : X -> X -> X;
  mul : X -> X -> X;
  entropy : X -> X;
  density : X -> X;
  regularize : X -> X;
  container_axiom : forall x : X, le x (regularize x);
  entropy_axiom : forall x y : X,
      le (entropy (add x y)) (add (entropy x) (entropy y));
  density_axiom : forall x : X, le (density (regularize x)) (density x);
  sparse_regularity_axiom : forall x : X, le (regularize (regularize x)) (regularize x);
  tverberg_axiom : forall x y : X, le (mul x y) (add x y);
  counting_axiom : forall x y : X,
      le (density (mul x y)) (mul (density x) (density y));
  extremal_axiom : forall x y : X,
      le (mul (density x) (density y)) (density (add x y))
}.

Arguments le {X} {_} _ _.
Arguments add {X} {_} _ _.
Arguments mul {X} {_} _ _.
Arguments entropy {X} {_} _.
Arguments density {X} {_} _. 
Arguments regularize {X} {_} _. 

Infix "<~" := le (at level 70).
Infix "+#" := add (at level 50, left associativity).
Infix "*#" := mul (at level 40, left associativity).

Definition HypergraphObj_flag_algebra_density
    {X : Type} `{CombStruct_flag_algebra_density X} (x : X) : X :=
  x.

Definition EntropyObj_flag_algebra_density
    {X : Type} `{CombStruct_flag_algebra_density X} (x : X) : X :=
  entropy x.

Definition DensityObj_flag_algebra_density
    {X : Type} `{CombStruct_flag_algebra_density X} (x : X) : X :=
  density x.

Definition RegularityObj_flag_algebra_density
    {X : Type} `{CombStruct_flag_algebra_density X} (x : X) : X :=
  regularize x.

Lemma container_step_flag_algebra_density
    {X : Type} `{CombStruct_flag_algebra_density X}
    (x : X) :
    HypergraphObj_flag_algebra_density x <~
      RegularityObj_flag_algebra_density x.
Proof.
  assert (hraw : x <~ regularize x).
  { exact (container_axiom x). }
  unfold HypergraphObj_flag_algebra_density, RegularityObj_flag_algebra_density.
  exact hraw.
Qed.

Lemma entropy_lemma_step_flag_algebra_density
    {X : Type} `{CombStruct_flag_algebra_density X}
    (x y : X) :
    EntropyObj_flag_algebra_density (x +# y) <~
      EntropyObj_flag_algebra_density x +#
        EntropyObj_flag_algebra_density y.
Proof.
  assert (hent : entropy (add x y) <~ add (entropy x) (entropy y)).
  { exact (entropy_axiom x y). }
  unfold EntropyObj_flag_algebra_density.
  exact hent.
Qed.

Lemma flag_density_step_flag_algebra_density
    {X : Type} `{CombStruct_flag_algebra_density X}
    (x : X)
    (hreg : RegularityObj_flag_algebra_density x <~
      RegularityObj_flag_algebra_density x) :
    DensityObj_flag_algebra_density
      (RegularityObj_flag_algebra_density x) <~
      DensityObj_flag_algebra_density x.
Proof.
  assert (hden : density (regularize x) <~ density x).
  { exact (density_axiom x). }
  assert (hkeep : RegularityObj_flag_algebra_density x <~
      RegularityObj_flag_algebra_density x).
  { exact hreg. }
  unfold DensityObj_flag_algebra_density, RegularityObj_flag_algebra_density.
  exact hden.
Qed.

Lemma sparse_regularity_step_flag_algebra_density
    {X : Type} `{CombStruct_flag_algebra_density X}
    (x : X) :
    RegularityObj_flag_algebra_density
      (RegularityObj_flag_algebra_density x) <~
      RegularityObj_flag_algebra_density x /\
    HypergraphObj_flag_algebra_density x <~
      RegularityObj_flag_algebra_density x.
Proof.
  assert (hsparse : regularize (regularize x) <~ regularize x).
  { exact (sparse_regularity_axiom x). }
  assert (hcont : HypergraphObj_flag_algebra_density x <~ RegularityObj_flag_algebra_density x).
  { apply container_step_flag_algebra_density. }
  split.
  - unfold RegularityObj_flag_algebra_density.
    exact hsparse.
  - exact hcont.
Qed.

Lemma tverberg_partition_step_flag_algebra_density
    {X : Type} `{CombStruct_flag_algebra_density X}
    (x y : X) :
    HypergraphObj_flag_algebra_density x *#
      HypergraphObj_flag_algebra_density y <~
      HypergraphObj_flag_algebra_density x +#
        HypergraphObj_flag_algebra_density y.
Proof.
  assert (htver : mul x y <~ add x y).
  { exact (tverberg_axiom x y). }
  unfold HypergraphObj_flag_algebra_density.
  exact htver.
Qed.

Lemma counting_upgrade_flag_algebra_density
    {X : Type} `{CombStruct_flag_algebra_density X}
    (x y : X) :
    DensityObj_flag_algebra_density (x *# y) <~
      DensityObj_flag_algebra_density x *#
        DensityObj_flag_algebra_density y.
Proof.
  assert (hcount : density (mul x y) <~ mul (density x) (density y)).
  { exact (counting_axiom x y). }
  unfold DensityObj_flag_algebra_density.
  exact hcount.
Qed.

Lemma extremal_conclusion_flag_algebra_density
    {X : Type} `{CombStruct_flag_algebra_density X}
    (x y : X) :
    DensityObj_flag_algebra_density (x *# y) <~
      DensityObj_flag_algebra_density (x +# y).
Proof.
  assert (h1 : DensityObj_flag_algebra_density (x *# y) <~
              DensityObj_flag_algebra_density x *# DensityObj_flag_algebra_density y).
  { apply counting_upgrade_flag_algebra_density. }
  assert (h2raw : mul (density x) (density y) <~ density (add x y)).
  { exact (extremal_axiom x y). }
  assert (h2 : DensityObj_flag_algebra_density x *# DensityObj_flag_algebra_density y <~
               DensityObj_flag_algebra_density (x +# y)).
  {
    unfold DensityObj_flag_algebra_density.
    exact h2raw.
  }
  exact (le_trans h1 h2).
Qed.
