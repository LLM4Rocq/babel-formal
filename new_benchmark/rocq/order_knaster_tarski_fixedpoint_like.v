(*
BENCHMARK_ID: TINY_MATHLIB_BATCH03_ORDER_KNASTER_TARSKI_FIXEDPOINT
PAIR_STEM: order_knaster_tarski_fixedpoint_like
MATH_DOMAIN: Order Theory
SOURCE_MATHLIB: Mathlib/Order/FixedPoints
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class CompleteLatticeLike (A : Type) := {
  leq : A -> A -> Prop;
  leq_refl : forall a : A, leq a a;
  leq_trans : forall a b c : A, leq a b -> leq b c -> leq a c;
  leq_antisymm : forall a b : A, leq a b -> leq b a -> a = b;
  sInf : (A -> Prop) -> A;
  sSup : (A -> Prop) -> A;
  sInf_le : forall S x, S x -> leq (sInf S) x;
  leq_sInf : forall S x, (forall y, S y -> leq x y) -> leq x (sInf S);
  leq_sSup : forall S x, S x -> leq x (sSup S);
  sSup_le : forall S x, (forall y, S y -> leq y x) -> leq (sSup S) x
}.

Infix "<=" := leq (at level 70).

Definition monotone {A : Type} `{CompleteLatticeLike A} (f : A -> A) : Prop :=
  forall a b : A, a <= b -> f a <= f b.

Definition IsPrefixed {A : Type} `{CompleteLatticeLike A} (f : A -> A) (x : A) : Prop :=
  f x <= x.

Definition IsPostfixed {A : Type} `{CompleteLatticeLike A} (f : A -> A) (x : A) : Prop :=
  x <= f x.

Definition lfp {A : Type} `{CompleteLatticeLike A} (f : A -> A) : A :=
  sInf (fun x => IsPrefixed f x).

Definition gfp {A : Type} `{CompleteLatticeLike A} (f : A -> A) : A :=
  sSup (fun x => IsPostfixed f x).

Lemma lfp_prefixed {A : Type} `{CompleteLatticeLike A}
    (f : A -> A) (hmono : monotone f) :
    IsPrefixed f (lfp f).
Proof.
  unfold IsPrefixed, lfp.
  apply leq_sInf.
  intros x hx.
  assert (h_lfp_le_x : sInf (fun y : A => IsPrefixed f y) <= x).
  {
    apply sInf_le.
    exact hx.
  }
  assert (h_flfp_le_fx : f (sInf (fun y : A => IsPrefixed f y)) <= f x).
  {
    apply hmono.
    exact h_lfp_le_x.
  }
  exact (leq_trans _ _ _ h_flfp_le_fx hx).
Qed.

Lemma lfp_least {A : Type} `{CompleteLatticeLike A}
    (f : A -> A) (hmono : monotone f) (x : A) (hx : IsPrefixed f x) :
    lfp f <= x.
Proof.
  assert (hbundle : lfp f <= x /\ f (lfp f) <= x).
  {
    assert (hlfp_le_x : lfp f <= x).
    {
      unfold lfp.
      apply sInf_le.
      exact hx.
    }
    assert (hmono_step : f (lfp f) <= f x).
    {
      apply hmono.
      exact hlfp_le_x.
    }
    assert (hfx_le_x : f x <= x).
    {
      exact hx.
    }
    assert (hflfp_le_x : f (lfp f) <= x).
    {
      exact (leq_trans _ _ _ hmono_step hfx_le_x).
    }
    split.
    - exact hlfp_le_x.
    - exact hflfp_le_x.
  }
  exact (proj1 hbundle).
Qed.

Lemma gfp_postfixed {A : Type} `{CompleteLatticeLike A}
    (f : A -> A) (hmono : monotone f) :
    IsPostfixed f (gfp f).
Proof.
  unfold IsPostfixed, gfp.
  apply sSup_le.
  intros x hx.
  assert (hx_le_gfp : x <= sSup (fun y : A => IsPostfixed f y)).
  {
    apply leq_sSup.
    exact hx.
  }
  assert (hfx_le_fgfp : f x <= f (sSup (fun y : A => IsPostfixed f y))).
  {
    apply hmono.
    exact hx_le_gfp.
  }
  exact (leq_trans _ _ _ hx hfx_le_fgfp).
Qed.

Lemma gfp_greatest {A : Type} `{CompleteLatticeLike A}
    (f : A -> A) (hmono : monotone f) (x : A) (hx : IsPostfixed f x) :
    x <= gfp f.
Proof.
  assert (hbundle : x <= gfp f /\ x <= f (gfp f)).
  {
    assert (hx_le_gfp : x <= gfp f).
    {
      unfold gfp.
      apply leq_sSup.
      exact hx.
    }
    assert (hmono_step : f x <= f (gfp f)).
    {
      apply hmono.
      exact hx_le_gfp.
    }
    assert (hx_le_fgfp : x <= f (gfp f)).
    {
      exact (leq_trans _ _ _ hx hmono_step).
    }
    split.
    - exact hx_le_gfp.
    - exact hx_le_fgfp.
  }
  exact (proj1 hbundle).
Qed.

Lemma lfp_mono {A : Type} `{CompleteLatticeLike A}
    (f g : A -> A) (hmono_f : monotone f) (hmono_g : monotone g)
    (hfg : forall x : A, f x <= g x) :
    lfp f <= lfp g.
Proof.
  assert (hbundle : lfp f <= lfp g /\ f (lfp f) <= lfp g).
  {
    assert (hpref_g : IsPrefixed g (lfp g)).
    {
      apply lfp_prefixed.
      exact hmono_g.
    }
    assert (hcomp_f : IsPrefixed f (lfp g)).
    {
      assert (hfg_at : f (lfp g) <= g (lfp g)).
      {
        apply hfg.
      }
      exact (leq_trans _ _ _ hfg_at hpref_g).
    }
    assert (hlfp_le : lfp f <= lfp g).
    {
      exact (@lfp_least A H f hmono_f (lfp g) hcomp_f).
    }
    assert (hmono_f_step : f (lfp f) <= f (lfp g)).
    {
      apply hmono_f.
      exact hlfp_le.
    }
    assert (hfg_at_lfp_g : f (lfp g) <= g (lfp g)).
    {
      apply hfg.
    }
    assert (hf_lfp_f_le_lfp_g : f (lfp f) <= lfp g).
    {
      assert (hf_lfp_f_le_g_lfp_g : f (lfp f) <= g (lfp g)).
      {
        exact (leq_trans _ _ _ hmono_f_step hfg_at_lfp_g).
      }
      exact (leq_trans _ _ _ hf_lfp_f_le_g_lfp_g hpref_g).
    }
    split.
    - exact hlfp_le.
    - exact hf_lfp_f_le_lfp_g.
  }
  exact (proj1 hbundle).
Qed.

Lemma gfp_mono {A : Type} `{CompleteLatticeLike A}
    (f g : A -> A) (hmono_f : monotone f) (hmono_g : monotone g)
    (hfg : forall x : A, f x <= g x) :
    gfp f <= gfp g.
Proof.
  assert (hbundle : gfp f <= gfp g /\ gfp f <= g (gfp g)).
  {
    assert (hpost_f : IsPostfixed f (gfp f)).
    {
      apply gfp_postfixed.
      exact hmono_f.
    }
    assert (hpost_g_on_gfp_f : IsPostfixed g (gfp f)).
    {
      assert (hfg_at : f (gfp f) <= g (gfp f)).
      {
        apply hfg.
      }
      exact (leq_trans _ _ _ hpost_f hfg_at).
    }
    assert (hgfp_le : gfp f <= gfp g).
    {
      exact (@gfp_greatest A H g hmono_g (gfp f) hpost_g_on_gfp_f).
    }
    assert (hmono_g_step : g (gfp f) <= g (gfp g)).
    {
      apply hmono_g.
      exact hgfp_le.
    }
    assert (hgfp_f_le_ggfp_g : gfp f <= g (gfp g)).
    {
      exact (leq_trans _ _ _ hpost_g_on_gfp_f hmono_g_step).
    }
    split.
    - exact hgfp_le.
    - exact hgfp_f_le_ggfp_g.
  }
  exact (proj1 hbundle).
Qed.
