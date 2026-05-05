(*
BENCHMARK_ID: TINY_MATHLIB_BATCH03_LINEAR_TENSOR_HOM_CURRYING_LIKE
PAIR_STEM: linear_tensor_hom_currying_like
MATH_DOMAIN: Linear Algebra
SOURCE_MATHLIB: Mathlib/LinearAlgebra/TensorProduct
ABSTRACTION_LEVEL: axiomatic
DECLARATION_COUNT: 12
*)

Set Universe Polymorphism.
Set Implicit Arguments.

Class RingLike (R : Type) := {
  rzero : R;
  radd : R -> R -> R;
  rneg : R -> R;
  rone : R;
  rmul : R -> R -> R;
  radd_assoc : forall a b c : R, radd (radd a b) c = radd a (radd b c);
  radd_comm : forall a b : R, radd a b = radd b a;
  radd_zero : forall a : R, radd a rzero = a;
  rzero_add : forall a : R, radd rzero a = a;
  radd_left_neg : forall a : R, radd (rneg a) a = rzero;
  rmul_assoc : forall a b c : R, rmul (rmul a b) c = rmul a (rmul b c);
  rone_mul : forall a : R, rmul rone a = a;
  rmul_one : forall a : R, rmul a rone = a;
  rleft_distrib : forall a b c : R, rmul a (radd b c) = radd (rmul a b) (rmul a c);
  rright_distrib : forall a b c : R, rmul (radd a b) c = radd (rmul a c) (rmul b c)
}.

Arguments rzero {R} _.
Arguments radd {R} _ _ _.
Arguments rneg {R} _ _.
Arguments rone {R} _.
Arguments rmul {R} _ _ _.

Class ModuleLike (R : Type) (M : Type) (RR : RingLike R) := {
  mzero : M;
  madd : M -> M -> M;
  mneg : M -> M;
  smul : R -> M -> M;
  madd_assoc : forall x y z : M, madd (madd x y) z = madd x (madd y z);
  madd_comm : forall x y : M, madd x y = madd y x;
  madd_zero : forall x : M, madd x mzero = x;
  mzero_add : forall x : M, madd mzero x = x;
  madd_left_neg : forall x : M, madd (mneg x) x = mzero;
  smul_add : forall a : R, forall x y : M, smul a (madd x y) = madd (smul a x) (smul a y);
  add_smul : forall a b : R, forall x : M,
      smul (radd RR a b) x = madd (smul a x) (smul b x);
  one_smul : forall x : M, smul (rone RR) x = x;
  mul_smul : forall a b : R, forall x : M,
      smul (rmul RR a b) x = smul a (smul b x);
  zero_smul : forall x : M, smul (rzero RR) x = mzero
}.

Arguments mzero {R M RR} _.
Arguments madd {R M RR} _ _ _.
Arguments mneg {R M RR} _ _.
Arguments smul {R M RR} _ _ _.

Definition BilinearLike {R M N P : Type}
    (RR : RingLike R)
    (MM : @ModuleLike R M RR)
    (MN : @ModuleLike R N RR)
    (MP : @ModuleLike R P RR)
    (b : M -> N -> P) : Prop :=
  (forall x1 x2 : M, forall y : N,
      b (madd MM x1 x2) y = madd MP (b x1 y) (b x2 y)) /\
  (forall a : R, forall x : M, forall y : N,
      b (smul MM a x) y = smul MP a (b x y)) /\
  (forall x : M, forall y1 y2 : N,
      b x (madd MN y1 y2) = madd MP (b x y1) (b x y2)) /\
  (forall a : R, forall x : M, forall y : N,
      b x (smul MN a y) = smul MP a (b x y)).

Definition TensorLike (R M N : Type)
    (RR : RingLike R)
    (MM : @ModuleLike R M RR)
    (MN : @ModuleLike R N RR) : Type :=
  (M * N)%type.

Definition TensorLiftLike {R M N P : Type}
    (RR : RingLike R)
    (MM : @ModuleLike R M RR)
    (MN : @ModuleLike R N RR)
    (MP : @ModuleLike R P RR)
    (b : M -> N -> P)
    (t : @TensorLike R M N RR MM MN) : P :=
  b (fst t) (snd t).

Definition CurryLike {R M N P : Type}
    (RR : RingLike R)
    (MM : @ModuleLike R M RR)
    (MN : @ModuleLike R N RR)
    (MP : @ModuleLike R P RR)
    (h : @TensorLike R M N RR MM MN -> P) : M -> N -> P :=
  fun x y => h (x, y).

Lemma curry_linear {R M N P : Type}
    (RR : RingLike R)
    (MM : @ModuleLike R M RR)
    (MN : @ModuleLike R N RR)
    (MP : @ModuleLike R P RR)
    (h : @TensorLike R M N RR MM MN -> P)
    (hAdd : forall x1 x2 : M, forall y : N,
      h (madd MM x1 x2, y) = madd MP (h (x1, y)) (h (x2, y)))
    (hSmul : forall a : R, forall x : M, forall y : N,
      h (smul MM a x, y) = smul MP a (h (x, y))) :
    (forall x1 x2 : M, forall y : N,
      @CurryLike R M N P RR MM MN MP h (madd MM x1 x2) y
        = madd MP (@CurryLike R M N P RR MM MN MP h x1 y) (@CurryLike R M N P RR MM MN MP h x2 y)) /\
    (forall a : R, forall x : M, forall y : N,
      @CurryLike R M N P RR MM MN MP h (smul MM a x) y
        = smul MP a (@CurryLike R M N P RR MM MN MP h x y)).
Proof.
  split.
  - intros x1 x2 y.
    assert (hStep : h (madd MM x1 x2, y) = madd MP (h (x1, y)) (h (x2, y))).
    { apply hAdd. }
    assert (hDefL : @CurryLike R M N P RR MM MN MP h (madd MM x1 x2) y = h (madd MM x1 x2, y)).
    { reflexivity. }
    assert (hDefR1 : @CurryLike R M N P RR MM MN MP h x1 y = h (x1, y)).
    { reflexivity. }
    assert (hDefR2 : @CurryLike R M N P RR MM MN MP h x2 y = h (x2, y)).
    { reflexivity. }
    assert (hPost : madd MP (h (x1, y)) (h (x2, y))
      = madd MP (@CurryLike R M N P RR MM MN MP h x1 y) (@CurryLike R M N P RR MM MN MP h x2 y)).
    {
      rewrite hDefR1.
      rewrite hDefR2.
      reflexivity.
    }
    transitivity (h (madd MM x1 x2, y)).
    + exact hDefL.
    + transitivity (madd MP (h (x1, y)) (h (x2, y))).
      * exact hStep.
      * exact hPost.
  - intros a x y.
    assert (hStep : h (smul MM a x, y) = smul MP a (h (x, y))).
    { apply hSmul. }
    assert (hDefL : @CurryLike R M N P RR MM MN MP h (smul MM a x) y = h (smul MM a x, y)).
    { reflexivity. }
    assert (hDefR : @CurryLike R M N P RR MM MN MP h x y = h (x, y)).
    { reflexivity. }
    assert (hPost : smul MP a (h (x, y)) = smul MP a (@CurryLike R M N P RR MM MN MP h x y)).
    { rewrite hDefR. reflexivity. }
    transitivity (h (smul MM a x, y)).
    + exact hDefL.
    + transitivity (smul MP a (h (x, y))).
      * exact hStep.
      * exact hPost.
Qed.

Lemma uncurry_linear {R M N P : Type}
    (RR : RingLike R)
    (MM : @ModuleLike R M RR)
    (MN : @ModuleLike R N RR)
    (MP : @ModuleLike R P RR)
    (b : M -> N -> P)
    (hb : @BilinearLike R M N P RR MM MN MP b) :
    (forall x1 x2 : M, forall y : N,
      @TensorLiftLike R M N P RR MM MN MP b (madd MM x1 x2, y)
        = madd MP (@TensorLiftLike R M N P RR MM MN MP b (x1, y)) (@TensorLiftLike R M N P RR MM MN MP b (x2, y))) /\
    (forall a : R, forall x : M, forall y : N,
      @TensorLiftLike R M N P RR MM MN MP b (smul MM a x, y)
        = smul MP a (@TensorLiftLike R M N P RR MM MN MP b (x, y))) /\
    (forall x : M, forall y1 y2 : N,
      @TensorLiftLike R M N P RR MM MN MP b (x, madd MN y1 y2)
        = madd MP (@TensorLiftLike R M N P RR MM MN MP b (x, y1)) (@TensorLiftLike R M N P RR MM MN MP b (x, y2))) /\
    (forall a : R, forall x : M, forall y : N,
      @TensorLiftLike R M N P RR MM MN MP b (x, smul MN a y)
        = smul MP a (@TensorLiftLike R M N P RR MM MN MP b (x, y))).
Proof.
  destruct hb as [hAddL [hSmulL [hAddR hSmulR]]].
  split.
  - intros x1 x2 y.
    assert (hStep : b (madd MM x1 x2) y = madd MP (b x1 y) (b x2 y)).
    { apply hAddL. }
    assert (hDefL : @TensorLiftLike R M N P RR MM MN MP b (madd MM x1 x2, y)
      = b (madd MM x1 x2) y).
    { reflexivity. }
    assert (hDefR1 : @TensorLiftLike R M N P RR MM MN MP b (x1, y) = b x1 y).
    { reflexivity. }
    assert (hDefR2 : @TensorLiftLike R M N P RR MM MN MP b (x2, y) = b x2 y).
    { reflexivity. }
    assert (hPost : madd MP (b x1 y) (b x2 y)
      = madd MP (@TensorLiftLike R M N P RR MM MN MP b (x1, y))
          (@TensorLiftLike R M N P RR MM MN MP b (x2, y))).
    {
      rewrite hDefR1.
      rewrite hDefR2.
      reflexivity.
    }
    transitivity (b (madd MM x1 x2) y).
    + exact hDefL.
    + transitivity (madd MP (b x1 y) (b x2 y)).
      * exact hStep.
      * exact hPost.
  - split.
    + intros a x y.
      assert (hStep : b (smul MM a x) y = smul MP a (b x y)).
      { apply hSmulL. }
      assert (hDefL : @TensorLiftLike R M N P RR MM MN MP b (smul MM a x, y)
        = b (smul MM a x) y).
      { reflexivity. }
      assert (hDefR : @TensorLiftLike R M N P RR MM MN MP b (x, y) = b x y).
      { reflexivity. }
      assert (hPost : smul MP a (b x y) = smul MP a (@TensorLiftLike R M N P RR MM MN MP b (x, y))).
      { rewrite hDefR. reflexivity. }
      transitivity (b (smul MM a x) y).
      * exact hDefL.
      * transitivity (smul MP a (b x y)).
        -- exact hStep.
        -- exact hPost.
    + split.
      * intros x y1 y2.
        assert (hStep : b x (madd MN y1 y2) = madd MP (b x y1) (b x y2)).
        { apply hAddR. }
        assert (hDefL : @TensorLiftLike R M N P RR MM MN MP b (x, madd MN y1 y2)
          = b x (madd MN y1 y2)).
        { reflexivity. }
        assert (hDefR1 : @TensorLiftLike R M N P RR MM MN MP b (x, y1) = b x y1).
        { reflexivity. }
        assert (hDefR2 : @TensorLiftLike R M N P RR MM MN MP b (x, y2) = b x y2).
        { reflexivity. }
        assert (hPost : madd MP (b x y1) (b x y2)
          = madd MP (@TensorLiftLike R M N P RR MM MN MP b (x, y1))
              (@TensorLiftLike R M N P RR MM MN MP b (x, y2))).
        {
          rewrite hDefR1.
          rewrite hDefR2.
          reflexivity.
        }
        transitivity (b x (madd MN y1 y2)).
        -- exact hDefL.
        -- transitivity (madd MP (b x y1) (b x y2)).
           ++ exact hStep.
           ++ exact hPost.
      * intros a x y.
        assert (hStep : b x (smul MN a y) = smul MP a (b x y)).
        { apply hSmulR. }
        assert (hDefL : @TensorLiftLike R M N P RR MM MN MP b (x, smul MN a y)
          = b x (smul MN a y)).
        { reflexivity. }
        assert (hDefR : @TensorLiftLike R M N P RR MM MN MP b (x, y) = b x y).
        { reflexivity. }
        assert (hPost : smul MP a (b x y) = smul MP a (@TensorLiftLike R M N P RR MM MN MP b (x, y))).
        { rewrite hDefR. reflexivity. }
        transitivity (b x (smul MN a y)).
        -- exact hDefL.
        -- transitivity (smul MP a (b x y)).
           ++ exact hStep.
           ++ exact hPost.
Qed.

Lemma curry_uncurry {R M N P : Type}
    (RR : RingLike R)
    (MM : @ModuleLike R M RR)
    (MN : @ModuleLike R N RR)
    (MP : @ModuleLike R P RR)
    (b : M -> N -> P) :
    forall x : M, forall y : N,
      @CurryLike R M N P RR MM MN MP (fun t : @TensorLike R M N RR MM MN => @TensorLiftLike R M N P RR MM MN MP b t) x y = b x y.
Proof.
  intros x y.
  assert (hDefC :
    @CurryLike R M N P RR MM MN MP
      (fun t : @TensorLike R M N RR MM MN => @TensorLiftLike R M N P RR MM MN MP b t) x y
      = @TensorLiftLike R M N P RR MM MN MP b (x, y)).
  { reflexivity. }
  assert (hDefT : @TensorLiftLike R M N P RR MM MN MP b (x, y) = b x y).
  { reflexivity. }
  transitivity (@TensorLiftLike R M N P RR MM MN MP b (x, y)).
  - exact hDefC.
  - exact hDefT.
Qed.

Lemma uncurry_curry {R M N P : Type}
    (RR : RingLike R)
    (MM : @ModuleLike R M RR)
    (MN : @ModuleLike R N RR)
    (MP : @ModuleLike R P RR)
    (h : @TensorLike R M N RR MM MN -> P) :
    forall t : @TensorLike R M N RR MM MN,
      @TensorLiftLike R M N P RR MM MN MP (@CurryLike R M N P RR MM MN MP h) t = h t.
Proof.
  intros t.
  destruct t as [x y].
  assert (hDefT :
    @TensorLiftLike R M N P RR MM MN MP (@CurryLike R M N P RR MM MN MP h) (x, y)
      = @CurryLike R M N P RR MM MN MP h x y).
  { reflexivity. }
  assert (hDefC : @CurryLike R M N P RR MM MN MP h x y = h (x, y)).
  { reflexivity. }
  transitivity (@CurryLike R M N P RR MM MN MP h x y).
  - exact hDefT.
  - exact hDefC.
Qed.

Lemma tensor_hom_adjunction_like {R M N P : Type}
    (RR : RingLike R)
    (MM : @ModuleLike R M RR)
    (MN : @ModuleLike R N RR)
    (MP : @ModuleLike R P RR) :
    (forall h : @TensorLike R M N RR MM MN -> P, forall t : @TensorLike R M N RR MM MN,
      @TensorLiftLike R M N P RR MM MN MP (@CurryLike R M N P RR MM MN MP h) t = h t) /\
    (forall b : M -> N -> P, forall x : M, forall y : N,
      @CurryLike R M N P RR MM MN MP (fun t : @TensorLike R M N RR MM MN => @TensorLiftLike R M N P RR MM MN MP b t) x y = b x y).
Proof.
  split.
  - intros h t.
    assert (hUncurry : @TensorLiftLike R M N P RR MM MN MP (@CurryLike R M N P RR MM MN MP h) t = h t).
    { apply uncurry_curry. }
    assert (hRefl : @TensorLiftLike R M N P RR MM MN MP (@CurryLike R M N P RR MM MN MP h) t
      = @TensorLiftLike R M N P RR MM MN MP (@CurryLike R M N P RR MM MN MP h) t).
    { reflexivity. }
    transitivity (@TensorLiftLike R M N P RR MM MN MP (@CurryLike R M N P RR MM MN MP h) t).
    + exact hRefl.
    + exact hUncurry.
  - intros b x y.
    assert (hCurry : @CurryLike R M N P RR MM MN MP (fun t : @TensorLike R M N RR MM MN => @TensorLiftLike R M N P RR MM MN MP b t) x y = b x y).
    { apply curry_uncurry. }
    assert (hRefl : @CurryLike R M N P RR MM MN MP (fun t : @TensorLike R M N RR MM MN => @TensorLiftLike R M N P RR MM MN MP b t) x y
      = @CurryLike R M N P RR MM MN MP (fun t : @TensorLike R M N RR MM MN => @TensorLiftLike R M N P RR MM MN MP b t) x y).
    { reflexivity. }
    transitivity (@CurryLike R M N P RR MM MN MP (fun t : @TensorLike R M N RR MM MN => @TensorLiftLike R M N P RR MM MN MP b t) x y).
    + exact hRefl.
    + exact hCurry.
Qed.

Lemma tensor_ext_like {R M N P : Type}
    (RR : RingLike R)
    (MM : @ModuleLike R M RR)
    (MN : @ModuleLike R N RR)
    (MP : @ModuleLike R P RR)
    (h1 h2 : @TensorLike R M N RR MM MN -> P)
    (hEq : forall x : M, forall y : N, h1 (x, y) = h2 (x, y)) :
    forall t : @TensorLike R M N RR MM MN, h1 t = h2 t.
Proof.
  intros t.
  destruct t as [x y].
  assert (hxy : h1 (x, y) = h2 (x, y)).
  { apply hEq. }
  assert (hLift : h1 (pair x y) = h2 (pair x y)).
  { exact hxy. }
  exact hLift.
Qed.
