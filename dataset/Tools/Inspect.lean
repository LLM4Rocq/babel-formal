
import Lean

open Lean Meta Elab Command

partial def collectConsts (e : Expr) : StateT (Std.HashSet Name) MetaM Unit := do
  match e with
  | .const n _ => modify (·.insert n)
  | .app f a   => collectConsts f; collectConsts a
  | .lam _ _ b _ => collectConsts b
  | .forallE _ d b _ => collectConsts d; collectConsts b
  | .letE _ t v b _ => collectConsts t; collectConsts v; collectConsts b
  | .mdata _ b => collectConsts b
  | .proj _ _ b => collectConsts b
  | _ => pure ()

def classify (env : Environment) (n : Name) : String :=
  match env.find? n with
  | none => "external"
  | some (ConstantInfo.axiomInfo _)  => "axiom"
  | some (ConstantInfo.thmInfo _)    => "theorem"
  | some (ConstantInfo.defnInfo _)   => "definition"
  | some (ConstantInfo.opaqueInfo _) => "opaque"
  | some (ConstantInfo.ctorInfo _)   => "constructor"
  | some (ConstantInfo.inductInfo _) => "inductive"
  | some (ConstantInfo.quotInfo _)   => "quot"
  | some (ConstantInfo.recInfo _)    => "recursor"

def isClass (env : Environment) (n : Name) : Bool :=
  env.isClass n

def toJsonName (n : Name) : Json :=
  Json.str n.toString

elab "#dumpDependencies" n:ident : command => do
  let env ← getEnv
  let some decl := env.find? n.getId
    | throwError "unknown declaration: {n}"
  let mut set : Std.HashSet Name := {}
  let ((), sType) ← (collectConsts decl.type).run set
  let ((), sAll)  ← (collectConsts decl.value!).run sType

  let typeNames := sType.toList.eraseDups
  let allNames  := sAll.toList.eraseDups

  let jDeps := allNames.map (fun nm =>
    Json.mkObj [("name", toJsonName nm), ("kind", Json.str (classify env nm)), ("isClass", Json.bool (isClass env nm))])

  let jType := typeNames.map (fun nm =>
    Json.mkObj [("name", toJsonName nm), ("kind", Json.str (classify env nm)), ("isClass", Json.bool (isClass env nm))])

  let axioms := (Lean.collectAxioms env decl.value!).toList
  let jAxioms := axioms.map (fun nm => Json.str nm.toString)

  let payload := Json.mkObj
    [ ("theorem", Json.str n.getId.toString)
    , ("typeConstants", Json.arr jType)
    , ("proofConstants", Json.arr jDeps)
    , ("axioms", Json.arr jAxioms)
    ]
  logInfo m!"{payload.pretty}"
