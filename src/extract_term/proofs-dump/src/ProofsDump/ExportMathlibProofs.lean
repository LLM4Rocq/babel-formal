/-
  Per-theorem dump split into two files, grouped by source "filename" (module).

  Usage in `src/ProofsDump.lean`:
    #export_proofs_each_to "proofs_by_src" progress
-/

import Lean
import Mathlib

open Lean
open Lean.Meta
open Lean.Elab
open Lean.Elab.Command

namespace ProofsDump

/-- Pretty-print an expression as a string via the delaborator. -/
def ppExprString (e : Expr) : MetaM String := do
  let fmt ← ppExpr e
  pure fmt.pretty

/-- Safe pretty-print that never throws (returns a fallback on error). -/
def tryPP (m : MetaM String) (fallback := "<pp failed>") : MetaM String := do
  try m catch _ => pure fallback

/-- Collect all `(name, ConstantInfo)` pairs from the current environment. -/
def collectDecls (env : Environment) : List (Name × ConstantInfo) :=
  env.constants.fold (init := []) (f := fun acc n ci => (n, ci) :: acc)

/-- Get the defining module name for a declaration, if any. -/
def moduleNameOf (env : Environment) (n : Name) : Option Name :=
  match env.getModuleIdxFor? n with
  | some midx => env.allImportedModuleNames.get? midx
  | none      => none

/-- Turn a module `Name` into a printable path-like string, e.g. `Mathlib/Topology/Basic`. -/
def modulePathString (m : Name) : String :=
  "/".intercalate (m.components.map (·.toString))

/-- Very conservative filename sanitizer. -/
def sanitize (s : String) : String :=
  let allowed (c : Char) := c.isAlphanum || c == '_' || c == '-' || c == '.'
  s.map (fun c => if allowed c then c else '_')

/-- Repeat a char `c` `n` times into a String. -/
def repeatChar (c : Char) (n : Nat) : String :=
  String.mk (List.replicate n c)

/-- Render a simple ASCII progress bar of fixed `width`. -/
def renderBar (i tot width : Nat) : String :=
  let pct  := if tot == 0 then 100 else (i * 100) / tot
  let fill := if tot == 0 then width else Nat.min width ((i * width) / tot)
  let rest := width - fill
  s!"[{repeatChar '#' fill}{repeatChar '-' rest}] {i}/{tot} ({pct}%)"

/--
  `#export_proofs_each_to "out_dir"`
  or
  `#export_proofs_each_to "out_dir" progress`  -- show terminal progress bar

  Writes TWO files per theorem:
    <theorem>_statement.txt    -- only the formal statement (type)
    <theorem>_proof_term.txt   -- only the pretty-printed proof term (value)

  Grouped under a folder named after the **source filename** (module path),
  e.g. `Mathlib/Topology/Basic` → folder `Mathlib_Topology_Basic`.
-/
elab "#export_proofs_each_to" outDirStr:str maybeProg?:("progress")? : command => do
  let showProg := maybeProg?.isSome
  let outRoot := System.FilePath.mk (outDirStr.getString)
  liftIO <| IO.FS.createDirAll outRoot

  -- Snapshot env & collect theorems (+module name if available)
  let env ← getEnv
  let theorems : List (Name × TheoremVal × Option Name) :=
    (collectDecls env).foldl
      (init := [])
      (fun acc (n, ci) =>
        match ci with
        | ConstantInfo.thmInfo ti =>
            let m? := moduleNameOf env n
            (n, ti, m?) :: acc
        | _ => acc)

  let total := theorems.length

  -- Pretty-printer options for TermElabM blocks
  let ppOpts := ({} : Options)
    |>.setNat `pp.width 100
    |>.setBool `pp.notation true
    |>.setBool `pp.beta false

  -- Prepare stdout handle for in-place updates (bar)
  let h ← liftIO IO.getStdout
  if showProg then
    liftIO do
      h.putStr (renderBar 0 total 40 ++ "\r")
      h.flush

  -- Iterate and write incrementally
  let mut i := 0
  for (n, ti, m?) in theorems do
    -- Pretty-print inside TermElabM (we only want raw type and proof term now)
    let (tyStr, prStr) ← Command.liftTermElabM <|
      withOptions (fun _ => ppOpts) do
        let t ← tryPP (ppExprString ti.type) "<pp type failed>"
        let v ← tryPP (ppExprString ti.value) "<pp proof failed>"
        pure (t, v)

    -- Folder name = source "filename" (module path), sanitized into ONE folder name
    -- e.g. Mathlib/Topology/Basic -> Mathlib_Topology_Basic
    let moduleStr := m?.map modulePathString |>.getD "UnknownModule"
    let folder    := outRoot / sanitize moduleStr
    liftIO <| IO.FS.createDirAll folder

    -- File names = theorem name, split in two
    let thmStem := sanitize n.toString
    let fileStmt := folder / s!"{thmStem}_statement.txt"
    let fileProof := folder / s!"{thmStem}_proof_term.txt"

    -- Write each part separately
    liftIO <| IO.FS.writeFile fileStmt tyStr
    liftIO <| IO.FS.writeFile fileProof prStr

    -- Progress update (throttled)
    i := i + 1
    if showProg then
      if i % 100 == 0 || i == total then
        liftIO do
          h.putStr (renderBar i total 40 ++ "\r")
          h.flush

  if showProg then
    liftIO do
      h.putStr (renderBar total total 40 ++ "\n")
      h.flush
  logInfo m!"Done. Wrote {total} theorems into {outRoot}"

end ProofsDump
