import re
import os
import json
import copy
from collections import defaultdict
from typing import Any, List, Dict, Optional, Tuple
import logging
logger = logging.getLogger(__name__)

from coqpyt.coq.structs import TermType
from coqpyt.coq.proof_file import _AuxFile
from src.coqpyt_extension.proof_file_light import ProofFileLight

def extract_all_notation(ast: Dict) -> List[str]:
    """
    Recursively extracts all notations from the given Rocq AST.

    Args:
        ast: A nested data structure (dict or list) representing parts of a Rocq AST.

    Returns:
        A list of notation strings extracted from the ast.
    """
    result = []
    if isinstance(ast, dict):
        if 'v' in ast:
            result += extract_all_notation(ast['v'])
        if 'expr' in ast:
            result += extract_all_notation(ast['expr'])
    elif isinstance(ast, list):
        if len(ast) > 2:
            if ast[0] == 'CNotation':
                result.append(ast[2][1])
        for entry in ast:
            result += extract_all_notation(entry)
    return result

def extract_all_types(ast: Dict) -> List[str]:
    """
    Recursively extracts all types from the given Rocq AST.

    Args:
        ast: A nested data structure (dict or list) representing parts of a Rocq AST.

    Returns:
        A list of type identifier strings extracted from the ast.
    """
    result = []
    if isinstance(ast, dict):
        if 'v' in ast:
            result += extract_all_types(ast['v'])
        if 'expr' in ast:
            result += extract_all_types(ast['expr'])
    elif isinstance(ast, list):
        if len(ast) > 2:
            if ast[0] == 'Ser_Qualid':
                result.append(ast[2][1])
        for entry in ast:
            result += extract_all_types(entry)
    return result

def get_queries_dict(
    aux_file: _AuxFile,
    queries_dict: Dict[int, Dict[str, str]],
    reduced_notation: bool = True
) -> Dict[str, Dict[str, List[str]]]:
    """
    Retrieves queries and diagnostic messages from the auxiliary file.

    Args:
        aux_file: An auxiliary file object given by CoqPyt.
        queries_dict: A dictionary mapping starting line numbers to query (a dict containing 'term_name', and 'category' keys).
        reduced_notation: If True, limits the extraction of notations to a reduced set.

    Returns:
        A dictionary mapping term names to a dictionary of message lists for each category:
        'notations', 'constants', and 'term'.
    """
    uri = f"file://{aux_file.path}"
    if uri not in aux_file.coq_lsp_client.lsp_endpoint.diagnostics:
        return []

    result = defaultdict(lambda : {"notations": [], "constants": [], "term": []})
    lines = aux_file.read().split("\n")
    for diagnostic in aux_file.coq_lsp_client.lsp_endpoint.diagnostics[uri]:
        command = lines[diagnostic.range.start.line : diagnostic.range.end.line + 1]
        if len(command) == 0:
            continue
        elif len(command) == 1:
            command[0] = command[0][
                diagnostic.range.start.character : diagnostic.range.end.character
                + 1
            ]
        else:
            command[0] = command[0][diagnostic.range.start.character :]
            command[-1] = command[-1][: diagnostic.range.end.character + 1]
        command = "".join(command).strip()

        start_line = diagnostic.range.start.line
        if start_line in queries_dict:
            query = queries_dict[start_line]
            term_name, category = query['term_name'], query['category']
            messages = []
            message = diagnostic.message
            match category:
                case "notations":
                    if not "not a defined object" in message and "Notation" in message:
                        messages = message.split('Notation')[1:]
                        if reduced_notation:
                            messages = [messages[0]]
                case "term":
                    if not "not a defined object" in message and 'Syntax error:' not in message:
                        messages = [message]
                case "constants":
                    if not "was not found" in message:
                        messages = [message]
            if messages:
                result[term_name][category] += messages
    for term_name in result:
        for category in result[term_name]:
            result[term_name][category].reverse()
    return result


class ProofFileMod(ProofFileLight):
    """"
    Extension of ProofFileLight to enable additionnal information retrieval (lambda-term, type, proof state etc.) from a Rocq file.
    """

    @staticmethod
    def _match_term_kind(term: str) -> Tuple[str, str]:
        """
        Extracts the term kind (lemma|theorem) and name from the given term string.
        """
        pattern = r'(Lemma|Theorem|Fact|Corollary|Proposition|Property) (\S+?)[:\ \n]'
        # Extract matches
        match = re.search(pattern, term)
        if match:
            return match.group(1), match.group(2)
        return "", ""

    def get_all_notations(self):
        notations_templates = {}
        pattern = r'Notation "?([^"]+)"? *:='
        submatch = r"'(\S*)'"
        for key in self.context.terms:
            term = self.context.terms[key]
            # HACK: for the moment, we ignore inductive notation, and replace re.fullmatch by re.match
            if term.type == TermType.NOTATION and 'Notation' in term.text:
                
                match = re.search(pattern, term.text)
                assert match, f'Notation is not parsable, look at {term.text}'
                notation_content = match.group(1)
                notation_content = re.sub(submatch, r'\1', notation_content)
                notations_templates[notation_content] = term.text
        return notations_templates

    def get_all_terms(self) -> List[Dict[str, Any]]:
        """
        Retrieves all valid terms from the current file.

        Iterates over proofs, extracting the term name, proposition, steps,
        notations, and constants. Proofs ending with 'Admitted' or 'Abort' are skipped.

        Returns:
            A list of dictionaries, each containing details of a term including:
                - name: The term's name.
                - proposition: The proposition text.
                - steps: A list of tuples (instruction, start_line, end_line) for each step.
                - notations: A list of notation.
                - constants: A list of type/constant.
                - idx: The index of the term in the proofs list.
        """
        all_terms = []
        for idx, term in enumerate(self.proofs):
            term_extract = self._match_term_kind(term.step.short_text)
            proposition = term.step.short_text
            steps = []
            for step in term.steps:
                range_start = step.step.ast.range.start
                range_end = step.step.ast.range.end
                instr = step.step.short_text
                steps.append((instr, range_start.line, range_end.line))
            
            if 'Admitted' in steps[-1][0] or 'Abort' in steps[-1][0]:
                continue
            if term_extract[0]:
                _, term_name = term_extract
                notations = extract_all_notation(term.ast.span)
                constants = extract_all_types(term.ast.span)
                for step in term.steps:
                    notations += extract_all_notation(step.ast.span)
                    constants += extract_all_types(step.ast.span)
                
                # hacky, extract_all should output sets directly
                notations = list(set(notations))
                constants = list(set(constants))
                all_terms.append({"name": term_name, "proposition": proposition, "steps": steps, "notations": notations, "constants": constants, "idx": idx})
        return all_terms

    def _extract_annotations(self, term, do_notations=True, do_constants=True):
        """
        Extracts information (term, notations, constants) for a given term.

        This method writes additional Rocq commands (Print and Check) into the auxiliary file to obtain annotations for the term's notations, constants, and the term itself.
        It then reads back diagnostics to extract the corresponding messages and updates the term dictionary.

        Args:
            term: A dictionary containing details of the term (obtained through get_all_terms).
            do_notations: Whether to process notations.
            do_constants: Whether to process constants.

        Returns:
            A copy of the term dictionary with added annotation fields.
        """
        aux_file = self._ProofFile__aux_file
        aux_saved = aux_file.read()
        aux_archive = aux_saved

        all_prints = []
        all_checks = []
        term_name = term['name']
        match_end = term['steps'][-1][2]
        
        instr_notations = []
        instr_constants = []
        instr_term = f'Print {term_name}.'

        for notation in term['notations']:
            instr_notations.append(f'Print "{notation}".')
        for constant in term['constants']:
            instr_constants.append(f'Check {constant}.')
        instr_all = [instr_term]
        
        if do_notations:
            instr_all += instr_notations
            all_prints = [{"instr": instr, "term_name": term_name, "category": 'notations'} for instr in instr_notations] + all_prints
        all_prints = [{"instr": instr_term, "term_name": term_name, "category": 'term'}] + all_prints

        if do_constants:
            instr_all += instr_constants
            all_checks = [{"instr": instr, "term_name": term_name, "category": 'constants'} for instr in instr_constants] + all_checks
        
        # a bit annoying: to avoid collision when adding instruction I start appending instruction from the end
        # so to keep things in order, and since we need to associate instruction with line number, things should be done in some weird reverse order.
        queries_dict = {}
        aux_saved_lines = aux_saved.splitlines()
        aux_saved_lines = aux_saved_lines[:match_end+1] + instr_all + aux_saved_lines[match_end+1:]
        aux_saved = "\n".join(aux_saved_lines)

        for k, instr in enumerate(all_prints + all_checks, start=1):
            queries_dict[k + match_end] = instr
        
        aux_file = self._ProofFile__aux_file
        aux_file.write(aux_saved)
        aux_file.didChange()
        result = get_queries_dict(aux_file, queries_dict)[term_name]
        
        aux_file.write(aux_archive)

        one_prop_one_term = len(result['term'])==1
        if not one_prop_one_term:
            with open('debug.v', 'w') as file_io:
                file_io.write(aux_saved)
        assert one_prop_one_term, "Issue with dict obtains from get_queries_dict, check if debug.v contains one Print of 'term'"

        result['term'] = result['term'][0]
        term = copy.deepcopy(term)
        term.update(result)
        return term
     
    def extract_one_by_one(self, export_path, metadata={}):
        """
        Extracts terms one by one and exports their details to JSON files.

        Iterates through all extracted terms, skips those already processed or forbidden,
        and saves each term's annotations and details in separate JSON files.
        It also maintains records of processed terms in 'done.json' and forbidden terms in 'forbidden.json',
        and writes the Rocq source file to 'source.v'.

        Args:
            export_path: The directory where the export files (JSON and source file) will be saved.

        Returns:
            None.
        """
        all_terms = self.get_all_terms()
        forbidden = set()
        done = set()

        forbidden_path = os.path.join(export_path, 'forbidden.json')
        done_path = os.path.join(export_path, "done.json")
        # path are stored as absolute path in class attributes, it was easier to give them again as parameters

        if os.path.exists(forbidden_path):
            with open(forbidden_path, 'r') as file:
                forbidden = set(json.load(file))
        if os.path.exists(done_path):
            with open(done_path, 'r') as file:
                done = set(json.load(file))
            
        
    
        for term in all_terms:
            term_name = term['name']
            if term_name in done or term_name in forbidden:
                continue
            forbidden.add(term_name)
            with open(forbidden_path, 'w') as file:
                json.dump(list(forbidden), file, indent=4)
            extract_term = self._extract_annotations(term)
            extract_term['steps'] = [s for s,_,_ in term["steps"] if 'Proof.' not in s and 'Qed.' not in s]
            extract_term['name'] = term_name
            extract_term['category'] = export_path.split('/')[-1]
            extract_term.update(metadata)
            forbidden.discard(term_name)
            done.add(term_name)

            idx = term['idx']
            term_path = os.path.join(export_path, f"term_{idx}.json")
            with open(term_path, 'w') as file:
                json.dump(extract_term, file, indent=4)
            with open(done_path, 'w') as file:
                json.dump(list(done), file, indent=4)
            with open(forbidden_path, 'w') as file:
                json.dump(list(forbidden), file, indent=4)