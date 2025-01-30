import re
from typing import Optional, Tuple, Union, List, Dict
import logging
import tempfile
from collections import defaultdict

from coqpyt.coq.structs import TermType, Step, Term, ProofStep, ProofTerm, Position
from coqpyt.coq.exceptions import *
from coqpyt.coq.changes import *
from coqpyt.coq.proof_file_light import ProofFileLight

from tqdm import tqdm


def extract_all_notation(data):
    result = []
    if isinstance(data, dict):
        if 'v' in data:
            result += extract_all_notation(data['v'])
        if 'expr' in data:
            result += extract_all_notation(data['expr'])
    elif isinstance(data, list):
        if len(data) > 2:
            if data[0] == 'CNotation':
                result.append(data[2][1])
        for entry in data:
            result += extract_all_notation(entry)
    return result

def extract_all_types(data):
    result = []
    if isinstance(data, dict):
        if 'v' in data:
            result += extract_all_types(data['v'])
        if 'expr' in data:
            result += extract_all_types(data['expr'])
    elif isinstance(data, list):
        if len(data) > 2:
            if data[0] == 'Ser_Qualid':
                result.append(data[2][1])
        for entry in data:
            result += extract_all_types(entry)
    return result

def get_queries_dict(aux_file, queries_dict, reduced_notation=True):
    uri = f"file://{aux_file.path}"
    if uri not in aux_file.coq_lsp_client.lsp_endpoint.diagnostics:
        return []

    result = defaultdict(lambda : {"proposition": [], "notation": [], "constant": [], "term": [], "state": [], "existentials": []})
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
        end_line = diagnostic.range.end.line
        if start_line in queries_dict:
            query = queries_dict[start_line]
            term_name, category = query['label'].split('#')
            messages = []
            message = diagnostic.message
            match category:
                case "notation":
                    if not "not a defined object" in message and "Notation" in message:
                        messages = message.split('Notation')[1:]
                        if reduced_notation:
                            messages = [messages[0]]
                case "term":
                    if not "not a defined object" in message:
                        messages = [message]
                case "proposition":
                    if not "was not found" in message:
                        messages = [message]
                case "constant":
                    if not "was not found" in message:
                        messages = [message]
                case "state":
                    if not "No goals to show" in message:
                        messages = [message]
                case "existentials":
                    if not "requires an open proof" in message:
                        messages = [message]
            if messages:
                result[term_name][category] += messages
    for term_name in result:
        for category in result[term_name]:
            result[term_name][category].reverse()
    return result


class ProofFileMod(ProofFileLight):
    """"Extension of ProofFileLight to enable additionnal information retrieval (lambda-term, type, proof state etc.)
    """

    @staticmethod
    def _match_term_name(term):
        pattern = r'(Lemma|Theorem) (\S+) *(.*) *: *(.*?)(.*).'
        # Extract matchess
        match = re.search(pattern, term)
        if match:
            return match.group(1), match.group(2), match.group(3), match.group(4)
        return None

    @staticmethod
    def remove_comments(text):
        cleaned = []
        i = 0
        n = len(text)
        stack_level = 0  # Tracks nesting level of comments

        while i < n:
            # Check for comment start '(*'
            if i + 1 < n and text[i] == '(' and text[i+1] == '*':
                stack_level += 1
                i += 2  # Skip the '(*'
                continue

            # Check for comment end '*)'
            if i + 1 < n and text[i] == '*' and text[i+1] == ')':
                if stack_level > 0:
                    stack_level -= 1
                    i += 2  # Skip the '*)'
                    continue
            # If not inside a comment, add the character to the result
            if stack_level == 0:
                cleaned.append(text[i])

            i += 1
        return ''.join(cleaned)

    @staticmethod 
    def remove_instr(text, instr_to_delete=['Print','Check', "Show"]):
        text_clean = text
        
        pattern_instr = f'({"|".join(instr_to_delete)}) .*?\.'
        all_matches = list(re.finditer(pattern_instr, text_clean, flags=re.DOTALL))
        for match in reversed(all_matches):
            match_start = match.start(0)
            match_end = match.end(0)
            text_clean = text_clean[:match_start] + text_clean[match_end:]
        
        return text_clean
    
    @classmethod
    def sanitize(cls, text):
        text_sanitize = cls.remove_comments(text)
        text_sanitize = cls.remove_instr(text_sanitize)
        return text_sanitize


    def _extract_all_terms_v2(self):
        all_terms = {}
        logging.info('Extract notations and constants')
        for term in self.proofs:
            term_extract = self._match_term_name(term.step.short_text)
            if term_extract:
                category, term_name, args, prop = term_extract
                notations = extract_all_notation(term.ast.span)
                constants = extract_all_types(term.ast.span)
                for step in term.steps:
                    notations += extract_all_notation(step.ast.span)
                    constants += extract_all_types(step.ast.span)
                
                # hacky, extract_all should output sets directly
                notations = list(set(notations))
                constants = list(set(constants))
                all_terms[term_name] = {"notations": notations, "constants": constants, "steps": [step.text for step in term.steps]}
        aux_file = self._ProofFile__aux_file
        aux_saved = aux_file.read()
        # Sanitize file: remove all Print/Check/Show and comments to avoid conflict with regex
        aux_saved = self.sanitize(aux_saved)
        pattern_lemma = "(Lemma|Theorem) (\S+) *.*[ \n]*:[ \n]*[\s\S]*?Proof.[ \n]*([\s\S]*?)(Qed\.)"
        all_matches = list(re.finditer(pattern_lemma, aux_saved))
        all_prints = []
        all_checks = []
        all_goals = []
        all_existentials = []
        for match in reversed(all_matches):
            term_name = match.group(2)
            beg_proof = match.start(3)
            end_proof = match.end(3)
            match_end = match.end(4)
            if term_name not in all_terms:
                print(all_terms.keys())
                raise Exception(f'Unfound term name, probably an issue with parser, look at {term_name} in {aux_file.path}')

            tactics = aux_saved[beg_proof:end_proof].split('\n')
            tactics = [tactic for tactic in tactics if '.' in tactic]
            term = all_terms[term_name]

            instr_prop = f'Check {term_name}'
            instr_term = f'Print {term_name}'
            instr_notations = []
            instr_constants = []
            instr_all = []
            for notation in term['notations']:
                instr_notations.append(f'Print "{notation}"')
            for constant in term['constants']:
                instr_constants.append(f'Check {constant}')

            all_prints = [{"instr": instr, "label": term_name + '#notation'} for instr in instr_notations] + all_prints
            all_prints = [{"instr": instr_term, "label": term_name + '#term'}] + all_prints
            
            all_checks = [{"instr": instr, "label": term_name + '#constant'} for instr in instr_constants] + all_checks
            all_checks = [{"instr": instr_prop, "label": term_name + '#proposition'}] + all_checks

            all_goals = [{"instr": 'Show Proof', "label": term_name + '#state'} for _ in tactics] + all_goals
            all_existentials = [{"instr": 'Show Existentials', "label": term_name + '#existentials'} for _ in tactics] + all_existentials
            # a bit annoying: to avoid collision when adding instruction I start appending instruction from the end
            # so to keep things in order, and since we need to associate instruction with line number, things should be done in some weird reverse order.
            instr_all += [instr_term]
            instr_all += instr_notations
            instr_all += [instr_prop]
            instr_all += instr_constants

            instr_tot = "\n" + ".\n".join(instr_all) + '.\n'

            new_tactics = ""
            for tactic in tactics:
                new_tactics += "Show Proof.\n"
                new_tactics += "Show Existentials.\n"
                new_tactics += tactic + '\n'
            aux_saved = aux_saved[:match_end] + instr_tot + aux_saved[match_end:]
            aux_saved = aux_saved[:beg_proof] + new_tactics + aux_saved[end_proof:]
        aux_lines = aux_saved.split('\n')
        idx_print = 0
        idx_check = 0
        idx_goals = 0
        idx_existentials = 0
        queries_dict = {}
        for num_line, line in enumerate(aux_lines):
            if line.startswith('Print'):
                queries_dict[num_line] = all_prints[idx_print]
                idx_print += 1
            if line.startswith('Check'):
                queries_dict[num_line] = all_checks[idx_check]
                idx_check += 1
            if line.startswith('Show Proof'):
                queries_dict[num_line] = all_goals[idx_goals]
                idx_goals += 1
            if line.startswith('Show Existentials'):
                queries_dict[num_line] = all_existentials[idx_existentials]
                idx_existentials += 1
        aux_file = self._ProofFile__aux_file
        aux_file.write(aux_saved)
        aux_file.didChange()
        result = get_queries_dict(aux_file, queries_dict)
        for entry in result:
            result[entry]['steps'] = all_terms[entry]["steps"]
        return result