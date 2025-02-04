import re
import os
import json
from tqdm import tqdm

from collections import defaultdict
from src.coqpyt_extension.proof_file_light import ProofFileLight

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

    result = defaultdict(lambda : {"proposition": [], "notations": [], "constants": [], "term": [], "states": [], "existentials": []})
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
                case "notations":
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
                case "constants":
                    if not "was not found" in message:
                        messages = [message]
                case "states":
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
    """"
    Extension of ProofFileLight to enable additionnal information retrieval (lambda-term, type, proof state etc.)
    """

    @staticmethod
    def _match_term_type(term: str):
        pattern = r'(Lemma|Theorem) (\S+) *(.*) *: *(.*?)(.*).'
        # Extract matchess
        match = re.search(pattern, term)
        if match:
            # TO DO, remove this check by improving the regex
            term_name = match.group(2)
            if term_name[-1] == ':':
                term_name = term_name[:-1]
            return match.group(1), term_name, match.group(3), match.group(4)
        return None

    @staticmethod
    def remove_comments(text: str):
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
    def remove_instr(text: str, instr_to_delete=['Print','Check', "Show"]):
        text_clean = text
        
        pattern_instr = f'({"|".join(instr_to_delete)})' + r' .*?\.'
        all_matches = list(re.finditer(pattern_instr, text_clean, flags=re.DOTALL))
        for match in reversed(all_matches):
            match_start = match.start(0)
            match_end = match.end(0)
            text_clean = text_clean[:match_start] + text_clean[match_end:]
        
        return text_clean
    
    def sanitize(self):
        aux_file = self._ProofFile__aux_file
        text = aux_file.read()
        text_sanitize = self.remove_comments(text)
        text_sanitize = self.remove_instr(text_sanitize)
        aux_file.write(text_sanitize)
    
    def get_all_terms(self):
        all_terms = {}
        for term in self.proofs:
            term_extract = self._match_term_type(term.step.short_text)
            if term_extract:
                _, term_name, _, _ = term_extract
                notations = extract_all_notation(term.ast.span)
                constants = extract_all_types(term.ast.span)
                for step in term.steps:
                    notations += extract_all_notation(step.ast.span)
                    constants += extract_all_types(step.ast.span)
                
                # hacky, extract_all should output sets directly
                notations = list(set(notations))
                constants = list(set(constants))
                all_terms[term_name] = {"name": term_name, "notations": notations, "constants": constants}
        pattern_lemma = r"(Lemma|Theorem) (\S+) *.*?[ \n]*:[ \n]*[\s\S]*?Proof.[ \n]*([\s\S]*?)(Qed\.)"
        aux_file = self._ProofFile__aux_file
        aux_saved = aux_file.read()
        all_matches = list(re.finditer(pattern_lemma, aux_saved))
        for idx, match in enumerate(all_matches):
            term_name = match.group(2)
            # TO DO, remove this check by improving the regex
            if term_name[-1] == ':':
                term_name = term_name[:-1]
            start_proof = match.start(3)
            end_proof = match.end(3)
            match_end = match.end(4)

            assert term_name in all_terms, f"Term {term_name} not found"

            all_terms[term_name]['start_proof'] = start_proof
            all_terms[term_name]['end_proof'] = end_proof
            all_terms[term_name]['match_end'] = match_end
            all_terms[term_name]['idx'] = idx
        return all_terms

    def _extract_annotations(self, term, do_notations=True, do_goals=True, do_existentials=True, do_constants=True):
        aux_file = self._ProofFile__aux_file
        aux_saved = aux_file.read()
        aux_archive = aux_saved
    
        all_prints = []
        all_checks = []
        all_goals = []
        all_existentials = []
        
        term_name = term['name']
        start_proof = term['start_proof']
        end_proof = term['end_proof']
        match_end = term['match_end']
        
        tactics = aux_saved[start_proof:end_proof].split('\n')
        tactics = [tactic for tactic in tactics if '.' in tactic]
        term['steps'] = tactics
        instr_notations = []
        instr_constants = []

        instr_term = f'Print {term_name}'
        instr_prop = f'Check {term_name}'
        for notation in term['notations']:
            instr_notations.append(f'Print "{notation}"')
        for constant in term['constants']:
            instr_constants.append(f'Check {constant}')
        instr_all = [instr_term]

        # add print instructions (in backward order)
        if do_notations:
            instr_all += instr_notations
            all_prints = [{"instr": instr, "label": term_name + '#notations'} for instr in instr_notations] + all_prints
        all_prints = [{"instr": instr_term, "label": term_name + '#term'}] + all_prints

        instr_all += [instr_prop]

        if do_constants:
            instr_all += instr_constants
            all_checks = [{"instr": instr, "label": term_name + '#constants'} for instr in instr_constants] + all_checks
        
        all_checks = [{"instr": instr_prop, "label": term_name + '#proposition'}] + all_checks
        if do_goals:
            all_goals = [{"instr": 'Show Proof', "label": term_name + '#states'} for _ in tactics] + all_goals
        if do_existentials:
            all_existentials = [{"instr": 'Show Existentials', "label": term_name + '#existentials'} for _ in tactics] + all_existentials
        # a bit annoying: to avoid collision when adding instruction I start appending instruction from the end
        # so to keep things in order, and since we need to associate instruction with line number, things should be done in some weird reverse order.
        
        instr_tot = "\n" + ".\n".join(instr_all) + '.\n'

        new_tactics = ""
        for tactic in tactics:
            if do_goals:
                new_tactics += "Show Proof.\n"
            if do_existentials:
                new_tactics += "Show Existentials.\n"
            new_tactics += tactic + '\n'
        aux_saved = aux_saved[:match_end] + instr_tot + aux_saved[match_end:]
        aux_saved = aux_saved[:start_proof] + new_tactics + aux_saved[end_proof:]
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
        result = get_queries_dict(aux_file, queries_dict)[term_name]
        aux_file.write(aux_archive)

        assert len(result['term'])==1 and len(result['proposition'])==1, "Issue with dict obtains from get_queries_dict, check if aux_saved contains only one Print of 'proposition' and/or 'term'"

        result['term'] = result['term'][0]
        result['proposition'] = result['proposition'][0]
        return result, aux_saved

        
    def extract_one_by_one(self, export_path, debug=False):
        # Sanitize file: remove all Print/Check/Show and comments to avoid conflict with regex
        self.sanitize()
        all_terms = self.get_all_terms()
        forbidden = set()
        done = set()

        forbidden_path = os.path.join(export_path, 'forbidden.json')
        done_path = os.path.join(export_path, "done.json")
        if os.path.exists(forbidden_path):
            with open(forbidden_path, 'r') as file:
                forbidden = set(json.load(file))
        if os.path.exists(done_path):
            with open(done_path, 'r') as file:
                done = set(json.load(file))
        
        for term_name, term in tqdm(list(all_terms.items())):
            if term_name in done or term_name in forbidden:
                continue
            forbidden.add(term_name)
            with open(forbidden_path, 'w') as file:
                json.dump(list(forbidden), file, indent=4)
            extract_term, aux_saved = self._extract_annotations(term)
            extract_term['steps'] = term["steps"]
            extract_term['name'] = term_name
            forbidden.discard(term_name)
            done.add(term_name)

            idx = term['idx']
            term_path = os.path.join(export_path, f"term_{idx}.json")
            with open(term_path, 'w') as file:
                json.dump(extract_term, file, indent=4)

            if debug:
                debug_path = os.path.join(export_path, f"term_{idx}_debug")
                with open(debug_path, 'w') as file:
                    file.write(aux_saved)
            with open(done_path, 'w') as file:
                json.dump(list(done), file, indent=4)
            with open(forbidden_path, 'w') as file:
                json.dump(list(forbidden), file, indent=4)