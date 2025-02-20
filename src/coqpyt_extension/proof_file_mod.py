import re
import os
import json
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)

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

    result = defaultdict(lambda : {"proposition": [], "notations": [], "constants": [], "term": []})
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
    def _admitted_all_before(source: str):
        pass

    @staticmethod
    def _match_term_type(term: str):
        pattern = r'(Lemma|Theorem) (\S+?)[:\ \n]'
        # Extract matches
        match = re.search(pattern, term)
        if match:
            return match.group(1), match.group(2)
        return None

    def get_all_terms(self):
        all_terms = []
        for idx, term in enumerate(self.proofs):
            term_extract = self._match_term_type(term.step.short_text)
            proposition = term.step.short_text
            steps = []
            for step in term.steps:
                range_start = step.step.ast.range.start
                range_end = step.step.ast.range.end
                instr = step.step.short_text
                steps.append((instr, range_start.line, range_end.line))
            
            if 'Admitted' in steps[-1][0] or 'Abort' in steps[-1][0]:
                continue
            if term_extract:
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
            all_prints = [{"instr": instr, "label": term_name + '#notations'} for instr in instr_notations] + all_prints
        all_prints = [{"instr": instr_term, "label": term_name + '#term'}] + all_prints

        if do_constants:
            instr_all += instr_constants
            all_checks = [{"instr": instr, "label": term_name + '#constants'} for instr in instr_constants] + all_checks
        
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
        if not one_prop_one_term or True:
            with open('debug.v', 'w') as file_io:
                file_io.write(aux_saved)
        assert one_prop_one_term, "Issue with dict obtains from get_queries_dict, check if debug.v contains one Print of 'term'"

        result['term'] = result['term'][0]
        term.update(result)
        return term
     
    def extract_one_by_one(self, export_path):
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
        
        source_str = self._ProofFile__aux_file.read()
        source_path = os.path.join(export_path, "source.v")
        with open(source_path, 'w') as file:
            file.write(source_str)
        for term in tqdm(all_terms):
            term_name = term['name']
            if term_name in done or term_name in forbidden:
                continue
            forbidden.add(term_name)
            with open(forbidden_path, 'w') as file:
                json.dump(list(forbidden), file, indent=4)
            extract_term = self._extract_annotations(term)
            extract_term['steps'] = term["steps"]
            extract_term['name'] = term_name
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