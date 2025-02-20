from pytanque import Pytanque, PetanqueError

def eval_tactics(thm, filepath, tactics, url="127.0.0.1", port=8765):
    with Pytanque(url, port) as pet:
        try:
            state = pet.start(file=filepath, thm=thm)
        except PetanqueError as e:
            return [], [{"status": "error", "goals": [], "message": e.message, "tactic": ""}]
        init_goals = [goal.pp for goal in pet.goals(state)]  
        res = []
        for tactic in tactics:
            entry = {"status": "", "goals": [], "message": "", "tactic": tactic}
            try:
                state = pet.run_tac(state, tactic, verbose=False)
                goals = pet.goals(state)

                entry['goals'] = [goal.pp for goal in goals]                
                if state.proof_finished:
                    entry['status'] = "finish"
                else:
                    entry['status'] = "ongoing"
            except PetanqueError as e:
                entry['status'] = "error"
                entry['message'] = e.message
            res.append(entry)
        return init_goals, res