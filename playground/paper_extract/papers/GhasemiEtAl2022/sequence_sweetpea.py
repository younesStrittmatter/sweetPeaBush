from sweetpea.primitives import Factor, DerivedLevel, WithinTrial, Transition
from sweetpea.constraints import MinimumTrials, Exclude
from sweetpea import fully_cross_block, synthesize_trials_uniform, synthesize_trials, print_experiments, \
    save_experiments_csv, synthesize_trials_non_uniform, CrossBlock, MultiCrossBlock


# main function
def main():
    create_sequence_exp1(113)


# replicate counterbalance of experiment
def create_sequence_exp1(nr_sequences=1):
    # 'sub_type', 'val', 'bel', 'instruction', 'conf'
    sub_type_list = ['MP', 'AC']
    val_list = ['Valid', 'Invalid']
    bel_list = ['Believable', 'Unbelievable']
    instruction_list = ['Logic', 'Belief']

    sub_type = Factor(name='sub_type', initial_levels=sub_type_list)
    val = Factor(name='val', initial_levels=val_list)
    bel = Factor(name='bel', initial_levels=bel_list)
    instruction = Factor(name='instruction', initial_levels=instruction_list)

    # add congruency as derived factora
    def is_conflict(v, b):
        return (v == 'Valid' and b == 'Unbelievable') or (v == 'Invalid' and b == 'Believable')

    def is_no_conflict(v, b):
        return not is_conflict(v, b)

    conflict = DerivedLevel(name='conflict', window=WithinTrial(predicate=is_conflict, factors=[val, bel]))
    non_conflict = DerivedLevel(name='non_conflict', window=WithinTrial(predicate=is_no_conflict, factors=[val, bel]))

    conf = Factor(name="conf", initial_levels=[conflict, non_conflict])

    # add transistions

    trial_constraints = MinimumTrials(trials=66)
    design = [sub_type, val, bel, instruction, conf]
    crossing = [sub_type, val, bel, instruction]
    constraints = [trial_constraints]
    block = CrossBlock(design, crossing, constraints)
    experiments = synthesize_trials_uniform(block, nr_sequences)
    save_experiments_csv(block, experiments, file_prefix='out/exp1/sweetpea/sequence')


if __name__ == '__main__':
    main()
