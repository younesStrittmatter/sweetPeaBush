from sweetpea.primitives import Factor, DerivedLevel, WithinTrial, Transition
from sweetpea.constraints import MinimumTrials, Exclude
from sweetpea import fully_cross_block, synthesize_trials_uniform, synthesize_trials, print_experiments, \
    save_experiments_csv, synthesize_trials_non_uniform, CrossBlock, MultiCrossBlock


# main function
def main():
    create_sequence_exp1(107)


# replicate counterbalance of experiment
def create_sequence_exp1(nr_sequences=1):
    positive_list = ['Happy', 'Good', 'Wonderful', 'Fantastic', 'Great', 'Nice', 'Pleasant', 'Amazing']
    negative_list = ['Awful', 'Terrible', 'Sick', 'Disgusting', 'Unpleasant', 'Horrible', 'Nasty', 'Sad']

    trialcode_list = ['attributeA', 'attributeB']
    stimulusitem1_list = positive_list + negative_list

    trialcode = Factor(name='trialcode', initial_levels=trialcode_list)
    stimulusitem1 = Factor(name='stimulusitem1', initial_levels=stimulusitem1_list)

    # add congruency as derived factora
    def is_positve(s):
        return s in positive_list

    def is_negative(s):
        return not is_positve(s)

    positive = DerivedLevel(name='positive', window=WithinTrial(predicate=is_positve, factors=[stimulusitem1]))
    negative = DerivedLevel(name='negative', window=WithinTrial(predicate=is_negative, factors=[stimulusitem1]))

    valence = Factor(name="valence", initial_levels=[positive, negative])

    # add transistions

    trial_constraints = MinimumTrials(trials=20)
    design = [trialcode, stimulusitem1, valence]
    crossing = [trialcode, valence]
    constraints = [trial_constraints]
    block = CrossBlock(design, crossing, constraints)
    experiments = synthesize_trials_uniform(block, nr_sequences)
    save_experiments_csv(block, experiments, file_prefix='out/exp1/sweetpea/sequence')


if __name__ == '__main__':
    main()
