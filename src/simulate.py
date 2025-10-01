import sweetpea as sp
import pandas as pd


def create_experiments():  # Placeholder for any setup code if needed

    word = sp.Factor("word", [sp.Level("red",2), "blue"])
    color = sp.Factor("color", ["red", sp.Level("blue",3)])

    def is_congruent(color, word):
        return color == word

    def is_incongruent(color, word):
        return color != word

    def is_switch(color):
        return color[0] != color[-1]

    def is_repeat(color):
        return color[0] == color[-1]

    def is_lagging(color):
        return color[0] != color[-2]

    def is_nonlagging(color):
        return color[0] == color[-2]

    congruent = sp.DerivedLevel('con', sp.WithinTrial(is_congruent, [color, word]))
    incongruent = sp.DerivedLevel('in_congruent', sp.WithinTrial(is_incongruent, [color, word]))

    switch = sp.DerivedLevel('switch', sp.Transition(is_switch, [color]))
    repeat = sp.DerivedLevel('repeat', sp.Transition(is_repeat, [color]))

    transition = sp.Factor('transition', [switch, repeat])

    lag = sp.DerivedLevel('lag', sp.Window(is_lagging, [color], 3, 1))
    non_lag = sp.DerivedLevel('nonlag', sp.Window(is_nonlagging, [color], 3, 1))

    lagging = sp.Factor('lagging', [lag, non_lag])

    congruency = sp.Factor('congruency', [congruent, incongruent])

    design = [word, color, congruency, transition, lagging]
    crossing = [word, color]
    constraints = []

    block = sp.CrossBlock(design=design, crossing=crossing, constraints=constraints)


    experiment = sp.synthesize_trials(block, 1)

    for e in experiment:
        df_t = pd.DataFrame(e)
        df_t.to_csv("output.csv", index=False)


if __name__ == "__main__":
    create_experiments()
