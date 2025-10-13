import sweetpea as sp

words = ['red', 'green', 'blue', 'yellow']
colors = ['red', 'green']


def get_ground_experiment():
    word_f = sp.Factor('word', words)
    color_f = sp.Factor('color', colors)

    design = [word_f, color_f]
    crossing = [word_f, color_f]
    constraints = []

    block = sp.CrossBlock(design=design, crossing=crossing, constraints=constraints)

    return block
