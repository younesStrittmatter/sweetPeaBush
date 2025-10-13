import sweetpea as sp
from ground_paradigm import get_ground_experiment


def main():
    block = get_ground_experiment()
    experiment = sp.synthesize_trials(block, 1)[0]

    import random
    import pandas as pd
    import numpy as np
    df = pd.DataFrame(experiment)
    df['correct'] = [random.choice([1, 0]) for _ in range(len(df))]
    df['rt'] = np.random.normal(loc=500, scale=50, size=len(df))
    df.to_csv("output.csv", index=False)






if __name__ == "__main__":
    main()