import sweetpea as sp
import pandas as pd
import random
import numpy as np


class Bush:
    def __init__(self, block):
        self.blocks = [block]
        self.dfs = []

    @property
    def current_block(self):
        return self.blocks[-1]

    @property
    def current_df(self):
        return self.dfs[-1] if self.dfs else None

    

    def random_simulate(self, n=1):
        experiments = sp.synthesize_trials(self.current_block, n)

        df_all = pd.DataFrame()
        for idx, experiment in enumerate(experiments):
            df = pd.DataFrame(experiment)
            df['subj_id'] = idx
            df['correct'] = [random.choice([1, 0]) for _ in range(len(df))]
            df['rt'] = np.random.normal(loc=500, scale=50, size=len(df))
            df_all = pd.concat([df_all, df], ignore_index=True)
        self.dfs.append(df_all)

    def get_experiments(self, n):
        sp.synthesize_trials(self.current_block, n)
