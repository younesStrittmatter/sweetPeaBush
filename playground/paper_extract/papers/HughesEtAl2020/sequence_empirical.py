import pandas as pd


def main():
    get_sequences_exp1()


def get_sequences_exp1():
    df = pd.read_excel('res/shared features - measures, data and code/experiment 1/data/raw/iat.xlsx')

    positive_list = ['Happy', 'Good', 'Wonderful', 'Fantastic', 'Great', 'Nice', 'Pleasant', 'Amazing']
    negative_list = ['Awful', 'Terrible', 'Sick', 'Disgusting', 'Unpleasant', 'Horrible', 'Nasty', 'Sad']

    def set_val(row):
        if row['stimulusitem1'] in positive_list: return 'positive'
        return 'negative'

    # extract only relevant columns from data
    df = df[df['blockcode'] == 'attributepractice']
    unique_subs = list(df['subject'].unique())
    k = 0

    for sub in unique_subs:
        df_subj = df[df['subject'] == sub]
        unique_blocks = list(df_subj['blocknum'].unique())
        for block in unique_blocks:
            df_block = df_subj[df_subj['blocknum'] == block]
            df_block = df_block[['trialcode', 'stimulusitem1']]
            df_block['valence'] = df_block.apply(lambda row: set_val(row), axis=1)
            df_block.to_csv('./out/exp1/empirical/sequence_' + str(k) + '.csv', index=False)
            k += 1


if __name__ == '__main__':
    main()
