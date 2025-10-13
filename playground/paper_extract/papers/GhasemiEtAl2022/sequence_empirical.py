import pandas as pd


def main():
    get_sequences_exp1()


def get_sequences_exp1():
    df = pd.read_csv('res/osfstorage-archive/tidy_data1.csv')

    # extract only relevant columns from data
    df = df[['subj_id', 'sub_type', 'val', 'bel', 'instruction', 'conf']]

    df['val'] = df.apply(lambda row: set_val(row), axis=1)

    unique_subs = list(df['subj_id'].unique())
    k = 0

    for sub in unique_subs:
        df_subj = df[df['subj_id'] == sub]
        df_subj = df_subj[['sub_type', 'val', 'bel', 'instruction', 'conf']]
        df_subj.to_csv('./out/exp1/empirical/sequence_' + str(k) + '.csv', index=False)
        k += 1


def set_val(row):
    if row['val'] == 'PS': return 'Valid'
    if row['val'] == 'PW': return 'Invalid'
    return row['val']


if __name__ == '__main__':
    main()
