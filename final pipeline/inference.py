import numpy as np
import pandas as pd
import argparse
import pickle

parser = argparse.ArgumentParser(description='get dataset file')
parser.add_argument('--train_ds', type=str, default='data/CAE_dataset.csv',
                    help='.')
parser.add_argument('--test_ds', type=str, default='data/CAE_test_dataset.csv',
                    help='..')
args = parser.parse_args()

def clean_nans(df):
    a = df.values
    b = np.argwhere(np.isnan(a))
    list_non_repeating = []
    for e in b:
        if e[0] not in list_non_repeating:
            list_non_repeating.append(e[0])
    df = df.drop(list_non_repeating)
    return df


def get_split(df):
    '''Big method to get pilot split'''
    def get_features(df):
        return [np.array(df.iloc[:, n]) for n in range(1, 12)]

    def list_of_indexes(df):
        pilot_id = np.array(df.iloc[:, -1])
        x0 = pilot_id[0]  # pilot id is the current
        x = [[0, pilot_id[0]]]
        for i in range(len(pilot_id)):
            if pilot_id[i] != x0:
                x.append([i, pilot_id[i]])
                x0 = pilot_id[i]
        # find the number of ids that were counted twice
        # len(x)
        count = 0
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                if x[i][1] == x[j][1]:
                    # print(i,":",x[i],"\n",j,":",x[j])
                    count += 1
        print("count", count)

        return x, count


    def get_features_by_test_run(df):
        features = np.transpose([np.array(df.iloc[:, n]) for n in range(1, 13)])

        indexes, count = list_of_indexes(df)
        features_by_run = []

    features = np.transpose([np.array(df.iloc[:, n]) for n in range(1, 13)])
    indexes, count = list_of_indexes(df)
    indexes = [i[0] for i in indexes]

    features_by_run = []
    j = 0

    for i in range(len(features)):

        if i == 0:
            test_run = [features[i]]
        elif features[i][-1] != features[i - 1][-1]:
            features_by_run.append(test_run)
            test_run = [features[i]]
        else:
            test_run.append(features[i])
        # if i%1000==0:#trace
        #   print(i)#trace

    features = get_features(df)
    x, count = list_of_indexes(df)
    features_by_run.append(test_run)
    feat = features_by_run

    defective_pilot = []
    good_pilot = []

    for i in features_by_run:
        if i[0][-2] == 0:
            good_pilot.append(i)
        elif i[0][-2] == 1:
            defective_pilot.append(i)
        else:
            raise Exception
    f = open("yeet_def.pickle", "wb")
    pickle.dump(defective_pilot, f)
    f.close()
    f = open("yeet_good.pickle", "wb")
    pickle.dump(good_pilot, f)
    f.close()


train_df = pd.read_csv(args.train_ds)
train_df = clean_nans(train_df)
get_split(train_df)