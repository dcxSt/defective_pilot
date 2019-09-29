import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from preprocessing_helper import get_features_map
import pandas as pd
import umap

parser = argparse.ArgumentParser(description='get dataset file')
parser.add_argument('--train_ds', type=str, default='data/CAE_dataset.csv',
                    help='.')
parser.add_argument('--test_ds', type=str, default='data/CAE_test_dataset.csv',
                    help='..')
args = parser.parse_args()


def remove_wrong_points(X, y):
    X_zeros = []
    for i in range(len(y)):
        if y[i] == 0:
            X_zeros.append(np.append(X[i], i))
    X_zeros = np.array(X_zeros)
    #X_zeros[:,:-1]
    Kmean = KMeans(n_clusters=1).fit(X_zeros[:,:-1])
    center = Kmean.cluster_centers_
    center.reshape(81)
    distance = []
    for zeros in X_zeros:
        dist = np.linalg.norm(zeros[:-1] - center)
        distance.append((dist,zeros[-1]))
    distance = sorted(distance, key=lambda a_entry: a_entry[0])
    wrong_points = distance[int(0.8*len(distance)):]
    wrong_points = np.array(wrong_points)
    wrong_points = wrong_points.transpose()
    #new_X = np.delete(X, wrong_points[1],0)
    #new_y = np.delete(y, wrong_points[1],0)
    #np.save("y_new.npy", new_y)
    #np.save("x_new.npy", new_X)
    for i in wrong_points[1]:
        y[int(i)] = 1
    return X, y


def graph_embedding(X_emb_1, X_emb_0):
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(1, 1, 1)
    # ones are in red
    ax.scatter(X_emb_1[:,0], X_emb_1[:,1], c='red', s=5)
    # zeros are in blue
    ax.scatter(X_emb_0[:,0], X_emb_0[:,1], c='blue', s=5)
    plt.show()


def clean_nans(df):
    a = df.values
    b = np.argwhere(np.isnan(a))
    list_non_repeating = []
    for e in b:
        if e[0] not in list_non_repeating:
            list_non_repeating.append(e[0])
    df = df.drop(list_non_repeating)
    return df


def get_split(df, test=False):
    '''Big method to get pilot split'''
    def get_features(df):
        if test:
            return [np.array(df.iloc[:, n]) for n in range(1, 11)]
        return [np.array(df.iloc[:, n]) for n in range(1, 12)]

    outter_pilot_id = None
    def list_of_indexes(df):
        pilot_id = np.array(df.iloc[:, -1])
        outter_pilot_id = pilot_id
        x0 = pilot_id[0]  # pilot id is the current
        x = [[0, pilot_id[0]]]
        # x is a list where each element is a list with two elements, [the , the pilot id]
        for i in range(len(pilot_id)):
            if pilot_id[i] != x0:
                x.append([i, pilot_id[i]])
                x0 = pilot_id[i]
        # find the number of ids that were counted twice
        # len(x)
        count = 0
        # this check if there are duplicates
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                if x[i][1] == x[j][1]:
                    # print(i,":",x[i],"\n",j,":",x[j])
                    count += 1

        return x, count


    def get_features_by_test_run(df):
        if test:
            features = np.transpose([np.array(df.iloc[:, n]) for n in range(1, 12)])
        else:
            features = np.transpose([np.array(df.iloc[:, n]) for n in range(1, 13)])
        indexes, count = list_of_indexes(df)
        features_by_run = []
    if test:
        features = np.transpose([np.array(df.iloc[:, n]) for n in range(1, 12)])
    else:
        features = np.transpose([np.array(df.iloc[:, n]) for n in range(1, 13)])
    indexes, count = list_of_indexes(df)
    real_pilot_id = [i[1] for i in indexes]
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
    if not test:
        for i in features_by_run:
            if i[0][-2] == 0:
                good_pilot.append(i)
            elif i[0][-2] == 1:
                defective_pilot.append(i)
            else:
                raise Exception
        return defective_pilot, good_pilot
    else:
        return features_by_run, real_pilot_id

def list_to_np(X_good, X_def):
    '''converts Steve's weird lists to normal human-readable formats'''
    # GOOD
    newlist = list()
    for i in range(len(X_good)):
        if len(X_good[i]) > 600:
            newlist.append(X_good[i][:600])
    X_good = np.array(newlist)[:, :, :10]
    y_good = np.zeros(len(X_good))

    # NOT SO GOOD
    newlist = list()
    for i in range(len(X_def)):
        if len(X_def[i]) > 600:
            newlist.append(X_def[i][:600])
    X_def = np.array(newlist)[:, :, :10]
    y_def = np.ones(len(X_def))

    X = np.concatenate([X_good, X_def], axis=0)
    y = np.concatenate([y_good, y_def], axis=0)

    np.save('data/X_raw.npy', X)
    np.save('data/y.npy', y)
    return X, y


def load_data():
    train_df = pd.read_csv(args.train_ds)
    train_df = clean_nans(train_df)
    defective_pilot, good_pilot = get_split(train_df)

    # get X and y time series, (n_samples, 600, 10)
    X, y = list_to_np(good_pilot, defective_pilot)
    return X, y


if __name__ == '__main__':
    X, y = load_data()
    # pull out dank features
    #X = get_features_map('data/X_raw.npy', pickle=True)
    X = np.load('data/x_feature_arima.npy')

    ### TRAINING ###
    X, y = remove_wrong_points(X, y)

    pca = PCA(n_components=15)
    X = pca.fit_transform(X, y)
    reducer = umap.UMAP()

    def repeated_umap(X, i):
        embeddings = []
        for e in range(i):
            X_emb = reducer.fit_transform(X)
            X_emb_1 = X_emb[np.argwhere(y == 1).reshape(np.argwhere(y == 1).shape[0])]
            X_emb_0 = X_emb[np.argwhere(y == 0).reshape(np.argwhere(y == 0).shape[0])]
            graph_embedding(X_emb_1, X_emb_0)
            embeddings.append(X_emb)
        embeddings = np.concatenate(embeddings, axis=1)
        X = np.concatenate([X, embeddings], axis=1)
        return X

    X = repeated_umap(X, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    model = LogisticRegression(solver='saga')
    model.fit(X_train, y_train)
    pred = model.predict(X_test)


    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    print('Accuracy score: {}\t F1 score: {}.'.format(acc, f1))

    ## done training, now streamlining testing process
    test_df = pd.read_csv(args.test_ds)
    test_df = clean_nans(test_df)
    features_by_run, pilot_id = get_split(test_df, test=True)
    print(len(pilot_id))
    #MANUAL shtuff
    newlist = list()
    new_pilot_list = list()
    for i in range(len(features_by_run)):
        if len(features_by_run[i]) > 600:
            newlist.append(features_by_run[i][:600])
            new_pilot_list.append(pilot_id[i])
    XX = np.array(newlist)[:, :, :10]
    np.save('data/XX.npy', XX)
    XX = get_features_map('data/XX.npy', pickle=False)
    XX = pca.transform(XX)
    XX_emb = reducer.transform(XX)
    XX = np.concatenate([XX, XX_emb], axis=1)
    pred = model.predict(XX)
    print()
    print()
    print(len(pred), len(new_pilot_list))
    import csv

    csvData = [['Pilot ID', 'Flag']]
    for i in range(len(pred)):
        csvData.append([new_pilot_list[i],pred[i]])

    with open('submission.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvData)
    csvFile.close()

