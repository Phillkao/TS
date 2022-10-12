import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from data_preprocessing import *


def slope(df, label):
    fig, ax = plt.subplots()
    bins = np.linspace(df.min(), df.max(), 50)
    for ii in np.unique(label):
        subset = df[label == ii]
        ax.hist(subset, bins=bins, alpha=0.5, label=f"Cluster {ii}")
    ax.legend()
    plt.show()
    scatter = plt.scatter(range(0, len(label)), df, c=label, cmap='Set1', s=5)
    plt.legend(handles=scatter.legend_elements()[0],
               labels=['label0', 'label1', 'label2'])
    plt.show()
    unique, counts = np.unique(label, return_counts=True)
    print(dict(zip(unique, counts)))
    return dict(zip(unique, counts))


def data_gen(path, rolling_window_size=24, time_step=120, target_time_length=24, target_diff=False, add_lags=True,
             add_current=True):
    df = pd.read_csv(path)
    X_raw, Y_raw, group_index = preprocessing(df, rolling=rolling_window_size)
    if add_lags:
        X_raw, Y_raw = add_lag(df, X_raw, Y_raw, shift=5)
    if add_current:
        X_raw['Current'] = Y_raw['SUM_STD_24']

    features_name = list(X_raw.columns)

    y = []
    for idx, i in enumerate(range(0, len(X_raw) - time_step, time_step)):
        for t in Y_raw.columns[3:]:
            y.append(Y_raw[t][i:i + time_step][-target_time_length:].describe()[['mean', 'std']].values)
    target_name = ['PWR_PMF_mean', 'PWR_PMF_std',
                   'PWR_FINER_mean', 'PWR_FINER_std',
                   'PWR_mean', 'PWR_std']

    y = np.array(y).reshape((int(len(y) / 3), 6))
    y = pd.DataFrame(y, columns=target_name)
    select_len = len(y) * time_step
    X_time_step = X_raw[:select_len].values.reshape(int(len(X_raw[:select_len]) / time_step), time_step,
                                                    len(features_name))[:-1]

    X_rolling = []
    Y_rolling = []
    for idx, i in enumerate(range(0, len(X_raw) - time_step, 1)):
        X_rolling.append(X_raw[i:i + time_step].values)
        # Y_rolling.append(Y_raw['PMF_SEC_PWR+FINER1_SEC_PWR'][i:i+time_step].values.std())
        for t in Y_raw.columns[3:]:
            Y_rolling.append(Y_raw[t][i:i + time_step][-target_time_length:].describe()[['mean', 'std']].values)

    Y_rolling = np.array(Y_rolling).reshape((int(len(Y_rolling) / 3), 6))
    X_rolling = np.array(X_rolling)[:-1]  # size, timestep, features

    if target_diff:
        Y_time_step = y.diff().shift(-1)[:-1]
        Y_rolling = pd.DataFrame(Y_rolling, columns=target_name).diff().shift(-1)[:-1]
    else:
        Y_time_step = y[:-1]
        Y_rolling = pd.DataFrame(Y_rolling, columns=target_name)[:-1]

    return X_raw, Y_raw, X_time_step, Y_time_step, X_rolling, Y_rolling, features_name


def kmeans_label(target, n_clusters=3, mode='PWR_std', vis=False):
    tmp = KMeans(n_clusters=n_clusters)
    cat_name = mode + ('_cat%s' % n_clusters)
    target[cat_name] = tmp.fit_predict(target[mode].values.reshape(-1, 1))

    threshold = []
    for i in range(n_clusters):
        threshold.append(target[mode][target[cat_name] == i].min())
    threshold.sort(reverse=True)
    threshold.pop()

    if len(threshold) == 2:
        target[cat_name][(target[mode] > threshold[1]) & (target[mode] < threshold[0])] = 0
        target[cat_name][target[mode] < threshold[1]] = 1
        target[cat_name][target[mode] > threshold[0]] = 2
        cls_name = ['Mid', 'Low', 'High']
    elif len(threshold) == 1:
        target[cat_name][target[mode] > threshold[0]] = 0
        target[cat_name][target[mode] < threshold[1]] = 1
        cls_name = ['Abnormal', 'Norm']

    if vis:
        slope(target[mode].values.reshape(-1, 1), target[cat_name])

    return target, cls_name
