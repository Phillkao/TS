def norm(x):
    return (x - x.min()) / (x.max() - x.min())


def corr_selection(df, col_list, feature_list, group_index):
    counter = 1

    while col_list:
        col_list.sort()
        col_name = col_list[0]
        corr_map = df[col_list].corr()
        group_list = list(corr_map[col_name][corr_map[col_name] >= 0.8].index)
        group_name = col_name + '_G' + str(counter)
        feature_list.append(group_name)
        group_index[group_name] = group_list

        df[group_name] = df[group_list].T.mean()
        col_list = list(set(col_list) - set(group_list))
        counter += 1
        feature_list.sort()

    return df, feature_list, group_index


def preprocessing(df, func_list=['AMPS', 'OHMS', 'VOLT', 'BRNR', 'PMB', 'PMC', 'AI'], corr_select=True, rolling=24,
                  rolling_method='top'):
    # Features
    raw_feature_list = list(df.columns)
    feature_list = []
    group_index = {}
    if corr_select:
        for func_name in func_list:
            cols = [col for col in df.columns if func_name in col]
            cols.sort()
            df, feature_list, group_index = corr_selection(df, cols, feature_list, group_index)
            raw_feature_list = list(set(raw_feature_list) - set(cols))
            raw_feature_list.sort()

        feature_list.extend(raw_feature_list)
        feature_list.remove('Date')
    else:
        for func_name in func_list:
            cols = [col for col in df.columns if func_name in col]
            cols.sort()
            feature_list.extend(cols)
            raw_feature_list = list(set(raw_feature_list) - set(cols))
            raw_feature_list.sort()

    # Target
    target_cols = [col for col in df.columns if 'SEC_PWR' in col]
    selected_target_list = []
    for target_col in target_cols:
        if '+' in target_col:
            df['SUM_STD_24'] = moving_std(df[target_col], window_size=rolling, method=rolling_method)
            # df['SUM_STD_24'] = df[target_col].rolling(rolling).std()
            selected_target_list.append('SUM_STD_24')
        else:
            name = target_col[6:] + '_STD'
            df[name] = moving_std(df[target_col], window_size=rolling, method=rolling_method)
            selected_target_list.append(name)
    selected_target_list.extend(target_cols)
    # df.fillna(method='backfill', inplace=True)
    feature_list = list(set(feature_list) - set(selected_target_list))
    feature_list.sort()
    selected_target_list.sort()

    X_raw = df.filter(feature_list)
    Y_raw = df.filter(selected_target_list)

    return X_raw, Y_raw, group_index


def add_lag(df, x, y, shift=5):
    for i in range(1, shift + 1, 1):
        name = 'T_%s' % abs(i)
        x[name] = df['SUM_STD_24'].shift(i)

    return x[shift:], y[shift:]


def moving_std(df, window_size, method='top'):
    if method == 'top':
        return df.rolling(window_size).std().fillna(method='backfill')
    if method == 'center':
        return df.rolling(window_size, center=True).std().fillna(method='backfill').fillna(method='ffill')
    if method == 'bottom':
        return df.iloc[::-1].rolling(window_size).std().fillna(method='backfill').values[::-1]
