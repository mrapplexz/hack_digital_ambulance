import catboost
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def get_funcs():
    res = []
    names = []
    for i in range(1, 6):
        res.append(lambda x: np.sin(i * x))
        res.append(lambda x: np.cos(i * x))
        res.append(lambda x: np.tanh(i * x))
        res.append(lambda x: np.sin(x / i))
        res.append(lambda x: np.cos(x / i))
        res.append(lambda x: np.tanh(x / i))

        names.append(f'sin({i}*x)')
        names.append(f'cos({i}*x)')
        names.append(f'tanh({i}*x)')

        names.append(f'sin(x/{i})')
        names.append(f'cos(x/{i})')
        names.append(f'tanh(x/{i})')
    return res, names


def make_features(df):
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['date'].dt.hour
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_morning'] = (df['hour'] >= 4) & (df['hour'] <= 12)
    df['is_day'] = (df['hour'] >= 13) & (df['hour'] <= 18)
    df['is_evening'] = df['hour'] >= 19
    df['is_night'] = df['hour'] < 4
    df['full_hours'] = ((df['date'].dt.year - 2015) * 365 + df['month'] * 30 + df['day']) * 24 + df['hour']
    funcs, nms = get_funcs()
    for i, func in enumerate(funcs):
        for col in ['hour', 'day', 'month']:
            df[f"{nms[i]}_func_{col}"] = func(df[col])
    return df.drop(columns=['date'])


def make_predictions(df: pd.DataFrame, model_dir: str):
    res = dict()
    res['date_time'] = df['date']
    targets = os.listdir(model_dir)
    good_df = make_features(df)
    shaps = {}
    for target in tqdm(targets):
        pool = catboost.Pool(good_df, cat_features=['hour', 'day', 'month', 'day_of_week'])
        models = [pickle.load(open(f"{model_dir}/{target}/{pth}", 'rb')) for pth in os.listdir(f"{model_dir}/{target}/")
                  if 'shrink' not in pth and 'trend' not in pth]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trend = \
            [pickle.load(open(f"{model_dir}/{target}/{pth}", 'rb')) for pth in os.listdir(f"{model_dir}/{target}/") if
             'trend' in pth][0]
            shrink = \
            [pickle.load(open(f"{model_dir}/{target}/{pth}", 'rb')) for pth in os.listdir(f"{model_dir}/{target}/") if
             'shrink' in pth][0]

        preds = np.mean([model.predict(pool) for model in models], axis=0) * shrink.predict(
            good_df[['full_hours']]) + trend.predict(good_df[['full_hours']])
        preds = np.round(preds+0.1)
        preds[preds < 0] = 0
        res[target] = preds

        imps = np.mean([model.get_feature_importance(pool, type='ShapValues') for model in models], axis=0)
        imps[:, -1] = imps[:, -1] * shrink.predict(good_df[['full_hours']]) + trend.predict(good_df[['full_hours']])
        imps[:, :-1] = np.transpose(np.transpose(imps[:, :-1], (1, 0)) * shrink.predict(good_df[['full_hours']]),
                                    (1, 0))
        shaps[target] = imps
    return pd.DataFrame(res), shaps, good_df