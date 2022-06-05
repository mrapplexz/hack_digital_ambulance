import datetime as dt
import logging
import os.path
import pickle
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap

from predictor import make_predictions
from substation import load_substations


class GraphFactory:
    predictions_daily: Optional[pd.DataFrame]
    predictions_hourly: Optional[pd.DataFrame]
    predictions: Optional[pd.DataFrame]
    shap_values: Optional[np.ndarray]
    features: Optional[pd.DataFrame]

    def __init__(self, substations_path: str, model_path: str, infer_from: dt.datetime, infer_to: dt.datetime,
                 cache_path: str):
        self.substations_path = substations_path
        self.model_path = model_path
        self.infer_from = infer_from
        self.infer_to = infer_to
        self.cache_path = cache_path
        self.logger = logging.getLogger()

        self.predictions_daily = None
        self.predictions_hourly = None
        self.shap_values = None
        self.features = None

    def load(self):
        if os.path.isfile(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                cache = pickle.load(f)
            self.predictions_daily = cache['predictions_daily']
            self.predictions_hourly = cache['predictions_hourly']
            self.predictions = cache['predictions']
            self.shap_values = cache['shap_values']
            self.features = cache['features']
        else:
            self.logger.info('Loading substations...')
            substations = load_substations(self.substations_path)

            self.logger.info('Making predictions...')
            predictions, shap_values, features = make_predictions(
                pd.DataFrame({'date': pd.date_range(self.infer_from, self.infer_to, freq='1H')}),
                self.model_path
            )
            self.predictions = predictions
            self.shap_values = shap_values
            self.features = features

            #  по дате
            pred_daily = predictions.groupby(predictions['date_time'].dt.date).sum().reset_index().melt(
                id_vars=['date_time'],
                var_name='substation',
                value_name='calls'
            )
            pred_daily = pred_daily.join(substations, on='substation')
            pred_daily['date_time'] = pd.to_datetime(pred_daily['date_time'])
            self.predictions_daily = pred_daily

            #  по часам
            pred_hourly = predictions.melt(id_vars=['date_time'], var_name='substation', value_name='calls')
            pred_hourly = pred_hourly.join(substations, on='substation')
            self.predictions_hourly = pred_hourly

            with open(self.cache_path, 'wb') as f:
                pickle.dump({
                    'predictions_daily': self.predictions_daily,
                    'predictions_hourly': self.predictions_hourly,
                    'predictions': self.predictions,
                    'shap_values': self.shap_values,
                    'features': self.features
                }, f)

    def create_total_figure(self):
        pred_daily = self.predictions_daily.copy()
        pred_daily['date_time'] = pred_daily['date_time'] - pd.to_timedelta(pred_daily['date_time'].dt.dayofweek, unit='d')
        pred_daily = pred_daily.groupby(['date_time', 'substation'])['calls'].sum().reset_index()
        unique = pred_daily.groupby('substation')['calls'].mean().sort_values(ascending=True).index.to_numpy()
        fig = go.Figure()
        for sub in unique:
            pred_sub = pred_daily[pred_daily['substation'] == sub]
            fig.add_trace(go.Bar(x=np.array(pred_sub['date_time']), y=np.array(pred_sub['calls']), name=sub))
        fig.update_layout(barmode='stack')
        fig.update_layout(showlegend=False)
        fig.update_layout(margin={"r": 1, "t": 1, "l": 1, "b": 1})
        fig.update_layout(height=256)
        return fig

    def get_densmap_figure(self, date, hour, show_hours):
        date = pd.to_datetime(date)
        if show_hours:
            cut_df = self.predictions_hourly[self.predictions_hourly['date_time'] == pd.to_datetime(date + dt.timedelta(hours=hour))]
        else:
            cut_df = self.predictions_daily[self.predictions_daily['date_time'] == date]
        # customdata =
        densmap = go.Densitymapbox(lat=cut_df['lat'], lon=cut_df['lon'], z=cut_df['calls'],
                                   customdata=cut_df['substation'],
                                   hovertemplate=r'''<b>Вызовов:</b> %{z}<br><b>Подстанция: </b>%{customdata} <extra></extra>''',
                                   radius=80)
        fig = go.Figure(densmap)
        # fig.update_layout(mapbox_opacity=0.75)
        fig.update_layout(mapbox_zoom=7)
        fig.update_layout(height=512)
        fig.update_layout(mapbox_center=(go.layout.mapbox.Center(lat=55.6264, lon=43.47)))
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r": 3, "t": 3, "l": 3, "b": 3})
        fig.update_layout(clickmode='event+select')
        return fig

    def create_substation_daily_figure(self, date):
        pred_hourly = self.predictions_hourly
        unique = pred_hourly.groupby('substation')['calls'].mean().sort_values(ascending=True).index.to_numpy()
        fig = go.Figure()
        pred_hrl = pred_hourly[(pd.to_datetime((pred_hourly['date_time']).dt.date) == pd.to_datetime(date))]
        for sub in unique:
            pred_sub = pred_hrl[(pred_hrl['substation'] == sub)]
            fig.add_trace(go.Bar(x=np.array(pred_sub['date_time']), y=np.array(pred_sub['calls']), name=sub))
        fig.update_layout(barmode='stack')
        fig.update_layout(margin={"r": 1, "t": 1, "l": 1, "b": 1})
        return fig

    def create_shap(self, substation, day, hour=None):
        pred = self.predictions
        shap_vs = self.shap_values
        feature_df = self.features
        if hour is not None:
            date = pd.to_datetime(day) + dt.timedelta(hours=hour)
            idx = pred[pred['date_time'] == date].index[0]
            return shap.force_plot(shap_vs[substation][idx][-1], shap_vs[substation][idx][:-1],
                                   features=feature_df.iloc[idx], show=False, matplotlib=False).html()
        else:
            # date = pd.to_datetime(day)
            # idx = pred[pd.to_datetime(pred['date_time'].dt.date) == date].index
            # shaps = shap_vs[substation][idx]
            # shap.summary_plot(shaps[:, :-1], features=feature_df.iloc[idx], show=False)
            # # plot = shap.plots.beeswarm(shaps[:, :-1])
            # # return plot.html()
            # fig = plt.gcf()
            # tmpfile = BytesIO()
            # fig.savefig(tmpfile, format='png')
            # encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

            # return '<img src=\'data:image/png;base64,{}\'>'.format(encoded)


            # canvas = FigureCanvas(fig)
            # png_output = StringIO()
            # canvas.print_png(png_output)
            # data = png_output.getvalue().encode('base64')
            # return '<img src="data:image/png;base64,{}">'.format(urllib.parse.quote(data.rstrip('\n')))

            return 'nope'

