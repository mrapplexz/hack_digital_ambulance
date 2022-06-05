from datetime import datetime as dt

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import shap
from dash.dependencies import Input, Output, State

import utils.dash_reusable_components as drc
from graph_factory import GraphFactory
from plotly_style import apply_plotly_style

apply_plotly_style()

graph_factory = GraphFactory('../fixed_substation.json', '../models', dt(2022, 5, 25), dt(2023, 5, 25), '../caches.pkl')
graph_factory.load()


app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}], external_stylesheets=[dbc.themes.BOOTSTRAP],
)
app.title = "ПРЕДСКАЗАТЕЛЬ ЗАГРУЖЕННОСТИ БРИГАД СКОРОЙ ПОМОЩИ"
app.scripts.append_script({"external_url": "https://cdn.plot.ly/plotly-locale-ru-latest.js"})
config_plots = dict(locale='ru')
server = app.server


# Layout of Dash App
app.layout = html.Div(
    children=[
        dbc.Modal(
            [
                dbc.ModalHeader(html.H4("Загрузка данных"), close_button=False),
                dbc.ModalBody("Новые журналы вызовов успешно добавлены!"),
                dbc.ModalFooter(),
            ],
            id="modal-upload",
            is_open=False,
        ),

        html.Div(
            className="row",
            children=[
                # Column for user controls
                html.Div(
                    className="three columns div-user-controls",
                    children=[
                        html.H1("ПРЕДСКАЗАТЕЛЬ ЗАГРУЖЕННОСТИ БРИГАД СКОРОЙ ПОМОЩИ"),
                        html.Div(
                            id='first-card',
                            children=[
                                html.H4("Загруженность подстанций за год"),
                                html.Div(
                                    id="div-for-all-year-graph",
                                    children=[
                                        dcc.Graph(id="all-year-graph", figure=graph_factory.create_total_figure()),
                                        html.Hr(),
                                    ],
                                ),
                               
                            ],
                        ),
                        html.Div(
                            id='second-card',
                            children=[
                                html.Div(
                                    id="div-for-shap-values-graph",
                                    children=[
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            id='third-card',
                            children=[
                                html.Div(
                                    id="div-for-dropdown",
                                    className="div-center",
                                    children=[
                                        html.H6(
                                    """Выберите день для просмотра ожидаемой загруженности"""
                                        ),
                                        dcc.DatePickerSingle(
                                            id="date-picker",
                                            min_date_allowed=dt(2022, 5, 25),
                                            max_date_allowed=dt(2023, 5, 25),
                                            initial_visible_month=dt(2022, 5, 25),
                                            date=dt(2022, 5, 25).date(),
                                            display_format="D MMMM, YYYY",
                                            style={"border": "0px solid black"},
                                        )
                                    ],
                                ),
                                # Change to side-by-side for mobile layout
                                html.Div(
                                    className="row div-center",
                                    children=[
                                        html.Div(
                                            id="day-or-hour-container",
                                            children=[
                                                html.H6(children="Предсказать загруженность за:"),
                                                dcc.RadioItems(
                                                    id="radio-hour-or-day",

                                                    labelStyle={
                                                        "margin-right": "7px",
                                                        "display": "inline-block",
                                                    },
                                                    options=[
                                                        {
                                                            "label": "Час",
                                                            "value": "True",
                                                        },
                                                        {
                                                            "label": "День",
                                                            "value": "False",
                                                        },
                                                    ],
                                                    value="True",
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            id="div-for-hour-slider",
                                            children=[
                                                drc.FormattedSlider(
                                                    id="hour-slider",
                                                    step=1,
                                                    min=0,
                                                    max=23,
                                                    value=14,
                                                    marks={
                                                        i: "{}".format(i)
                                                        for i in range(0, 24)
                                                    },
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                html.Div(
                                    id="div-uploading",
                                    className='div-center',
                                    children=[
                                        html.H6("Добавьте ещё журналы вызовов, если требуется"),
                                        dcc.Upload(
                                            id='upload-data',
                                            children=html.Div([
                                                'Перетащите или ',
                                                html.A('Выберите файлы')
                                            ]),
                                            style={
                                                'width': '100%',
                                                'height': '60px',
                                                'lineHeight': '60px',
                                                'borderWidth': '1px',
                                                'borderStyle': 'dashed',
                                                'borderRadius': '5px',
                                                'textAlign': 'center',
                                                'margin': '10px'
                                            },
                                            multiple=True
                                        ),
                                        html.Hr(),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                # Column for app graphs and plots
                html.Div(
                    className="nine columns div-for-charts bg-grey",
                    children=[
                        html.H2("Карта загруженности"),
                        dcc.Graph(id="map-graph", className="openstreetmap"),
                        html.H2("Распределение за день"),
                        dcc.Graph(id="histogram"),
                    ],
                ),
            ],
        )
    ]
)


@app.callback(
    Output('div-for-hour-slider', 'style'),
    [Input('radio-hour-or-day', 'value')]
)
def radio_container(radio_value):
    if radio_value == "False":
        return {'display': 'none'}
    else:
        return {'display': 'block'}


@app.callback(
    Output('map-graph', 'figure'),
    [Input('date-picker', 'date'), Input('hour-slider', 'value'), Input('radio-hour-or-day', 'value')]
)
def graph_densmap(date, hour, show_hour):
    date = pd.to_datetime(date)
    show_hour = show_hour == 'True'
    return graph_factory.get_densmap_figure(date, hour, show_hour)


@app.callback(
    Output('histogram', 'figure'),
    [Input('date-picker', 'date')]
)
def graph_histogram(date):
    date = date
    return graph_factory.create_substation_daily_figure(date)


@app.callback(
    Output(component_id='div-for-shap-values-graph', component_property='children'),
    [Input('date-picker', 'date'), Input('hour-slider', 'value'), Input('radio-hour-or-day', 'value'), Input('map-graph', 'clickData')])
def display_click_data(date, hour, show_hour, click_data):
    if click_data is not None:
        date = pd.to_datetime(date)
        hour = hour if show_hour == 'True' else None
        shap_el = graph_factory.create_shap(click_data['points'][0]['customdata'], date, hour)
        shap_html = f"<head>{shap.getjs()}</head><body>{shap_el}</body>"
        if shap_el == 'nope':
            return [html.Div()]
        return [html.Div(
                    children=[
                        html.Iframe(
                            srcDoc=shap_html,
                            style={"width": "100%", "height": "175px", "border": 0},
                        ),
                        html.Hr(),
                    ]
                )
        ]
    return [html.Div()]

@app.callback(Output('modal-upload', 'is_open'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'),
              State("modal-upload", "is_open"))
def update_output(list_of_contents, list_of_names, list_of_dates, is_open):
    if list_of_contents is not None:
        return not is_open
    return is_open


if __name__ == "__main__":
    app.run_server(debug=False)
