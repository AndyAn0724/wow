"""Use "Export" in Auctionator to save data as file and accumlate.
Make a UI to easy upload and save. Then read and plot time series.

C:\Program Files (x86)\World of Warcraft\_classic_\WTF\Account\643972072#1\龙之召唤\Luiofdead\SavedVariables
"""

import os
import glob
import datetime
import pandas as pd
from io import StringIO
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'browser'

from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

# dir
DIR_ROOT = r"C:\myGit\wow"
DIR_DATA = os.path.join(DIR_ROOT, 'data')
_params_update_layout = dict(paper_bgcolor='white', plot_bgcolor='white')
_params_update_axes = dict(linecolor='#e6e6e6', gridcolor='#e6e6e6')
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])


# HELPERS
def check_data(df):
    col1 = df.columns.tolist()[0]
    if col1 != '价格':
        checked = 0
    else:
        checked = 1

    return checked


def load_all_data():
    ls_files = glob.glob(os.path.join(DIR_DATA, '*.csv'))
    ls_dfs = []
    for f in ls_files:
        tmp = pd.read_csv(f)
        dt = f.split('\\')[-1].replace('.csv', '')
        dt = datetime.datetime.strptime(dt, '%Y%m%d.%H%M%S')
        # check data
        checked = check_data(tmp)
        if checked:
            tmp['Datetime'] = dt
            ls_dfs.append(tmp)
    df = pd.concat(ls_dfs, sort=False)

    return df


def get_unique_items():
    df = load_all_data()
    items = df['名称'].unique().tolist()
    items = sorted(items)

    return items


def parse_to_gsc(n):
    c = n % 100
    s = n // 100 % 100
    g = n // 10000 % 100
    str_out = ''
    for key, val in zip(['g', 's', 'c'], [g, s, c]):
        if val != 0:
            str_out += f'{val}{key}'

    return str_out


# APP LAYOUT
input_textarea = dcc.Textarea(
    id='input_textarea',
    placeholder='Enter...',
    style={'height': 200},
)
button_upload = dbc.Button(
    "Upload", id='button_upload',
    color="primary", className="me-1", n_clicks=0
)
button_refresh_graph = dbc.Button(
    'Refresh', id='button_refresh_graph',
    color='primary', className='me-1', n_clicks=0
)
button_select_all = dbc.Button(
    "Select All", id='button_select_all',
    color="primary", className="me-1", n_clicks=0
)
output_feedback = html.Div(
    id='output_feedback',
    style={'whiteSpace': 'pre-line'}
)
_LS_ITEMS = get_unique_items()
_LS_ITEMS_NUM = 10
input_items = html.Div([
    dbc.Label('Items: '),
    dcc.Dropdown(
        id='input_items',
        options=_LS_ITEMS,
        value=_LS_ITEMS[:_LS_ITEMS_NUM],
        multi=True
    ),
], className='mb-3')
input_value = html.Div([
    dbc.Label('Value: '),
    dcc.Dropdown(
        id='input_value',
        options=['价格', '可购买'],
        value='价格'
    )
], className='mb-3')
graph_price = dcc.Graph(id='graph_price')
app.title = 'AuctionViewer'
app.layout = dbc.Container(
    children=[
        html.H2('💰Auction Viewer'),
        html.Hr(),
        html.H4('Uploader: '),
        dbc.Row(input_textarea),
        html.Hr(),
        dbc.Row(button_upload),
        dbc.Row(output_feedback),
        html.Hr(),
        html.H4('Viewer: '),
        dbc.Row(input_items),
        dbc.Row(input_value),
        dbc.Row([
            dbc.Col(button_refresh_graph, width=1),
            dbc.Col(button_select_all, width=2),
        ]),
        html.Hr(),
        dbc.Row(graph_price),
    ]
)


@app.callback(
    Output('output_feedback', 'children'),
    Input('button_upload', 'n_clicks'),
    State('input_textarea', 'value')
)
def upload_data(n_clicks, str_csv):
    print(str_csv)
    if (n_clicks <= 0):
        raise PreventUpdate

    if (str_csv is None) or (len(str_csv) == 0):
        str_out = 'Nothing to be uploaded.'
        return str_out

    str_io = StringIO(str_csv)
    df = pd.read_csv(str_io, sep=',')
    checked = check_data(df)
    if not checked:
        str_out = f"Wrong file/string format. Not uploaded. "
    else:
        now = datetime.datetime.now()
        fname = f"{now:%Y%m%d.%H%M%S}.csv"
        fpath = os.path.join(DIR_DATA, fname)
        try:
            df.to_csv(fpath, index=False)
            str_out = f"SUCCESS. Uploaded to: {fpath}"
        except Exception as e:
            str_out = f"FAILED with Error: {e}"

    return str_out


@app.callback(
    Output('input_items', 'options'),
    Input('button_refresh_graph', 'n_clicks'),
)
def update_items(n_clicks):
    items = get_unique_items()

    return items


@app.callback(
    Output('input_items', 'value'),
    Input('button_select_all', 'n_clicks'),
)
def update_items_2(n_clicks):
    items = get_unique_items()

    return items


@app.callback(
    Output('graph_price', 'figure'),
    [
        Input('button_refresh_graph', 'n_clicks'),
        Input('input_items', 'value'),
        Input('input_value', 'value'),
    ]
)
def graph_items(n_clicks, items, value):
    print(items)
    if (items is None) or (value is None):
        raise PreventUpdate
    if isinstance(items, str):
        items = [items]
    # load data
    df = load_all_data()
    df = df[df['名称'].isin(items)]
    df = df.sort_values(['名称', 'Datetime'])
    df = df.drop_duplicates(subset=['名称', 'Datetime'])
    df = df.reset_index(drop=True)
    # parse the price to g/s/c for easier view
    df['gsc'] = df['价格'].apply(parse_to_gsc)
    # convert unit to g
    df['价格'] = df['价格'] / 10000

    # plot more details if only one input
    if len(items) == 1:
        # plot volume as well
        df['Datetime'] = df['Datetime'].apply(
            lambda x: f"{x:%Y/%m/%d-%H:%M:%S}"
        )
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02
        )
        fig.add_trace(go.Scatter(
            x=df['Datetime'], y=df['价格'],
            name=items[0], mode='markers+lines'
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df['Datetime'], y=df['可购买'],
            mode='markers+lines',
            name='可购买'
        ), row=2, col=1)
        fig.update_yaxes(title_text='价格(g)', **_params_update_axes,
                         row=1, col=1)
        fig.update_yaxes(title_text='可购买', **_params_update_axes,
                         row=2, col=1)
        fig.update_xaxes(**_params_update_axes)
        title = items[0]
    else:
        fig = px.line(df, x='Datetime', y=value, color='名称', markers=True,
                      hover_data=['gsc'])
        y_title_text = '价格(g)' if value == '价格' else '可购买'
        fig.update_yaxes(title_text=y_title_text, **_params_update_axes)
        fig.update_xaxes(**_params_update_axes)
        title = f'{value}历史'
    # global layout adjustment
    fig.update_layout(
        title=title,
        # hovermode="x",
        autosize=False,
        height=800,
        **_params_update_layout
    )

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)

