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
# pio.renderers.default = 'browser'
#
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

# dir
DIR_ROOT = r"C:\myGit\wow"
DIR_DATA = os.path.join(DIR_ROOT, 'data')
# dir to store single files (for archive)
DIR_DATA_SINGLE = os.path.join(DIR_DATA, 'single')
# file path to store all combined data
FPATH_DATA_ALL = os.path.join(DIR_DATA, 'data_all.csv')
FPATH_DATA_REMOVE = os.path.join(DIR_DATA, 'remove.csv')
_params_update_layout = dict(paper_bgcolor='white', plot_bgcolor='white')
_params_update_axes = dict(linecolor='#e6e6e6', gridcolor='#e6e6e6')
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])


# HELPERS
def generate_remove_list():
    # save a list to be removed
    ls_remove = ['割裂雕文']
    df = pd.DataFrame({'名称': ls_remove})
    df.to_csv(FPATH_DATA_REMOVE, index=False)


def check_data(df):
    col1 = df.columns.tolist()[0]
    if col1 != '价格':
        checked = 0
    else:
        checked = 1

    return checked


def load_all_data():
    df = pd.read_csv(FPATH_DATA_ALL)
    # remove
    ls_remove = pd.read_csv(FPATH_DATA_REMOVE)
    ls_remove = ls_remove['名称'].unique().tolist()
    df = df[~df['名称'].isin(ls_remove)]
    df['Datetime'] = pd.to_datetime(df['Datetime'])

    return df

def load_all_files():
    ls_files = glob.glob(os.path.join(DIR_DATA_SINGLE, '*.csv'))
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


def process_all_files():
    df = load_all_files()
    # refresh the combined data file
    df.to_csv(FPATH_DATA_ALL, index=False)


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


def calc_change(col_value):
    df = load_all_data()
    df = df.sort_values(['名称', 'Datetime']).reset_index(drop=True)
    df['chg'] = df.groupby('名称')[col_value].diff()
    df_chg = df.groupby('名称')['chg'].nth(-1)
    df_chg = df_chg.sort_values(ascending=False)
    df_chg = df_chg.dropna()

    return df_chg


def calc_dist_to_mean(col_value, n_obs=10):
    df = load_all_data()
    df = df.sort_values(['名称', 'Datetime']).reset_index(drop=True)
    df['mean'] = df.groupby('名称')[col_value].apply(
        lambda x: x.rolling(window=n_obs, min_periods=int(n_obs/2)).mean()
    )
    df['dtm'] = df[col_value] - df['mean']
    df_dtm = df.groupby('名称')['dtm'].nth(-1)
    df_dtm = df_dtm.sort_values(ascending=False)
    df_dtm = df_dtm.dropna()

    return df_dtm


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
button_clearall = dbc.Button(
    'Clear All', id='button_clearall',
    color='primary', className='me-1', n_clicks=0
)
alert_upload = dbc.Alert(
    id='alert_upload',
    is_open=False,
    dismissable=True,
    duration=4000,
)
button_refresh_graph = dbc.Button(
    'Refresh', id='button_refresh_graph',
    color='primary', className='me-1', n_clicks=0
)
button_select_all = dbc.Button(
    "Select All", id='button_select_all',
    color="primary", className="me-1", n_clicks=0
)
# output_feedback = html.Div(
#     id='output_feedback',
#     style={'whiteSpace': 'pre-line'}
# )
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
graph_pricechg = dcc.Graph(id='graph_pricechg')
graph_volumechg = dcc.Graph(id='graph_volumechg')
graph_pricedtm = dcc.Graph(id='graph_pricedtm')
graph_volumedtm = dcc.Graph(id='graph_volumedtm')
app.title = 'AuctionViewer'
app.layout = dbc.Container(
    children=[
        html.H2('💰Auction Viewer'),
        html.Hr(),
        html.H4('Uploader: '),
        dbc.Row(input_textarea),
        html.Hr(),
        dbc.Row([
            dbc.Col(button_upload, width='auto'),
            dbc.Col(button_clearall, width='auto'),
            dbc.Col(button_refresh_graph, width='auto'),
        ]),
        dbc.Row(alert_upload),
        html.Hr(),
        html.H4('Top Movers: '),
        dbc.Row([
            dbc.Col(graph_pricechg),
            dbc.Col(graph_volumechg)
        ]),
        dbc.Row([
            dbc.Col(graph_pricedtm),
            dbc.Col(graph_volumedtm),
        ]),
        html.Hr(),
        dbc.Row(input_items),
        dbc.Row(input_value),
        dbc.Row([
            dbc.Col(button_select_all, width=2),
        ]),
        html.Hr(),
        dbc.Row(graph_price),
    ]
)


@app.callback(
    [
        Output('alert_upload', 'is_open'),
        Output('alert_upload', 'children'),
        Output('alert_upload', 'color'),
    ],
    Input('button_upload', 'n_clicks'),
    State('input_textarea', 'value')
)
def upload_data(n_clicks, str_csv):
    print(str_csv)
    if (n_clicks <= 0):
        raise PreventUpdate

    if (str_csv is None) or (len(str_csv) == 0):
        alert_open = True
        alert_str = 'Nothing to be uploaded.'
        alert_color = 'warning'
        return [alert_open, alert_str, alert_color]

    str_io = StringIO(str_csv)
    df = pd.read_csv(str_io, sep=',')
    checked = check_data(df)
    if not checked:
        alert_str = f"Wrong file/string format. Not uploaded. "
        alert_color = 'danger'
    else:
        now = datetime.datetime.now()
        fname = f"{now:%Y%m%d.%H%M%S}.csv"
        fpath = os.path.join(DIR_DATA_SINGLE, fname)
        try:
            # save to file first
            df.to_csv(fpath, index=False)
            # process and concat all files into one for faster process
            process_all_files()
            alert_str = f"[{n_clicks}] SUCCESS. Uploaded to: {fpath}"
            alert_color = 'success'
        except Exception as e:
            alert_str = f"FAILED with Error: {e}"
            alert_color = 'danger'

    return [True, alert_str, alert_color]


@app.callback(
    Output('input_textarea', 'value'),
    Input('button_clearall', 'n_clicks')
)
def clear_textarea(n_clicks):
    print(f"clear_textarea.n_clicks: {n_clicks}")
    if (n_clicks is None) or (n_clicks == 0):
        raise PreventUpdate

    return ''


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
            mode='markers+lines',
            name=items[0],
            line_shape='hv',
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df['Datetime'], y=df['可购买'],
            mode='markers+lines',
            name='可购买',
            line_shape='hv',
        ), row=2, col=1)
        fig.update_yaxes(title_text='价格(g)', **_params_update_axes,
                         row=1, col=1)
        fig.update_yaxes(title_text='可购买', **_params_update_axes,
                         row=2, col=1)
        fig.update_xaxes(**_params_update_axes)
        title = items[0]
    else:
        fig = px.line(
            df, x='Datetime', y=value, color='名称', markers=True,
            hover_data = ['gsc'],
            # line_shape='hv'
        )
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


@app.callback(
    [
        Output('graph_pricechg', 'figure'),
        Output('graph_volumechg', 'figure'),
        Output('graph_pricedtm', 'figure'),
        Output('graph_volumedtm', 'figure'),
    ],
    Input('button_refresh_graph', 'n_clicks'),
)
def graph_analysis(n_clicks):
    # set threshold
    thre_pxchg = 0.5 # 50silver
    thre_volchg = 50
    n_obs = 10
    # price change
    df_pxchg = calc_change(col_value='价格')
    # convert to gold
    df_pxchg /= 10000
    # top movers
    df_pxchg = pd.concat([df_pxchg.head(), df_pxchg.tail()])
    fig_pxchg = px.bar(df_pxchg)
    fig_pxchg.add_hline(y=thre_pxchg, line_color='green', line_dash='dot')
    fig_pxchg.add_hline(y=-thre_pxchg, line_color='red', line_dash='dot')
    fig_pxchg.update_traces(showlegend=False)
    fig_pxchg.update_xaxes(**_params_update_axes)
    fig_pxchg.update_yaxes(title_text='价格(g)', **_params_update_axes)
    fig_pxchg.update_layout(title=f"价格 Top Movers", **_params_update_layout)
    # fig_pxchg.show()

    # volume change
    df_volchg = calc_change(col_value='可购买')
    df_volchg = pd.concat([df_volchg.head(), df_volchg.tail()])
    fig_volchg = px.bar(df_volchg)
    fig_volchg.add_hline(y=thre_volchg, line_color='green', line_dash='dot')
    fig_volchg.add_hline(y=-thre_volchg, line_color='red', line_dash='dot')
    fig_volchg.update_traces(showlegend=False)
    fig_volchg.update_xaxes(**_params_update_axes)
    fig_volchg.update_yaxes(title_text='可购买', **_params_update_axes)
    fig_volchg.update_layout(title=f"可购买 Top Movers", **_params_update_layout)
    # fig_volchg.show()

    # price dtm
    df_pxdtm = calc_dist_to_mean(col_value='价格', n_obs=n_obs)
    # convert to gold
    df_pxdtm /= 10000
    # top movers
    df_pxdtm = pd.concat([df_pxdtm.head(), df_pxdtm.tail()])
    fig_pxdtm = px.bar(df_pxdtm)
    fig_pxdtm.add_hline(y=thre_pxchg, line_color='green', line_dash='dot')
    fig_pxdtm.add_hline(y=-thre_pxchg, line_color='red', line_dash='dot')
    fig_pxdtm.update_traces(showlegend=False)
    fig_pxdtm.update_xaxes(**_params_update_axes)
    fig_pxdtm.update_yaxes(title_text='价格DTM(g)', **_params_update_axes)
    fig_pxdtm.update_layout(title=f"价格 Dist-to-mean", **_params_update_layout)

    # volume dtm
    df_voldtm = calc_dist_to_mean(col_value='可购买', n_obs=n_obs)
    df_voldtm = pd.concat([df_voldtm.head(), df_voldtm.tail()])
    fig_voldtm = px.bar(df_voldtm)
    fig_voldtm.add_hline(y=thre_volchg, line_color='green', line_dash='dot')
    fig_voldtm.add_hline(y=-thre_volchg, line_color='red', line_dash='dot')
    fig_voldtm.update_traces(showlegend=False)
    fig_voldtm.update_xaxes(**_params_update_axes)
    fig_voldtm.update_yaxes(title_text='可购买DTM', **_params_update_axes)
    fig_voldtm.update_layout(title=f"可购买 Dist-to-mean",
                             **_params_update_layout)

    return fig_pxchg, fig_volchg, fig_pxdtm, fig_voldtm


if __name__ == '__main__':
    app.run_server(debug=True)

