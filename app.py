# -*- coding: utf-8 -*-

import plotly.graph_objects as go

import dash_core_components as dcc
import dash_html_components as html
import dash
import dash_auth
from dash.dependencies import Input, Output, State

from tensorflow import keras
import numpy as np
import json
import os

from auth import VALID_USERNAME_PASSSWORD

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
colors = {"gray": "#4C5760", "bone": "#D7CEB2", "blue": "#00A7E1"}


def load_ids():
    id_jsons = sorted(os.listdir("ids"))
    id_dicts = []
    for id_json in id_jsons:
        with open(os.path.join("ids", id_json), "r") as f_json:
            id_dict = json.load(f_json)
            id_dicts.append(id_dict)
    return id_dicts


p_ids, t_ids = load_ids()
round_ids = {
    "RR": 1,
    "BR": 2,
    "R128": 3,
    "R64": 4,
    "R32": 5,
    "R16": 6,
    "QF": 7,
    "SF": 8,
    "F": 9,
}
round_list = ["RR", "BR", "R128", "R64", "R32", "R16", "QF", "SF", "F"]

model = keras.models.load_model("models")

app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True,
)
auth = dash_auth.BasicAuth(app, VALID_USERNAME_PASSSWORD)

server = app.server

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        dcc.Link(
            href="/",
            children=[
                html.Img(
                    style={
                        "height": "25px",
                        "display": "inline",
                        "margin": "0 15px 0 15px",
                    },
                    src="https://ttt.studio/wp-content/uploads/2018/11/favicon.png",
                ),
            ],
        ),
        html.H1(
            className="page-title",
            style={"display": "inline", "color": "#5993C3"},
            children="Tennis Match Prediction",
        ),
        html.Div(
            className="link-container",
            children=[
                dcc.Link("Home", className="link", href="/"),
                dcc.Link("Code", className="link", href="/code"),
                dcc.Link("Demo", className="link", href="/demo"),
            ],
        ),
        html.Div(id="page-content"),
    ]
)

home_page = html.Div(
    [
        html.H2(className="title", children=["About", html.Div(className="underline")]),
        html.P(
            className="home-p",
            children=[
                """
                This web application uses a machine learning model to predict the winner in a tennis match given sufficient information about the match. The model will pick a winner based on the tournament, round number, the two players, their age and rank. To view an in-depth explenation click on the 'Code' link, to try a demo click on the 'Demo' link.
                """
            ],
        ),
        html.P(
            className="home-p",
            children=[
                """
                This web application was created using Dash. Dash can be used to seamlessly create data science and machine learning web applications with very little web development experience. This allows Data Scientists to be able to deploy full blown web applications without requiring separate front-end or back-end teams.
                """
            ],
        ),
    ]
)

code_page = html.Div(
    [
        html.H2(className="title", children=["Code", html.Div(className="underline")]),
        html.Div(
            className="code-container",
            children=[
                html.H5(
                    className="code-desc", children=["Front-end: Demo-Page Layout"]
                ),
                html.Code(
                    className="code",
                    children=[
                        """
demo_page = html.Div(
    style={"width": "90%", "margin": "auto"},
    children=[
        html.H3(
            style={"color": colors["bone"]},
            children="Tournament Details",
        ),
        html.H6(style={"color": "white"}, children="Name"),
        dcc.Dropdown(
            id="t_name",
            style={"color": "white"},
            options=[
                {"label": t_name, "value": t_name}
                for t_name in t_ids.keys()
            ],
            placeholder="Wimbledon",
        ),
        html.H6(style={"color": "white"}, children="Round"),
        dcc.Dropdown(
            id="t_round",
            style={"color": "white"},
            options=[
                {"label": t_round, "value": t_round}
                for t_round in round_ids.keys()
            ],
            placeholder="Final",
        ),
        html.Div(
            style={
                "display": "flex",
                "justifyContent": "space-between",
            },
            children=[
                html.Div(
                    style={"width": "45%"},
                    children=[
                        html.H3(
                            style={"color": colors["bone"]},
                            children="Player 1 Details",
                        ),
                        html.H6(
                            style={"color": "white"}, children="Name"
                        ),
                        dcc.Dropdown(
                            id="p1_name",
                            style={"color": "white"},
                            options=[
                                {"label": p_name, "value": p_name}
                                for p_name in p_ids.keys()
                            ],
                        ),
                        html.H6(
                            style={"color": "white"}, children="Age"
                        ),
                        dcc.Input(
                            style={
                                "backgroundColor": colors["gray"],
                                "color": "white",
                            },
                            id="p1_age",
                            type="number",
                        ),
                        html.H6(
                            style={"color": "white"}, children="Rank"
                        ),
                        dcc.Input(
                            style={
                                "backgroundColor": colors["gray"],
                                "color": "white",
                            },
                            id="p1_rank",
                            type="number",
                        ),
                    ],
                ),
                html.Div(
                    style={"width": "45%"},
                    children=[
                        html.H3(
                            style={"color": colors["bone"]},
                            children="Player 2 Details",
                        ),
                        html.H6(
                            style={"color": "white"}, children="Name"
                        ),
                        dcc.Dropdown(
                            id="p2_name",
                            className="ken-dropdown",
                            style={"color": "white"},
                            options=[
                                {"label": p_name, "value": p_name}
                                for p_name in p_ids.keys()
                            ],
                        ),
                        html.H6(
                            style={"color": "white"}, children="Age"
                        ),
                        dcc.Input(
                            style={
                                "backgroundColor": colors["gray"],
                                "color": "white",
                            },
                            id="p2_age",
                            type="number",
                        ),
                        html.H6(
                            style={"color": "white"}, children="Rank"
                        ),
                        dcc.Input(
                            style={
                                "backgroundColor": colors["gray"],
                                "color": "white",
                            },
                            id="p2_rank",
                            type="number",
                        ),
                    ],
                ),
            ],
        ),
        html.Button(
            style={
                "margin": "20px auto",
                "display": "block",
                "color": colors["bone"],
            },
            children="Submit",
            id="submit",
            n_clicks=0,
        ),
        html.H6(
            style={
                "width": "fit-content",
                "margin": "auto",
                "color": colors["bone"],
            },
            id="result",
        ),
    ],
)
"""
                    ],
                ),
                html.H5(
                    className="code-desc",
                    children="Back-end: Prediction Calculation using Trained Model",
                ),
                html.Code(
                    className="code",
                    children=[
                        """
@app.callback(
    Output("result", "children"),
    Input("submit", "n_clicks"),
    State("t_name", "value"),
    State("t_round", "value"),
    State("p1_name", "value"),
    State("p1_age", "value"),
    State("p1_rank", "value"),
    State("p2_name", "value"),
    State("p2_age", "value"),
    State("p2_rank", "value"),
)
def calc_winner(_, t_name, t_round, p1_name, p1_age, p1_rank, p2_name, p2_age, p2_rank):
    if not t_name:  # Return nothing if values empty
        return ""
    t_round_id = round_ids[t_round]
    data1 = [
        t_ids[t_name]
        + [t_round_id]
        + p_ids[p1_name]
        + [p1_age, p1_rank]
        + p_ids[p2_name]
        + [p2_age, p2_rank]
    ]
    data2 = [
        t_ids[t_name]
        + [t_round_id]
        + p_ids[p2_name]
        + [p2_age, p2_rank]
        + p_ids[p1_name]
        + [p1_age, p1_rank]
    ]
    pred1 = model.predict(data1)
    pred2 = model.predict(data2)
    prediction = np.asarray([(pred1[0][i] + pred2[0][1 - i]) / 2 for i in range(2)])
    players = [p1_name, p2_name]
    winner = players[prediction.argmax()]
    probability = int(np.amax(prediction) * 100)
    return f"Predicted winner: {winner}, with probability {probability}%"
"""
                    ],
                ),
            ],
        ),
    ]
)

demo_page = html.Div(
    style={"width": "90%", "margin": "auto"},
    children=[
        html.H2(className="title", children=["Demo", html.Div(className="underline")]),
        html.H3(style={"color": colors["bone"]}, children="Tournament Details"),
        html.H6(style={"color": "white"}, children="Name"),
        dcc.Dropdown(
            id="t_name",
            style={"color": "white"},
            options=[{"label": t_name, "value": t_name} for t_name in t_ids.keys()],
            placeholder="Wimbledon",
        ),
        html.H6(style={"color": "white"}, children="Round"),
        dcc.Dropdown(
            className="dropdown",
            id="t_round",
            style={"color": "white"},
            options=[
                {"label": t_round, "value": t_round} for t_round in round_ids.keys()
            ],
            placeholder="Final",
        ),
        html.Div(
            style={"display": "flex", "justifyContent": "space-between"},
            children=[
                html.Div(
                    style={"width": "45%"},
                    children=[
                        html.H3(
                            style={"color": colors["bone"]}, children="Player 1 Details"
                        ),
                        html.H6(style={"color": "white"}, children="Name"),
                        dcc.Dropdown(
                            id="p1_name",
                            style={"color": "white"},
                            options=[
                                {"label": p_name, "value": p_name}
                                for p_name in p_ids.keys()
                            ],
                        ),
                        html.H6(style={"color": "white"}, children="Age"),
                        dcc.Input(
                            style={"backgroundColor": colors["gray"], "color": "white"},
                            id="p1_age",
                            type="number",
                        ),
                        html.H6(style={"color": "white"}, children="Rank"),
                        dcc.Input(
                            style={"backgroundColor": colors["gray"], "color": "white"},
                            id="p1_rank",
                            type="number",
                        ),
                    ],
                ),
                html.Div(
                    style={"width": "45%"},
                    children=[
                        html.H3(
                            style={"color": colors["bone"]}, children="Player 2 Details"
                        ),
                        html.H6(style={"color": "white"}, children="Name"),
                        dcc.Dropdown(
                            id="p2_name",
                            className="ken-dropdown",
                            style={"color": "white"},
                            options=[
                                {"label": p_name, "value": p_name}
                                for p_name in p_ids.keys()
                            ],
                        ),
                        html.H6(style={"color": "white"}, children="Age"),
                        dcc.Input(
                            style={"backgroundColor": colors["gray"], "color": "white"},
                            id="p2_age",
                            type="number",
                        ),
                        html.H6(style={"color": "white"}, children="Rank"),
                        dcc.Input(
                            style={"backgroundColor": colors["gray"], "color": "white"},
                            id="p2_rank",
                            type="number",
                        ),
                    ],
                ),
            ],
        ),
        html.Button(
            style={"margin": "20px auto", "display": "block", "color": colors["bone"]},
            children="Submit",
            id="submit",
            n_clicks=0,
        ),
        dcc.Graph(id="results-bar"),
    ],
)


@app.callback(
    Output("results-bar", "figure"),
    Input("submit", "n_clicks"),
    State("t_name", "value"),
    State("t_round", "value"),
    State("p1_name", "value"),
    State("p1_age", "value"),
    State("p1_rank", "value"),
    State("p2_name", "value"),
    State("p2_age", "value"),
    State("p2_rank", "value"),
)
def calc_winner(_, t_name, t_round, p1_name, p1_age, p1_rank, p2_name, p2_age, p2_rank):
    if not t_name:  # Return nothing if values empty
        return go.Figure()

    t_round_id = round_ids[t_round]

    data1 = [
        t_ids[t_name]
        + [t_round_id]
        + p_ids[p1_name]
        + [p1_age, p1_rank]
        + p_ids[p2_name]
        + [p2_age, p2_rank]
    ]
    data2 = [
        t_ids[t_name]
        + [t_round_id]
        + p_ids[p2_name]
        + [p2_age, p2_rank]
        + p_ids[p1_name]
        + [p1_age, p1_rank]
    ]
    pred1 = model.predict(data1)
    pred2 = model.predict(data2)
    prediction = np.asarray([(pred1[0][i] + pred2[0][1 - i]) / 2 for i in range(2)])
    players = [p1_name, p2_name]
    probs = [int(prob * 100) for prob in prediction]
    fig = go.Figure([go.Bar(x=players, y=probs, text=probs, textposition="auto")])
    fig.update_layout(title_text=f"{players[0]} vs {players[1]} Match Prediction")
    fig.update_xaxes(title_text="Players")
    fig.update_yaxes(title_text="Predicted Chance of Winning (%)")
    return fig


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def display_page(pathname):
    if pathname == "/":
        return home_page
    elif pathname == "/code":
        return code_page
    elif pathname == "/demo":
        return demo_page
    return "404"


if __name__ == "__main__":
    app.run_server(debug=True)