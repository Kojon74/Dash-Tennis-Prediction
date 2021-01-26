# -*- coding: utf-8 -*-

import dash_core_components as dcc
import dash_html_components as html
import dash
from dash.dependencies import Input, Output, State

from tensorflow import keras
import numpy as np
import json
import os

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]


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

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.layout = html.Div(
    [
        html.H2("Tennis Match Predtiction"),
        html.H3("Tournament Details"),
        html.H6("Name"),
        dcc.Dropdown(
            id="t_name",
            options=[{"label": t_name, "value": t_name} for t_name in t_ids.keys()],
            placeholder="Wimbledon",
        ),
        html.H6("Round"),
        dcc.Dropdown(
            id="t_round",
            options=[
                {"label": t_round, "value": t_round} for t_round in round_ids.keys()
            ],
            placeholder="Final",
        ),
        html.Div(
            style={"display": "flex", "justifyContent": "space-around"},
            children=[
                html.Div(
                    style={"width": "45%"},
                    children=[
                        html.H3("Player 1 Details"),
                        html.H6("Name"),
                        dcc.Dropdown(
                            id="p1_name",
                            options=[
                                {"label": p_name, "value": p_name}
                                for p_name in p_ids.keys()
                            ],
                        ),
                        html.H6("Age"),
                        dcc.Input(id="p1_age", type="number"),
                        html.H6("Rank"),
                        dcc.Input(id="p1_rank", type="number"),
                    ],
                ),
                html.Div(
                    style={"width": "45%"},
                    children=[
                        html.H3("Player 2 Details"),
                        html.H6("Name"),
                        dcc.Dropdown(
                            id="p2_name",
                            options=[
                                {"label": p_name, "value": p_name}
                                for p_name in p_ids.keys()
                            ],
                        ),
                        html.H6("Age"),
                        dcc.Input(id="p2_age", type="number"),
                        html.H6("Rank"),
                        dcc.Input(id="p2_rank", type="number"),
                    ],
                ),
            ],
        ),
        html.Button(
            style={"margin": "10px auto", "display": "block"},
            children="Submit",
            id="submit",
            n_clicks=0,
        ),
        html.Div(id="result"),
    ]
)


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
    probability = np.amax(prediction) * 100
    return f"Predicted winner: {winner}, with probability {probability}%"


if __name__ == "__main__":
    app.run_server(debug=True)