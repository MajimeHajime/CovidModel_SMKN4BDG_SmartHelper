from flask import Flask, render_template, url_for
import plotly
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import json
#import PredictionModel // Masih buggy

app = Flask(__name__)


@app.route('/')
def front():
    return render_template(
        'frontpage.html'
        )

@app.route('/stats')
def stats():
    bar = create_plot()
    return render_template(
        'stats.html',
        plot=bar
        )


def create_plot():
    df = pd.read_csv('testfile.csv')
    fig = px.line(df, x="date", y="predicted", color=px.Constant("Prediction"))
    fig.add_bar(x=df["date"], y=df["total"], name="Total Cases")
    fig.add_trace(go.Scatter(x=df["date"], y=df['y_fut'], name="Future Prediction", mode='lines+markers'))
    data = fig
    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


if __name__ == '__main__':
    app.run(debug=True)

#  ++=========================================++
#  || KKSI AI - SMKN 4 Bandung - Smart Helper ||
#  ||-----------------------------------------||
#  ||   Pembina:                              ||
#  ||       Ibu Tyas                          ||
#  ||   Anggota(Berdasarkan Alphabet):        ||
#  ||       Aldo Fakhry                       ||
#  ||       Firdaus Haqiqi                    ||
#  ||       Muhammad Azmi                     ||
#  ||-----------------------------------------||
#  ||   Terimakasih Pada:                     ||
#  ||       # Sentdex                         ||
#  ||       # Pandas Documentation            ||
#  ||       # Diwas Padley                    ||
#  ||       # Stack Overflow                  ||
#  ||       # Our World In Data               ||
#  ++=========================================++