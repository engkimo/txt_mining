import streamlit as st
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
from bokeh.plotting import figure, ColumnDataSource, from_networkx
import networkx as nx

from src import code


@st.cache
def load_time_series_data():
    """
    ランダムに時系列データを生成する。
    """
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['a', 'b', 'c']
    )
    return chart_data


@st.cache
def load_iris_data():
    """
    データ読み込み, cacheにして最適化を行う
    """
    iris_data = load_iris()
    df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    labels = iris_data.target_names[iris_data.target]
    return df, labels


def show_heatmap(df: pd.DataFrame):
    """
    各特徴の相関ヒートマップをみる
    """
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(df.corr(), annot=True, ax=ax)
    st.pyplot(fig)


def show_distplot_columns(df: pd.DataFrame, labels: np.ndarray):
    """
    それぞれの特徴について、各ラベルごとの分布を見る
    """
    for column in df.columns:
        st.write(f'- {column}')
        series = df[column]
        target_names = list(set(labels))
        hist_data = [series[labels == name] for name in target_names]
        fig = ff.create_distplot(
            hist_data, target_names, bin_size=[.1, .25, .5])
        st.plotly_chart(fig, use_container_width=True)


def show_distplot(df: pd.DataFrame, labels: np.ndarray):
    column = df.columns[0]
    st.write(f'- {column}')
    series = df[column]
    target_names = list(set(labels))
    hist_data = [series[labels == name] for name in target_names]
    fig = ff.create_distplot(
        hist_data, target_names, bin_size=[.1, .25, .5])
    st.plotly_chart(fig, use_container_width=True)


@st.cache(allow_output_mutation=True)
def fit_transform_pca(df: pd.DataFrame):
    """
    主成分分析した結果の第二主成分ベクトルを可視化した結果を得る
    """
    pca = PCA(n_components=2)
    X = pca.fit_transform(df)
    return X


def show_scatter2d(X: np.ndarray, labels: np.ndarray):
    fig, ax = plt.subplots(figsize=(10,10))
    target_names = list(set(labels))
    for i, target in enumerate(target_names):
        X_ = X[labels == target]
        x = X_[:, 0]
        y = X_[:, 1]
        ax.scatter(x, y, cmap=[i]*len(X_), label=target)

    ax.legend()
    st.pyplot(fig)


def show_scatter2d_bokeh(X: np.ndarray, labels: np.ndarray):
    # bokeh向け前処理
    tools = 'hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,'
    colormap = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
    colors = [colormap[t] for t in labels]
    x = X[:, 0]
    y = X[:, 1]
    source = ColumnDataSource(data=dict(
        x=x,
        y=y,
        desc=labels,
        colors=colors,
        radius=[0.05]*len(labels)
    ))
    tooltips = [
        ("index", "$index"),
        ("(x,y)", "($x, $y)"),
        ("desc", "@desc"),
    ]
    p = figure(tools=tools, tooltips=tooltips)

    p.scatter(
        'x', 'y', source=source, radius='radius',fill_alpha=0.6, fill_color = 'colors', line_color=None
    )
    st.bokeh_chart(p, use_container_width=True)


def view():
    st.title('Visualization')

    st.header('Line plot of random time series data')
    col1, col2 = st.columns(2)
    with col1:
        chart_data = load_time_series_data()
        st.line_chart(chart_data)

    with col2:
        st.code(code.VIS_LINEPLOT, language='python')

    # Load Iris data
    st.header('Visualization iris data')
    df_iris, labels = load_iris_data()

    # DataFrame
    st.subheader('Describe DataFrame')
    col1, col2 = st.columns(2)
    with col1:
        st.write(df_iris)
    with col2:
        st.code("""st.write(df_iris)""", language='python')

    # Heatmap
    st.subheader('Heatmap of correlations by feature')
    col1, col2 = st.columns(2)
    with col1:
        show_heatmap(df_iris)
    with col2:
        st.code(code.VIS_HEATMAP, language='python')

    # Feature dist by plotly
    st.subheader('Feature distribution for each label by visualizing plotly')
    col1, col2 = st.columns(2)
    with col1:
        show_distplot(df_iris, labels)
    with col2:
        st.code(code.VIS_PLOTLY, language='python')

    '''
    # PCA plot with bokeh
    st.subheader('Scatter plots from Principal Component Analysis with bokeh')
    df_iris_pca = fit_transform_pca(df_iris)
    col1, col2 = st.columns(2)
    with col1:
        show_scatter2d_bokeh(df_iris_pca, labels)
    with col2:
        st.code(code.VIS_BOKEH, language='python')


    # Graph plot
    st.subheader('Graph plots with bokeh')
    
    G = nx.karate_club_graph()
    tools = 'hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,'
    plot = figure(
        title='Graph plots with bokeh',
        x_range=(-1.1,1.1), y_range=(-1.1,1.1),
        tools=tools)
    graph = from_networkx(G, nx.spring_layout, scale=2, center=(0,0))
    plot.renderers.append(graph)

    st.bokeh_chart(plot, use_container_width=True)

    st.code(code.VIS_BOKEH_GRAPH, language='python')
    '''

if __name__ == '__main__':
    view()
