import streamlit as st
from bokeh.plotting import figure, from_networkx
import networkx as nx


st.title('Test')
x = [1, 2, 3, 4, 5]
y = [6, 7, 2, 4, 5]
p = figure(
    title='simple line example',
    x_axis_label='x',
    y_axis_label='y')
p.line(x, y, legend_label='Trend', line_width=2)

st.bokeh_chart(p, use_container_width=True)

G = nx.karate_club_graph()

tools = "hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,"
plot = figure(
    x_range=(-1.1,1.1), y_range=(-1.1,1.1),
    tools=tools)

graph = from_networkx(G, nx.spring_layout, scale=2, center=(0,0))
plot.renderers.append(graph)

st.bokeh_chart(plot, use_container_width=True)
