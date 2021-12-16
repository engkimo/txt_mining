WIDGET_CHECKBOX = """checkbox_state = st.checkbox('Show text')
if checkbox_state:
    st.write('checkbox enable')
"""

WIDGET_BUTTON = """button_state = st.button('Say hello')
if button_state:
    st.write('Why hello there')
else:
    st.write('Goodbye')
"""

WIDGET_SELECTBOX = """option = st.selectbox(
    'select box:',
    [1, 2, 3]
)

st.write('You selected: ', option)
"""

WIDGET_INPUT = """title = st.text_input('inputbox', 'hello')
st.write('inputbox:', title)
"""

WIDGET_FILEUPLOADER = """uploaded_file = st.file_uploader('Choose a image file')
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    st.image(
        image, caption='upload images',
        use_column_width=True
    )
"""

WIDGET_EXPANDER = """with st.expander('See detail'):
    st.write('Hello expander!')
"""

WIDGET_DOWNLOAD = """binary_contents = b'example content'
st.download_button('Download binary file', binary_contents)
"""

VIS_LINEPLOT = """chart_data = load_time_series_data()
st.line_chart(chart_data)
"""

VIS_HEATMAP = """fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(df_iris.corr(), annot=True, ax=ax)
st.pyplot(fig)
"""

VIS_PLOTLY = """column = df_iris.columns[0]
st.write(f'- {column}')
series = df_iris[column]
target_names = list(set(labels))
hist_data = [series[labels == name] for name in target_names]
fig = ff.create_distplot(
    hist_data, target_names, bin_size=[.1, .25, .5])
st.plotly_chart(fig, use_container_width=True)"""

VIS_BOKEH = """df_iris_pca = fit_transform_pca(df_iris)
# bokeh向け前処理
tools = 'hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,'
colormap = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
colors = [colormap[t] for t in labels]
x = df_iris_pca[:, 0]
y = df_iris_pca[:, 1]
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
st.bokeh_chart(p, use_container_width=True)"""

VIS_BOKEH_GRAPH = """G = nx.karate_club_graph()

tools = 'hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,'
plot = figure(
    x_range=(-1.1,1.1), y_range=(-1.1,1.1),
    tools=tools)

graph = from_networkx(G, nx.spring_layout, scale=2, center=(0,0))
plot.renderers.append(graph)

st.bokeh_chart(plot, use_container_width=True)
"""

CACHE_EXAMPLE = """@st.cache
def progress_cache(i):
    time.sleep(0.05)

def progress_no_cache(i):
    time.sleep(0.05)

def view_bar(func):
    # Add a placeholder
    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(20):
        # Update the progress bar with each iteration.
        latest_iteration.text(f'Iteration {i+1}')
        bar.progress((i + 1)*5)
        func(i)

st.write('Starting a long computation with cache...')
view_bar(progress_cache)

st.write('Starting a long computation without cache...')
view_bar(progress_no_cache)
"""

SESSION_BAD = """count = 0
increment_count = st.button('count +')
decrement_count = st.button('count -')
if increment_count:
    count += 1
if decrement_count:
    count -= 1

st.write(f'count: {count}')
"""

SESSION_FIXED = """if 'count' not in st.session_state:
    st.session_state.count = 0

st.write('This app revert counter value by application event.')
increment_count = st.button('count +')
decrement_count = st.button('count -')
if increment_count:
    st.session_state.count += 1
if decrement_count:
    st.session_state.count -= 1

st.write(f'count: {st.session_state.count}')
"""