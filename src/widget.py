import io 
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd
import nlplot
import plotly.figure_factory as ff
import plotly.graph_objects as go
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sus 
from matplotlib.font_manager import FontProperties
import nlplot
import streamlit as st
from PIL import Image, ImageDraw
import base64

from src import code

def view():
    st.title('CITY-INSIGHT　Text Mining DEMO')

    df_edu = pd.read_csv('/opt/streamlit/src/nightley_sns_data_tokenized.csv')
    #df = pd.read_csv('/opt/streamlit/src/nightley_sns_data_tokenized.csv')
    # show_df = st.sidebar.checkbox('Show dataFrame')
    show_df = st.checkbox('Show dataFrame')
    
    # df_edu = df.head(100)

    if show_df == True:
        st.write(df_edu)

    npt = nlplot.NLPlot(df_edu, target_col = 'trimmed_text2')

    stopwords = npt.get_stopword(top_n = 1, min_freq = 1)

    item_select = st.selectbox(
    label = 'Select item',
    options = ['Word count','Tree Map', 'Word Distribution'
                , 'Sunburst Chart', 'Co-occurrence network'
                , 'Word Cloud', 'Circle Chart'
                ])

    if item_select == 'Word count':
        st.write(npt.bar_ngram(title='Word count(one word)', ngram = 1, top_n = 20, stopwords = stopwords, width = 1000, height = 700))
        st.write(npt.bar_ngram(title='Word count(two word)', ngram = 2, top_n = 20, stopwords = stopwords, width = 1000, height = 700))

    if item_select == 'Tree Map':
        st.write(npt.treemap(title='Tree Map', ngram=1, top_n = 20, stopwords = stopwords, width = 1000, height = 700))

    if item_select == 'Word Distribution':
        st.write(npt.word_distribution(title='Word Distribution', xaxis_label='count', width = 1000, height = 700))

    if item_select == 'Co-occurrence network':
        npt.build_graph(stopwords = stopwords, min_edge_frequency = 1)
        # st.write(npt.co_network(title = 'Co-occurrence network',width = 1000, height = 700))
        st.write(npt.co_network(title = 'Co-occurrence network',width = 1000, height = 700))
        #st.pyplot()

    if item_select == 'Sunburst Chart':
        npt.build_graph(stopwords = stopwords, min_edge_frequency=1)
        st.write(npt.sunburst(title='Sunburst Chart', width = 1000, height=700))

    if item_select == 'Word Cloud':
        st.set_option('deprecation.showPyplotGlobalUse', False)
        npt.build_graph(stopwords = stopwords, min_edge_frequency=2)
        st.write(npt.wordcloud(max_words = 100, max_font_size = 100, stopwords = stopwords, 
        colormap = 'tab20_r', width = 1000, height = 600))
        st.pyplot()
        
    if item_select == 'Circle Chart':
        df_target = df_edu[['post_id', 'geo_0']].groupby('geo_0').count() / len(df_edu)
        fig_target = go.Figure(data=[go.Pie(labels=df_target.index,
                                            values=df_target['post_id'],
                                            hole=.3)])
        fig_target.update_layout(showlegend=False,
                                height=400,
                                margin={'l': 20, 'r': 60, 't': 0, 'b': 0})
        fig_target.update_traces(textposition='inside', textinfo='label+percent')

        # Layout (Sidebar)
        '''
        st.markdown("## Settings")
        cat_selected = st.selectbox('Categorical Variables', vars_cat)
        cont_selected = st.selectbox('Continuous Variables', vars_cont)
        cont_multi_selected = st.multiselect('Correlation Matrix', vars_cont,
                                            default=vars_cont)
        '''
        st.markdown("## 都道府県別比率")
        st.plotly_chart(fig_target, use_container_width=True)

    '''
    # import
    st.header('Import streamlit')
    st.code("""import streamlit as st""", language='python')

    # checkbox
    st.header('Checkbox')
    col1, col2 = st.columns(2)
    with col1:
        checkbox_state = st.checkbox('Show text')
        if checkbox_state:
            st.write('checkbox enable')

    with col2:
        st.code(code.WIDGET_CHECKBOX, language='python')

    # button
    st.header('Button')
    col1, col2 = st.columns(2)
    with col1:
        button_state = st.button('Say hello')
        if button_state:
            st.write('Why hello there')
        else:
            st.write('Goodbye')
    
    with col2:
        st.code(code.WIDGET_BUTTON, language='python')

    # selectbox
    st.header('Selectbox')
    col1, col2 = st.columns(2)
    with col1:
        option = st.selectbox(
            'select box:',
            [1, 2, 3]
        )
        st.write('You selected: ', option)
    
    with col2:
        st.code(code.WIDGET_SELECTBOX, language='python')

    # inputbox
    st.header('Inputbox')
    col1, col2 = st.columns(2)
    with col1:
        title = st.text_input('inputbox', 'hello')
        st.write('inputbox:', title)
    
    with col2:
        st.code(code.WIDGET_INPUT, language='python')

    # file upload
    st.header('File uploader')
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader('Choose a image file')
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            st.image(
                image, caption='upload images',
                use_column_width=True
            )
    
    with col2:
        st.code(code.WIDGET_FILEUPLOADER, language='python')

    # Expander
    st.header('Expander (New in 0.86.0)')
    col1, col2 = st.columns(2)
    with col1:
        with st.expander('See detail'):
            st.write('Hello expander!')

    with col2:
        st.code(code.WIDGET_EXPANDER, language='python')

    # Download button
    st.header('Download button (New in 0.88.0)')
    col1, col2 = st.columns(2)
    with col1:
        binary_contents = b'example content'
        st.download_button('Download binary file', binary_contents)

    with col2:
        st.code(code.WIDGET_DOWNLOAD, language='python')
    '''

if __name__ == '__main__':
    view()
