import logging

import streamlit as st

from src import widget, visualization, cache, session_state


if 'count' not in st.session_state:
    logging.info('initialize variable "count"')
    st.session_state.count = 0

''''''
selectbox_result = st.sidebar.selectbox(
    'Menu',
    ('Widget', 'Visualization', 'Cache', 'Session State')
)

if selectbox_result == 'Widget':
    widget.view()
elif selectbox_result == 'Visualization':
    visualization.view()
elif selectbox_result == 'Cache':
    cache.view()
elif selectbox_result == 'Session State':
    session_state.view()
