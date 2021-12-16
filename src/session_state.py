import streamlit as st

from src import code


def view():
    st.title('Session state (New in 0.84.0)')
    st.header('BAD counter app')
    col1, col2 = st.columns(2)
    with col1:
        st.write('This app revert counter value by application event.')
        count = 0
        increment_count = st.button('count +', key='bad_app_increment_button')
        decrement_count = st.button('count -', key='bad_app_decrement_button')
        if increment_count:
            count += 1
        if decrement_count:
            count -= 1

        st.write(f'count: {count}')
    
    with col2:
        st.code(code.SESSION_BAD, language='python')
    
    st.header('Fixed counter app by session state')
    col1, col2 = st.columns(2)
    with col1:
        increment_count = st.button('count +')
        decrement_count = st.button('count -')
        if increment_count:
            st.session_state.count += 1
        if decrement_count:
            st.session_state.count -= 1

        st.write(f'count: {st.session_state.count}')
    
    with col2:
        st.code(code.SESSION_FIXED, language='python')


if __name__ == '__main__':
    view()