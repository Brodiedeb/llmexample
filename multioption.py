import streamlit as st

if 'dummy_data' not in st.session_state.keys():
    dummy_data = ['Influenza','ARDS','BRA','MEX','ARG']
    st.session_state['dummy_data'] = dummy_data
else:
    dummy_data = st.session_state['dummy_data']

def checkbox_container(data):
    st.header('Select A country')
    new_data = st.text_input('Enter country Code to add')
    cols = st.columns(10)
    if cols[0].button('Add Coutry'):
        dummy_data.append(new_data)
    if cols[1].button('Select All'):
        for i in data:
            st.session_state['dynamic_checkbox_' + i] = True
        st.experimental_rerun()
    if cols[2].button('UnSelect All'):
        for i in data:
            st.session_state['dynamic_checkbox_' + i] = False
        st.experimental_rerun()
    for i in data:
        st.checkbox(i, key='dynamic_checkbox_' + i)

def get_selected_checkboxes():
    return [i.replace('dynamic_checkbox_','') for i in st.session_state.keys() if i.startswith('dynamic_checkbox_') and st.session_state[i]]


checkbox_container(dummy_data)
st.write('You selected:')
st.write(get_selected_checkboxes())