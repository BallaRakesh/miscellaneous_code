import streamlit as st
from genai import calling_llm
# streamlit
st.title('Restaurant Name Generator')
cuisine = st.sidebar.selectbox("pick a cuisine", ('Indian', 'Italian', 'spanish'))
def generate_responce(cuisine):
    return calling_llm(cuisine)

if cuisine:
    responce_ = generate_responce(cuisine)
    st.header(responce_['restaurant_name'].strip())
    menu_items = responce_['menu_items_name'].strip().split(',')
    st.write('**MENU ITEMS**')
    for idx, items in enumerate(menu_items):
        st.write(idx+1, ')', items.strip())
        
        
api_key = 'AIzaSyBfkAGgG1xqlzkNeE_IxGDY5ATSwTEQpTw'