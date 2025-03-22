import streamlit as st
from implementation import compiled_agent

user_msg = st.chat_input("Choose field or skill you want to learn about")
if user_msg:
    st.write(user_msg)
    state = compiled_agent.invoke({"user_msg": user_msg})
    st.write(state["df"])