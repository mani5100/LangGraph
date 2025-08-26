import streamlit as st
from backend import chatbot
from langchain_core.messages import HumanMessage
if "message_history" not in st.session_state:
    st.session_state["message_history"]=[]

for message in st.session_state["message_history"]:
    with st.chat_message(message['role']):
        st.text(message['content'])

userInput=st.chat_input("Type Here")
config={
    "configurable":{
        "thread_id":1
    }
}
if userInput:
    st.session_state["message_history"].append({
        "role":"user",
        "content":userInput
    })
    with st.chat_message("user"):
        st.text(userInput)
    response=chatbot.invoke({
        'messages':[
            HumanMessage(content=userInput)
        ]
    },config=config)
    ai_message= response['messages'][-1].content
    st.session_state["message_history"].append({
        "role":"ai",
        "content":ai_message
    })
    with st.chat_message("ai"):
        st.text(ai_message)
    