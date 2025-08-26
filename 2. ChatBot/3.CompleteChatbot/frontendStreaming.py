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
    with st.chat_message("ai"):
        aimessage=st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {
                    "messages":[HumanMessage(content=userInput)]
                },
                config=config,
                stream_mode="messages"
            )
        )

    
    st.session_state["message_history"].append({
        "role":"ai",
        "content":aimessage
    })
    
    