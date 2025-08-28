import streamlit as st
import uuid
from backend import get_unique_threads,chatbot
from langchain_core.messages import HumanMessage
# ==================================================================
# =======================Utility Functions==========================
# ==================================================================
def reset_chat():
    st.session_state['message_history']=[]
def generate_thread():
    return str(uuid.uuid4())
def load_message_history(thread_id):
    config={"configurable":{"thread_id":thread_id}}
    messages=chatbot.get_state(config=config)
    for message in messages.values['messages']:
        if isinstance(message,HumanMessage):
            role="human"
        else:
            role="ai"
        st.session_state['message_history'].append({
            "role":role,
            "content":message.content
        })
# ==================================================================
# ========================Session Settings==========================
# ==================================================================
if "session_thread_id" not in st.session_state:
    st.session_state['session_thread_id']=generate_thread()
if "chat_threads" not in st.session_state:
    st.session_state['chat_threads']=get_unique_threads()
if "message_history" not in st.session_state:
    st.session_state["message_history"]=[]
    
# ==================================================================
# ========================Sidebar Settings==========================
# ==================================================================
st.sidebar.header("LangGraph Chatbot")
if st.sidebar.button("New Chat"):
    reset_chat()
    thread_id=generate_thread()
    st.session_state['session_thread_id']=thread_id
    st.session_state['chat_threads'].append(thread_id)
st.sidebar.title("My Conversations")

for thread_id in st.session_state['chat_threads']:
    if st.sidebar.button(thread_id):
        reset_chat()
        st.session_state["session_thread_id"]=thread_id
        load_message_history(st.session_state["session_thread_id"])
        

for message in st.session_state["message_history"]:
    with st.chat_message(message['role']):
        st.text(message['content'])

userInput=st.chat_input("Type Here")
config={
    "configurable":{
        "thread_id":st.session_state["session_thread_id"]
    }
}
if userInput:
    st.session_state["message_history"].append({
        "role":"human",
        "content":userInput
    })
    with st.chat_message("human"):
        st.text(userInput)
    with st.chat_message("ai"):
        ai_message=st.write_stream(
            message_chunk.content for message_chunk,metadata in chatbot.stream(
                {
                    "messages":[HumanMessage(content=userInput)]
                    },
                config=config,
                stream_mode="messages"
            )
        )    
        st.session_state["message_history"].append({
            "role":"ai",
            "content":ai_message
            })
        
    
    # response=chatbot.invoke({"messages":[HumanMessage(content=userInput)]},config=config)
    # st.session_state["message_history"].append({
    #     "role":"ai",
    #     "content":response['messages'][-1].content
    # })
    # with st.chat_message("ai"):
    #     st.text(response['messages'][-1].content)