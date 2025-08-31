import streamlit as st
import uuid
from backend import get_unique_threads,chatbot
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage,ToolMessage

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
            st.session_state['message_history'].append({
            "role":role,
            "content":message.content
        })
        elif isinstance(message,AIMessage):
            role="ai"
            st.session_state['message_history'].append({
            "role":role,
            "content":message.content
        })
        elif isinstance(message, ToolMessage) or isinstance(message, SystemMessage):
            continue
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

for thread_id in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(thread_id):
        reset_chat()
        st.session_state["session_thread_id"]=thread_id
        load_message_history(st.session_state["session_thread_id"])
        
# ==================================================================
# ========================Loading Messages==========================
# ==================================================================
for message in st.session_state["message_history"]:
    with st.chat_message(message['role']):
        st.text(message['content'])

# ==================================================================
# ========================Chatbot Section===========================
# ==================================================================
userInput=st.chat_input("Type Here")
config={
    "configurable":{
        "thread_id":st.session_state["session_thread_id"],
        "metadata": {"thread_id": st.session_state["session_thread_id"]},
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
        status_holder = {"box": None}
        def get_ai_message():
            for message_chunk,metadata in chatbot.stream(
                {
                    "messages":[HumanMessage(content=userInput)]
                    },
                config=config,
                stream_mode="messages"
            ):
                if isinstance(message_chunk,AIMessage):
                    yield message_chunk.content
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )
        ai_message=st.write_stream(get_ai_message()) 
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
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