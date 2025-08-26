import streamlit as st
from backend import chatbot
from langchain_core.messages import HumanMessage
import uuid
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()

# =============================================================================
# ==============================Utility Functions==============================
# =============================================================================
def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id=generate_thread_id()
    st.session_state['thread_id']=thread_id
    st.session_state['message_history']=[]
    add_thread(st.session_state['thread_id'])
    
    
def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
    if thread_id not in st.session_state['thread_titles']:
        st.session_state['thread_titles'][thread_id] = "New Chat"
        st.session_state['thread_title_set'][thread_id]=False

def load_coversation(thread_id):
    config={
        "configurable":{
            "thread_id":thread_id
        }
    }
    data=chatbot.get_state(config=config)
    return data.values['messages']
# =============================================================================
# ==============================Session Setup==================================
# =============================================================================
if "message_history" not in st.session_state:
    st.session_state["message_history"]=[]
if "thread_id" not in st.session_state:
    st.session_state["thread_id"]=generate_thread_id()
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"]=[]
if "thread_titles" not in st.session_state:
    st.session_state["thread_titles"]={}
if "thread_title_set" not in st.session_state:
    st.session_state["thread_title_set"]={}

add_thread(st.session_state['thread_id'])

    
# =============================================================================
# ==============================Sidebar UI=====================================
# =============================================================================

st.sidebar.title("LangGraph Chatbot")
if st.sidebar.button("New Chat"):
    reset_chat()
st.sidebar.header("My Conversations")
for thread_id in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(label=st.session_state["thread_titles"][thread_id],key=thread_id):
        st.session_state['thread_id']=thread_id
        messages=load_coversation(thread_id)
        temp_messages=[]
        for message in messages:
            if isinstance(message,HumanMessage):
               role='user'
            else:
               role="ai"
            temp_messages.append({
                "role":role,
                "content":message.content
            })
        st.session_state['message_history']=temp_messages
# =============================================================================
# ==============================Chatbot Setup==================================
# =============================================================================
config={
    "configurable":{
        "thread_id":st.session_state['thread_id']
    }
}

for message in st.session_state["message_history"]:
    with st.chat_message(message['role']):
        st.text(message['content'])

userInput=st.chat_input("Type Here")
if userInput:
    cur_thread_id=st.session_state['thread_id']
    title_set=st.session_state['thread_title_set'][cur_thread_id]
    if not title_set:
        st.session_state['thread_title_set'][cur_thread_id]=True
        prompt=f"""
        You will be given a User Input. I want to implement a feature like ChapGPT. What is does it uses the first user input and make a title of 3 to 4 words that is then used in sidebar. 
        you have to help me with that.
        Yo have to only give the 4 words answer. Don't add any thing like `Title: and then the answer.`
        User input is as follows:
        UserInput= {userInput}
        """
        model=ChatOpenAI(model="gpt-3.5-turbo")
        response=model.invoke(prompt)
        st.session_state['thread_titles'][cur_thread_id]=response.content
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
    
    