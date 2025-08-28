from langgraph.graph import StateGraph,START,END
from langchain_openai import ChatOpenAI
from typing import TypedDict,Annotated
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage,HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
load_dotenv()
model=ChatOpenAI()
class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]
def chat_node(state:ChatState)->ChatState:
    response=model.invoke(state['messages'])
    return {"messages" : [response]}
con=sqlite3.connect(database='chatbot.db',check_same_thread=False)
checkpointer=SqliteSaver(con)
graph=StateGraph(ChatState)
graph.add_node("chat_node",chat_node)
graph.add_edge(START,"chat_node")
graph.add_edge("chat_node",END)
chatbot=graph.compile(checkpointer=checkpointer)

config={
    "configurable":{
        "thread_id":"thread-1"
    }
}
def get_unique_threads():
    unique_thread=set()
    for checkpoint in checkpointer.list(None):
        unique_thread.add(checkpoint.config['configurable']['thread_id'])
    return list[unique_thread]

# chatbot.invoke({'messages':[HumanMessage(content="What is my name")]},config=config)
# messages=chatbot.get_state(config=config)