from langgraph.graph import state,START,END,StateGraph
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict,Annotated
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import BaseMessage,HumanMessage
from langgraph.graph.message import add_messages
load_dotenv()

llm=ChatOpenAI()

class chatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

def chat_node(state:chatState)->chatState:
    messages=state['messages']
    response=llm.invoke(messages)
    return {
        "messages":[response]
    }

checkpointer=InMemorySaver()

graph=StateGraph(chatState)
graph.add_node("chat_node",chat_node)
graph.add_edge(START,"chat_node")
graph.add_edge("chat_node",END)

chatbot=graph.compile(checkpointer=checkpointer)