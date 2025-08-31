from langgraph.graph import StateGraph,START,END
from langchain_openai import ChatOpenAI
from typing import TypedDict,Annotated
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode,tools_condition
import requests
import sqlite3
import os
# ==================================================================
# =======================Loading env================================
# ==================================================================
load_dotenv()
model=ChatOpenAI()
# ==================================================================
# =======================Making State Class=========================
# ==================================================================
class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

# ==================================================================
# ============================Tools=================================
# ==================================================================
search_tool=TavilySearch(tavily_api_key=os.environ["TRAVILY_API_KEY"])
@tool()
def get_weather_data(city:str):
    "This tool Fetches the current weather for a given city, including temperature, humidity, wind, air quality, sunrise/sunset times, and other local conditions. Useful for answering questions about the weather, air quality, or daily forecasts in specific locations."
    api_key=os.environ['WEATHERSTACK_API_KEY']
    url=f"https://api.weatherstack.com/current?access_key={api_key}&query={city}"
    r=requests.get(url)
    return r.json()
@tool
def get_stockmarket_data(symbol:str):
    """Get the latest stock quote for a company by its ticker symbol.
    Returns open, high, low, current price, volume, previous close,
    change amount, and percent change for the latest trading day.
    Example: symbol='TSLA' (Tesla), 'AAPL' (Apple), 'MSFT' (Microsoft).
    """
    api_key=os.environ['ALPHAVANTAGE_API_KEY']
    url=f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    r=requests.get(url)
    return r.json()
tools=[search_tool,get_stockmarket_data,get_weather_data]
toolModel=model.bind_tools(tools)
# ==================================================================
# ============================Graph=================================
# ==================================================================

# ============================Nodes=================================
def chat_node(state:ChatState)->ChatState:
    """LLM node that may answer or request a tool call."""
    response=toolModel.invoke(state['messages'])
    return {"messages" : [response]}
tool_node=ToolNode(tools)

# ============================Database==============================
con=sqlite3.connect(database='chatbot.db',check_same_thread=False)

checkpointer=SqliteSaver(con)
# ============================Nodes=================================
graph=StateGraph(ChatState)
graph.add_node("chat_node",chat_node)
graph.add_node("tools",tool_node)
graph.add_edge(START,"chat_node")
graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge("tools","chat_node")


chatbot=graph.compile(checkpointer=checkpointer)

# config={
#     "configurable":{
#         "thread_id":"thread-1"
#     }
# }
def get_unique_threads():
    unique_thread=set()
    for checkpoint in checkpointer.list(None):
        unique_thread.add(checkpoint.config['configurable']['thread_id'])
    return list(unique_thread)

# chatbot.invoke({'messages':[HumanMessage(content="What is my name")]},config=config)
# messages=chatbot.get_state(config=config)
# print(messages.values['messages'])