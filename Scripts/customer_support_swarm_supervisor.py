import datetime
from collections import defaultdict
from typing import Callable

from langchain_core.runnables import RunnableConfig
#from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

from langgraph_swarm import create_handoff_tool, create_swarm
from langgraph_supervisor import create_supervisor

from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
import torch
import pandas as pd

import os
from dotenv import load_dotenv

load_dotenv()

# API Key Configuration 
groq_api = os.getenv('GROQ_API_KEY')
hf_api = os.getenv('HF_TOKEN')

torch.classes.__path__ = [os.path.join(torch.__path__[0], 'torch', '_classes.py')]

#model = ChatOpenAI(model="gpt-4o")
model = ChatGroq(
    model="llama3-70b-8192",
    temperature=0
)

# Mock data for tools
RESERVATIONS = defaultdict(lambda: {"flight_info": {}, "hotel_info": {}})

'''TOMORROW = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
FLIGHTS = [
    {
        "departure_airport": "BOS",
        "arrival_airport": "JFK",
        "airline": "Jet Blue",
        "date": TOMORROW,
        "id": "1",
    }
]
HOTELS = [
    {
        "location": "New York",
        "name": "McKittrick Hotel",
        "neighborhood": "Chelsea",
        "id": "1",
    }
]'''

df_flight = pd.read_excel(r"C:\Tasks\Multimodal_Agentic_RAG\LangGraph_Supervisor_Swarm\Inputs\Sample_LG_Swarm_Data.xlsx",sheet_name="Flight_Info")
df_hotel = pd.read_excel(r"C:\Tasks\Multimodal_Agentic_RAG\LangGraph_Supervisor_Swarm\Inputs\Sample_LG_Swarm_Data.xlsx",sheet_name="Hotel_Info")

FLIGHTS = []
HOTELS = []

for i in range(len(df_flight)):
    df_f_d = {}
    df_f_d['departure_airport'] = df_flight['departure_airport'].iloc[i]
    df_f_d['arrival_airport'] = df_flight['arrival_airport'].iloc[i]
    df_f_d['airline'] = df_flight['airline'].iloc[i]
    df_f_d['date'] = df_flight['date'].iloc[i].date().isoformat()
    df_f_d['id'] = df_flight['id'].iloc[i]
    FLIGHTS.append(df_f_d)

for i in range(len(df_hotel)):
    df_f_h = {}
    df_f_h['location'] = df_hotel['location'].iloc[i]
    df_f_h['name'] = df_hotel['name'].iloc[i]
    df_f_h['neighborhood'] = df_hotel['neighborhood'].iloc[i]
    df_f_h['date'] = df_hotel['date'].iloc[i].date().isoformat()
    df_f_h['id'] = df_hotel['id'].iloc[i]
    HOTELS.append(df_f_h)

print("FLIGHTS = = = = ", FLIGHTS)
print("HOTELS = = = = ", HOTELS)


# Flight tools
def search_flights(
    departure_airport: str,
    arrival_airport: str,
    date: str,
) -> list[dict]:
    """Search flights.

    Args:
        departure_airport: 3-letter airport code for the departure airport. If unsure, use the biggest airport in the area
        arrival_airport: 3-letter airport code for the arrival airport. If unsure, use the biggest airport in the area
        date: YYYY-MM-DD date
    """
    # return all flights for simplicity
    return FLIGHTS


def book_flight(
    flight_id: str,
    config: RunnableConfig,
) -> str:
    """Book a flight."""
    user_id = config["configurable"].get("user_id")
    flight = [flight for flight in FLIGHTS if flight["id"] == flight_id][0]
    RESERVATIONS[user_id]["flight_info"] = flight
    return "Successfully booked flight"


# Hotel tools
def search_hotels(location: str) -> list[dict]:
    """Search hotels.

    Args:
        location: offical, legal city name (proper noun)
    """
    # return all hotels for simplicity
    return HOTELS


def book_hotel(
    hotel_id: str,
    config: RunnableConfig,
) -> str:
    """Book a hotel"""
    user_id = config["configurable"].get("user_id")
    hotel = [hotel for hotel in HOTELS if hotel["id"] == hotel_id][0]
    RESERVATIONS[user_id]["hotel_info"] = hotel
    return "Successfully booked hotel"


# Define handoff tools
transfer_to_hotel_assistant = create_handoff_tool(
    agent_name="hotel_assistant",
    description="Transfer user to the hotel-booking assistant that can search for and book hotels.",
)
transfer_to_flight_assistant = create_handoff_tool(
    agent_name="flight_assistant",
    description="Transfer user to the flight-booking assistant that can search for and book flights.",
)


# Define agent prompt
def make_prompt(base_system_prompt: str) -> Callable[[dict, RunnableConfig], list]:
    def prompt(state: dict, config: RunnableConfig) -> list:
        user_id = config["configurable"].get("user_id")
        current_reservation = RESERVATIONS[user_id]
        system_prompt = (
            base_system_prompt
            + f"\n\nUser's active reservation: {current_reservation}"
            + f"Today is: {datetime.datetime.now()}"
        )
        return [{"role": "system", "content": system_prompt}] + state["messages"]

    return prompt


# Define agents
flight_assistant = create_react_agent(
    model,
    [search_flights, book_flight, transfer_to_hotel_assistant],
    prompt=make_prompt("You are a flight booking assistant and nothing else"),
    name="flight_assistant",
)

hotel_assistant = create_react_agent(
    model,
    [search_hotels, book_hotel, transfer_to_flight_assistant],
    prompt=make_prompt("You are a hotel booking assistant and nothing else"),
    name="hotel_assistant",
)

# Compile and run!
checkpointer = InMemorySaver()
builder = create_swarm([flight_assistant, hotel_assistant], default_active_agent="flight_assistant")

# Important: compile the swarm with a checkpointer to remember
# previous interactions and last active agent
app_swarm = builder.compile(checkpointer=checkpointer, name="Swarm_Agent")
config = {"configurable": {"thread_id": "1", "user_id": "1"}}
# result = app.invoke({
#     "messages": [
#         {
#             "role": "user",
#             "content": "i am looking for a flight from boston to ny tomorrow"
#         }
#     ],
# }, config)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

loader = PyPDFLoader(file_path=r"C:\Tasks\Multimodal_Agentic_RAG\LangGraph_Supervisor_Swarm\Inputs\Las_Vegas_Wikipedia.pdf")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)
vector = Chroma.from_documents(documents, embeddings, persist_directory="./chromapdf_db")
retriever = vector.as_retriever(search_type="similarity_score_threshold",
                                                        search_kwargs={"k": 2, "score_threshold": 0.3})

retriever_tool = create_retriever_tool(
    retriever,
    "lasvegas_search",
    "Search for information about Las Vegas. For any questions about Las Vegas, you must use this tool which is RAG based and not Web based!",
)

tv_search = DuckDuckGoSearchRun()

tools = [retriever_tool, tv_search]

system_message="""You are helpful assisstant which will route and answer using tools(RAG or vectorstore tool, web search tool) based on user question. 
If user question is based on RAG that is retrieval from vectorstore, so you need to find the relevant answer, for this use the vectorstore tool which is already mentioned. 
If user question is not available in vectorstore then search from internet or web and answer, for this use tool duckduckgosearch which is already mentioned.
Also mention the tools used."""

rag_web_agent = create_react_agent(model, tools, name = "rag_web_agent", prompt=system_message)

llm_img = ChatGroq(
    model="llama3-70b-8192",
    temperature=0
)

image_agent = create_react_agent(
    model=llm_img,
    tools=[],
    name="image_expert",
    prompt="Read the image and describe about the image as in which place it belongs to and what does it have from path C:\\Tasks\\Multimodal_Agentic_RAG\\LangGraph_Supervisor_Swarm\\Inputs\\Yosemite.jpg"
)

# Create supervisor workflow
workflow_secsup = create_supervisor(
    [rag_web_agent, image_agent],
    model=model,
    prompt=(
        "You are a secondary supervisor managing a rag or web and a image expert. "
        "For searching from web or pdf vectorstore as rag based on user question, use rag_web_agent which further uses tools. "
        "For image analysis or anything with respect to image, use image_expert image_agent."
    )
)

# Compile and run
app_secondary_sup = workflow_secsup.compile(name="Secondary_Supervisor")


workflow_main_sup = create_supervisor(
    [app_swarm, app_secondary_sup],
    model=model,
    prompt=(
        "You are the main team supervisor managing another supervisor named Secondary_Supervisor(which manages rag or web and a image expert) and then swarm of flight and hotel booking agents. "
        "For searching from web or pdf vectorstore as rag based or image analysis on user question, use the secondary supervisor named Secondary_Supervisor app_secondary_sup which further manages the agents. "
        "For any user question strictly related to only flight and hotel booking information, use the swarm of agents named Swarm_Agent app_swarm which can give info as well as book flights or hotels. Also use the config dictionary which has user info for only swarm agent"
    )
)

# Compile and run
app_main_sup = workflow_main_sup.compile(name="Main_Supervisor")

result = app_main_sup.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What is Las Vegas"
        }
    ]
}, config)

for m in result["messages"]:
    m.pretty_print()