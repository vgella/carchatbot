import streamlit as st
import faiss
import torch
from langchain_community.chat_models import ChatOpenAI
import numpy as np
import json
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModel
#import os; os.environ["OPENAI_API_KEY"] = "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"

faiss_model = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(faiss_model)
model = AutoModel.from_pretrained(faiss_model)
# Set Streamlit page configuration
st.set_page_config(page_title='CarChatBot🤖', layout='wide')

# Initialize session states
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []


# Load synthetic car data from JSON file
def load_synthetic_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

data = load_synthetic_data("synthetic_data.json")

# Build FAISS index
def embed_text(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        embeddings = model(**tokens).pooler_output
    return embeddings[0].numpy()

def build_faiss_index(data):
    documents = []
    embeddings = []

    for car in data:
        # Combine all relevant fields into a single text representation
        text = f"{car['make']} {car['model']} ({car['year']}) - Specs: {car['specifications']} | Price: {car['pricing']} | Special Offers: {car['special_offers']} | Reviews: {car['reviews']}"
        documents.append(text)
        embeddings.append(embed_text(text))  # Replace with your embedding logic

    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, documents

index, documents = build_faiss_index(data)

# Retrieve documents from FAISS index
def retrieve_documents(query, index, documents, top_k=2):
    query_embedding = np.random.rand(384).reshape(1, -1)  # Replace with your embedding logic
    distances, indices = index.search(query_embedding, top_k)
    return [documents[i] for i in indices[0]]

# Define function to get user input
def get_text():
    input_text = st.text_input("You: ", st.session_state["input"], key="input",
                               placeholder="Your AI assistant here! Ask me anything ...",
                               label_visibility='hidden')
    return input_text

# Define function to start a new chat
def new_chat():
    save = []
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    st.session_state.entity_memory.entity_store = {}
    st.session_state.entity_memory.buffer.clear()

# Set up sidebar with various options
with st.sidebar.expander("🛠️ ", expanded=False):
    if st.checkbox("Preview memory store"):
        with st.expander("Memory-Store", expanded=False):
            st.session_state.entity_memory.store
    if st.checkbox("Preview memory buffer"):
        with st.expander("Bufffer-Store", expanded=False):
            st.session_state.entity_memory.buffer
    MODEL = st.selectbox(label='Model', options=['gpt-3.5-turbo', 'gpt-4o-mini', 'gpt-4o'])
    K = st.number_input(' (#)Summary of prompts to consider', min_value=3, max_value=1000)

# Set up the Streamlit app layout
st.title("🤖 Car Dealership Chatbot with Memory 🧠")
st.subheader("Powered by 🦜 LangChain + OpenAI + Streamlit")

# Ask the user to enter their OpenAI API key
from langchain.prompts import PromptTemplate

# Define the custom ENTITY_MEMORY_CONVERSATION_TEMPLATE
CAR_TEMPLATE = PromptTemplate(
    template="""
    You are a helpful assistant for a car dealership. Use the context provided to answer questions, remember past interactions, and optimize for actionable results. Consider modern advertising methods to provide responses that are concise, retain the viewer's attention, and convert leads. Optimize for conciseness; the user is the average american and will lose attention with every additional word. response's should optimize user engagement and provide meaningful information. After each response, include a list of features of the car that the user might be interested in based on the conversation so far (this should be formatted like a python list and only include 1 to 3 word feature descriptions per entry). Keep conversation limited to these purposes, but feel free to ask additional questions that provide insight into the features the user is looking for. The features should not be formatted in markdown, and only should appear as a list such as [towing capacity, mileage, additional seating]. The list should always be presented in this format explicitly, a comma seperated list with brackets. This list of features should be specific to the user's responses and will provide insight into our backend, indicating which features to display for cars viewed. Subtly suggest mazda occasionally where it is natural, break away from this if and when it seems conversationally unnatural.

    Entities:
    {entities}

    Conversation History:
    {history}

    User Query:
    {input}

    Respond concisely and ensure user engagement with follow-up questions when appropriate.
    """,
    input_variables=["entities", "history", "input"]
)


openai_api_key=st.secrets["OPENAI_API_KEY"]
print(openai_api_key)
if True:
    # Create an OpenAI instance
    llm = ChatOpenAI(temperature=0,
                 model=MODEL,
                 openai_api_key=openai_api_key,
                 verbose=False)

    # Create a ConversationEntityMemory object if not already created
    if 'entity_memory' not in st.session_state:
        st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=K)
        print("here")

    # Define a custom prompt for the car dealership chatbot
    CAR_DEALERSHIP_PROMPT = PromptTemplate(
        template="""
        You are an AI car dealership assistant chatbot. Engage users in natural conversation, provide accurate car-related information, 
        and aim to convert leads into actionable outcomes like test drives, financing details, or booking service appointments.

        Always include relevant vehicle information based on context and user preferences. End responses with a follow-up question to 
        encourage user engagement. Be friendly, professional, and concise.

        {context}
        Chat History:
        {history}
        User Query: {query}
        """,
        input_variables=["context", "history", "query"]
    )

    # Create the ConversationChain object with the specified configuration
    Conversation = ConversationChain(
        llm=llm,
        prompt=CAR_TEMPLATE,
        memory=st.session_state.entity_memory
    )
else:
    st.sidebar.warning('API key required to try this app. The API key is not stored in any form.')

# Add a button to start a new chat

# Get the user input
user_input = get_text()

if user_input:
    # Retrieve relevant context using FAISS
    retrieved_docs = retrieve_documents(user_input, index, documents)
    context = "\n".join(retrieved_docs)
    print(context)

    human_input = f"User Query: {user_input}\nContext: {context}\nHistory: {st.session_state['past']}"
    print(human_input)
    # Generate the output using the ConversationChain object and the user input
    output = Conversation.run(human_input)

    # Store the conversation in session state
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

# Display the conversation history using an expander
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        st.info(st.session_state["past"][i], icon="🧐")
        st.success(st.session_state["generated"][i], icon="🤖")

