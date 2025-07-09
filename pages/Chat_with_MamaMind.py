import os
import time
import json
from dotenv import load_dotenv
import streamlit as st
from utils import *
from textwrap import dedent
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

st.set_page_config(page_title="MamaMind", page_icon=":female-doctor:")

col1, col2 = st.columns([3, 1])

with st.sidebar:
    st.image("static\image1-removebg-preview.png")

# Apply custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("styles.css")

# Define constants
DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'
SYSTEM_PROMPT = dedent("""
    You are a mental health adviser. You should use your knowledge of cognitive behavioral therapy,
    meditation techniques, mindfulness practices, and other therapeutic methods to guide the user through 
    their feelings and improve their well-being. 
    You will respond to the user's questions based on the {input}, {severity} of their perinatal depression.

    * For mild severity, offer supportive advice and practical coping strategies.
    * For moderate severity, provide empathetic responses and suggest helpful resources.
    * For severe severity, advise the user to seek professional help and share urgent resources.
    After addressing the user's question, include a relevant follow-up question only if it feels natural to continue the conversation. 
    Avoid overwhelming the user with too many questions.

    If the conversation ends with 'thank you,' 'thanks,' 'bye,' or 'goodbye,' conclude in a friendly manner with: 
    'I hope this helps. If you have more questions, feel free to ask!'
    
    "Context: {context}"
    "Input: {input},{severity}"
    """
)

# Function to load questions from the JSON file
def load_questions(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    
def display_welcome_message(llm):
    prompt = """
            You are an AI assistant designed to welcome users to the "MamaMind-Gentle Guidance for new beginnings" app.
            Your task is to greet the user with an appealing welcome message and then prompt them to provide their query or concern. Follow these steps:

            1. Start with a warm and friendly welcome message.
            2. Briefly explain what the "MamaMind" app does. 
            3. Invite the user to answer the Edinurgh depression questionnaire first so as to assess the depression level.

            The app provides the expecting and new mothers with mental health recommendations based on cognitive behavioral therapy, 
            assesses the severity of perinatal depression using Edinburgh Depression Scale and provides the response accordingly.
            
            Here is an example format you can follow:
            ---
            **Welcome to MamaMind!**

            We are here to support you through your journey with perinatal depression. Whether you are expecting a baby or navigating the postpartum period, MamaMind offers compassionate advice, helpful resources, and a listening ear.

            **How can we assist you today?**

            Please share your questions or concerns, and let us provide the support you need.
            ---
            """
    response = llm.invoke(prompt)
    message_text = response.content
    lines = message_text.split(" ")
    for line in lines:
        yield line + " "
        time.sleep(0.02)

def interpret_epds_score(score):
    if score <= 9:
        return "None to minimal depression"
    elif 10 <= score <= 12:
        return "Mild depression"
    elif 13 <= score <= 14:
        return "Moderate depression"
    else:
        return "Severe depression"

def begin_chat(llm, severity):
    with st.chat_message("assistant"):
        st.write("How can I help you?")
    
    query = st.chat_input("Tell me about your problems")

    if query:
        if os.path.exists(DB_FAISS_PATH):
            context_docs = get_similar_docs(query)
            context = " ".join(context_docs.page_content)
            with col2:
                st.write("RAG output")
                # st.markdown(context_docs.page_content)
                st.markdown(context_docs.metadata)
        else:
            directory = './data'
            documents = load_docs(directory)
            docs = split_docs(documents)
            generate_embeddings(docs)


        # Initialize LLM
        llm = llm
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "{input}"),
            ]
        )
        
        # Create QA chain
        db = FAISS.load_local(DB_FAISS_PATH, HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings", model_kwargs={'device': 'cpu'}), allow_dangerous_deserialization=True)
        qa_chain = create_qa_chain(llm, db, prompt)

        # Get response
        response = qa_chain.invoke({"context": context, "input": query, "severity": severity})
        answer = response['answer']

        # Insert new messages at the beginning of the list
        st.session_state.messages.insert(0, {"role": "assistant", "content": answer})
        st.session_state.messages.insert(0, {"role": "user", "content": query})

        # Display messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])

def main():    
    
    # Get user's Groq api key
    with st.sidebar:
        groq_api_key = st.text_input(label = "**Groq API key**", placeholder="Ex gsk-2twmA8tfCb8un4...",
        key ="groq_api_key_input", help = "How to get a Groq api key: Visit https://console.groq.com/login")

        # Initialize session state for the model if it doesn't already exist
        if 'selected_model' not in st.session_state:
            st.session_state['selected_model'] = ""
        # Container for markdown text
        with st.container():
            st.markdown("""Make sure you have entered your API key.
                        Don't have an API key yet?
                        Visit https://console.groq.com/login and Get your API key""")
            st.session_state['selected_model'] = st.selectbox("Choose the model",("","llama3-70b-8192","gemma2-9b-it"),key="tab2_sidebar_selectbox")
            model_chosen = st.session_state['selected_model']
        if model_chosen:
            st.write(f"You have selected the model: {model_chosen}")
        else:
            st.write("Please select a model.")

    with col1:
        st.markdown("<div style='text-align: center; font-size: 32px; font-weight: bold;'>👩‍⚕️ MamaMind</div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align: center; font-size: 24px;'>Gentle Guidance for New Beginnings</div", unsafe_allow_html=True)
        st.divider() 

        if groq_api_key and model_chosen:
            
            llm = initialize_llm(model_chosen, groq_api_key)

            # Initialize session state for messages
            if "messages" not in st.session_state:
                st.session_state.messages = []
            if "query" not in st.session_state:
                st.session_state.query = ""
            # Initialize session state to keep track of question index and responses
            if 'started' not in st.session_state:
                st.session_state.started = False
            if 'question_index' not in st.session_state:
                st.session_state.question_index = 0  # to start from the first question in the list
            if 'responses' not in st.session_state:
                st.session_state.responses = []
            if 'scores' not in st.session_state:
                st.session_state.scores = []

            # Load the EPDS questions
            epds_questions = load_questions('static/epds_questions.json')
            if "welcome_message_displayed" not in st.session_state:
                #Generate Welcome Message with User Input
                with st.chat_message("assistant"):
                    llm = llm
                    st.write_stream(display_welcome_message(llm))
                    st.session_state["welcome_message_displayed"] = True

            # Function to move to the next question
            def next_question(response, score):
                st.session_state.responses.append(response)
                st.session_state.scores.append(score)
                if st.session_state.question_index < len(epds_questions) - 1:
                    st.session_state.question_index += 1
                else:
                    st.session_state.question_index = 'completed'

            # Ask if the user wants to start the questionnaire
            if not st.session_state.started:
                with st.chat_message("assistant"):
                    # st.write("To assess the level of severity of the depression, we would like you to answer the Ediburgh Depression questionnaire.")
                    start = st.radio("Would you like to answer the EPDS questionnaire?", ("Yes", "No"), index=None, horizontal=True, key='start_radio')
                #st.write("Start: ", start)
                if start == "Yes":
                    st.session_state.started = True
                    st.session_state.question_index = 0
                    st.session_state.responses = []
                    st.session_state.scores = []
                    st.rerun()
                elif start == "No":
                    st.session_state.started = False
                    severity = "None"
                    
                    st.write("Thank you for your time. Ask any questions you have to MamaMind 🙂.")
                    begin_chat(llm, severity)
            else:
                # st.write("Session state started: ", st.session_state.started)
                # st.write("Question index: ", st.session_state.question_index)
                if st.session_state.question_index == 'completed':
                    with st.chat_message('assistant'):
                        st.write("You have completed the EPDS questionnaire.")
                    epds_score = sum(st.session_state.scores)
                    severity = interpret_epds_score(epds_score)
                    answer = llm.invoke(severity)
                    with st.chat_message('assistant'):
                        st.write(answer.content)
    
                    begin_chat(llm, severity)
                    # st.write("Your total score:", epds_score)
                    # st.write("Severity:", severity)
                else:
                    current_question = epds_questions[st.session_state.question_index]
                    with st.chat_message("assistant"):
                        st.write(current_question["question"])
                        response = st.radio("Select your response:", current_question["options"], index=None, horizontal=True)
                    if st.button("**Next**"):
                        if response:  # Ensure a response is selected
                            score = current_question["scores"][current_question["options"].index(response)]
                            next_question(response, score)
                            st.rerun()
                        else:
                            st.warning("Please select a response before proceeding.")

    with col2:
        st.write("Example Prompts")
        with st.expander("View Example Prompts"):
            st.markdown("""
            - **How can I manage my anxiety during pregnancy?**
            - **What are some coping strategies for postpartum depression?**
            - **Can you suggest activities to improve my mental health?**
            - **What resources are available for new mothers?**
            - **How do I recognize the signs of perinatal depression?**
            """)
        
if __name__ == "__main__":
    main()