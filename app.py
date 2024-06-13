# Necessary Imports
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv
import PyPDF2
import os
import io
import speech_recognition as sr
import pyttsx3
from concurrent.futures import ThreadPoolExecutor
from htmlTemplates import user_template, bot_template, css  # Import from your template file

#st.title("Chat Your PDFs")
st.image(open("logo2.jpg", "rb").read())
st.markdown(css, unsafe_allow_html=True)  # Apply custom CSS for message styling

# Load environment variables
load_dotenv()

# Retrieve API key from environment variable
google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key is None:
    st.warning("API key not found. Please set the google_api_key environment variable.")
    st.stop()

# Initialize session state
if 'vector_index' not in st.session_state:
    st.session_state.vector_index = None

def initialize_vector_index(uploaded_file):
    if uploaded_file is not None:
        pdf_data = uploaded_file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_data))
        context = "\n\n".join(page.extract_text() for page in pdf_reader.pages)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
        texts = text_splitter.split_text(context)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        return FAISS.from_texts(texts, embeddings).as_retriever()
    return None

st.sidebar.header("PDF File Processor")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:  # Check if the file has been uploaded      
    st.sidebar.success("File uploaded successfully!") # Show success message immediately after file upload
if st.sidebar.button("Process PDF"):
    st.session_state.vector_index = initialize_vector_index(uploaded_file)
    st.sidebar.success("PDF is processed successfully!")                           

# Initialize speech recognition and TTS
r = sr.Recognizer()
executor = ThreadPoolExecutor(max_workers=1)

# Initialize session state for storing conversation history
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

if 'text_input' not in st.session_state:
    st.session_state['text_input'] = ''

def display_message():
    for message, is_user in st.session_state.conversation:
        template = user_template if is_user else bot_template
        formatted_message = template.replace('{{MSG}}', message)
        st.markdown(formatted_message, unsafe_allow_html=True)

def capture_audio():
    st.info("🎤 Listening... Please speak your question now!")
    r = sr.Recognizer()
    with sr.Microphone(device_index=2) as source:
        audio = r.listen(source, timeout=10)
    st.empty()

    try:
        recognized_text = r.recognize_google(audio)
        if recognized_text:
            st.success('Your audio is captured.')  # Display success message             
            return recognized_text
    except sr.UnknownValueError:
        st.warning("Could not understand audio, please try again.")
    except sr.RequestError:
        st.error("API unavailable. Please check your internet connection and try again.")
    return ""

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def process_question(user_question, docs, is_voice=False):
    prompt_template = """
    Answer the question as detailed as possible from the provided context,
    make sure to provide all the details, if the answer is not in the provided context just say, 
    'answer is not available in the context', don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.8, api_key=google_api_key)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    response_text = response['output_text']
    st.session_state.conversation.append((response_text, False))  # Add bot response to history
    display_message()  # Display the entire conversation
    if is_voice:
        executor.submit(text_to_speech, response_text)

def clear_text_input():
    st.session_state['text_input'] = ''

# Display the conversation chain
display_message()

st.subheader("Ask a Question:")
user_question = st.text_input("Type your question here:", key='text_input', value=st.session_state['text_input'])

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    if st.button("Type Question ⌨️"):
        if st.session_state.vector_index is None:  # Check if the vector index is initialized
            st.warning("Please upload and process a PDF first.")
        elif user_question:  # Check if there is a question entered
            with st.spinner('Processing your query...'):
                st.session_state.conversation.append((user_question, True))
                process_question(user_question, st.session_state.vector_index.get_relevant_documents(user_question))
            st.rerun()

with col2:
    if st.button("Ask Question? 🗣️"):
        if st.session_state.vector_index is None:  # Check if the vector index is initialized
            st.warning("Please upload and process a PDF first.")
        else:
            user_question = capture_audio()
            if user_question:
                with st.spinner('Processing your query...'):
                    st.session_state.conversation.append((user_question, True))
                    process_question(user_question, st.session_state.vector_index.get_relevant_documents(user_question), is_voice=True)
                st.rerun()

with col3:
    if st.button("Clear 🗑️", on_click=clear_text_input):
        pass