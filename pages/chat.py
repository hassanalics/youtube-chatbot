import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import os

# Load token from secrets
hf_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# Optional: set it as environment variable (if required by a library)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

st.header("Youtube Chatbot")

# Store video_id before calling this script
video_id = st.session_state.get("video_id")

if not video_id:
    st.error("No video ID provided.")
    st.stop()

# Proxy config
proxies = {
    "http": "http://ab1ff93281556f2305ebc82c384d736f:@proxy-server.scraperapi.com:8001",
    "https": "http://ab1ff93281556f2305ebc82c384d736f:@proxy-server.scraperapi.com:8001"
}

@st.cache_data(show_spinner="Loading transcript...")
def get_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"], proxies=proxies)
        return " ".join(chunk["text"] for chunk in transcript_list)
    except TranscriptsDisabled:
        return None

@st.cache_resource(show_spinner="Indexing transcript...")
def get_vector_store(transcript):
    print('************ inside get vecotr store *****************')
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return FAISS.from_documents(chunks, embeddings)

transcript = get_transcript(video_id)

if not transcript:
    st.error("Transcript not available for this video.")
    st.stop()

vector_store = get_vector_store(transcript)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

# LangChain setup
parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

parser = StrOutputParser()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
)
model = ChatHuggingFace(llm=llm, verbose=True)

prompt = PromptTemplate(
    template="""
    You are a helpful assistant.
    Answer ONLY from the provided transcript context.
    If the context is insufficient, just say you don't know.

    {context}
    Question: {question}
    """,
    input_variables = ['context', 'question']
)

main_chain = parallel_chain | prompt | model | parser

# ---------------- Chat UI ----------------

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if user_input := st.chat_input("Ask a question about the video..."):
    # Show user message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = main_chain.invoke(user_input)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})