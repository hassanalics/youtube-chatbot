import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

st.header("Youtube Chatbot")

with st.form(key='my_form'):
	video_id_input = st.text_input(label='Enter Youtube Video Id')
	id_submit_button = st.form_submit_button(label='Submit')

if id_submit_button:
    # video_id = "5v6u_U6QRQA&ab" # only the ID, not full URL
    st.session_state["video_id"] = video_id_input

    # st.write("Id Successfully Entered")
    st.switch_page('pages/chat.py')

# question = st.chat_input("Ask Question")
# print(question)
# if question:
#     main_chain = parallel_chain | prompt | model | parser
#     result = main_chain.invoke(question)
#     st.write(result)