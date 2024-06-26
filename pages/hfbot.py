import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import Qdrant
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.utils.math import cosine_similarity

from langchain_community.document_loaders import ArxivLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import PubMedLoader
from langchain_community.chat_models import ChatCohere
import re
from qdrant_client import QdrantClient
import time 

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("🫀 Cardiology Guideline Recommendations Chatbot")
st.caption("🚀 A chatbot powered by OpenAI LLM integrated with RAG. This is for eductional purposes. Do not input PHI.")

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()

openai_chat_model = ChatOpenAI(openai_api_key=openai_api_key,model='gpt-4-turbo')

model_name="kamalkraj/BioSimCSE-BioLinkBERT-BASE"

@st.cache_resource
def creating_db(model_name):
    base_embeddings = HuggingFaceEmbeddings(model_name=model_name)
    qdrant_client = QdrantClient(
        url="https://5fff5f2d-b0f0-4ecc-aefa-81905ce94dc9.us-east4-0.gcp.cloud.qdrant.io:6333", 
        api_key="YYDLXtDyw75MKxpErAvcIqkGPxo_66qZILNb-EDLeoFJPgi8LbdKFQ")
    qdrant_vectorstore = Qdrant(
        client=qdrant_client, collection_name="Cardiology ACC guidelines", 
        embeddings=base_embeddings,
    )
    retriever = qdrant_vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.1})
    return retriever,qdrant_vectorstore

retriever,qdrant_vectorstore = creating_db(model_name)

def format_docs(docs):
    cleaned_content =  "\n\n".join([doc.page_content for doc in docs])
    cleaned_content = re.sub(r'\s+', ' ', cleaned_content)  # Replace one or more whitespace characters with a single space
    cleaned_content = re.sub(r'\\u[\dA-Fa-f]{4}', '', cleaned_content)  # Remove unicode escape sequences
    cleaned_content = re.sub(r'\\x[\dA-Fa-f]{2}', '', cleaned_content)  # Remove hexadecimal escape sequences
    return cleaned_content

# Simulated RAG function
def qa_end2end(question):

  diagnosis_template = """You are a physician. Convert the question given into a list of relevant diagnosis and themes. Dont't answer the question yet. Here is the scenario.{question}"""
  diagnosis_prompt = ChatPromptTemplate.from_template(diagnosis_template)
  output_parser = StrOutputParser()
  openai_diagnosis_chain = diagnosis_prompt | openai_chat_model | output_parser
#   topics = openai_diagnosis_chain.invoke(question)
  topics = st.write_stream(openai_diagnosis_chain.stream(question))


#   print(question)
#   print(100*'*')
#   print(topics)
#   print(100*'*')

  rag_template = """[INST]Guideline committee said the following: "{context}". Based on what the guideline committee said, summarize the following question and cite the study from the guidelines relevant to these topics.\n\nHere are the topics: \n{topics}.[/INST]"""
  rag_prompt = ChatPromptTemplate.from_template(rag_template)
  openai_retrieval_chain = (
      {"context": retriever | format_docs,
      "topics": RunnablePassthrough()}
      | rag_prompt | openai_chat_model | output_parser)
#   reference = openai_retrieval_chain.invoke(topics)
  reference = st.write_stream(openai_retrieval_chain.stream(topics))
  context = qdrant_vectorstore.similarity_search_with_score(topics)
  references = ' ; '.join(list(set([i[0].metadata['content'] for i in context])))
  st.write('Reference:')
  st.write(references)

  

#   print(reference)
#   print(100*'*')

  rag_template = """[INST]Based on the references provided, answer the following question and cite the study from the guidelines relevant to these topics.\n\nHere are the references:{ref}. \n\nHere is the question: \n{question}.[/INST]"""
  rag_prompt = ChatPromptTemplate.from_template(rag_template)

  output_parser = StrOutputParser()

  qa_retrieval_chain = (
      {"ref": RunnablePassthrough(),
      "question": RunnablePassthrough()}
      | rag_prompt | openai_chat_model | output_parser

  )


  answer = qa_retrieval_chain.invoke({'ref': reference, 'question': question})
  st.write("Final Answer is:")
  answer = st.write_stream(qa_retrieval_chain.stream({'ref': reference, 'question': question}))
#   print(answer)
#   print(100*'*')

  return {'topics':topics, 'reference':reference, 'answer':answer}


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
# Simulate RAG processing
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)  # Immediately display the user's prompt

    rag_response = qa_end2end(prompt)


    # Optionally, add a small delay to simulate processing time
    time.sleep(1)  # Sleep for 1 second before continuing

    # # # Display RAG System Topics and Reference before showing the answer
    # # if 'topics' in rag_response:
    # #     st.session_state.messages.append({"role": "assistant", "content": f"Topics Identified: {rag_response['topics']}"})
    # #     st.write(f"Topics Identified: {rag_response['topics']}")
    # #     time.sleep(1)  # Sleep for 1 second

    # # Display the final answer from the RAG system
    # st.session_state.messages.append({"role": "assistant", "content": rag_response['answer']})
    # # st.write(f"Final answer is: {rag_response['answer']}")

    # if 'reference' in rag_response:
    #     st.session_state.messages.append({"role": "assistant", "content": f"Reference Information: {rag_response['reference']}"})
    #     st.write(f"Reference Information and explanation: {rag_response['reference']}")
    #     time.sleep(1)  # Sleep for 1 second


