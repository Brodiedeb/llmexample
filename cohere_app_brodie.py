import streamlit as st

import requests
import pandas as pd
import base64
import ast

from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_cohere import ChatCohere
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from streamlit_sortables import sort_items

from qdrant_client import QdrantClient


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

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

retriever, qdrant_vectorstore = creating_db(model_name)

cohere_chat_model = ChatCohere(cohere_api_key='GTBBhzYzLyvtESHCSiKSFgcU1QhWHhviGFhVo9Y3')

template = """You are a physician.
    You will be provided a section of the case scenario and the chief complaint that the patient came in with.
    Using the chief complaint as a guide, give a ranked list of top differential diagnoses that you can interpret from the section which fit best with the chief complaint.
    Be as clear and precise as possible. No explanation needed.
    Format the output as a python list as follows ['Diagnosis 1, Diagnosis 2, Diagnosis 3, ...]'>
    Here is the section of the scenario {context}"""


# Add the context to your user query
custom_rag_prompt = PromptTemplate.from_template(template)


LOGO_IMAGE = "bot-transformed.png"

st.markdown(
    """
    <style>
    .container {
        display: flex;
        text-align: center;
    }
    .logo-text {
        font-weight:700 !important;
        font-size:75px !important;
        color: #f9a01b !important;
        padding-top: 25px !important;
                margin-left: 20px;

    }
    .logo-img {
        float:right;
        width: 150px;
        height: 150px;
    }
    .outer-center {
    float: right;
    right: 50%;
    position: relative;
}
.inner-center {
    float: right;
    right: -50%;
    position: relative;
}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
        <div class="outer-center">
        <div class="product inner-center">
    <div class="container">
        <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open(LOGO_IMAGE, "rb").read()).decode()}">
        <p class="logo-text">Medbot</p>
    </div>
    """,
    unsafe_allow_html=True
)

setting = st.radio(
    "What's setting of this patient encounter?",
    ["Clinic", "ICU", "Floor/wards"],
    captions = ["Any type of clinic", "In the ICU/CCU", "On the wards"])
st.write("You selected:", setting)

chief_complaint = st.text_area(
    "Please enter patient chief complaint here.",
    )

HPI = st.text_area(
    "Please enter patient HPI here.",
    )

Allergies_PMH_PSH_FamilyHx_SocialHx_Meds = st.text_area(
    "Please enter patient allergies, PMH, PSH, FamilyHx, SocialHx, and Medications here.",
    )

HemodynamicMonitoring_FluidBalance_RespiratoryDeviceSettings_PhysicalExamination = st.text_area(
    "Please enter patient hemodynamic monitoring, fluid balance, respiratory device settings, and physical exam info here. ",
    )

LabsAndDiagnostics = st.text_area(
    "Please enter labs and diagnostics.",
    )


submit_button = st.button("Submit", type="primary")

if setting == "Clinic":
    introduction = "You are a physician seeing this patient in the clinic."
elif setting == "ICU":
    introduction = "You are a physician seeing this patient in the ICU."
else:
    introduction = "You are a physician seeing this patient in the medicine wards."

plan_template = introduction + """The guidelines said, "{context}." Based on what the guidelines said, write an assessment and plan for the problem: {problem} provided \n
    The plans should be concrete and specific with what you would do in that situation. The plan should me appropriate for the setting in which you are seeing the patient. \n
    For example, instead of saying start antibiotics, say which antibiotics you would use. \n
    Be as concise as possible. Format with following structure - Problem: Plan."""


# Add the context to your user query
plan_rag_prompt = PromptTemplate.from_template(plan_template)

wt = []
diagnoses_dictionary = {"chief complaint": None, "HPI": None, "Allergies_PMH_PSH_FamilyHx_SocialHx_Meds": None, "HemodynamicMonitoring_FluidBalance_RespiratoryDeviceSettings_PhysicalExamination":None, "LabsAndDiagnostics": None}

def checkbox_container(data):
        st.header('Select, Add and Sort Diagnoses')
        new_data = st.text_input('Enter Diagnosis to add')
        cols = st.columns(3)
        if cols[0].button('Add Diagnosis'):
            data.append(new_data)
        if cols[1].button('Select All'):
            for i in data:
                st.session_state['dynamic_checkbox_' + i] = True
            st.experimental_rerun()
        if cols[2].button('UnSelect All'):
            for i in data:
                st.session_state['dynamic_checkbox_' + i] = False
            st.experimental_rerun()
        for i in data:
            st.checkbox(i, key='dynamic_checkbox_' + i)

def get_selected_checkboxes():
    return [i.replace('dynamic_checkbox_','') for i in st.session_state.keys() if i.startswith('dynamic_checkbox_') and st.session_state[i]]

if submit_button: # and (chief_complaint is not None and HPI is not None and Allergies_PMH_PSH_FamilyHx_SocialHx_Meds is not None and HemodynamicMonitoring_FluidBalance_RespiratoryDeviceSettings_PhysicalExamination is not None and LabsAndDiagnostics is not None):
    wt.append(chief_complaint)
    wt.append(HPI) 
    wt.append(Allergies_PMH_PSH_FamilyHx_SocialHx_Meds)
    wt.append(HemodynamicMonitoring_FluidBalance_RespiratoryDeviceSettings_PhysicalExamination)
    wt.append(LabsAndDiagnostics)
    if len(wt) > len(diagnoses_dictionary):
    	print("Error: The list 'wt' contains more elements than there are keys in the dictionary.")
    else:
    	keys = list(diagnoses_dictionary.keys())  # Get a list of dictionary keys
    	for i in range(len(wt)):
    		diagnoses_dictionary[keys[i]] = wt[i]

    # body = {
    #     "prompt": diagnoses_dictionary
    # }
    rag_chain = (
        {"context": lambda x: diagnoses_dictionary}
        | custom_rag_prompt
        | cohere_chat_model
        | StrOutputParser()
    )

    diagnoses = rag_chain.invoke(diagnoses_dictionary)
    st.session_state.diagnoses = diagnoses
    # st.subheader("Output")
    # st.write(st.session_state.diagnoses)
    # out = requests.post("https://xgedf7kieezvkvgdw6qb7igvfe0iznnz.lambda-url.us-west-2.on.aws/", json=body)
    # out = out.json()
    diagnoses_list = st.session_state.diagnoses 
    diagnoses_list = ast.literal_eval(diagnoses_list)
    st.session_state.diagnoses_list=[item.strip() for item in diagnoses_list]

      
if "diagnoses" not in st.session_state:
    st.session_state.diagnoses = False
elif st.session_state.diagnoses:

    st.title("Medical Diagnosis Selector")
    checkbox_container(st.session_state.diagnoses_list)
    selected_diagnoses = get_selected_checkboxes()


    # st.title("Add additional diagnoses/problems")

    # append_diagnosis_form = st.text_input("Add an additional diagnosis")
    # append_diagnosis_button = st.button("Add diagnosis")

    # if append_diagnosis_button:
    #     st.session_state.diagnoses_list.append(append_diagnosis_form)

    
    sorted_items = sort_items(
        selected_diagnoses,
        direction='vertical',  # Ensure the items are sorted vertically
    )

    st.title("Plan Generation")
    generate_plan_button = st.button("Generate Plan")
    if generate_plan_button:

        plan_rag_chain = (
        {"context": retriever | format_docs,
        "problem": RunnablePassthrough()}
        | plan_rag_prompt
        | cohere_chat_model
        | StrOutputParser()
    )
        for problem in sorted_items:
            st.write_stream(plan_rag_chain.stream(problem))
            st.write('Reference:')
            context = qdrant_vectorstore.similarity_search_with_score(problem)
            references = ' ; '.join(list(set([i[0].metadata['content'] for i in context])))
            st.write(references)
            st.write(context)

#Note: need to add callback=progress somehow


# with st.container():
# st.write(diagnoses_dictionary)




