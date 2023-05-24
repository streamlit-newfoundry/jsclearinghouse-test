import streamlit as st
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
import os


def load_documents_from_file(file_path):
    with open(file_path, 'r') as file:
        documents = [line.strip() for line in file]
    return documents

#  only load the data once and then cache it for subsequent runs.
@st.cache_data
def load_data():
    print('loading data...')
    file_path = 'llama_data.txt'
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

    return SimpleDirectoryReader(input_files=[file_path]).load_data()


@st.cache_resource
def get_vector(documents):
    print('vectorizing vector_index')
    vector_index = GPTVectorStoreIndex.from_documents(documents, organization_id=st.secrets["ORG_ID"])

    return vector_index


def main():
    documents = load_data()

    vector_index = get_vector(documents)

    query_engine = vector_index.as_query_engine()

    st.title("Jump$tart with GPT")

    user_input = st.text_input("Search the Jump$tart Clearinghouse system here: ")

    if st.button('Search'):

        response = query_engine.query(user_input)
        st.write('Response', response)


if __name__ == "__main__":
    main()