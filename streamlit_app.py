import streamlit as st
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
import os

def load_documents_from_file(file_path):
    with open(file_path, 'r') as file:
        documents = [line.strip() for line in file]
    return documents

# @st.cache_data only load the data once and then cache it for subsequent runs.
def load_data():
    file_path = 'llama_data.txt'
    documents = load_documents_from_file(file_path)

    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

    index = GPTVectorStoreIndex.from_documents(documents, organization_id=st.secrets["ORG_ID"])

    return index.as_query_engine()

def main():
    query_engine = load_data()

    st.title("Jump$tart Streamlit App with GPT")

    user_input = st.text_input("Search the Jump$tart Clearinghouse system here: ")

    if st.button('Search'):
        response = query_engine.query(user_input)
        st.write('Response', response)

if __name__ == "__main__":
    main()