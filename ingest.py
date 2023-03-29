"""Load html from files, clean up, split, ingest into Weaviate.""",
import pickle

from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
import os

def ingest_docs(documentation_path: str = None):
    """Get documents from local markdown files"""
    documents = []

    # list all files in a directory and subdirectories
    for path, subdirs, files in os.walk(documentation_path):
        for name in files:
            if name.endswith('mdx'):
                print(f"Loading file {name} from {path}")
                loader = UnstructuredMarkdownLoader(os.path.join(path, name))
                raw_documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                )
                documents.extend(text_splitter.split_documents(raw_documents)) 

    if documents == []:
        raise ValueError("No documents found.")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)


if __name__ == "__main__":
    ingest_docs('/Users/zi/Work/Rasa/rasa/docs/docs')
