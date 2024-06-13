from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

def load_pdf(data_path):
    loader = DirectoryLoader(path=data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents
def text_split(extracted_pdf):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                   chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_pdf)
    return text_chunks 
def download_huggingfaceembedding():
    embedding = HuggingFaceEmbeddings(model_name="TheBloke/Llama-2-7B-Chat-GGML")
    return embedding