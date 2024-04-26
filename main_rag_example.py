#Bibliografy: https://github.com/ollama/ollama/blob/main/docs/tutorials/langchainpy.md

from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader


#Se obtiene un archivo con el texto del libro "La Odisea" de Homero, el cual se encuentra
# en el proyecto Gutenberg

URL = "https://www.gutenberg.org/files/58221/58221-h/58221-h.htm"

loader = WebBaseLoader(URL)
data = loader.load()

#Dividir el texto en padazos de 500 caracteres para permitir un mejor ajuste
#de la tecinca RAG
text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)


#Cargar un conector a la Base de Datos vectorial Chroma

oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="phi3")
vectorstore = Chroma.from_documents(
                  documents = all_splits,
                  embedding = oembed
              )

#Hacer una busqueda en la base vectorial Chroma al almacenar
#los chuncks para ver la cantidad de documentos que coinciden con este

question = "Quién es Helena?"
docs = vectorstore.similarity_search(question)
len(docs)

llm = Ollama( model = "phi3" )

#Realizar RAG con LangChain y recuperar los chunks similares
qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = vectorstore,
        return_source_documents = True
     )


response = qa.invoke({"query": question})
print(f"Respuesta a la pregunta: ${response}")
print(response)