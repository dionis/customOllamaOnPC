#Bibliografy: https://github.com/ollama/ollama/blob/main/docs/tutorials/langchainpy.md

from langchain_community.llms import Ollama

prompt = "Como hacer un postgrado sobre Inteligencia Artificial Generativa?"

llm = Ollama(model="phi3")

response = llm.invoke(prompt)
print(response)
