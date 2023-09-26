import os
import sys

import constants
import openai
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.indexes import VectorstoreIndexCreator


os.environ["OPENAI_API_KEY"] = constants.APIKEY

query = None
if len(sys.argv) > 1:
  query = sys.argv[1]

print(query)

loader = DirectoryLoader("data/")
index = VectorstoreIndexCreator().from_loaders([loader])

chain = ConversationalRetrievalChain.from_llm(
  llm=ChatOpenAI(model="gpt-3.5-turbo"),
  retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)

chat_history = []
while True:
    if not query:
        query = input("Prompt: ")
    if query in ['quit', 'q', 'exit']:
        sys.exit()
    
    result = chain({"question": query, "chat_history": chat_history})
    print(result['answer'])

    chat_history.append((query, result['answer']))
    query = None