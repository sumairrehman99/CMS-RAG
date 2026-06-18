from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document
import os
import json


from dotenv import load_dotenv


load_dotenv(override=True)

MODEL = "gpt-4.1-mini"
DB_NAME = str(Path(__file__).parent.parent / "vector_db")

open_ai_key = os.getenv('OPENAI_API_KEY')
if open_ai_key:
    print('OpenAI API key found')
else:
    print('OpenAI key not found')


from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
#from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document


SYSTEM_PROMPT = """
You are a CMS manual question-answering assisstant. Answer the questions using the context provided to you. 
You can infer answers from the context provided. Try and use the context as much as you can. 

Before answering, carefully analyze the retrieved context.
If multiple rules seem relevant, compare them before giving the answer.
Do not rush to answer from the first matching chunk.
Explain the reasoning briefly using only the provided context.

If the context 
provided truly can not answer the question, respond with: 'Sorry, I could not find this information in the CMS manuals provided to me.'

Context: {context}

"""

embeddings_OpenAI = OpenAIEmbeddings(model="text-embedding-3-large")
#embeddings_HF = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
RETRIEVAL_K = 15


vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings_OpenAI)
retriever = vectorstore.as_retriever(search_kwargs={'k' : RETRIEVAL_K})
llm = ChatOpenAI(temperature=0, model_name=MODEL)

def fetch_context(question: str):
    return retriever.invoke(question)


def fetch_sources(docs):
    sources = []

    for doc in docs:
        source = doc.metadata.get('source', 'Unknown Source')       # If no source, return 'Unknown Source'
        page_number = doc.metadata.get('page', None)                # If no page, return None

        if page_number is not None:
            page_number += 1           # Page numbers start at 0

        sources.append(f'- {source}, Page {page_number}')
    
    return sorted(set(sources))



from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

def combined_question(question: str, history: list = None) -> str:
    history = history or []

    prior = "\n".join(user_msg for user_msg, assistant_msg in history)
    return prior + "\n" + question


def convert_to_messages(history: list = None):
    history = history or []

    messages = []
    for user_msg, assistant_msg in history:
        messages.append(HumanMessage(content=user_msg))
        messages.append(AIMessage(content=assistant_msg))

    return messages


def answer_question(question: str, history: list = None) -> str:
    history = history or []

    combined = combined_question(question, history)
    
    rewrite_prompt = f"""
    Optimize this query for document retrieval.
    Query: {combined}
    """
    
    new_prompt = llm.invoke([HumanMessage(content=rewrite_prompt)]).content
    print(new_prompt)

    docs = fetch_context(new_prompt)

    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)

    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))

    response = llm.invoke(messages)
    #print('CONTEXT SENT TO LLM')
    #print(context[:3000])
    sources = fetch_sources(docs)

    if response.content == 'Sorry, I could not find this information in the CMS manuals provided to me.':
        return response.content, []
    else:
        return response.content, sources


def answer_eval(question):
    docs = fetch_context(question)
    context = '\n\n'.join(doc.page_content for doc in docs)

    system_prompt = SYSTEM_PROMPT.format(context=context)

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=question)]

    response = llm.invoke(messages)

    return response.content, [doc.page_content for doc in docs]

# if __name__ == '__main__':
#     print('RAG active...')
#     print('Launching Interface...')
#     gr.ChatInterface(answer_question).launch()