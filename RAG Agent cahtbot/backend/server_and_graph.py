
# file contain graph and fastapi server

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.tools.retriever import create_retriever_tool
# create_retriver_tool(retriver,name='serach_info',description="..." )
# it return raw string as context .
from langgraph.graph import StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict
import asyncio
import json
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
import os

os.environ["GOOGLE_API_KEY"] = "********"  



embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = InMemoryVectorStore(embedding)


llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash')



rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """you are an intelligent RAG assistant, use context if available, else answer from general knowledge"""),
    ("human", "{input}"),
    ("system", "context: {context}")
])

class State(TypedDict):
    answer: str
    query: str

async def rag_node(state):
    query = state['query']
    docs = vectorstore.as_retriever().get_relevant_documents(query)
    context = " ".join([d.page_content for d in docs])
    prompt = rag_prompt.format_messages(input=query, context=context)
    response = await llm.ainvoke(prompt)
    return {"answer": response}

builder = StateGraph(State)
builder.add_node("rag_node", rag_node)
builder.add_edge("__start__", "rag_node")
graph = builder.compile()





# fastapi server 


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow origin for local development
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)


# handle posted file  and data indexing 
@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        loader = PyPDFLoader(file_path)
        docs = loader.load()
         # chunk_size determine how long chunk string length when retrieve from  vectrostore.as_retriever('user_query') ( in this project i use 
        splitter =  RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = splitter.split_documents(docs)
        
        vectorstore.add_documents(split_docs)
        print('file uploaded by user')
        return {"status": "PDF added", "num_chunks":"hello"}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.get("/stream-query/")
async def stream_query(q: str):
    async def event_generator():
        # Stream response in chunks
        async for chunk in graph.astream({"query": q}):
            # print(chunk['rag_node']['answer'].content)
            yield f"data: {chunk['rag_node']['answer'].content}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
