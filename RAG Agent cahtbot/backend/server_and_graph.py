
# Rag agent graph and fastapi server

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
# from langchain.tools.retriever import create_retriever_tool
# create_retriver_tool(retriver,name='serach_info',description="..." )
# it return relevant text as context 
from langgraph.graph import StateGraph ,START,END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict ,List ,Optional
import asyncio
import json
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate 
import os
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import BaseMessage ,ToolMessage,HumanMessage , SystemMessage

os.environ["GOOGLE_API_KEY"] = "********"  



embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = InMemoryVectorStore(embedding)


llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash')



from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage ,BaseMessage ,SystemMessage
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS

from langchain.prompts import ChatPromptTemplate

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import TypedDict, List, Literal ,Optional
from langchain_core.vectorstores import InMemoryVectorStore







# === Define LLM ===
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash',api_key="AIzaSyDK1CNcAhSrM4qy3UVIXLu7J7Qk2U51Rug")




embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key="AIzaSyDK1CNcAhSrM4qy3UVIXLu7J7Qk2U51Rug")
vectorstore = InMemoryVectorStore(embedding)



retriever = vectorstore.as_retriever()

@tool
def retrieve_docs(query: str) -> str:
    """Search relevant context for a given query"""
    docs = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in docs[:3]])

retrieve_docs.invoke('')



llm_with_tools = llm.bind_tools([retrieve_docs])
tools_by_name = {"retrieve_docs": retrieve_docs}


# define State
class RAGState(TypedDict,total=False):
    messages: Optional[List[BaseMessage]]
    user_input: List[BaseMessage]
    iteration:int =0

  


#  agent 
def llm_node(state: RAGState):
    messages = state.get("messages",[]) 
    print(messages) 
    user_input = state.get('user_input',[])
    # for first invoke 
    if not messages:
        message = [SystemMessage(content='you are an intelligent RAG assistant, use retirver tool if needed , otherwise give general answer'),
                   HumanMessage(content=f"{user_input}")
                   ]
        messages.extend(message)



    
    # llm invoke input must be a PromptValue, str, or list of BaseMessages for example : llm.invoke(input) ; input  =  " hi " or  [BaseMessage] or ChatPromptTemplate().format_messages()    
    result = llm_with_tools.invoke(messages)
    messages.append(result)
    return {"messages": messages, "iteration": state.get("iteration", 0) + 1, "user_input": state["user_input"]}


#tool_node
def tool_node(state: RAGState):
    print('tool call')
    messages = state["messages"]
    tool_calls = messages[-1].tool_calls
    outputs = []
    for call in tool_calls:
        tool = tools_by_name[call["name"]]
        output = tool.invoke(call["args"])
        outputs.append(ToolMessage(tool_call_id=call["id"], content=output))
    return {"messages": messages + outputs, "iteration": state["iteration"], "user_input": state["user_input"]}


# routes
def should_continue(state: RAGState) -> Literal["tool", END]:
    if state["iteration"] > 2:
        return END
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tool"
    return END


# === Build Graph ===
graph_builder = StateGraph(RAGState)
graph_builder.add_node("llm", llm_node)
graph_builder.add_node("tool", tool_node)

graph_builder.set_entry_point("llm")
graph_builder.add_edge("tool", "llm")
graph_builder.add_conditional_edges("llm", should_continue, {"tool": "tool", END: END})

graph= graph_builder.compile()


# rag_prompt = ChatPromptTemplate.from_messages([
#     ("system", """you are an intelligent RAG assistant, use context if available, else answer from general knowledge"""),
#     ("human", "{input}"),
#     ("system", "context: {context}")
# ])

# class State(TypedDict,total=False):
#     answer:Optional[List[BaseMessage]]
#     query: str
#     context:Optional[str]


# # Retrieve Node
# async def retrieve(state:State):
#     query = state['query']
#     docs = vectorstore.similarity_search(query, k=3)
#     context = " ".join([d.page_content for d in docs])
#     return {'context': context}

# #  Generate Node
# async def generate(state:State):
#     query = state.get('query')
#     context = state.get('context')
#     prompt =  rag_prompt.format_messages(input=query, context=context)
#     response = await llm.ainvoke(prompt)
#     return {"answer": response}


# # async def rag_node(state):
# #     query = state['query']
# #     # docs = vectorstore.as_retriever().get_relevant_documents(query)
# #     context = " ".join([d.page_content for d in docs])
# #     prompt = rag_prompt.format_messages(input=query, context=context)
# #     response = await llm.ainvoke(prompt)
# #     return {"answer": response}



# graph_builder = StateGraph(State)

# graph_builder.add_node("retrieve", retrieve)
# graph_builder.add_node("generate", generate)


# graph_builder.add_edge(START, "retrieve")
# graph_builder.add_edge("retrieve", "generate")
# graph_builder.add_edge("generate", END)

# # Compile
# # graph = graph_builder.compile(checkpointer=InMemorySaver())
# graph = graph_builder.compile()









# fastapi server 


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow  local development
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
        return {"status": "PDF added", "num_chunks":f"{len(split_docs)}"}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.get("/stream-query/")
async def stream_query(query: str):
    async def event_generator():
        # Stream response in chunks
        async for chunk in graph.astream({"query": query}):
            # print(chunk['rag_node']['answer'].content)
            yield f"data: {chunk['rag_node']['answer'].content}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
