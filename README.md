# RAG-chatbot agent 
chatbot with docs upload features, user can upload docs and agent give you relevant information 

if context is missing(pdf not upload by user) , agent give you general answer.



how it work

[1]client post a pdf file and server receive pdf file. 
at backent pdf file split into small docs  and embedding , then embedding add to vectorestore
[2] graph node first retrieve relevant information from vectorstore based on user query 
and add this relevant string to query and paas to model/llm
llm generate infomation answer based on query and context.
if context is missing , model give you a general answer ( based on llm trained )




