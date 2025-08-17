# RAG-chatbot
chatbot with docs upload  , user can upload docs and agent give you relevant  information 
if context is missing(pdf not upload by user) , agent give you general answer .



how it work

[1] at client client user post a pdf file and server receive pdf file 
server convert pdf into chunk and embedding then add to vectorestore
[2] graph node first retrieve relevant information from vectorstore based on user query 
and add this relevant string to query and paas to model/llm
llm generate infomation answer based on query and context.



