from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_openrouter import ChatOpenRouter

load_dotenv() 

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="chroma_DB", 
    embedding_function = embedding_model
)

retriver = vectorstore.as_retriever(
    search_type = "mmr", 
    search_kwargs = {"k" : 2, "fetch_k" : 5, "lambda_mult" : 0.5}, 
)

llm =ChatOpenRouter(model="openrouter/free")
prompt =ChatPromptTemplate.from_messages(
    [
        ("system", """ You are a helpful AI assistant.
            Use ONLY the provided context to answer the question.
            If the answer is not present in the context,
            say: "I could not find the answer in the document."""),
        ("human", """ 
                    Context: {context}
                    Question: {question}"""),
    ]
)

print("Rag System created")
print("Press 0 to exit")

while True:
    query = input("You: ")
    if query == "0":
        break
    docs = retriver.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = prompt.invoke({
        "context" : context, 
        "question" : query
    })

    responce = llm.invoke(final_prompt)
    print(f"\n AI: {responce.text}")