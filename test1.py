from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline


ds = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")
#whole dataset is in this variable
df = ds["test"].to_pandas()
print(df.head())

#step 2 : Splitting the data, or converting the dataset into Chunks
# these chunks are stored in the vector database

all_docs = [{"text" : row["answer"], "metadata" : {"id" : row["id"]}} for row in ds['test']]

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
documents = splitter.create_documents([d["text"] for d in all_docs], metadatas=[d["metadata"] for d in all_docs])

for i, doc in enumerate(documents[:10]):
    print("Text chunk", i, ":", doc.page_content, "\n")
    print("Metadata:", doc.metadata, "\n")


embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embedding)


retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

qa_model = pipeline("text2text-generation", model="google/flan-t5-base") #this is generator(llm)



def get_answer(question : str, retriever, llm) -> str :
    relevant_docs = retriever.get_relevant_documents(question)
    context = "\n".join([i.page_content for i in relevant_docs])
    prompt = f"""Answer the following question based on the context below:

    Context:
    {context}

    Question: {question}
    Answer:"""

    return llm(prompt)

print("\n\n\n")


choice = 'y'

while(choice == 'y'):
    if(choice == 'y'):
        question = input("Enter your question(s)\n")
        answer = get_answer(question, retriever, qa_model)
        print(answer[0]["generated_text"])

    print("Do you want to continue(y/n)?")
    choice = input()
    if(choice == 'n'):
        break
    