import os
from langchain.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface.llms.huggingface_endpoint import HuggingFaceEndpoint

huggingface_token = os.getenv("HF_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    temperature=0.7,
    huggingfacehub_api_token=huggingface_token
)


urls = [
    'https://ar.wikipedia.org/wiki/%D8%B3%D9%83%D9%83_%D8%AD%D8%AF%D9%8A%D8%AF_%D9%85%D8%B5%D8%B1',
    'https://en.wikipedia.org/wiki/Egyptian_National_Railways'
]
pdf = 'Document.pdf'  

def build_retriever(urls, pdf):
    all_docs = []
    for url in urls:
        loader = WebBaseLoader(url)
        data = loader.load()
        all_docs.extend(data)

    loader = PyPDFLoader(pdf)
    data_pdf = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(all_docs)

    text_splitter_pdf = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts_pdf = text_splitter_pdf.split_documents(data_pdf)

    all_texts = texts + texts_pdf

    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(all_texts, embeddings)
    retriever = db.as_retriever()

    return retriever

def chat_with_rag_retriever(retriever, query):
    template = """
    إذا كان الإدخال تحيّة أو جملة قصيرة (مثل “صباح الخير”، “مرحباً”، “Hello”), فأجب بردّ قصير مناسب بنفس اللغة:
    - إذا كانت التحية عربيّة: “صباح النور!” أو “مرحباً!”
    - إذا كانت بالإنجليزية: “Good morning!” أو “Hello!”

أجب بناءً على المعلومات الواردة في السياق، **وبنفس لغة السؤال**.
إذا لم تُذكر الإجابة في السياق، فاكتب "لا أعلم" (إذا كان السؤال عربيًّا) أو "I don't know." (إذا كان السؤال إنجليزيًّا).

Context: {context}

Question: {question}
Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    chain_type_kwargs = {"prompt": prompt}
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_type_kwargs)

    response = chain.run(query)
    answer = response.split("Answer:")[-1].strip()
    return f"Question: {query}\nAnswer: {answer}"

if __name__ == "__main__":
    retriever = build_retriever(urls, pdf)
    query = "التواصل و الشكاوي؟"
    print(chat_with_rag_retriever(retriever, query))
