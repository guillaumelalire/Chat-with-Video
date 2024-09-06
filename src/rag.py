from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from huggingface_hub import login
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnablePassthrough
import os
from langchain_core.documents import Document

import transformers
load_dotenv()

hugging_face_token = os.getenv('HUGGINGFACE_TOKEN')
login(token=hugging_face_token) # Login to the Hugging Face Hub

def split_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ' '], chunk_size=300, chunk_overlap=50)
    document = Document(
        page_content=text,
    )
    return text_splitter.split_documents([document])

def create_vector_storage(text):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    chunks = split_into_chunks(text)
    return FAISS.from_documents(chunks, embeddings)
    
def launch_model(db):
    model_name='mistralai/Mistral-7B-v0.1'

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    text_generation_pipeline = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=True,
        max_new_tokens=200,
    )

    prompt_template = """Answer the question based only on the following context:

    {context}

    Question: {question}
     """

    mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    # Create prompt from prompt template 
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    # Create llm chain 
    llm_chain = LLMChain(llm=mistral_llm, prompt=prompt)

    retriever = db.as_retriever(
        search_type='similarity',
        search_kwargs={'k': 2}
    )
    
    rag_chain = ( 
        {"context": retriever, "question": RunnablePassthrough()}
        | llm_chain
    )

    return rag_chain

def llm_response(rag_chain, prompt):
    response = rag_chain.invoke(prompt)
    return response['text']#.split("[/INST]")[-1]
