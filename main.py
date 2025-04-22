import nest_asyncio
import pprint
import re
import time
import os
import getpass
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.schema import MetadataMode
from llama_index.llms.groq import Groq
from llama_index.llms.google_genai import GoogleGenAI
from dotenv import load_dotenv
from llama_index.core.extractors import TitleExtractor, QuestionsAnsweredExtractor
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.llms.mistralai import MistralAI
from llama_index.llms.ollama import Ollama


nest_asyncio.apply()

# Load environment variables
load_dotenv()

# extracting text from every doc in dir (each page is an individual doc)
docs = SimpleDirectoryReader(input_dir="./data").load_data()

# print(len(docs))
# pprint.pprint(docs)

# transformations
""" 
expand metadata of each doc before indexing it

each doc holds a lot of data, not all of it is needed

when node is sent to embeddings model before putting it into vector db, metadata is sent with content of text
- relevant part of metadata should be included not all 
"""

# define which keys from metadata should be sent to embeddings model and language model
document = Document(
    text="This is a super-customized document",
    metadata={
        "file_name": "super_secret_document.txt",
        "category": "engineering",
        "author": "LlamaIndex",
    },
    excluded_embed_metadata_keys=["file_name"],
    excluded_llm_metadata_keys=["category"],
    metadata_separator="\n",
    metadata_template="{key}:{value}",
    text_template="Metadata:\n{metadata_str}\n-----\nContent:\n{content}",
)

# print(
#     "The LLM sees this: \n",
#     document.get_content(metadata_mode=MetadataMode.LLM),
# )
# print(
#     "The Embedding model sees this: \n",
#     document.get_content(metadata_mode=MetadataMode.EMBED),
# )


# excluding page label for embedding
for doc in docs:
    # defining context/metadata template
    doc.text_template = "Metadata:\n{metadata_str}\n---\nContent:\n{content}"

    # exclude page label from embedding
    if "page_label" not in doc.excluded_embed_metadata_keys:
        doc.excluded_embed_metadata_keys.append("page_label")

# print(docs[5].get_content(metadata_mode=MetadataMode.EMBED))

"""
applying more sophisticated transformations

- using language models to extract information from nodes
- extracting some info from each doc before embedding to add more info to embeddings (making it even more granular)
- adding control and augmenting data to ensure it is easier to retrieve 

technique used to improve rag (using language model)
- subtract summary/example questions and answers that particular sections of the data could solve

language model with lower context needs longer chunk size

"""


# Initialize Groq
groq_llm = Groq(model="qwen-qwq-32b", api_key=os.getenv("GROQ_API_KEY"))

# Initialize Google Gemini
gemini_llm = GoogleGenAI(model="gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY"))

llm = HuggingFaceInferenceAPI(
    model_name="mistralai/Mistral-7B-v0.1", token=os.getenv("HUGGINGFACE_TOKEN")
)

mistral_llm = MistralAI(
    model="codestral-latest",
    api_key=os.getenv("MISTRAL_API_KEY"),  # First 1M tokens free
)

# Initialize local LLM
local_llm = Ollama(
    model="tinyllama",
    base_url="http://localhost:11434",
    request_timeout=600,
    options={"num_gpu": -1, "num_thread": 2},  # Force CPU-only  # Limit CPU threads
)


# overlap ensures paragraphs are not cut in half
text_splitter = SentenceSplitter(
    separator=" ", chunk_size=1024, chunk_overlap=128  # 1024  # 128
)

# nodes reduced from 5 and questions reduced from 3
title_extractor = TitleExtractor(llm=local_llm, nodes=4)
qa_extractor = QuestionsAnsweredExtractor(llm=local_llm, questions=2)

pipeline = IngestionPipeline(
    transformations=[text_splitter, title_extractor, qa_extractor]
)

nodes = pipeline.run(documents=docs, in_place=True, show_progress=True)

print("NODES: ", len(nodes))
if nodes:
    print(nodes[2].get_content(metadata_mode=MetadataMode.EMBED))
