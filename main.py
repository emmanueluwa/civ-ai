import nest_asyncio
import pprint
import os
import getpass
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.schema import MetadataMode
from llama_index.llms.groq import Groq


nest_asyncio.apply()

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
        "category": "finance",
        "author": "LlamaIndex",
    },
    # excluded_embed_metadata_keys=["file_name"],
    excluded_llm_metadata_keys=["category"],
    metadata_separator="\n",
    metadata_template="{key}:{value}",
    text_template="Metadata:\n{metadata_str}\n-----\nContent:\n{content}",
)

print(
    "The LLM sees this: \n",
    document.get_content(metadata_mode=MetadataMode.LLM),
)
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

"""

os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")
