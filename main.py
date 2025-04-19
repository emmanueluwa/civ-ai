import nest_asyncio
from llama_index.core import SimpleDirectoryReader

nest_asyncio.apply()

# extracting text from every doc in dir
docs = SimpleDirectoryReader(input_dir="./data").load_data()
