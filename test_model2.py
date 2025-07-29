from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceHub(
    repo_id="google/flan-t5-xxl",
    task="text2text-generation",
    model_kwargs={"max_length":512},
    # huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)
print(llm("Explain the significance of Fourier transforms"))
