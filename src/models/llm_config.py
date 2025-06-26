import os
from dotenv import load_dotenv
from langchain_aws.chat_models import ChatBedrock
from langchain_aws.embeddings import BedrockEmbeddings

load_dotenv()

def get_llm():
    return ChatBedrock(
        region_name=os.getenv("AWS_REGION"),
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        model_kwargs={
            "temperature": 0.3,
            "top_p": 0.9
        }
    )

# def get_embeddings():
#     from langchain_google_genai import GoogleGenerativeAIEmbeddings
#     return GoogleGenerativeAIEmbeddings(
#         model="models/text-embedding-004",
#         google_api_key=os.getenv("GOOGLE_API_KEY")
#     )

def get_embeddings():
    return BedrockEmbeddings(
        region_name=os.getenv("AWS_REGION"),
        model_id="amazon.titan-embed-text-v1"
    )