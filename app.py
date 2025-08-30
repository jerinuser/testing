import os 
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

class Person(BaseModel):
    name: str = Field(description="The name of the person")
    age: int = Field(description="The age of the person")
    city: str = Field(description="The city where the person lives")
    
parser = PydanticOutputParser(pydantic_object=Person)

prompt=PromptTemplate(
    input_variables=["input"],
    template="Extract the name, age, and city from the following text: \n{input}\n{format_instructions}",
    partial_variables={"format_instructions": parser.get_format_instructions()}
)


llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=api_key,
)

input_text="My name is Alice, I am 11 years old and I live in New York."

formatted_prompt = prompt.format(input=input_text)

response = llm.invoke(formatted_prompt)

parsed_response = parser.parse(response.content)

print(f"Parsed Response:{parsed_response}")

print(f"{parsed_response.name}")
# print(f"{parsed_response.age}")
# print(f"{parsed_response.city}")