import json
import traceback

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

def generate_hw01(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )

    message__ = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )

    examples = [
        {
            "input": "2024年台灣10月紀念日有哪些?",
            "output": """
            "Result": [
                {
                    "date": "2024-10-10",
                    "name": "國慶日"
                }
            ]
            """
        }
    ]

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )

    few_shot_prompt  = FewShotChatMessagePromptTemplate(
        examples=examples,
        example_prompt=example_prompt
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "請根據問題列出台灣的紀念日，以 JSON 格式輸出"),
            few_shot_prompt,
            ("human", "{input}"),
        ]
    )

    chain = final_prompt | llm 
    response = chain.invoke(input=message__)
    return response

    
def generate_hw02(question):
    pass
    
def generate_hw03(question2, question3):
    pass
    
def generate_hw04(question):
    pass
    
def demo(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
      
    response = llm.invoke([message])
    
    return response
