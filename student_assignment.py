import json
import traceback

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.agents import create_openai_functions_agent, AgentExecutor
import requests
from langchain.tools import tool

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

    examples = [
        {
            "input":"2024年台灣10月紀念日有哪些?",
            "output":
            """{
            "Result": [
                {
                    "date": "2024-10-10",
                    "name": "國慶日"
                },
                {
                    "date": "2024-10-11",
                    "name": "重陽節"
                }
            ]
            }"""
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
    response = chain.invoke({"input":question})
    return response.content

# 將工具轉換為 LangChain 工具
@tool
def holiday_lookup_tool(country: str, year: str) -> str:
    """
    查詢指定國家和年份的假期列表。
    
    :param country: ISO 3166-1 Alpha-2 國家代碼，例如 "TW" 表示台灣
    :param year: 查詢年份，格式為 "YYYY"
    :return: 假期的名稱和日期列表，格式化為字符串。如果發生錯誤，則返回錯誤信息。
    """
    api_key = "mrMdz6nfGAfUMxw849dlRWJeEHgbAXck"
    api_url = "https://calendarific.com/api/v2/holidays"
    parameters = {
        "api_key": api_key,
        "country": country,
        "year": year
    }
    calendarific_requests=requests.get(api_url, params=parameters)
    if calendarific_requests.status_code != 200:
        return {"error": f"API error: {calendarific_requests.status_code}"}
    holidays = calendarific_requests.json().get("response", {}).get("holidays", [])
    if "error" in holidays:
        return {"error": holidays["error"]}
    
    results = []
    for holiday in holidays:
        name = holiday.get("name", "Unknown Holiday")
        date = holiday.get("date", {}).get("iso", "Unknown Date")
        results.append({"date": date, "name": name})
    return results

def generate_hw02(question):
    llm = AzureChatOpenAI(
        model=gpt_config['model_name'],
        deployment_name=gpt_config['deployment_name'],
        openai_api_key=gpt_config['api_key'],
        openai_api_version=gpt_config['api_version'],
        azure_endpoint=gpt_config['api_base'],
        temperature=gpt_config['temperature']
    )

    examples = [
        {
            "input":"2024年台灣10月紀念日有哪些?",
            "output": 
            """
            {
                "Result": [
                    {
                        "date": "2024-10-10",
                        "name": "國慶日"
                    },
                    {
                        "date": "2024-10-09",
                        "name": "重陽節"
                    },
                    {
                        "date": "2024-10-21",
                        "name": "華僑節"
                    },
                    {
                        "date": "2024-10-25",
                        "name": "台灣光復節"
                    },
                    {
                        "date": "2024-10-31",
                        "name": "萬聖節"
                    }
                ]
            }
            """
        }
    ]

    example_prompt = FewShotChatMessagePromptTemplate(
        examples=examples,
        example_prompt=ChatPromptTemplate.from_messages([
            ("human", "{input}"),
            ("ai", "{output}")
        ]),
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("ai", "{agent_scratchpad}"),
            ("system", "請根據問題列出台灣的紀念日，以 JSON 格式輸出:"),
            example_prompt,
            ("human", "{input}"),
        ]
    )

    tools = [holiday_lookup_tool]
    agent = create_openai_functions_agent(llm, tools, final_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    agent_response = agent_executor.invoke({"input": question})
    response_json = json.dumps(agent_response.get("output"), indent=4, ensure_ascii=False).encode('utf8').decode().replace("```json\\n","").replace("\\n```","")
    response = json.loads(response_json)
    return response
    
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
