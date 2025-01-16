import json
import traceback
import requests

from model_configurations import get_model_configuration

from langchain import hub
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.tools import tool
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory



gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

llm = AzureChatOpenAI(
        model=gpt_config['model_name'],
        deployment_name=gpt_config['deployment_name'],
        openai_api_key=gpt_config['api_key'],
        openai_api_version=gpt_config['api_version'],
        azure_endpoint=gpt_config['api_base'],
        temperature=gpt_config['temperature']
    )

prompt = hub.pull("hwchase17/openai-functions-agent")

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

store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

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

def generate_hw01(question):

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "請根據問題列出台灣的紀念日，以 JSON 格式輸出"),
            example_prompt,
            ("human", "{input}"),
        ]
    )

    chain = final_prompt | llm 
    response_chain = chain.invoke({"input":question})
    response_json = json.dumps(response_chain.content, indent=4, ensure_ascii=False).encode('utf8').decode().replace("```json\\n","").replace("\\n```","")
    response = json.loads(response_json)
    return response

def generate_hw02(question):
 
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
    
    tools = [holiday_lookup_tool]
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    history_handler = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    history_handler.invoke(
        {"input": question2},
        config={"configurable": {"session_id": "holidays"}},
    )

    examples_hw3 = [
        {
            "input":"蔣公誕辰紀念日是否有在十月的節日清單中?",
            "output":
            """{
                "Result": 
                    {
                        "add": false,
                        "reason": "蔣中正誕辰紀念日並未包含在十月的節日清單中。目前十月的現有節日包括國慶日、重陽節、華僑節、台灣光復節和萬聖節。因此，如果該日被認定為節日，應該將其新增至清單中。"
                    }
            }"""
        }
    ]

    example_prompt_hw3 = FewShotChatMessagePromptTemplate(
        examples=examples_hw3,
        example_prompt=ChatPromptTemplate.from_messages([
            ("human", "{input}"),
            ("ai", "{output}")
        ]),
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "請根據問題回答，以 JSON 格式輸出:add : 這是一個布林值，表示是否需要將節日新增到節日清單中。根據問題判斷該節日是否存在於清單中，如果不存在，則為 true；否則為 false。reason : 描述為什麼需要或不需要新增節日，具體說明是否該節日已經存在於清單中，以及當前清單的內容。"),
            example_prompt_hw3,
            ("human", "{input}"),
        ]
    )
    
    runnable = final_prompt | llm
    history_handler = RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    
    response = history_handler.invoke(
        {"input": question3},
        config={"configurable": {"session_id": "holidays"}},
    )

    return response.content
    
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

#response = generate_hw01("2024年台灣10月紀念日有哪些?")
#response = generate_hw02("2024年台灣10月紀念日有哪些?")
#response = generate_hw03("2024年台灣10月紀念日有哪些?", "蔣公誕辰紀念日是否有在剛剛詢問的回答中?")
#print(response)