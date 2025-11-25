from google.adk.agents.llm_agent import Agent
from google.adk.tools import google_search, AgentTool
from google.adk.runners import InMemoryRunner
import asyncio
from google.genai import types


research_agent = Agent(
    model = 'gemini-2.0-flash',
    name = 'research_agent',
    description = 'An agent with the task of searching stuff on google',
    instruction= """You are a specialized research agent.
    Your only job is to use the google_search tool to find 2-3 pieces of relevant information
    on the given topic and present the findings with citations.""",
    tools=[google_search],
    output_key="research_findings"
)

summarizer_agent = Agent(
    model = 'gemini-2.0-flash',
    name = 'summarizer_agent',
    description = 'An agent with the task of searching stuff on google',
    instruction= """Read the provided research findings: {research_findings}
Create a concise summary as a bulleted list with 3-5 key points.""",
    output_key="final_summary"
)

root_agent = Agent(
    model='gemini-2.0-flash',
    name='root_agent',
    description='A helpful assistant for user questions.',
    instruction="""You are a research coordinator. Your goal is to answer the user's query by orchestrating a workflow.
1. First, you MUST call the `ResearchAgent` tool to find relevant information on the topic provided by the user.
2. Next, after receiving the research findings, you MUST call the `SummarizerAgent` tool to create a concise summary.
3. Finally, present the final summary clearly to the user as your response.""",
    tools=[AgentTool(research_agent), AgentTool(summarizer_agent)],
)

runner = InMemoryRunner(agent=root_agent)

async def chat_loop():
    while True:
        user_input = input("You:")
        if user_input == "quit":
            break
        response = await runner.run_debug(user_input)
        

if __name__ == "__main__":
    asyncio.run(chat_loop())


