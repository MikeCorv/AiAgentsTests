from google.adk.agents import Agent, ParallelAgent, SequentialAgent
from google.adk.tools import google_search
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.genai import types
import asyncio


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)


tech_researcher = Agent(
    model='gemini-3-pro-preview',
    name = "tech_agent",
    description = "A tech agent searching tech related stuff",
    instruction = """You are a specialized research agent, focused on tech news, articles and research.
    Your only job is to use the google_search tool to find 2-3 pieces of relevant information
    on the given topic and present the findings with citations.""",
    tools = [google_search],
    output_key="tech_research", 
)

protein_researcher = Agent(
    model='gemini-3-pro-preview',
    name = "protein_agent",
    description = "A protein agent searching tech related stuff",
    instruction = """You are a specialized research agent, focused on protein and biotech news, articles and research.
    Your only job is to use the google_search tool to find 2-3 pieces of relevant information
    on the given topic and present the findings with citations.""",
    tools = [google_search],
    output_key="protein_research", 
)

algorithms_researcher = Agent(
    model='gemini-3-pro-preview',
    name = "algorithms_agent",
    description = "An algorithms agent searching tech related stuff",
    instruction = """You are a specialized research agent, focused on algorithms news, articles and research.
    Your only job is to use the google_search tool to find 2-3 pieces of relevant information
    on the given topic and present the findings with citations.""",
    tools = [google_search],
    output_key="algorithms_research", 
)

aggregator_agent = Agent(
    name="AggregatorAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
       # It uses placeholders to inject the outputs from the parallel agents, which are now in the session state.
    instruction="""Combine these three research findings into a single executive summary:

    **Technology Trends:**
    {tech_research}
    
    **Health Breakthroughs:**
    {protein_research}
    
    **Finance Innovations:**
    {algorithms_research}
    
    Your summary should highlight common themes, surprising connections, and the most important key takeaways from all three reports. The final summary should be around 200 words.""",
    output_key="executive_summary",  # This will be the final output of the entire system.
)

parallel_research_team = ParallelAgent(
    name="ParallelResearchTeam",
    sub_agents=[tech_researcher, protein_researcher, algorithms_researcher],
)

root_agent = SequentialAgent(
    name="ResearchSystem",
    sub_agents=[parallel_research_team, aggregator_agent],
)


runner = InMemoryRunner(agent=root_agent)

async def parallel_starter():
    response = await runner.run_debug(
        "Run the daily executive briefing"
    )
    return response

if __name__ == "__main__":
    asyncio.run(parallel_starter())
