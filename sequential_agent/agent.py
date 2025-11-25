from google.adk.agents import Agent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools import google_search
from google.genai import types
from google.adk.runners import InMemoryRunner
import asyncio

retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)

#Next time you run this define a function to calculate the time Gemini 3 needs.
outline_agent = Agent(
    name="OutlineAgent",
    model=Gemini(
        model="gemini-3-pro-preview",
        retry_options= retry_config
    ),
    instruction="""From a google search, you should come up with the most powerful algorithms for the given competition description.""",
    tools = [google_search], 
    output_key="algorithms_outline",  # The result of this agent will be stored in the session state with this key.
)


# Writer Agent: Writes the full blog post based on the outline from the previous agent.
writer_agent = Agent(
    name="WriterAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    # The `{blog_outline}` placeholder automatically injects the state value from the previous agent's output.
    instruction="""Following this outline strictly: {algorithms_outline}
    Write a brief, 200 to 300-word explanation post with an engaging and informative tone.""",
    output_key="algorithms_draft",  # The result of this agent will be stored with this key.
)

editor_agent = Agent(
    name="EditorAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    # This agent receives the `{blog_draft}` from the writer agent's output.
    instruction="""Edit this draft: {algorithms_draft}
    Your task is to make the algorithms explainable to a 5 years old.
    So the main goal must be to decompose the algorithms (with some analogies) and enhancing overall clarity.""",
    output_key="final_algorithms_explanation",  # This is the final output of the entire pipeline.
)

root_agent = SequentialAgent(
    name='algorithms_pipeline',
    sub_agents=[outline_agent, writer_agent, editor_agent],
)


runner = InMemoryRunner(agent=root_agent)
async def runner_starter():
    response = await runner.run_debug(
       """CAFA 6 Protein Function Prediction:
In this competition, you will build a model that predicts
Gene Ontology (GO) terms for protein sequences. The CAFA
(Critical Assessment of Function Annotation) is a community-wide
experiment to benchmark and improve methods for protein
function prediction. The challenge is to predict the function
of "unknown" proteins. You are given protein sequences and
their associated GO terms (from the BPO, CCO, and MFO ontologies)
for the training set. Your goal is to predict these GO terms
for a test set of proteins."""
    )
    return response

if __name__ == "__main__":
    asyncio.run(runner_starter())