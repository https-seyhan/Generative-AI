# Here we see two specialised agents collaborating — this pattern is increasingly used 
# in enterprise automation pipelines, such as compliance monitoring or report generation.
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
# Import the core CrewAI building blocks + OpenAI’s LLM wrapper.

# Define LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
# All agents will share the same GPT-4o brain, 
# but with a tiny bit of temperature (0.2) so they sound slightly different and more natural.

# Define agents
researcher = Agent(name="Researcher", role="Finds and summarises info", goal="Provide accurate background", llm=llm)
analyst = Agent(name="Analyst", role="Interprets data and writes insights", goal="Create a clear summary", llm=llm)
# Researcher → knows it has access to search tools (CrewAI automatically gives every agent a web-search tool ).
# Analyst → great at reasoning and writing clear explanations.

# Define tasks
task1 = Task(description="Find the latest AI regulations in Australia", agent=researcher)
task2 = Task(description="Summarise and explain implications for businesses", agent=analyst)
# Tasks are assigned to specific agents.
# By default, CrewAI runs them sequentially (task1 → passes its output → task2).
# Important: task2 automatically receives the full output of task1 as context — no extra code needed.

# Create Crew
crew = Crew(agents=[researcher, analyst], tasks=[task1, task2])
# This is the “team”.
# CrewAI wires everything together:

#       Gives both agents search tools (and math if you add them)
#       Handles the hand-off of information between agents
#       Manages the ReAct loops internally for each agent
results = crew.kickoff()
print(results)
# What actually happens when you run it (verbose mode would show this):
# Researcher wakes up
# → Searches Google (via SerpAPI or any backend you configured) for “Australia AI regulations 2025”
# → Reads the latest government bills, summaries from law firms, etc.
# → Writes a clean bullet-point background
# Its output is automatically sent to Analyst
# Analyst wakes up
# → Receives Researcher’s full report
# → Thinks: “Now I need to explain what this means for businesses”
# → Produces a business-friendly summary with implications, risks, and opportunities
# Crew finishes and returns the final result (Analyst’s output by default)

