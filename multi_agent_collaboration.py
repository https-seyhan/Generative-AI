from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

# Define LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

# Define agents
researcher = Agent(name="Researcher", role="Finds and summarises info", goal="Provide accurate background", llm=llm)
analyst = Agent(name="Analyst", role="Interprets data and writes insights", goal="Create a clear summary", llm=llm)

# Define tasks
task1 = Task(description="Find the latest AI regulations in Australia", agent=researcher)
task2 = Task(description="Summarise and explain implications for businesses", agent=analyst)

# Create Crew
crew = Crew(agents=[researcher, analyst], tasks=[task1, task2])
results = crew.kickoff()

print(results)