from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

researcher = Agent(name="Researcher", role="Finds data", llm=llm)
analyst = Agent(name="Analyst", role="Summarises insights", llm=llm)

crew = Crew(agents=[researcher, analyst],
            tasks=[
                Task(description="Find AI regulations in Australia", agent=researcher),
                Task(description="Explain implications for businesses", agent=analyst)
            ])

results = crew.kickoff()
print(results)