from crewai import Agent, Task, Crew
import litellm

# -----------------------------
# Wrapper class for local LLM
# -----------------------------
class OllamaLLM:
    def __init__(self, model_name="llama3", temperature=0.2, max_tokens=500):
        self.model_name = f"ollama/{model_name}"  # Prepend provider to model name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def call(self, prompt: str) -> str:
        """CrewAI expects a .call() method"""
        response = litellm.completion(
            model=self.model_name,  # Use the formatted model name
            messages=[{"role": "user", "content": prompt}],  # Format prompt as messages
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content  # Extract the response content

# -----------------------------
# Initialise LLM
# -----------------------------
llm = OllamaLLM()

# -----------------------------
# Define Agents
# -----------------------------
researcher = Agent(
    name="Researcher",
    role="AI Policy Researcher",
    goal="Find and summarise AI regulations in Australia",
    backstory="You are an expert policy researcher specialising in Australian AI law and ethics.",
    llm=llm
)

analyst = Agent(
    name="Analyst",
    role="Business Analyst",
    goal="Explain AI regulations to businesses",
    backstory="You are a business analyst specialising in technology compliance and risk.",
    llm=llm
)

# -----------------------------
# Define Tasks
# -----------------------------
task1 = Task(
    description="Identify the latest AI regulations in Australia.",
    agent=researcher,
    expected_output="A list or summary of AI regulations and policies."
)

task2 = Task(
    description="Summarise implications of AI regulations for Australian businesses.",
    agent=analyst,
    expected_output="Concise summary of business implications of regulations."
)

# -----------------------------
# Create Crew and run
# -----------------------------
crew = Crew(
    agents=[researcher, analyst],
    tasks=[task1, task2],
    verbose=True
)

results = crew.kickoff()

print("\n=== Crew Results ===\n")
print(results)