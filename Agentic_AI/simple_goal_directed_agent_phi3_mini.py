from crewai import Agent, Task, Crew
import litellm

# --------------------------------------------
# Lightweight LLM Wrapper (Single Shared Model)
# --------------------------------------------
class LocalLiteLLM:
    def __init__(self, model_name="ollama/phi3:mini", temperature=0.3, max_tokens=200):
        # Use smaller, quantised models from Ollama
        # e.g. phi3:mini, gemma:2b, or mistral:instruct
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def call(self, prompt: str) -> str:
        """Simple call method compatible with CrewAI"""
        try:
            response = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                # Stream disabled to reduce memory
                stream=False
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[Error calling model: {e}]"

# --------------------------------------------
# Use one shared instance of a small local model
# --------------------------------------------
llm = LocalLiteLLM(model_name="ollama/phi3:mini", temperature=0.2, max_tokens=150)

# --------------------------------------------
# Define lightweight Agents
# --------------------------------------------
researcher = Agent(
    name="Researcher",
    role="AI Policy Researcher",
    goal="Find and summarise AI regulations in Australia briefly.",
    backstory="Expert in Australian AI policy and law, concise writing.",
    llm=llm
)

analyst = Agent(
    name="Analyst",
    role="Business Analyst",
    goal="Explain AI regulations simply for small businesses.",
    backstory="Business analyst focusing on tech compliance, uses simple language.",
    llm=llm
)

# --------------------------------------------
# Define light Tasks
# --------------------------------------------
task1 = Task(
    description="Identify key AI regulations in Australia (keep short).",
    agent=researcher,
    expected_output="Short list or paragraph of main AI policies."
)

task2 = Task(
    description="Summarise business implications of these regulations in simple terms.",
    agent=analyst,
    expected_output="Concise 2–3 sentence summary."
)

# --------------------------------------------
# Create Crew
# --------------------------------------------
crew = Crew(
    agents=[researcher, analyst],
    tasks=[task1, task2],
    verbose=False  # Turn off verbose logging to reduce console overhead
)

if __name__ == "__main__":
    results = crew.kickoff()
    print("\n=== Crew Results ===\n")
    print(results)
