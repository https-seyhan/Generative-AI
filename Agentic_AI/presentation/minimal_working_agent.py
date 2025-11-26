# This simple example shows reactive reasoning. However, a true agentic system extends beyond this — it plans, recalls, and adapts.
# It reacts step-by-step but doesn’t do long-term planning or memory by default.

from langchain.agents import initialize_agent, AgentType, load_tools
from langchain.chat_models import ChatOpenAI

# initialize_agent → the factory that builds an agent
# AgentType → defines different agent behaviors
# load_tools → quick way to plug in external tools
# ChatOpenAI → wrapper around GPT-4o (or any OpenAI model)


# 1 Setup model and tools
llm = ChatOpenAI(model="gpt-4o", temperature=0)
# Creates the "brain": GPT-4o with temperature=0 → deterministic, no randomness (important for reliable tool use).

tools = load_tools(["serpapi", "llm-math"], llm=llm)  # The agent decides by itself when to search, when to calculate, and when to stop.
# serpapi → Google search (so the agent can look up current facts like 2024 GDP)
# llm-math → a calculator tool powered by the LLM (can do addition, percentages, etc.)

# 2 Initialise the agent
agent = initialize_agent(tools, llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
# ReAct = Reason + Act 
# Zero-shot" = no examples in the prompt, just a description of each tool
# It works by repeatedly thinking in this loop inside the LLM:
#           Thought: Do I need a tool or can I answer now?
#           Action: [tool_name]
#           Action Input: {query for the tool}
#           Observation: [result from tool]
#           Thought: Now I have new info...
#           (repeat until it can give Final Answer)
# verbose=True → show exact thought process printed in the terminal.

# 3 Run an autonomous task
response = agent.run("Find the GDP growth rate of Australia in 2024 and explain its impact on small businesses.")
print(response)

