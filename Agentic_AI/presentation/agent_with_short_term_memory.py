# This model remembers past interactions — a step towards persistent, self-improving systems.
# this is no longer a stateless chatbot. It remembers everything forever (or until you clear the memory).

from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOpenAI
# LangGraph = the framework from LangChain as graphs (nodes + edges) with persistent state.
# StateGraph = a graph where every node receives and returns a shared dictionary (the "state").
# END = special marker meaning "we're done".

# Define state and model
graph = StateGraph()
llm = ChatOpenAI(model="gpt-4o-mini")
# Empty graph created.
# Using the cheap, fast gpt-4o-mini model

@graph.add_node("chat")
# This defines the only node in the graph, called "chat".

def chat_node(state):
# Every time this node runs, it receives the current state (a dictionary) that travels through the entire graph.
    user_input = state.get("input") # "input": the new message from the user
    history = state.get("memory", []) # "memory": the full conversation history so far (list of strings). Defaults to empty list first time.
    response = llm.invoke(f"Conversation history: {history}\nUser: {user_input}\nAgent:") # The LLM gets the entire past conversation injected into its prompt every single time
    history.append(user_input)
    history.append(response.content) # We update the history with the latest exchange.
    return {"memory": history, "output": response.content}
#    The node returns an updated state:
#    New memory (now longer)
#    The agent’s reply in "output"
#    This updated state will be passed to the next loop if you call the graph again.

graph.set_entry_point("chat")
graph.add_edge("chat", END)
# First node to run = "chat"
# After "chat" finishes → go to END (stop)
# So the flow is: start → chat → END

# Run
# First user message
result = graph.run({"input": "Tell me about renewable energy trends in Australia.", "memory": []})

# A fresh conversation (memory: [])
print(result["output"])

# Second turn — it remembers the first question!
result2 = graph.run({
    "input": "Which state is leading and why?",
    "memory": result["memory"]   # ← pass the old memory forward!
})
print(result2["output"])
# This is true persistent memory. Similar to conversations with chatGPT


