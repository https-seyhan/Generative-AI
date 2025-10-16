from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOpenAI

graph = StateGraph()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

@graph.add_node("conversation")
def chat_node(state):
    user_input = state.get("input")
    history = state.get("memory", [])
    response = llm.invoke(f"History: {history}\nUser: {user_input}\nAgent:")
    history.append(user_input)
    history.append(response.content)
    return {"memory": history, "output": response.content}

graph.set_entry_point("conversation")
graph.add_edge("conversation", END)

# Example run
session = {"memory": []}
session["input"] = "Explain how Agentic AI differs from traditional LLMs."
result = graph.run(session)
print(result["output"])


# Example: summarising a business report
user_query = """
Summarise the following report into key insights for executives:
'The Australian renewable sector grew 14% in 2024, driven by solar uptake, while wind lagged behind due to maintenance delays.'
"""

response = llm.invoke(user_query)
print(response.content)