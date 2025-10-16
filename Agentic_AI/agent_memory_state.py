from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOpenAI

graph = StateGraph()
llm = ChatOpenAI(model="gpt-4o-mini")

@graph.add_node("chat")
def chat_node(state):
    user_input = state.get("input")
    memory = state.get("memory", [])
    response = llm.invoke(f"History: {memory}\nUser: {user_input}\nAgent:")
    memory.append(user_input)
    memory.append(response.content)
    return {"memory": memory, "output": response.content}

graph.set_entry_point("chat")
graph.add_edge("chat", END)

result = graph.run({"input": "Tell me about renewable energy trends in Australia.", "memory": []})
print(result["output"])