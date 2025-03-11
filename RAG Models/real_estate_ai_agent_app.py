import streamlit as st
import numpy as np

# --- Prospect Theory Functions ---

def pt_utility(x, alpha=0.88, beta=0.88, lambda_=2.25):
    return x ** alpha if x >= 0 else -lambda_ * (-x) ** beta

def pt_weight(p, gamma=0.61):
    return (p ** gamma) / ((p ** gamma + (1 - p) ** gamma) ** (1 / gamma))

# --- Decision Tree Node Class ---
class DecisionNode:
    def __init__(self, name, outcomes, probs, alpha, beta, lambda_, gamma):
        self.name = name
        self.outcomes = outcomes
        self.probs = probs
        self.ev = self.compute_ev()
        self.pt_value = self.compute_pt_value(alpha, beta, lambda_, gamma)

    def compute_ev(self):
        return np.dot(self.outcomes, self.probs)

    def compute_pt_value(self, alpha, beta, lambda_, gamma):
        pt_utilities = [pt_utility(x, alpha, beta, lambda_) for x in self.outcomes]
        pt_weights = [pt_weight(p, gamma) for p in self.probs]
        return np.dot(pt_utilities, pt_weights)

    def summary(self):
        return {
            "Option": self.name,
            "Expected Value": round(self.ev, 2),
            "Prospect Theory Value": round(self.pt_value, 2)
        }

# --- Streamlit UI ---
st.set_page_config(page_title="Real Estate AI Agent", layout="centered")
st.title("🏘️ Real Estate AI Decision Tree using Prospect Theory")

st.sidebar.header("🎯 AI Agent Risk Profile (Prospect Theory Parameters)")
alpha = st.sidebar.slider("Gain Sensitivity (α)", 0.5, 1.0, 0.88)
beta = st.sidebar.slider("Loss Sensitivity (β)", 0.5, 1.0, 0.88)
lambda_ = st.sidebar.slider("Loss Aversion (λ)", 1.0, 4.0, 2.25)
gamma = st.sidebar.slider("Probability Distortion (γ)", 0.1, 1.0, 0.61)

st.markdown("Select decision options with outcomes and probabilities:")

# --- Options Definition (Add More If Needed) ---
default_options = {
    "Option A: Buy in Premium Suburb": {
        "outcomes": [150000, -50000],
        "probs": [0.7, 0.3]
    },
    "Option B: Rent + Stock Market": {
        "outcomes": [80000, -10000],
        "probs": [0.6, 0.4]
    },
    "Option C: Buy in Growth Zone": {
        "outcomes": [100000, -20000],
        "probs": [0.65, 0.35]
    },
    "Option D: Real Estate Trust (REIT)": {
        "outcomes": [60000, -5000],
        "probs": [0.85, 0.15]
    }
}

# --- Compute and Display Results ---
results = []
for key, val in default_options.items():
    node = DecisionNode(
        name=key,
        outcomes=val["outcomes"],
        probs=val["probs"],
        alpha=alpha,
        beta=beta,
        lambda_=lambda_,
        gamma=gamma
    )
    results.append(node.summary())

# --- Sort by PT value
sorted_results = sorted(results, key=lambda x: x["Prospect Theory Value"], reverse=True)

# --- Show Table
st.subheader("📊 AI Agent Decision Outcomes")
st.table(sorted_results)

# --- Recommend Best Option
best_option = sorted_results[0]["Option"]
st.success(f"🔍 AI Agent recommends: **{best_option}** based on Prospect Theory utility.")

# --- Bonus Add-On
if st.checkbox("Show EV-based (rational) recommendation"):
    ev_sorted = sorted(results, key=lambda x: x["Expected Value"], reverse=True)
    ev_best = ev_sorted[0]["Option"]
    st.info(f"💡 If using classical Expected Value (EV), best option is: **{ev_best}**")
