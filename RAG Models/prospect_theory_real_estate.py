import numpy as np

# --- Prospect Theory Utility Functions (Kahneman & Tversky) ---

def pt_utility(x, alpha=0.88, beta=0.88, lambda_=2.25):
    """Value function in Prospect Theory."""
    return x**alpha if x >= 0 else -lambda_ * (-x)**beta

def pt_weight(p, gamma=0.61):
    """Probability weighting function in Prospect Theory."""
    return (p ** gamma) / ((p ** gamma + (1 - p) ** gamma) ** (1 / gamma))

# --- Decision Tree Node with EV & PT Logic ---

class DecisionNode:
    def __init__(self, name, outcomes, probs):
        self.name = name
        self.outcomes = outcomes
        self.probs = probs
        self.ev = self.compute_ev()
        self.pt_value = self.compute_pt_value()

    def compute_ev(self):
        return np.dot(self.outcomes, self.probs)

    def compute_pt_value(self):
        pt_utilities = [pt_utility(x) for x in self.outcomes]
        pt_weights = [pt_weight(p) for p in self.probs]
        return np.dot(pt_utilities, pt_weights)

    def summary(self):
        return {
            'Decision': self.name,
            'Expected Value (EV)': round(self.ev, 2),
            'Prospect Theory Utility': round(self.pt_value, 2)
        }

# --- Real Estate Decision Tree (AI Agent Layer) ---

def real_estate_decision_agent(decision_options):
    decision_nodes = []
    for option, details in decision_options.items():
        node = DecisionNode(name=option, outcomes=details['outcomes'], probs=details['probs'])
        decision_nodes.append(node)

    # Sort by PT value (behavioral preference)
    sorted_nodes = sorted(decision_nodes, key=lambda x: x.pt_value, reverse=True)

    print("=== Real Estate Decision Recommendations (AI Agent Tree) ===")
    for i, node in enumerate(sorted_nodes, start=1):
        summary = node.summary()
        print(f"\nOption {i}: {summary['Decision']}")
        print(f"  - EV: ${summary['Expected Value (EV)']}")
        print(f"  - PT Value: {summary['Prospect Theory Utility']} (weighted by behavioral bias)")

    print("\n🔍 AI Agent Recommendation → Choose:", sorted_nodes[0].name)

# --- Example Use Case: Real Estate Options

options = {
    "Option A: Buy House in Premium Suburb": {
        "outcomes": [150000, -50000],  # gain, loss
        "probs": [0.7, 0.3]
    },
    "Option B: Rent + Invest in Stocks": {
        "outcomes": [80000, -10000],
        "probs": [0.6, 0.4]
    },
    "Option C: Buy Affordable House in Growth Zone": {
        "outcomes": [100000, -20000],
        "probs": [0.65, 0.35]
    },
    "Option D: Real Estate Trust (REIT Investment)": {
        "outcomes": [60000, -5000],
        "probs": [0.85, 0.15]
    }
}

# --- Run AI Agent Tree
real_estate_decision_agent(options)
