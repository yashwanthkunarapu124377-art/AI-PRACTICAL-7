# Run: python P7_bayesian_network.py
# Library: none. Only plain Python 3

# Simple Bayesian Network for Medical Diagnosis:
# Cold (C) -> Fever (F), Cough (G)

P_C = {True: 0.2, False: 0.8}

# P(Fever | Cold)
P_F_given_C = {
    True: {True: 0.7, False: 0.3},
    False: {True: 0.1, False: 0.9}
}

# P(Cough | Cold)
P_G_given_C = {
    True: {True: 0.8, False: 0.2},
    False: {True: 0.15, False: 0.85}
}

def infer_cold(fever, cough):
    # Joint probabilities: P(C, F, G) = P(C) * P(F|C) * P(G|C)
    joint_true = P_C[True] * P_F_given_C[True][fever] * P_G_given_C[True][cough]
    joint_false = P_C[False] * P_F_given_C[False][fever] * P_G_given_C[False][cough]
    
    # Normalize using Bayes' Rule: P(C=True | F, G)
    total = joint_true + joint_false
    return joint_true / total

if __name__ == "__main__":
    scenarios = [(True, True), (True, False), (False, True), (False, False)]
    print("--- Bayesian Network Medical Diagnosis ---")
    for fever, cough in scenarios:
        prob = infer_cold(fever, cough)
        print(f"Fever={str(fever):5} | Cough={str(cough):5} => P(Cold) = {prob:.4f}")
