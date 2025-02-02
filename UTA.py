import pandas as pd
import numpy as np
import gurobipy as gp


####
# UTA test
####


data = pd.read_csv("./data/Preferences.csv",delimiter=";" )

new_rep = np.array([[104.67, 1.14, 2],
                    [110.5, 1.34, 2]])


ref_rep = data.drop(columns=["rank"]).to_numpy()
ranking = data["rank"].to_list()
ranking = [r-1 for r in ranking]

max_vals = ref_rep.max(axis=0)
ref_rep = ref_rep/max_vals
new_rep = new_rep/max_vals
print(new_rep)
num_pieces = 5

model = gp.Model("UTA")

u = model.addVars(ref_rep.shape[1], num_pieces, lb=0, name="u")
sigma = model.addVars(len(ranking), lb=0, name="sigma")

model.setObjective(gp.quicksum(sigma[i] for i in range(len(ranking))), gp.GRB.MINIMIZE)

for i in range(len(ranking)):
    for j in range(len(ranking)):
        if i < j:
            model.addConstr(
                gp.quicksum(u[k, l] * ref_rep[ranking[i], k] for k in range(ref_rep.shape[1]) for l in range(num_pieces)) + sigma[i] <=
                gp.quicksum(u[k, l] * ref_rep[ranking[j], k] for k in range(ref_rep.shape[1]) for l in range(num_pieces)) + sigma[j] - 1
            )

model.optimize()

u_values = model.getAttr('x', u)

def compute_value(solution, u_values):
    value = 0
    for k in range(len(solution)):
        for l in range(num_pieces):
            value += u_values[k, l] * solution[k]
    return value


reference_values = [compute_value(uni, u_values) for uni in ref_rep]
new_values = [compute_value(uni, u_values) for uni in new_rep]

all_solution = np.vstack((ref_rep, new_rep))
all_values = reference_values + new_values

labels= ["old_"+str(r) for r in ranking]+["new_"+str(i) for i in range(len(new_rep))]

ranked_solution = sorted(range(len(all_values)), key=lambda k: all_values[k])

# print("Reference solution values:", reference_values)
# print("New solution values:", new_values)
print("Ranked solution:")
for i in ranked_solution:
    print(f"{labels[i]}: {all_values[i]}")

