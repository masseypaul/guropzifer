import pandas as pd
import numpy as np
import gurobipy as gp


####
# UTA test
####


data = pd.read_csv("./data/Preferences.csv",delimiter=";" )

new_rep = np.array([[104.67, 1.14, 2],
                    [110.5, 1.34, 2],
                    [101.51,1.18,3]])


ref_rep = data.drop(columns=["rank"]).to_numpy()
ranking = data["rank"].to_list()
ranking = [r-1 for r in ranking]

max_vals = ref_rep.max(axis=0)
ref_rep = ref_rep/max_vals
new_rep = new_rep/max_vals
num_pieces = 20
model = gp.Model("UTA")
margin = 0.001

def evaluate_piecewise_linear(x, y_points):
    x_points = np.array([i/(num_pieces-1) for i in range(num_pieces)])
    idx = np.searchsorted(x_points, x, side='left')

    if idx == 0:
        return y_points[0]
    elif idx == len(x_points):
        return y_points[-1]
    else:
        x0, x1 = x_points[idx - 1], x_points[idx]
        y0, y1 = y_points[idx - 1], y_points[idx]
        slope = (y1 - y0) / (x1 - x0)
        return y0 + slope * (x - x0)

def sum_piecewise_linear_functions(vector, piecewise_functions):
    total_sum = 0
    for x, function in zip(vector, piecewise_functions):
        s = evaluate_piecewise_linear(x, function)
        total_sum += s

    return total_sum

u = model.addVars(ref_rep.shape[1], num_pieces, lb=0, name="u")
sigma_neg = model.addVars(len(ranking), lb=0, name="sigma")
v = model.addVars(len(ranking), ref_rep.shape[1], lb=0, name="v")

model.setObjective(gp.quicksum(sigma_neg[i] for i in range(len(ranking))), gp.GRB.MINIMIZE)

for k in range(ref_rep.shape[1]):
    for l in range(num_pieces - 1):
        model.addConstr(u[k, l] <= u[k, l + 1],name=f"monotone_{k}_{l}")

model.addConstr(gp.quicksum(u[k, num_pieces - 1] for k in range(ref_rep.shape[1])) == 1, name="normal")

breakpoints = np.linspace(0, 1, num_pieces)

for i in range(len(ranking)):
    for k in range(ref_rep.shape[1]):
        for l in range(num_pieces - 1):
            if breakpoints[l] <= ref_rep[ranking[i], k] <= breakpoints[l + 1]:
                alpha = (ref_rep[ranking[i], k] - breakpoints[l]) / (breakpoints[l + 1] - breakpoints[l])
                model.addConstr(v[i, k] == (1 - alpha) * u[k, l] + alpha * u[k, l + 1])

# Add ranking constraints
for i in range(len(ranking)):
    for j in range(i + 1, len(ranking)):
            model.addConstr(
                gp.quicksum(v[i, k] for k in range(ref_rep.shape[1])) + sigma_neg[i] <=
                gp.quicksum(v[j, k] for k in range(ref_rep.shape[1])) + sigma_neg[j] - margin,
                name="rank"
            )


# Optimize the model
model.optimize()



u_values = model.getAttr('x', u)
new_values = []
for k in range(ref_rep.shape[1]):
    new_values.append([u_values[k, l] for l in range(num_pieces)])
    print(f"Utility for criterion {k}: {[u_values[k, l] for l in range(num_pieces)]}")


def compute_value(solution, u_values):
    value = 0
    for k in range(len(solution)):
        for l in range(num_pieces):
            value += u_values[k, l] * solution[k]
    return value


reference_values = [sum_piecewise_linear_functions(list(uni), new_values) for uni in ref_rep]
new_values = [sum_piecewise_linear_functions(list(uni), new_values) for uni in new_rep]

all_solution = np.vstack((ref_rep, new_rep))
all_values = reference_values + new_values

labels= ["old_"+str(r) for r in ranking]+["new_"+str(i) for i in range(len(new_rep))]

ranked_solution = sorted(range(len(all_values)), key=lambda k: all_values[k])

print("Ranked solution:")
for i in ranked_solution:
    print(f"{labels[i]}: {all_values[i]}")

