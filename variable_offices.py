
import pandas as pd
import numpy as np
import gurobipy as gp

###############
# Group 5
# When you have no fixed center offices
###############

index_values = pd.read_csv("data/bricks_index_values.csv",index_col=0)
distances = pd.read_csv("data/brick_rp_distances.csv",index_col=0)
all_distances = pd.read_csv("data/distances.csv",delimiter=";")
distance_matrix= np.genfromtxt('data/dis.csv', delimiter=',', skip_header=0)  # skip_header=1 if there is a header

representation = [
    [4,5,6,7,8,15],
    [10,11,12,13,14],
    [9,16,17,18],
    [1,2,3,19,20,21,22]
]

NB_SR = len(representation)
N_BRICKS = sum(len(repres) for repres in representation)
new_work_balance = (len(representation))/NB_SR

representation_minus_one = [[x-1 for x in row] for row in representation]

distances = np.array(distances)
workload = index_values

# Create Model and Variables
m = gp.Model("Step 1")
v = m.addMVar(shape=(N_BRICKS,NB_SR),vtype=gp.GRB.BINARY,name="V")
center_bricks = m.addVars(N_BRICKS,vtype = gp.GRB.BINARY, name="C")

# Create Constraints
# Center brick constraints
old_center = [3,13,15,21]
old_center_one_hot = [1 if i in old_center else 0 for i in range(N_BRICKS)]

m.addConstr(gp.quicksum(center_bricks[i] for i in range(len(center_bricks))) == NB_SR, name="nb center")

# One SR per area
m.addConstrs(
    (sum(v[i, j] for j in range(NB_SR)) == 1 for i in range(N_BRICKS)), 
    name="RowSum"
)

# Center can only belong to one SR
m.addConstrs(
    (gp.quicksum(center_bricks[i]*v[i,j] for i in range(N_BRICKS)) == 1 for j in range(NB_SR)),
    name="OCPSR" # One Center Per SR
)


# MinMax workload objective
workload_repeated = np.repeat(workload, NB_SR, axis=0).reshape(N_BRICKS, NB_SR)
resulting_workload = m.addVars(N_BRICKS, NB_SR, vtype=gp.GRB.CONTINUOUS, name="RW")
m.addConstrs(
    (resulting_workload[i, j] == workload_repeated[i, j] * v[i, j] for i in range(N_BRICKS) for j in range(NB_SR)),
    name="workload"
)
weighted_sums = m.addVars(NB_SR, lb=0, vtype=gp.GRB.CONTINUOUS, name="WS")
m.addConstrs(
    (weighted_sums[j] == gp.quicksum(resulting_workload[i, j] for i in range(N_BRICKS)) for j in range(NB_SR)),
    name="V=WS"
)
max_workload = m.addVar(vtype=gp.GRB.CONTINUOUS, name="max_workload")
m.addConstrs(
    (weighted_sums[j] <= max_workload for j in range(NB_SR)),
    name="MaxWorkloadConstraint"
)
m.setObjective(max_workload, gp.GRB.MINIMIZE)

# Disruption objective
diff_var = m.addVars(N_BRICKS, vtype=gp.GRB.INTEGER, name="Diff")
m.addConstrs(
    (diff_var[i] >= center_bricks[i] - old_center_one_hot[i] for i in range(N_BRICKS)),
    name="diff_constr"
)
abs_var = m.addVars(N_BRICKS, vtype=gp.GRB.INTEGER, name="abs")
m.addConstrs((abs_var[i] == gp.abs_(diff_var[i]) for i in range(N_BRICKS)), name="abs_constr")
sum_abs_var = m.addVar(vtype=gp.GRB.INTEGER, name="sum_abs")
m.addConstr(sum_abs_var == gp.quicksum(abs_var[i] for i in range(N_BRICKS)), name="sum_abs_constr")
scaled_sum_abs_var = m.addVar(vtype=gp.GRB.CONTINUOUS, name="scaled_sum_abs")
m.addConstr(scaled_sum_abs_var == 0.5 * sum_abs_var, name="scaled_sum_constr")
m.setObjective(scaled_sum_abs_var, gp.GRB.MINIMIZE)

# Total distance objective
mult_mat = m.addVars(N_BRICKS,NB_SR,vtype=gp.GRB.CONTINUOUS)
m.addConstrs(mult_mat[i,j] == v[i,j]*distances[i,j] for i in range(N_BRICKS) for j in range(NB_SR))
dist_sum = gp.quicksum(mult_mat[i,j] for i in range(N_BRICKS) for j in range(NB_SR))
m.setObjective(dist_sum, gp.GRB.MINIMIZE)
m.optimize() 

# restemp = []
# for i in range(N_BRICKS):
#     l=[]
#     for j in range(NB_SR):
#         l.append(abs_var[i,j].x)
#     restemp.append(l)
# print(restemp)

# dist_mat = []
# for i in range(N_BRICKS):
#     row = []
#     for j in range(NB_SR):
#         row.append(mult_mat[i,j].x)
#     dist_mat.append(row)


# dist_mat = np.array(dist_mat)
# print(dist_mat)

# new_center = -1
# if nb_additional_SR > 0:
#     for i in range(N_BRICKS):
#         if center_bricks[i].x > 0:
#             new_center = i
#     print(f"nouveau SR en {i}")
print("La solution est :")
res = []
for j in range(NB_SR):
    sale_j = []
    for i in range(N_BRICKS):
        if v[i,j].x == 1:
            sale_j.append(i+1)
    res.append(sale_j)
res2 = []
for i in range(N_BRICKS):
    if center_bricks[i].x == 1:
        res2.append(i+1)
print(res)
print("Centers :", res2)
# print(f"new_center : {res2} // old : [3,13,15,21]")

#print workload matrix
# workload_res = []
# for i in range(N_BRICKS):
#     row = []
#     for j in range(NB_SR):
#         row.append(resulting_workload[i,j].x)
#     workload_res.append(row)

# workload_res = np.array(workload_res)
# print(workload_res)

# sums = []
# for i in range(NB_SR):
#     sums.append(weighted_sums[i].x)
# print(sums)


################################
# Display part
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS


# automatic definition to do

# keys = [4, 14, 16, 22]
keys = res2
# keys.append(new_center+1)

# Create a dictionary by zipping keys with the sublists
center_to_associates = dict(zip(keys, res))

center_to_assigned = {key - 1: [value - 1 for value in values] for key, values in center_to_associates.items()}

# Apply MDS
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
points_2d = mds.fit_transform(distance_matrix)

# Assign colors 
colors = ['red', 'green', 'blue', 'purple', 'orange','cyan']
color_map = {}
for idx, center in enumerate(center_to_assigned.keys()):
    color_map[center] = colors[idx % len(colors)]

# Plot the map
plt.figure(figsize=(8, 6))
for center_idx, associates in center_to_assigned.items():
    center_x, center_y = points_2d[center_idx]
    center_color = color_map[center_idx]
    
    # Plot center point
    plt.scatter(center_y, center_x, c=center_color, s=150, label=f"Center {center_idx+1}", edgecolor='black')

    # Plot associated points
    for assoc_idx in associates:
        assoc_x, assoc_y = points_2d[assoc_idx]
        plt.scatter(assoc_y, assoc_x, c=center_color, s=100, label=f"Assigned {assoc_idx+1}" if assoc_idx == associates[0] else "")
        # Draw connection lines
        plt.plot([center_y, assoc_y], [center_x, assoc_x], c=center_color, linestyle='--', lw=1.5)

# Annotate
for i, (x, y) in enumerate(points_2d):
    plt.text(y, x, f"Area {i+1}", fontsize=10, ha="right", weight="bold")

# Plot formatting
plt.title("Areas Map with Multiple Centers and Different Colors", fontsize=14)
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.grid()
plt.show()