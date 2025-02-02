###############
# Group 5
###############



import pandas as pd
import numpy as np
import gurobipy as gp

index_values = pd.read_csv("data/bricks_index_values_100.csv",index_col=0)
distances = pd.read_csv("data/brick_rp_distances_100.csv",index_col=0)
# pfitzer_100 = pd.read_csv("data/Pfitzer10-100.csv")
distance_matrix= np.genfromtxt('data/dis_100.csv', delimiter=',', skip_header=0)  # skip_header=1 if there is a header

#### change here the type of optimization
OBJECTIVE = "dist" #or "disrupt" or "dist"
# print(index_values.head())
# print(distances)
# print(pfitzer_100.head())

init_affect = pd.read_csv("data/init_affect_100.csv",index_col=0)
representation = {}
for i in init_affect.index:
    if init_affect.loc[i,"rp"] not in representation:
        representation[init_affect.loc[i,"rp"]] = [i]
    else:
        representation[init_affect.loc[i,"rp"]].append(i)
representation = list(representation.values())
representation_minus_one = [[x-1 for x in row] for row in representation]
print(representation_minus_one)
N_BRICKS = index_values.shape[0]
N_RPS = distances.shape[1]

distances = np.array(distances)
workload = index_values

#Create Model and Variables
m = gp.Model("Step 1")
v = m.addMVar(shape=(N_BRICKS,N_RPS),vtype=gp.GRB.BINARY,name="V")

#Create Constraints
# Center brick constraints
center = [2,
17,
29,
32,
58,
65,
72,
78,
81,
97,
]
for i,k in enumerate(center):
    m.addConstr(v[k,i] == 1)

#One SR per area
m.addConstrs(
    (sum(v[i, j] for j in range(N_RPS)) == 1 for i in range(N_BRICKS)), 
    name="RowSum"
)

# Workload constraints
workload_repeated = np.repeat(workload,N_RPS,axis=0).reshape(N_BRICKS,N_RPS)
resulting_workload = m.addVars(N_BRICKS,N_RPS,vtype=gp.GRB.CONTINUOUS, name="RW")
m.addConstrs(
    (resulting_workload[i,j] == workload_repeated[i,j] * v[i,j] for i in range(N_BRICKS) for j in range(N_RPS)),
    name="workload"
)

minWL = 0.95
maxWL = 1.05
weighted_sums = m.addVars(N_RPS, lb=0, vtype=gp.GRB.CONTINUOUS,name="WS")
m.addConstrs(
    (weighted_sums[j] == gp.quicksum(resulting_workload[i,j] for i in range(N_BRICKS)) for j in range(N_RPS)), 
    name="V=WS"
)

m.addConstrs(
    (weighted_sums[j] >= minWL for j in range(N_RPS)), 
    name="MinWeightedSums"
)
m.addConstrs(
    (weighted_sums[j] <= maxWL for j in range(N_RPS)), 
    name="MaxWeightedSums"
)

#Create Objective
#Objective minimum disruption
m_orig = np.zeros((N_BRICKS,N_RPS),dtype=int)
for i in range(N_RPS):
    m_orig[representation_minus_one[i],i] = 1
    
#Objective 1
if OBJECTIVE == "disrupt":
    diff_var = m.addVars(N_BRICKS,N_RPS,vtype=gp.GRB.INTEGER,name="Diff")
    m.addConstrs((diff_var[i,j] >= v[i,j] - m_orig[i,j] for i in range(N_BRICKS) for j in range(N_RPS)),name="diff?")
    abs_var = m.addVars(N_BRICKS,N_RPS,vtype=gp.GRB.INTEGER,name="abs")
    m.addConstrs((abs_var[i,j] == gp.abs_(diff_var[i,j]) for i in range(N_BRICKS) for j in range(N_RPS)),name="abs?")
    disrupt_sum = gp.quicksum(abs_var[i,j] for i in range(N_BRICKS) for j in range(N_RPS))
    
    m.setObjective(disrupt_sum, gp.GRB.MINIMIZE)
    m.optimize()
    # restemp = []
    # for i in range(N_BRICKS):
    #     l=[]
    #     for j in range(N_RPS):
    #         l.append(abs_var[i,j].x)
    #     restemp.append(l)
    # print(restemp)
    print("Disrupt sum result: ",disrupt_sum.getValue())

elif OBJECTIVE == "dist":
    #Objective 2
    mult_mat = m.addVars(N_BRICKS,N_RPS,vtype=gp.GRB.CONTINUOUS)
    m.addConstrs(mult_mat[i,j] == v[i,j]*distances[i,j] for i in range(N_BRICKS) for j in range(N_RPS))
    dist_sum = gp.quicksum(mult_mat[i,j] for i in range(N_BRICKS) for j in range(N_RPS))
    m.setObjective(dist_sum, gp.GRB.MINIMIZE)
    m.optimize()

    # dist_mat = []
    # for i in range(N_BRICKS):
    #     row = []
    #     for j in range(N_RPS):
    #         row.append(mult_mat[i,j].x)
    #     dist_mat.append(row)

    # dist_mat = np.array(dist_mat)
    # print(dist_mat)

print("La solution est :")
res = []
for j in range(N_RPS):
    sale_j = []
    for i in range(N_BRICKS):
        if v[i,j].x == 1:
            sale_j.append(i+1)
    res.append(sale_j)

print(res)

#print workload matrix
# workload_res = []
# for i in range(N_BRICKS):
#     row = []
#     for j in range(N_RPS):
#         row.append(resulting_workload[i,j].x)
#     workload_res.append(row)

# workload_res = np.array(workload_res)
# print(workload_res)

# sums = []
# for i in range(N_RPS):
#     sums.append(weighted_sums[i].x)

# print(sums)


################################
# Display part
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

#ids is tha distance csv stripped of headers data

# automatic definition to do

keys = [x+1 for x in center]

# Create a dictionary by zipping keys with the sublists
center_to_associates = {}
for cluster in res:
    for key in keys:
        if key in cluster:
            center_to_associates[key] = cluster

center_to_assigned = {key - 1: [value - 1 for value in values] for key, values in center_to_associates.items()}

# Apply MDS
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
points_2d = mds.fit_transform(distance_matrix)

# Assign colors 
colors = ['red', 'green', 'blue', 'purple', 'orange','cyan','pink','yellow','black','brown']
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


