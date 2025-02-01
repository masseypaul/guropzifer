
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


#### change here the type of optimization: Will be removed because 3-objectives
OBJECTIVE = "dist" #or "disrupt" or "dist"

representation = [
    [4,5,6,7,8,15],
    [10,11,12,13,14],
    [9,16,17,18],
    [1,2,3,19,20,21,22]
]

nb_SR = len(representation)
nb_bricks = sum(len(repres) for repres in representation)
new_work_balance = (len(representation))/nb_SR

representation_minus_one = [[x-1 for x in row] for row in representation]

distances = np.array(distances)
workload = index_values

# Create Model and Variables
m = gp.Model("Step 1")
v = m.addMVar(shape=(nb_bricks,nb_SR),vtype=gp.GRB.BINARY,name="V")

center_bricks = m.addVars(nb_bricks,vtype = gp.GRB.BINARY, name="C")

# Create Constraints
# Center brick constraints
old_center = [3,13,15,21]

m.addConstr(gp.quicksum(center_bricks[i] for i in range(len(center_bricks))) == nb_SR, name="nb center")

# One SR per area
m.addConstrs(
    (sum(v[i, j] for j in range(nb_SR)) == 1 for i in range(nb_bricks)), 
    name="RowSum"
)

# Center can only belong to one SR
m.addConstrs(
    (gp.quicksum(center_bricks[i]*v[i,j] for i in range(nb_bricks)) == 1 for j in range(nb_SR)),
    name="OCPSR" # One Center Per SR
)


# TO DO: Workload constraints: to reimplement as objective
workload_repeated = np.repeat(workload,nb_SR,axis=0).reshape(nb_bricks,nb_SR)
resulting_workload = m.addVars(nb_bricks,nb_SR,vtype=gp.GRB.CONTINUOUS, name="RW")
m.addConstrs(
    (resulting_workload[i,j] == workload_repeated[i,j] * v[i,j] for i in range(nb_bricks) for j in range(nb_SR)),
    name="workload"
)

minWL = 0.8*new_work_balance
maxWL = 1.2*new_work_balance
print(minWL,maxWL)
weighted_sums = m.addVars(nb_SR, lb=0, vtype=gp.GRB.CONTINUOUS,name="WS")
m.addConstrs(
    (weighted_sums[j] == gp.quicksum(resulting_workload[i,j] for i in range(nb_bricks)) for j in range(nb_SR)), 
    name="V=WS"
)

m.addConstrs(
    (weighted_sums[j] >= minWL for j in range(nb_SR)), 
    name="MinWeightedSums"
)
m.addConstrs(
    (weighted_sums[j] <= maxWL for j in range(nb_SR)), 
    name="MaxWeightedSums"
)


#Create Objective
#Objective minimum disruption
m_orig = np.zeros((nb_bricks,nb_SR),dtype=int)
for i in range(len(representation)):
    m_orig[representation_minus_one[i],i] = 1
    
#Objective 1
if OBJECTIVE == "disrupt":
    m_orig = np.zeros((nb_bricks,nb_SR),dtype=int)
    for i in range(nb_SR):
        m_orig[old_center[i],i] = 1

    diff_var = m.addVars(nb_bricks,nb_SR,vtype=gp.GRB.INTEGER,name="Diff")
    m.addConstrs((diff_var[i,j] >= v[i,j] - m_orig[i,j] for i in range(nb_bricks) for j in range(nb_SR)),name="diff?")
    abs_var = m.addVars(nb_bricks,nb_SR,vtype=gp.GRB.INTEGER,name="abs")
    m.addConstrs((abs_var[i,j] == 0.5*gp.abs_(diff_var[i,j]) for i in range(nb_bricks) for j in range(nb_SR)),name="abs?")
    disrupt_sum = gp.quicksum(abs_var[i,j] for i in range(nb_bricks) for j in range(nb_SR))
    m.setObjective(disrupt_sum, gp.GRB.MINIMIZE)

    m.write("myModel.lp")
    #Create Objective
    #Solve
    m.optimize()
    restemp = []
    for i in range(nb_bricks):
        l=[]
        for j in range(nb_SR):
            l.append(abs_var[i,j].x)
        restemp.append(l)
    print(restemp)


# elif OBJECTIVE == "dist":
    #TO DO: Objective 2:still broken because of center
    # mult_mat = m.addVars(nb_bricks,nb_SR,vtype=gp.GRB.CONTINUOUS,name="M")
    
    
    # dist_temp = m.addMVar(shape=(nb_bricks, nb_unfixed_center), vtype=gp.GRB.CONTINUOUS, name="D")
    
    # m.addConstrs((dist_temp[i,k] == gp.quicksum(center_bricks[j]*distance_matrix[i,j] 
    #                                             for j in range(nb_bricks) if j not in center)
    #               for i in range(nb_bricks) for k in range(nb_unfixed_center)), name="tempdist")
    

    
    # m.addConstrs((mult_mat[i,j] == v[i,j]*distances[i,j] for i in range(nb_bricks) for j in range(len(center))), name="ee")
    # m.addConstrs((mult_mat[i,j] == v[i,j]*dist_temp[i] for i in range(nb_bricks) for j in range(len(center),nb_SR)),
    #              name="Part2")
    # dist_sum = gp.quicksum(mult_mat[i,j] for i in range(nb_bricks) for j in range(nb_SR))
    
    # m.setObjective(dist_sum, gp.GRB.MINIMIZE)
    # m.write("myModel.lp")
    # #Create Objective
    # #Solve
    # m.optimize() 

    # dist_mat = []
    # for i in range(nb_bricks):
    #     row = []
    #     for j in range(nb_SR):
    #         row.append(mult_mat[i,j].x)
    #     dist_mat.append(row)
    

    # dist_mat = np.array(dist_mat)
    # print(dist_mat)


# new_center = -1
# if nb_additional_SR > 0:
#     for i in range(nb_bricks):
#         if center_bricks[i].x > 0:
#             new_center = i
#     print(f"nouveau SR en {i}")
print("La solution est :")
res = []
for j in range(nb_SR):
    sale_j = []
    for i in range(nb_bricks):
        if v[i,j].x == 1:
            sale_j.append(i+1)
    res.append(sale_j)
res2 = []
for i in range(nb_bricks):
    if center_bricks[i].x == 1:
        res2.append(i+1)

print(res)
# print(f"new_center : {res2} // old : [3,13,15,21]")

#print workload matrix
workload_res = []
for i in range(nb_bricks):
    row = []
    for j in range(nb_SR):
        row.append(resulting_workload[i,j].x)
    workload_res.append(row)

workload_res = np.array(workload_res)
print(workload_res)

sums = []
for i in range(nb_SR):
    sums.append(weighted_sums[i].x)

print(sums)


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


