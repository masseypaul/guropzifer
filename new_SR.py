###############
# Group 5
# When you want to add a new SR
###############



import pandas as pd
import numpy as np
import gurobipy as gp

index_values = pd.read_csv("data/bricks_index_values.csv",index_col=0)
distances = pd.read_csv("data/brick_rp_distances.csv",index_col=0)
all_distances = pd.read_csv("data/distances.csv",delimiter=";")
distance_matrix= np.genfromtxt('data/dis.csv', delimiter=',', skip_header=0)  # skip_header=1 if there is a header
# pfitzer_100 = pd.read_csv("Pfitzer10-100.csv")

#### change here the type of optimization
OBJECTIVE = "disrupt" #or "disrupt" or "dist"

# print(index_values.head())
# print(distances)
# print(all_distances.head())
# print(pfitzer_100.head())

nb_additional_SR = 1

representation = [
    [4,5,6,7,8,15],
    [10,11,12,13,14],
    [9,16,17,18],
    [1,2,3,19,20,21,22]
]

nb_center = len(representation)+nb_additional_SR
nb_bricks = sum(len(repres) for repres in representation)
# print(nb_center,nb_bricks)

representation_minus_one = [[x-1 for x in row] for row in representation]

distances = np.array(distances)
workload = index_values

#Create Model and Variables
m = gp.Model("Step 1")
v = m.addMVar(shape=(nb_bricks,nb_center),vtype=gp.GRB.BINARY,name="V")

center_bricks = m.addVars(nb_bricks,vtype = gp.GRB.BINARY, name="C")

#Create Constraints
# Center brick constraints
center=[3,13,15,21]
m.addConstr(v[3,0] == 1)
m.addConstr(v[13,1] == 1)
m.addConstr(v[15,2] == 1)
m.addConstr(v[21,3] == 1)

m.addConstrs((center_bricks[c] == 1 for c in center), name="Initial_Center")
m.addConstr(gp.quicksum(center_bricks[i] for i in range(len(center_bricks))) == nb_center, name="nb center")

#One SR per area
m.addConstrs(
    (sum(v[i, j] for j in range(nb_center)) == 1 for i in range(nb_bricks)), 
    name="RowSum"
)

# Center can only belong to one SR
m.addConstrs(
    (gp.quicksum(center_bricks[i]*v[i,j] for i in range(nb_bricks)) == 1 for j in range(nb_center)),
    name="OCPSR" # One Center Per SR
)

# Workload constraints
workload_repeated = np.repeat(workload,nb_center,axis=0).reshape(nb_bricks,nb_center)
resulting_workload = m.addVars(nb_bricks,nb_center,vtype=gp.GRB.CONTINUOUS, name="RW")
m.addConstrs(
    (resulting_workload[i,j] == workload_repeated[i,j] * v[i,j] for i in range(nb_bricks) for j in range(nb_center)),
    name="workload"
)

minWL = 0.8*0.8
maxWL = 1.2*0.8
print(minWL,maxWL)
weighted_sums = m.addVars(nb_center, lb=0, vtype=gp.GRB.CONTINUOUS,name="WS")
m.addConstrs(
    (weighted_sums[j] == gp.quicksum(resulting_workload[i,j] for i in range(nb_bricks)) for j in range(nb_center)), 
    name="V=WS"
)

m.addConstrs(
    (weighted_sums[j] >= minWL for j in range(nb_center)), 
    name="MinWeightedSums"
)
m.addConstrs(
    (weighted_sums[j] <= maxWL for j in range(nb_center)), 
    name="MaxWeightedSums"
)

#Create Objective
#Objective minimum disruption
m_orig = np.zeros((nb_bricks,nb_center),dtype=int)
for i in range(len(representation)):
    m_orig[representation_minus_one[i],i] = 1
    
#Objective 1
if OBJECTIVE == "disrupt":
    diff_var = m.addVars(nb_bricks,nb_center,vtype=gp.GRB.INTEGER,name="Diff")
    m.addConstrs((diff_var[i,j] >= v[i,j] - m_orig[i,j] for i in range(nb_bricks) for j in range(nb_center)),name="diff?")
    abs_var = m.addVars(nb_bricks,nb_center,vtype=gp.GRB.INTEGER,name="abs")
    m.addConstrs((abs_var[i,j] == gp.abs_(diff_var[i,j]) for i in range(nb_bricks) for j in range(nb_center)),name="abs?")
    disrupt_sum = gp.quicksum(abs_var[i,j] for i in range(nb_bricks) for j in range(nb_center))
    
    m.setObjective(disrupt_sum, gp.GRB.MINIMIZE)
    #Create Objective
    #Solve
    m.optimize()
    restemp = []
    for i in range(nb_bricks):
        l=[]
        for j in range(nb_center):
            l.append(abs_var[i,j].x)
        restemp.append(l)
    print(restemp)

elif OBJECTIVE == "dist":
    #Objective 2
    mult_mat = m.addVars(nb_bricks,nb_center,vtype=gp.GRB.CONTINUOUS,name="M")
    
    
    dist_temp = m.addMVar(shape=(nb_bricks, nb_additional_SR), vtype=gp.GRB.CONTINUOUS, name="D")
    
    m.addConstrs((dist_temp[i,k] == gp.quicksum(center_bricks[j]*distance_matrix[i,j] 
                                                for j in range(nb_bricks) if j not in center)
                  for i in range(nb_bricks) for k in range(nb_additional_SR)), name="tempdist")
    

    
    m.addConstrs(mult_mat[i,j] == v[i,j]*distances[i,j] for i in range(nb_bricks) for j in range(len(representation)))
    m.addConstrs((mult_mat[i,j] == v[i,j]*dist_temp[i] for i in range(nb_bricks) for j in range(len(representation),nb_center)),
                 name="Part2")
    dist_sum = gp.quicksum(mult_mat[i,j] for i in range(nb_bricks) for j in range(nb_center))
    
    m.setObjective(dist_sum, gp.GRB.MINIMIZE)
    m.write("myModel.lp")
    #Create Objective
    #Solve
    m.optimize() 

    dist_mat = []
    for i in range(nb_bricks):
        row = []
        for j in range(nb_center):
            row.append(mult_mat[i,j].x)
        dist_mat.append(row)
    

    dist_mat = np.array(dist_mat)
    print(dist_mat)
    
new_center = -1
for i in range(nb_bricks):
    if center_bricks[i].x > 0 and i not in center:
        new_center = i
    if center_bricks[i].x > 0:
        print(i+1)
print(f"nouveau SR en {i}")
print("La solution est :")
res = []
for j in range(nb_center):
    sale_j = []
    for i in range(nb_bricks):
        if v[i,j].x == 1:
            sale_j.append(i+1)
    res.append(sale_j)

print(res)

#print workload matrix
workload_res = []
for i in range(nb_bricks):
    row = []
    for j in range(nb_center):
        row.append(resulting_workload[i,j].x)
    workload_res.append(row)

workload_res = np.array(workload_res)
print(workload_res)

sums = []
for i in range(nb_center):
    sums.append(weighted_sums[i].x)

print(sums)


################################
# Display part
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS


# automatic definition to do

keys = [4, 14, 16, 22]
keys.append(new_center+1)

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


