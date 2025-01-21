import pandas as pd
import numpy as np

pfitzer_100 = pd.read_csv("Pfitzer10-100.csv",delimiter=";", index_col=0)
pfitzer_100 = pfitzer_100.iloc[1:]
pfitzer_100.index = pfitzer_100.index.astype(int)


workload = pd.DataFrame(columns=["index_value"],index=pfitzer_100.index)
#rename zone column index to brick
workload.index.name = "brick"
workload["index_value"] = pfitzer_100["workload index"].astype(float)
workload.to_csv('bricks_index_values_100.csv')

distances_df = pd.DataFrame(index=pfitzer_100.index)
distances_df.index.name = "brick"

centers_coords = {}
is_for_zone_cols = [
    "current zones",
    "Unnamed: 6",
    "Unnamed: 7",
    "Unnamed: 8",
    "Unnamed: 9",
    "Unnamed: 10",
    "Unnamed: 11",
    "Unnamed: 12",
    "Unnamed: 13",
    "Unnamed: 14"]
for i in pfitzer_100.index:
    if int(pfitzer_100.loc[i,"current office"]) == 1:
        id = np.where(pfitzer_100.loc[i,is_for_zone_cols])[0][0]
        centers_coords[id] = (pfitzer_100.loc[i,"x"],pfitzer_100.loc[i,"y"])

distances = []
for i in pfitzer_100.index:
    x,y = pfitzer_100.loc[i,"x"],pfitzer_100.loc[i,"y"]
    distance_i = []
    for id in range(len(is_for_zone_cols)):
        xc,yc = centers_coords[id]
        dist = np.sqrt((x-xc)**2+(y-yc)**2)
        distance_i.append(dist)
    distances.append(distance_i)
    
distances = np.asarray(distances)
distances = distances.T

for id in range(len(is_for_zone_cols)):
    distances_df["rp"+str(id+1)] = distances[id]
distances_df.to_csv('brick_rp_distances_100.csv')

init_affect = pd.DataFrame(index=pfitzer_100.index)
init_affect.index.name = "brick"
rps = []
for i in pfitzer_100.index:
    id = np.where(pfitzer_100.loc[i,is_for_zone_cols])[0][0]
    rps.append(id)

init_affect["rp"] = rps
init_affect.to_csv('init_affect_100.csv')

centers = np.where(pfitzer_100["current office"] == 1)[0]
with open("center_ids.txt","w") as f:
    for c in centers:
        f.write(str(c)+"\n")
        
#Make matrix 100*100 of all distances

all_distances = pd.DataFrame()

for i in pfitzer_100.index:
    for j in pfitzer_100.index:
        x1,y1 = pfitzer_100.loc[i,"x"],pfitzer_100.loc[i,"y"]
        x2,y2 = pfitzer_100.loc[j,"x"],pfitzer_100.loc[j,"y"]
        dist = np.sqrt((x1-x2)**2+(y1-y2)**2)
        all_distances.loc[i,j] = dist

all_distances.to_csv("dis_100.csv",header=False,index=False)
