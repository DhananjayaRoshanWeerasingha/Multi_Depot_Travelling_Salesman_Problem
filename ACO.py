import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the data from the Excel sheet
data = pd.read_excel('Book1.xlsx', sheet_name='Sheet1', names=['id', 'x', 'y'])
depot = pd.read_excel('Book1.xlsx', sheet_name='Sheet2', names=['id', 'x', 'y'])

# Assign names to the matrices
matrix_1_name = 'A'
matrix_2_name = 'B'

# Print the data and depot dataframes
print(matrix_1_name)
print(data)

print(matrix_2_name)
print(depot)

# Calculate the square root of the Euclidean distance between each depot and each data point
distances = np.zeros((len(depot), len(data)))
for i in range(len(depot)):
    for j in range(len(data)):
        distances[i][j] = np.sqrt((depot.iloc[i]['x'] - data.iloc[j]['x'])**2 + (depot.iloc[i]['y'] - data.iloc[j]['y'])**2)

print(distances)

# Assign data points to the nearest depot
assignments = {}
for j in range(len(data)):
    nearest_depot = np.argmin(distances[:, j])
    if nearest_depot in assignments:
        assignments[nearest_depot].append(data.iloc[j]['id'])
    else:
        assignments[nearest_depot] = [data.iloc[j]['id']]

print(assignments)

# Calculate the new depot locations
new_depots = []
for depot_id, assigned_data_points in assignments.items():
    assigned_data = data[data['id'].isin(assigned_data_points)]
    new_depot_x = assigned_data['x'].mean()
    new_depot_y = assigned_data['y'].mean()
    new_depots.append({'id': depot_id, 'x': new_depot_x, 'y': new_depot_y})

# Convert the new depots to a pandas dataframe
new_depot_df = pd.DataFrame(new_depots, columns=['id', 'x', 'y'])

# Print the new depot locations
print(new_depot_df)

# Recalculate the distance between each data point and the new obtained depot points
for i in range(len(new_depots)):
    for j in range(len(data)):
        distances[i][j] = np.sqrt((new_depots[i]['x'] - data.iloc[j]['x'])**2 + (new_depots[i]['y'] - data.iloc[j]['y'])**2)

print(distances)

# Recalculate the distance between each data point and the new obtained depot points
for i in range(len(new_depots)):
    for j in range(len(data)):
        distances[i][j] = np.sqrt((new_depots[i]['x'] - data.iloc[j]['x'])**2 + (new_depots[i]['y'] - data.iloc[j]['y'])**2)

# Assign data points to the nearest depot
assignments2 = {}
for j in range(len(data)):
    nearest_depot = np.argmin(distances[:, j])
    if nearest_depot in assignments2:
        assignments2[nearest_depot].append(data.iloc[j]['id'])
    else:
        assignments2[nearest_depot] = [data.iloc[j]['id']]

print(assignments2)

# Initialize the ACO algorithm
num_ants = 50
num_iterations = 100
alpha = 1.0
beta = 2.0
pheromone_init = 0.01

# Create scatter plot
plt.scatter(new_depots['x'], new_depots['y'], c='b')
# Initialize the graph matrix
graph = np.zeros((len(new_depots), len(new_depots))) 
# Add edges to scatter plot based on graph
for i in range(len(new_depots)):
    for j in range(i+1, len(new_depots)):
        if graph[i][j] > 0:
            plt.plot([new_depots.iloc[i]['x'], new_depots.iloc[j]['x']],
                     [new_depots.iloc[i]['y'], new_depots.iloc[j]['y']],
                     c='r', alpha=0.3)

plt.show()

