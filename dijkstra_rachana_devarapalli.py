import cv2
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# Create an empty image canvas for visualization
canvas = np.zeros((500, 1200, 3), dtype=np.uint8)

# Define obstacle shapes using polygons
rect1 = np.array([[95, 0], [180, 0], [180, 405], [95, 405]], np.int32)
rect2 = np.array([[270, 1200], [355, 1200], [355, 195], [270, 195]], np.int32)
center = (650, 250)
length = 150 + 2 * 5 * np.arctan(np.radians(30))
vertices = [(int(center[0] + length * np.cos((i + 0.5) * 2 * np.pi / 6)),
             int(center[1] + length * np.sin((i + 0.5) * 2 * np.pi / 6))) for i in range(6)]
hexagon = np.array(vertices)

rect3 = np.array([[895, 45], [1105, 45], [1105, 130], [895, 130]], np.int32)
rect4 = np.array([[895, 455], [1105, 455], [1105, 370], [895, 370]], np.int32)
rect5 = np.array([[1015, 130], [1105, 130], [1105, 370], [1015, 370]], np.int32)

# Fill the canvas with obstacle shapes
cv2.fillPoly(canvas, [rect1, rect2, hexagon, rect3, rect4, rect5], (255, 0, 0))

# Split the canvas into channels (B, G, R) and transpose the blue channel
b, g, r = cv2.split(canvas)
b = b.T

# Function to generate possible actions for children from a given node eight total (-1,1),(1,0 etc)
def actions_for_children(OL_top_node, closed_nodes):
    children = []  # (c2c, parent ID, x, y)

    # Check adjacent pixels for obstacles and add valid actions_for_children to children list
    def check_and_add(x_offset, y_offset, cost):
        nonlocal children
        x, y = int(OL_top_node[3]) + x_offset, int(OL_top_node[4]) + y_offset
        if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0] and b[x][y] != 255:
            val = np.where((closed_nodes[:, 3] == x) & (closed_nodes[:, 4] == y))[0]
            if val.size == 0:
                children.append([OL_top_node[0] + cost, OL_top_node[1], x, y])

    # Check and add actions_for_children for all possible moves
    check_and_add(-1, 1, 1.4)
    check_and_add(0, 1, 1)
    check_and_add(1, 1, 1.4)
    check_and_add(1, 0, 1)
    check_and_add(1, -1, 1.4)
    check_and_add(0, -1, 1)
    check_and_add(-1, -1, 1.4)
    check_and_add(-1, 0, 1)

    return children

# Take user input for starting and ending coordinates with validation
correct_input = False
while not correct_input:
    initial_x_config = int(input('Enter Start x point: '))
    initial_y_config = int(input('Enter Start y point: '))
    final_x_config = int(input('Enter End x point: '))
    final_y_config = int(input('Enter End y point: '))

    # Adjust y-coordinates to match the canvas coordinate system
    initial_y_config = canvas.shape[0] - initial_y_config - 1
    final_y_config = canvas.shape[0] - final_y_config - 1

    if (
        initial_x_config > canvas.shape[1]
        or initial_y_config > canvas.shape[0]
        or final_x_config > canvas.shape[1]
        or final_y_config > canvas.shape[0]
    ):
        print('Given input out of bounds')
    elif b[initial_x_config][initial_y_config] == 255 or b[final_x_config][final_y_config] == 255:
        print('Given coordinates located on obstacle')
    else:
        correct_input = True


# Initialize open list, closed list, and NodeID
open_nodes = np.array([[0, 1, 0, initial_x_config, initial_y_config]])
closed_nodes = np.array([[-1, -1, -1, -1, -1]])
NodeID = 1

# Start the loop to find the shortest path
while (
    not (closed_nodes[-1][3] == final_x_config and closed_nodes[-1][4] == final_y_config)
    and not open_nodes.shape[0] == 0
):
    open_nodes = open_nodes[open_nodes[:, 0].argsort()]  # Sort open list based on cost-to-come
    children = actions_for_children(open_nodes[0], closed_nodes)  # Generate children nodes

    for i in range(len(children)):
        val = np.where(
            (open_nodes[:, 3] == children[i][2]) & (open_nodes[:, 4] == children[i][3])
        )[0]  # Check if the child is present in the open list
        if val.size > 0:
            if children[i][0] < open_nodes[int(val.item())][0]:
                open_nodes[int(val.item())][0] = children[i][0]
                open_nodes[int(val.item())][2] = children[i][1]
        else:
            open_nodes = np.vstack(
                [
                    open_nodes,
                    [children[i][0], NodeID + 1, children[i][1], children[i][2], children[i][3]],
                ]
            )
            NodeID += 1

    closed_nodes = np.vstack([closed_nodes, open_nodes[0]])
    open_nodes = np.delete(open_nodes, 0, axis=0)


# Backtrack to find the path
backtrack = np.array([[final_x_config, final_y_config]])
val = np.where(
    (closed_nodes[:, 3] == final_x_config) & (closed_nodes[:, 4] == final_y_config)
)[0]
parent = closed_nodes[int(val)][2]

while parent:
    val = np.where(closed_nodes[:, 1] == parent)[0]
    backtrack = np.vstack([backtrack, [closed_nodes[int(val)][3], closed_nodes[int(val)][4]]])
    parent = closed_nodes[int(val)][2]

backtrack = np.flip(backtrack, axis=0)
backtrack = backtrack.astype(int)

# Create a video of node exploration and path
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('node_exploration1.mp4', fourcc, 20.0, (canvas.shape[1], canvas.shape[0]))
cv2.circle(
    canvas, (final_x_config,  final_y_config), 1, (255, 255, 255), 2
)
cv2.circle(
    canvas, (initial_x_config, initial_y_config), 1, (255, 255, 255), 2
)

# Visualize the exploration of nodes
for i in range(1, closed_nodes.shape[0]):
    canvas[int(closed_nodes[i][4])][int(closed_nodes[i][3])][0] = 255
    canvas[int(closed_nodes[i][4])][int(closed_nodes[i][3])][1] = 255
    out.write(canvas)

# Visualize the final path
for i in range(backtrack.shape[0]):
    cv2.circle(
        canvas,
        (int(backtrack[i][0]),  int(backtrack[i][1])),
        1,
        (0, 200, 0),
        1,
    )
    out.write(canvas)

out.release()
print("Goal reached and vizualization complete!")
