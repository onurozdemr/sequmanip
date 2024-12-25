import robotic as ry
import numpy as np

from utils import (sample_uniform_points, compute_heuristic, find_path_between_configurations, 
                   move_on_path, ik_for_agent_to_object, 
                   solve_ik_for_all_points, filter_solutions_for_agent_to_object, reachable, move_agent_away_from_object, object_faces_goal)


C = ry.Config()
task = "p8-corner.g"
C.addFile(task)
EGO_NAME = "ego"
OBJ_NAME = "obj"
SUB_GOAL_NAME = "sub-goal1"
GOAL_NAME = "goal_visible"
CAMERA_NAME = "camera_gl"
TOOL_NAME = "tool"

ry.params_add({
    "rrt/stepsize": 0.01,
    "rrt/verbose": 0
})

print(C.getFrameNames())
C.view(True)
C.view_close()

class Node:
    def __init__(self, C, level, id, type, parentId = None):
        self.config = ry.Config()
        self.config.addConfigurationCopy(C) 
        self.parentId = parentId
        self.id = id
        self.level = level
        self.type = type # type = "pick" or "place"
        if self.type == "place":
            self.config.attach(EGO_NAME, OBJ_NAME)

def branch_from_place_node(node: Node, id):
    sampled_points = sample_uniform_points(node.config, num_samples= 300)
    level = node.level
    scores = []
    for point in sampled_points:
        score = compute_heuristic(node.config, point, EGO_NAME, GOAL_NAME)
        scores.append((point, score))
    scores.sort(key=lambda x: x[1], reverse=True)

    path = []
    for idx in range(len(scores)):
        path = find_path_between_configurations(node.config, node.config.getJointState(), scores[idx][0])
        if isinstance(path, np.ndarray):
            if path.ndim < 1:  
                print("empty path")
                continue
            else:
                found = True
                break
    if found:
        converged = move_on_path(node.config, path)
        node.config.frame(OBJ_NAME).unLink()
        newNode = Node(node.config, node.level + 1, id, "pick", node.id)
        return newNode
    return None

allNodes = {}
allConfigs = {}
id = 0
node = Node(C, 0, id, "pick", parentId=-1) # parentId = -1 ==> root node
allNodes[id] = node
allConfigs[id] = node.config
id +=1
path_nodes = []
L = [node]
node = None

while len(L) > 0:    
    node = L.pop(0)

    print(f"################################    PROCESSING NODE {node.id}, level: {node.level}   ###############################################")
    print(f"Node type: {node.type}")
    print(f"Parent of current node: {node.parentId}")
    print(f"Level of current node: {node.level}")
    #node.config.view(True)
    #node.config.view_close()
    if node.type == "place" and reachable(node.config, node.config.frame(GOAL_NAME)) and object_faces_goal(node.config, node.config.frame(GOAL_NAME)):

        found = False
        while not found:
            path = find_path_between_configurations(node.config, node.config.getJointState(), node.config.frame(GOAL_NAME).getPosition()[0:2])
            if isinstance(path, np.ndarray):
                if path.ndim >= 1:  
                    found = True             
        print("******************************* GOAL REACHABLE ***************************************")

        node.config.view(True)
        node.config.view_close()
        move_on_path(node.config, path, found=True)
        path_nodes = []
        current_id = node.id
        break
    else:
        if node.type == "pick":

            config_temp = ry.Config()
            config_temp.addConfigurationCopy(node.config)

            newPos = move_agent_away_from_object(config_temp, EGO_NAME, OBJ_NAME, 0.08) # for rrt to find path better, otherwise rrt struggles
            config_temp.frame(EGO_NAME).setJointState(newPos[0:2])
            qSol, feasible = ik_for_agent_to_object(config_temp, EGO_NAME, OBJ_NAME)
            solutions = []
            notFeasible = False
            if feasible == 1:
                config_temp.setJointState(qSol)
                solutions = solve_ik_for_all_points(config_temp, EGO_NAME, OBJ_NAME)
            else:
                print("IK problem not feasible.")
                notFeasible = True

            if notFeasible:
                continue

            filtered_solutions = filter_solutions_for_agent_to_object(config_temp, solutions)
            config_temp.clear()
            del config_temp
            print(f"filtered: {filtered_solutions}")
            paths = []
            for i in range(len(filtered_solutions)):
                path = find_path_between_configurations(node.config, node.config.getJointState(),  filtered_solutions[i])
                if isinstance(path, np.ndarray):
                    if path.ndim < 1:  
                        print("empty path")
                        continue
                paths.append(path)
            print(len(paths))
            print(paths[0])
            for path in paths:
                initialJointState = node.config.getJointState()
                converged = move_on_path(node.config, path)
                newNode = Node(node.config, node.level + 1, id, type="place", parentId= node.id)
                print(f"added place node with id: {newNode.id}")
                allNodes[id] = newNode
                #newNode.config.view(True)
                id += 1
                node.config.setJointState(initialJointState)
                L.append(newNode)
            
            print(L)

        elif  node.type == "place":            
            node_place = branch_from_place_node(node, id)  
            if node_place is not None:
                L.append(node_place)
                #node_place.config.view(True)
                allNodes[id] = node_place

                id +=1 

print("############################# END OF THE LOOP ################################")


for node in allNodes:
    del node

for node in L:
    del node





