import robotic as ry
import numpy as np


from utils_two_agent import (sample_uniform_points, compute_heuristic, find_path_between_configurations, 
                   move_on_path, ik_for_agent_to_object, 
                   solve_ik_for_all_points, filter_solutions_for_agent_to_object, reachable, move_agent_away_from_object, distance_constraint)
C = ry.Config()
task = "p6-wall-tool.g"
C.addFile(task)
ry.params_add({
    "rrt/stepsize": 0.05,
    "rrt/verbose": 0
})
EGO_NAME = "ego"
EGO_NAME1 = "ego1"
EGO_NAME2 = "ego2"

OBJ_NAME = "obj"
GOAL_NAME = "goal_visible"
CAMERA_NAME = "camera_gl"
C.view(True)

class Node:
    def __init__(self, C, level, id, type, agent_turn, parentId = None):
        self.config = ry.Config()
        self.config.addConfigurationCopy(C) 
        self.parentId = parentId
        self.id = id
        self.agent_turn = agent_turn
        self.level = level
        self.type = type # type = "pick" or "place"
        if self.type == "place" and self.agent_turn  == 1:
            self.config.attach(EGO_NAME1, OBJ_NAME)
        if self.type == "place" and self.agent_turn  == 2:
            self.config.attach(EGO_NAME2, OBJ_NAME)

id = 0
node = Node(C, 0, id, "pick", 1, parentId=-1) # parentId = -1 ==> root node

id +=1
path_nodes = []
L = [node]
config_temp = ry.Config()
config_temp.addConfigurationCopy(node.config)

jointState = move_agent_away_from_object(config_temp, EGO_NAME + str(node.agent_turn), OBJ_NAME, 0.08)
print(jointState)
print("######################## PRINTING AFTER MOVING AWAY   THE OBJECT #############################")
config_temp.setJointState(jointState)
qSol, feasible = ik_for_agent_to_object(config_temp, EGO_NAME + str(node.agent_turn), OBJ_NAME)
solutions = []
notFeasible = False
if feasible == 1:
    config_temp.setJointState(qSol)
    config_temp.view(True)
    solutions = solve_ik_for_all_points(config_temp, EGO_NAME + str(node.agent_turn), OBJ_NAME)
else:
    print("IK problem not feasible.")
    notFeasible = True

filtered_solutions = filter_solutions_for_agent_to_object(config_temp, solutions, node.agent_turn)
config_temp.clear()
del config_temp
print(filtered_solutions)
print(len(filtered_solutions))

paths = []

for i in range(len(filtered_solutions)):
    #print(i)
    #print(filtered_solutions[i])
    #print(f"jointState = { node.config.getJointState()}, sol = {filtered_solutions[i]}")
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
    converged = move_on_path(node.config, path, node.agent_turn)
    newNode = Node(node.config, node.level + 1, id, type="place", agent_turn=node.agent_turn, parentId= node.id)
    print(f"added place node with id: {newNode.id}")
    #newNode.config.view(True)
    id += 1
    node.config.setJointState(initialJointState)
    L.append(newNode)

for node in L:
    node.config.view(True)
