import robotic as ry
import numpy as np
import time
from bottleneck import bottleneck_via_betweenness_approx, grid_vertex_to_env

from utils import (sample_uniform_points, compute_heuristic, polar_to_cartesian, find_path_between_configurations, 
                   move_on_path, get_grasp_positions, filter_solutions_for_agent_to_object, reachable, object_faces_goal)


C = ry.Config()
task = "p6-wall.g"
C.addFile(task)
EGO_NAME = "ego"
OBJ_NAME = "obj"
SUB_GOAL_NAME = "sub-goal1"
GOAL_NAME = "goal_visible"
CAMERA_NAME = "camera_above"
TOOL_NAME = "tool"

ry.params_add({
    "rrt/stepsize": 0.01,
    "rrt/verbose": 0
})


print(C.getFrameNames())



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



def branch_from_place_node_filter(node: Node, id):

    sampled_points = sample_uniform_points(node.config, num_samples= 500)

    level = node.level
    
    scores = []
    for point in sampled_points:
        score = compute_heuristic(node.config, point, EGO_NAME, GOAL_NAME)
        scores.append((point, score))
    scores.sort(key=lambda x: x[1], reverse=True)

    path = []
    for idx in range(len(scores)):

        delta = np.array(node.config.frame(OBJ_NAME).getPosition()[:2]) - np.array(node.config.getJointState())
        q_newgoal = np.array(scores[idx][0]) - delta
        path = find_path_between_configurations(node.config, node.config.getJointState(), q_newgoal)
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

def branch_from_place_node_btl(node: Node, id):
    
    inner_radius = 1.0
    outer_radius = 3.0
    step_size = 0.05
    threshold = 0.05  

    bottleneck_nodes, bc_values, obj_vertex = bottleneck_via_betweenness_approx(
        node.config, inner_radius, outer_radius, step_size, threshold
    )
    found_positions = []
    for b_node in bottleneck_nodes:
        pos_world = grid_vertex_to_env(b_node, step_size)  # as defined above
        found_positions.append(pos_world)

    level = node.level

    path = []
    found = False
    np.random.shuffle(found_positions)
    for pos in found_positions:
        print(f"pos: {pos}")
        delta = np.array(node.config.frame(OBJ_NAME).getPosition()[:2]) - np.array(node.config.getJointState())
        q_newgoal = pos - delta


        path = find_path_between_configurations(node.config, node.config.getJointState(), q_newgoal)
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

C.view(True)
C.view_close()

agent_gap = C.getFrame(EGO_NAME).getSize()[1] * 2
print(agent_gap)
allNodes = {}
allConfigs = {}
id = 0
node = Node(C, 0, id, "pick", parentId=-1) # parentId = -1 ==> root node
allNodes[id] = node
allConfigs[id] = node.config
id +=1
path_nodes = []


Q = [node]
node = None


start_time = time.perf_counter()

while len(Q) > 0:    
    node = Q.pop(0)

    print(f"################################    PROCESSING NODE {node.id}, level: {node.level}   ###############################################")
    print(f"Node type: {node.type}")
    print(f"Parent of current node: {node.parentId}")
    print(f"Level of current node: {node.level}")
    #node.config.view(True)
    #node.config.view_close()
    within_reach, trajectory = reachable(node.config, node.config.frame(GOAL_NAME))
    if node.type == "place" and within_reach:
        node.config.view(True)
        node.config.view_close()
        move_on_path(node.config, trajectory, found=True)
        break
    else:
        if node.type == "pick":

            config_temp = ry.Config()
            config_temp.addConfigurationCopy(node.config)

            init_state = config_temp.getJointState()
            agent_coords = config_temp.getFrame(EGO_NAME).getPosition()[:2]
            obj_coords = config_temp.getFrame(OBJ_NAME).getPosition()[:2]
            goal_coords = config_temp.getFrame(GOAL_NAME).getPosition()[:2]

            tries = 18
            candidate_positions = []
            angle_step = 2 * np.pi / tries
            random_shift = np.random.uniform(0, 2 * np.pi)
            for sample_idx in range(tries):
                polar_shift = np.array([agent_gap, sample_idx * angle_step + random_shift])

                ## polar_shift -> cartesian_shift:
                r, theta = polar_shift
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                cartesian_shift = np.array([x, y])

                q_temp = obj_coords + cartesian_shift - agent_coords + init_state
                config_temp.setJointState(q_temp)

                grasp_candidate = get_grasp_positions(config_temp)
                if grasp_candidate is not None:
                    candidate_positions.append(grasp_candidate)

            valid_candidates = filter_solutions_for_agent_to_object(config_temp, candidate_positions)
            print(valid_candidates)

            config_temp.clear()
            del config_temp

            paths = []
            for i in range(len(valid_candidates)):
                path = find_path_between_configurations(node.config, node.config.getJointState(),  valid_candidates[i])
                if isinstance(path, np.ndarray):
                    if path.ndim < 1:  
                        print("empty path")
                        continue
                paths.append(path)
            #print(len(paths))
            #print(paths[0])
            for path in paths:
                initialJointState = node.config.getJointState()
                converged = move_on_path(node.config, path)
                newNode = Node(node.config, node.level + 1, id, type="place", parentId= node.id)
                #print(f"added place node with id: {newNode.id}")
                allNodes[id] = newNode
                #newNode.config.view(True)
                id += 1
                node.config.setJointState(initialJointState)
                Q.append(newNode)
            
            print(len(Q))

        elif  node.type == "place":            
            node_place = branch_from_place_node_btl(node, id)  
            if node_place is not None:
                Q.append(node_place)
                #node_place.config.view(True)
                allNodes[id] = node_place

                id +=1 

print("############################# END OF THE LOOP ################################")

end_time = time.perf_counter()
print(f"Loop execution time: {end_time - start_time:.6f} seconds")

for node in allNodes:
    del node

for node in Q:
    del node





