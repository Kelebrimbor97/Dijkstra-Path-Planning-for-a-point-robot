import numpy as np
import matplotlib.pyplot as plt
import cv2

from math import cos,sin
from queue import PriorityQueue

open_list = PriorityQueue()
closed_list = []
parent_node_dict = {}
node_dict = {}
action_states = np.array([[1,0], [-1,0], [0,1], [0,-1], [1,1], [-1,1], [1,-1], [-1,-1]])

################################################################################

def gen_map(show_map):

    map_area = np.zeros((250,600,3), np.uint8)
    cv2.rectangle(map_area, (0,0), (600,250), (255,0,0), -1)
    cv2.rectangle(map_area, (5,5), (595,245), (0,0,0),-1)

    #Rect 1
    cv2.rectangle(map_area, (100,100),(150,0),(255,0,0),5)
    cv2.rectangle(map_area, (100,100),(150,0),(0,255,0),-1)

    #Rect 
    cv2.rectangle(map_area, (100,250),(150,150),(255,0,0),5)
    cv2.rectangle(map_area, (100,250),(150,150),(0,255,0),-1)

    #Inner Hex
    hex_center = np.array([300,125])
    v_up = np.array([0,75])
    theta = np.deg2rad(60)
    rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    hex_pts = []

    for i in range(6):
        v_up = np.dot(v_up, rot)
        new_pt = hex_center + v_up
        hex_pts.append(new_pt)

    hex_pts = np.array(hex_pts, np.int32)

    #Outer Hex 
    hex_bdr = []
    for i in hex_pts:

        curr_vect = i - hex_center
        curr_vect_norm = np.linalg.norm(curr_vect)
        curr_vect = curr_vect + (curr_vect/curr_vect_norm)*5
        curr_vect = curr_vect + hex_center
        hex_bdr.append(curr_vect)

    hex_bdr = np.array(hex_bdr, np.int32)

    cv2.fillPoly(map_area, [hex_bdr], (255,0,0))
    cv2.fillPoly(map_area, [hex_pts], (0,255,0))

    #Triangle
    tri_pts = np.array([[460,25],[560,125],[460,225]])

    tri_center = np.average(tri_pts, axis=0)

    tri_bdr = []

    for i in tri_pts:

        curr_vect = i - tri_center
        curr_vect_norm = np.linalg.norm(curr_vect)
        curr_vect = curr_vect + (curr_vect/curr_vect_norm)*5
        curr_vect = curr_vect + tri_center
        tri_bdr.append(curr_vect)

    tri_bdr = np.array(tri_bdr, np.int32)
    cv2.fillPoly(map_area, [tri_bdr], (255,0,0))
    cv2.fillPoly(map_area, [tri_pts], (0,255,0))

    map_area = cv2.cvtColor(map_area,cv2.COLOR_BGR2RGB)
    # cv2.circle(map_area, (50,150), 20, (0,255,255), 5)

    

    if show_map:
        plt.figure('Generated Map')
        plt.imshow(map_area, cmap='gray')
        plt.show()

    return map_area

################################################################################

def backtrack(map_area):

    parent_id = int(parent_node_dict['goal_state'])
    curr_node_loc = node_dict[parent_id]
    accumulator = [curr_node_loc]

    while(parent_id!=None):

        curr_node_loc = node_dict[parent_id]
        accumulator.append(curr_node_loc)
        parent_id = parent_node_dict[parent_id][0]
        map_area[curr_node_loc[0], curr_node_loc[1]] = [255,0,0]

    accumulator = np.array(accumulator, np.int32)
    # print(accumulator)
    return map_area

################################################################################
def main():


    #Get initial state coordinates
    initial_state_x = int(input('Please enter x coordinate of the inital state: '))
    initial_state_y = int(input('Please enter y coordinate of the inital state: '))

    #Get goal state coordinates
    goal_state_x = int(input('Please enter x coordinate of the goal state: '))
    goal_state_y = int(input('Please enter y coordinate of the goal state: '))

    initial_state = [initial_state_y, initial_state_x]
    goal_state = [goal_state_y, goal_state_x]

    map_area = gen_map(False)        #Creaste map area and display it

    if np.array_equal(map_area[initial_state],[0,0,0]) or np.array_equal(map_area[goal_state],[0,0,0]):
        print('Incorrect coordinates, please try again')
    
    root_node = (0, initial_state, None)    #c2c, own location, parent_loc
    open_list.put(root_node)
    curr_node = root_node
    node_id = 0


    while not(open_list.empty()) and not np.array_equal(curr_node[1],goal_state):

        node_id += 1
        curr_node = open_list.get()
        curr_node_loc = curr_node[1]
        prev_node_id = curr_node[2]
        cv2.imshow('Dijkstra_Visualiser', map_area)
        if cv2.waitKey(1) & 0XFF == ord('q'):cv2.destroyAllWindows()
        cv2.imwrite('img_accumulator/'+str(node_id)+'.jpg',map_area)

        # print('Current Parent node:', curr_node)

        closed_list.append(curr_node_loc)
        parent_node_dict[node_id] = [prev_node_id]
        node_dict[node_id] = curr_node_loc

        if np.array_equal(curr_node[1],goal_state):
            parent_node_dict['goal_state'] = prev_node_id
            node_dict['goal_state'] = curr_node_loc
            print('Reached')
            map_area = backtrack(map_area)
            node_id += 1
            cv2.imshow('Dijkstra_Visualiser', map_area)
            if cv2.waitKey(0) & 0XFF == ord('q'):cv2.destroyAllWindows()
            cv2.imwrite('img_accumulator/'+str(node_id)+'.jpg',map_area)

        for i in action_states:

            child_loc = np.add(curr_node_loc, i)
            child_cost = curr_node[0] + np.linalg.norm(i)
            child_node = (child_cost, child_loc.tolist(), node_id)
            is_obst = map_area[child_loc[0], child_loc[1]] 

            if not child_loc.tolist() in closed_list and np.array_equal(is_obst,[0,0,0]):

                # print('\nChild node:', child_node)
                open_list.put(child_node)
                map_area[child_loc[0], child_loc[1]] = [150,200,255]
                # print('Reached here!!! location:', child_loc, ' value:', is_obst,'\n')
                # cv2.imshow('map_update',map_area)


################################################################################

if __name__=="__main__":
    main()