from cc3d.core.PySteppables import *
import numpy as np
from pathlib import Path
import secrets
np.random.seed(secrets.randbelow(10000))

phi0 = {{phi0}}
m = {{m}}
half_cone= {{half_cone}}
max_alpha_thresh= {{max_alpha_thresh}}
CE= {{CE}}
LAS_ablation= {{LAS_ablation}} 
ARP_depletion= {{ARP_depletion}}

if CE != 15:
   phi0= 0.8


def FindAngle(pix, pop_com): #BOTH NP ARRAYS
    # Get angle (greater than 0) in radians between pix and pop_com
    vec_to_pix = pix - pop_com
    angle_raw = np.arctan2(vec_to_pix[1], vec_to_pix[0])
    angle_raw = angle_raw + (angle_raw < 0)*(2*3.14) # CONVERT TO 0-2PI
    return angle_raw

def FindAlpha(angle360, cell_angle_list, alpha_list):
    # Get alpha from minimum absolute difference
    temp = np.absolute(cell_angle_list-angle360) # Get list of absolute differences
    cell_location = np.argmin(temp) # Get minimum of absolute differences
    alpha = alpha_list[cell_location]  # Get alpha at minimum absolute difference
    return alpha

def GetProbOfUsingProb(max_alpha):
    # Get minimum between 1 and f(alpha)
    return min(1, (max_alpha**2)/2)

def CalculateFeedback(max_alpha):      
    # Get feedback based on maximum alpha and and alpha threshold
    # Dependent on whether m is equal to 110 or not
    # Also dependent on whether laser ablation variable is 1 or not      
    feedback= 0.0 #soft but thresh crossed
    if max_alpha > max_alpha_thresh and m== 110:
        feedback= 1
    elif max_alpha > max_alpha_thresh and m!= 110:
        feedback= 0.1
    else:
        feedback= 0
    
    if LAS_ablation ==1:
        feedback= 0.01#OR THERE COULD BE A MULTIPLIER TO REDUCE THE FEEDBACK: F=F*FACTOR

    return feedback

class UpdatePhi(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self,frequency)

    def start(self):
        self.shared_steppable_vars['chosen_angle']= 0.0
        self.shared_steppable_vars['max_alpha']= 0.0
        

    def step(self, mcs):

        # 
        count_boundary= 0
        pop_com = np.array([0.0,0.0]) # COM of all cells
        count = 0
        for cell in self.cell_list_by_type(self.CELL):
            # ['IsBoundary'] is to determine if cell is at periphery of collective
            cell.dict['IsBoundary'] = 0 
            pop_com += np.array([cell.xCOM,cell.yCOM])
            for neighbor, common_surface_area in self.get_cell_neighbor_data_list(cell):
                if not neighbor:
                    cell.dict['IsBoundary'] = 1
                    count_boundary+= 1
                    break
            count+= 1

        pop_com /= count # COM of all cells
        self.shared_steppable_vars['pop_com'] = pop_com

        # Initialize fields for phi, rm_abi, and alpha
        field_phi = self.field.phi_field
        field_rm_abi = self.field.rm_abi
        field_alpha = self.field.alpha
        field_IsBoundary = self.field.IsBoundary

        field_phi[:,:,:] = 0.0
        field_rm_abi[:,:,:] = 0.0
        field_IsBoundary[:,:,:] = 0.0

        # Get chosen_angle and make it between 0 and 2*pi radians
        chosen_angle= self.shared_steppable_vars['chosen_angle']
        if chosen_angle >= 2*3.14:
            chosen_angle= chosen_angle - 2*3.14
        

        # Get lists of cell angles and alphas
        cell_angle_list = []
        alpha_list= []
        angle_cell_dict = {}
        count_active= 0
        max_alpha= self.shared_steppable_vars['max_alpha']
        # print("max_alpha",max_alpha)
        feedback= CalculateFeedback(max_alpha)

        for cell in self.cell_list_by_type(self.CELL):
            phi = phi0
            rm_abi = 0
            # cell.dict['phi'] = phi
            cell.dict['rm_abi'] = rm_abi
            
            # Check cell only if on periphery of the collective
            IsBoundary = cell.dict['IsBoundary']
            if IsBoundary == 1:
                cell_com = np.array([cell.xCOM, cell.yCOM]) # Get cell COM
                cell_angle = FindAngle(cell_com, pop_com) # Get angle between COM_cell and COM_collective
                alpha_cell= field_alpha[cell.xCOM, cell.yCOM, cell.zCOM] # Get alpha only at the cell's COM
                angle_cell_dict[cell_angle]= alpha_cell # Make dictionary - {angle : alpha}
                
                cell_angle_list.append(cell_angle) # Append angle to list of angles
                alpha_list.append(alpha_cell) # Append cell's alpha to list of alphas
                
                # Get unit vector from boundary cell's angle
                cell_unit_vec= np.array([np.cos(cell_angle),np.sin(cell_angle)])

                # Get vector for whole cell's migration direction
                chosen_angle_unit_vec= np.array([np.cos(chosen_angle), np.sin(chosen_angle)])

                # if (np.absolute(cell_angle-chosen_angle) < half_cone*3.14/180 ) or (np.absolute(2*3.14-(cell_angle-chosen_angle)) < half_cone*3.14/180 ):

                # Dot product of unit vectors give cos(theta) in radians
                # Compare result to cos(half_cone) in radians
                if np.dot(cell_unit_vec, chosen_angle_unit_vec) >= np.cos(half_cone*(3.14/180) ):
                    rm_abi= 1
                    # phi= (mcs >= 50)*(m_current == 110)*(depth_sensed) + phi0
                    phi= feedback + phi0
                    # print(phi)
                    count_active+= 1
                else:
                    rm_abi=0.1
                cell.dict['rm_abi'] = rm_abi
                cell.dict['phi'] = phi

            # For each pixel of cell, assign phi, rm_abi and IsBoundary
            pixel_list = self.get_cell_pixel_list(cell)
            for pixel_tracker_data in pixel_list:
                px = pixel_tracker_data.pixel
                field_phi[px.x,px.y,px.z] = phi
                field_rm_abi[px.x,px.y,px.z] = rm_abi
                field_IsBoundary[px.x,px.y,px.z] = IsBoundary
        
        # f_active= (1 - 2.71**(count_active/count_boundary))/(1 + 2.71**(count_active/count_boundary))
        # print(f_active)
        for cell in self.cell_list_by_type(self.CELL):# SET PHI FOR NON ACTIVELY REMODELING CELLS
            if cell.dict['rm_abi'] != 1:
                phi= 0.8*(feedback + phi0)# 
                cell.dict['phi']= phi
                pixel_list = self.get_cell_pixel_list(cell)
                for pixel_tracker_data in pixel_list:
                    px = pixel_tracker_data.pixel
                    field_phi[px.x,px.y,px.z] = phi

        self.shared_steppable_vars['cell_angle_list'] = cell_angle_list
        self.shared_steppable_vars['alpha_list'] = alpha_list
        self.shared_steppable_vars['max_alpha']= float(max(alpha_list))
        self.shared_steppable_vars['cell_angle_sort'] = sorted(angle_cell_dict.items())
        
class DoRemodeling(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self,frequency)
    def start(self):
        # Determine the COM of all cells at start of simulation
        pop_com = np.array([0.0,0.0])
        count= 0
        for cell in self.cell_list_by_type(self.CELL):
            pop_com += np.array([cell.xCOM,cell.yCOM])
            count+= 1
        pop_com /= count
        
        self.shared_steppable_vars['old_pop_com'] = pop_com

    def step(self, mcs):

        field_alpha = self.field.alpha
        field_cell = self.cell_field
        pop_com = self.shared_steppable_vars['pop_com']
        cell_angle_list = self.shared_steppable_vars['cell_angle_list']
        alpha_list = self.shared_steppable_vars['alpha_list']

        # Get pixels for medium/ECM
        for x, y, z in self.every_pixel():
            if not field_cell[x,y,z]: # ONLY ECM PIXELS
                pix = np.array([x,y])
                angle = FindAngle(pix,pop_com) # Find angle between ECM pixel and COM of cell collective
                alpha_propagate = FindAlpha(angle, cell_angle_list, alpha_list)
                # cluster_num = FindCluster(pix, pop_com)
                # alpha_LE = cluster_alpha[cluster_num]
                dist = np.linalg.norm(pix-pop_com)#POP_COM OR COM OF THE CELLS IN THAT GROUP
                if alpha_propagate/(dist**0.5) > field_alpha[x,y,z]:
                    field_alpha[x,y,z] = alpha_propagate/(dist**0.5) # THIS CANT BE CLUSTER PHI, IT HAS TO BE ALPHA_AVG/R

        output_dir= self.output_dir
        if output_dir is not None:
            output_path= Path(output_dir).joinpath('tracks' + '.dat')
        
        if mcs%100 == 0 and mcs !=0 :#YOU DONT WANT ANYTHING AT ZERO AS ITS TRIVIAL
            delta_d= pop_com - self.shared_steppable_vars['old_pop_com']
            self.shared_steppable_vars['old_pop_com'] = pop_com

            with open(output_path, 'a') as fout:
                fout.write('{} {} \n'.format(pop_com[0],pop_com[1]))


    def finish(self):
        pass

class FindECMpolarity(SteppableBasePy): 
#FIND THE NET DIRECTION OF ECM POLARITY, 
# DO THE RANDOM DISTRIBUTIONS AND PICK A NEW FRONT/BACK THETA, TO BE FED INTO UPDATE_RM
    def __init__(self, frequency=200):
        SteppableBasePy.__init__(self,frequency)

    def step(self, mcs):     

        cell_angle_sort= np.array(self.shared_steppable_vars['cell_angle_sort'])
        num_boundary= len(cell_angle_sort) # get length of list of cell_angle_sort

        angle_avg_alpha_avg= []
        # For each cluster in number of boundaries (num_boundary)
        for cluster_num in range(num_boundary):
            # cell_angle_sort has cluster_num in rows of array
            # cell_angle_sort has angle as first column of array
            start_angle= cell_angle_sort[cluster_num,0]
            final_angle= start_angle
            sum_angles= 0
            sum_alphas= 0
            next_cluster= cluster_num
            flag_add_pi= 0
            counter=0

            # Check if difference between angles is less than twice half_cone (in radians)
            while final_angle-start_angle < half_cone*2*(3.14/180): 
                sum_angles+= final_angle
                #make sure its not part of the wrap-around
                sum_alphas+= (cell_angle_sort[next_cluster - flag_add_pi*(num_boundary),1])**3 
                #Making probability proportional to sum of cube of alphas, rather than cube of sum of alphas
                if next_cluster== (num_boundary - 1): # i.e. IT IS AT THE LAST POINT
                    flag_add_pi= 1 # Add 2pi to the initial angles when wrapping around
                next_cluster+= 1  
                final_angle= (2*3.14)*flag_add_pi + cell_angle_sort[next_cluster - flag_add_pi*(num_boundary),0] # WRAP ARONUD THE LIST
                counter+= 1
            
            # Append tuple of average angle and average alpha
            angle_avg_alpha_avg.append( ((sum_angles)/counter, (sum_alphas)/counter) )

        # Make numpy array from list of tuples of average angle and average alpha    
        angle_avg_alpha_avg= np.array(angle_avg_alpha_avg)
        angles= angle_avg_alpha_avg[:,0]
        alphas= angle_avg_alpha_avg[:,1]

        direction_index= 0
        # cell_angle_sort has angle as first column of array
        max_alpha= max(np.array(cell_angle_sort)[:,1])
        prob_of_using_prob= GetProbOfUsingProb(max_alpha) 
        
        ### THE SENSITIVITY TO CHOOSE REMODELING RELATED DIRECTION IS FORCES DEPENDENT
        ### Only when ARP depletion is False(0)
        if (np.sum(alphas)!= 0 and (secrets.randbelow(10000))/10000 < prob_of_using_prob) and (ARP_depletion==0):
            # alphas_p= (alphas**3)/np.sum(alphas**3)
            alphas_p= alphas/np.sum(alphas)
            # print(alphas_p)
            direction_index= np.random.choice(int(len(alphas)),1,p= alphas_p)
            
        else:
            # Pick random number from numbers between 0 and length of alphas list
            direction_index= np.random.choice(int(len(alphas)),1)

        # chosen_angle is picked from average angles list, witch each angle corresponding to specific alpha
        chosen_angle= angles[direction_index]
        self.shared_steppable_vars['chosen_angle']= float(chosen_angle)
        # print("chosen_direcion",180/3.14*float(chosen_angle))


            # extra_angles_modified= []
            # end_loc= start_loc+num_boundary//2
            # if end_loc > num_boundary:#NO MINUS ONE BECASUE WE ARE USING SLICES, THERE AUTOMATIC-1 HAPPENS AT THE LAST INDEX
                # end_loc2= end_loc - num_boundary
                # end_loc= num_boundary
                # # print("end_loc2",end_loc2)
                # # print("end_loc", end_loc)
                
                # for i in range(end_loc2):#ANGLE= ANGLE+2*PI
                    # extra_angles_modified.append( (2*3.14+cell_angle_sort[i][0],cell_angle_sort[i][1]) )
                # # print("extra", extra_angles_modified)
                

            # total_collection= extra_angles_modified + cell_angle_sort[start_loc:end_loc]
            # # print("tot", total_collection)
            # # print("mean", np.mean(total_collection,0))
            # angle_avg_alpha_avg.append(np.mean(total_collection,0))
        # angle_avg_alpha_avg= np.array(angle_avg_alpha_avg)
        # print("ang_alp", angle_avg_alpha_avg)
        # angles= angle_avg_alpha_avg[:,0]
        # alphas= angle_avg_alpha_avg[:,1]
        # print("alphas", alphas )
        # print("angles", angles)
        # print("ang_avg", angle_avg_alpha_avg)
            
        # print("slice", np.array(angle_avg_alpha_avg)[:,1])
        # direction_index= 0
        # max_alpha= max(np.array(cell_angle_sort)[:,1])
        # prob_of_using_prob= GetProbOfUsingProb(max_alpha) 
        
        # ### THE SENSITIVITY TO CHOOSE REMODELING RELATED DIRECTION IS FORCES DEPENDENT
        
        # if np.sum(alphas)!= 0 and (secrets.randbelow(10000))/10000 < prob_of_using_prob:
            # alphas_p= (alphas**3)/np.sum(alphas**3)
            # # print(alphas_p)
            # direction_index= np.random.choice(int(len(alphas)),1,p= alphas_p)
            # # print("1",direction_index)
        # else:
            # direction_index= np.random.choice(int(len(alphas)),1)
            # # print("2", direction_index)
        # chosen_angle= angles[direction_index]
        # self.shared_steppable_vars['chosen_angle']= float(chosen_angle)
        # # print("chosen_direcion",float(chosen_angle))



# Set forces for cells using "chosen_angle"
# Magnitude of protrusive force is equal to 10*phi
class SetCellForces(SteppableBasePy):
    def __init__(self, frequency=1):

        SteppableBasePy.__init__(self,frequency)

    def start(self):
        """
        Called before MCS=0 while building the initial simulation
        """

    def step(self, mcs):
        chosen_angle = self.shared_steppable_vars['chosen_angle']
        for cell in self.cell_list_by_type(self.CELL):
            cell.lambdaVecX = -10*cell.dict['phi']*np.cos(chosen_angle)  # force component pointing along X axis - towards positive X's
            cell.lambdaVecY = -10*cell.dict['phi']*np.sin(chosen_angle)  # force component pointing along Y axis - towards negative Y's
            cell.lambdaVecZ = 0.0  # force component pointing along Z axis

    def finish(self):
        """
        Called after the last MCS to wrap up the simulation
        """

    def on_stop(self):
        """
        Called if the simulation is stopped before the last MCS
        """
