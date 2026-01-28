#!/usr/bin/env python
# coding: utf-8
"""
OpenFOAM Data Extraction Script

This script extracts flow field data from OpenFOAM simulation results and converts them
into PyTorch Geometric Data format for use with graph neural networks.

The script supports multiple geometry configurations:
- Single airfoils
- Tandem airfoils (cruise and takeoff configurations)
- Race car configurations
- Three-airfoil configurations
- Configurations with ground effect

Author: W.X. Lim
Date: 2023-2025
"""

import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from getDID import getSV, getDSDF, getDSDF2, getDSDF3
from timeit import default_timer


class Empty_Class():
    """Empty class placeholder for mesh objects."""
    pass




def rotate_tensor(pos, theta, resize=1.0, clock_wise=True):
    """
    Rotate 2D tensor positions around a pivot point.
    
    This function rotates 2D coordinates by a specified angle around a pivot point.
    It is used to apply angle of attack (AoA) rotations to airfoil positions.
    
    Args:
        pos (torch.Tensor): 2D tensor of shape [N, 2] containing (x, y) coordinates
        theta (float): Rotation angle in degrees
        resize (float, optional): X-coordinate of pivot point. Defaults to 1.0 (trailing edge)
        clock_wise (bool, optional): If True, rotate clockwise; if False, counter-clockwise.
                                    Defaults to True.
    
    Returns:
        torch.Tensor: Rotated positions with same shape as input
    """
    import torch
    phi = torch.tensor(theta * torch.pi / 180)
    s = torch.sin(phi)
    c = torch.cos(phi)
    rot = torch.stack([torch.stack([c, -s]),
                       torch.stack([s, c])])  # counter-clockwise rotation matrix
    pos = pos - torch.tensor([resize, 0])  # Translate to pivot point
    if clock_wise:
        pos_rot = pos @ rot.t()  # Transpose for clockwise rotation
    else:
        pos_rot = pos @ rot  # Counter-clockwise rotation
    pos_rot = pos_rot + torch.tensor([resize, 0])  # Translate back
    return pos_rot

def fun_mesh2data4(file, path, xlim=None, ylim=None):
    """
    Convert OpenFOAM mesh data to PyTorch Geometric Data format (enhanced version).
    
    This is the main function for extracting flow field data from OpenFOAM simulations.
    It supports multiple geometry configurations including single airfoils, tandem airfoils,
    race car configurations, and three-airfoil setups with various flow conditions.
    
    Args:
        file (str): Name of the OpenFOAM case directory
        path (str): Path to the directory containing OpenFOAM cases
        xlim (tuple, optional): X-axis limits for domain filtering [xmin, xmax]
        ylim (tuple, optional): Y-axis limits for domain filtering [ymin, ymax]
    
    Returns:
        tuple: (data, mesh0) where:
            - data: PyTorch Geometric Data object with flow fields and geometry features
            - mesh0: Mesh object containing raw OpenFOAM data
    
    Supported path patterns:
        - 'single': Single airfoil configurations
        - 'takeoff': Tandem airfoil takeoff configurations with ground effect
        - 'wground_single_randomFields': Single airfoil with ground effect
        - 'raceCar_randomFields': Race car configurations
        - 'Three': Three-airfoil configurations
        - Other: Standard tandem airfoil cruise configurations
    """
    import numpy as np
    import re
    import networkx as nx
    import torch
    from torch_geometric.data import Data
    from fun_LoadData import fun_foam2mesh, fun_foam2graph, fun_new_edges2
    class Empty_Class(): pass
    
    # Parse case file based on configuration type
    if 'wground_single_randomFields' in path:
        # Single airfoil with ground effect and random flow conditions
        fname = path + file + '/all/'
        tmp = file.split('_')
        naca = [tmp[1]]
        Re = float(tmp[2][2:])
        AoA = float(tmp[3][3:])
        hc_net = float(tmp[4][2:])
        # Extract height from Allrun script
        with open(path + file + '/airfoil/Allrun') as f:
            last_line = f.readlines()[-2]
            match = re.search(r'transformPoints -translate "\(0 0 ([\d.]+)\)"', last_line)
            if match:
                height = float(match.group(1))
        af_pos = rotate_tensor(torch.tensor([[0., 0.]]), theta=AoA, clock_wise=False)
        af_pos = af_pos + torch.tensor([0., height])
        print("Case = %i, parsing NACA-%s ... " % (k0, naca[0]), end=" ")
    elif 'raceCar_randomFields' in path:
        # Race car configuration with random flow conditions
        fname = path + file + '/all/'
        tmp = file.split('_')
        naca = [tmp[1], tmp[2]]
        Re = float(tmp[3][2:])
        AoA = [float(tmp[4][3:]), float(tmp[9][4:])]
        hc_net = float(tmp[5][2:])
        resize = float(tmp[6][6:])
        hcb_net = float(tmp[7][3:])
        scb_net = float(tmp[8][3:])
        # Extract front airfoil height
        with open(path + file + '/airfoil_f/Allrun') as f:
            last_line = f.readlines()[-2]
            match = re.search(r'transformPoints -translate "\(0 0 ([\d.]+)\)"', last_line)
            if match:
                height = float(match.group(1))
        # Extract back airfoil stagger and gap
        with open(path + file + '/airfoil_b/Allrun') as f:
            last_line = f.readlines()[-2]
            match = re.search(r'transformPoints -translate "\(([\d.]+) 0 ([\d.]+)\)"', last_line)
            if match:
                stagger = float(match.group(1))
                gap = float(match.group(2))
        # Calculate airfoil positions with rotation
        af_pos = rotate_tensor(torch.tensor([[0., 0.]]), theta=AoA[0], clock_wise=False)
        af_pos = af_pos + torch.tensor([0., height])
        back_af_pos = rotate_tensor(torch.tensor([[0., 0.]]), theta=AoA[1], resize=resize, clock_wise=False)
        back_af_pos = back_af_pos + torch.tensor([stagger, gap])
        af_pos = torch.tensor([[*af_pos[0]], [*back_af_pos[0]]])
        print("Case = %i, parsing NACA-%s ... " % (k0, naca[0] + '-' + naca[1]), end=" ")

    elif 'Three' in path:
        # Three-airfoil configuration
        fname = path + file + '/all/'
        tmp = file.split('_')
        naca = [tmp[1], tmp[2], tmp[3]]  # 3 airfoil codes
        Re = 500.0
        AoA = [5.0, 5.0, 5.0]
        # Parse geometric configuration
        stagger = float(tmp[4][1:])      # S0.33 - stagger between first and second
        gap = float(tmp[5][1:])          # G0.11 - gap between first and second
        stagger2 = float(tmp[6][2:])     # Sc0.21 - stagger between second and third
        gap2 = float(tmp[7][2:])         # Gc-0.11 - gap between second and third
        # Position airfoils (relative to first at (0, 0))
        af1 = rotate_tensor(torch.tensor([[0.0, 0.0]]), theta=AoA[0], clock_wise=False)
        af2 = af1 + torch.tensor([[1.0 + stagger, gap]])
        af3 = af1 + torch.tensor([[2.0 + stagger + stagger2, gap2]])
        af_pos = torch.tensor([[*af1[0]], [*af2[0]], [*af3[0]]])
        print(f"Case = {k0}, parsing NACA-{naca[0]}-{naca[1]}-{naca[2]} ... ", end=" ")

    elif 'single' in path:  # Single-Airfoil Cases
        fname = path + file + '/'
        tmp = file.split('_')
        if 'random' in path:
            naca = [tmp[1]]
            Re = float(tmp[2][2:])
            AoA = float(tmp[3][3:])
        elif 'aoa5' in path:
            naca = [tmp[1]]
            Re = 500.0
            AoA = 5.0
        else:
            naca = [tmp[1]]
            Re = 500.0
            AoA = 0.0
        # Rotate by AOA at trailing edge coordinates [1,0]
        af_pos = torch.tensor([[0., 0]])
        af_pos = rotate_tensor(af_pos, theta=AoA, clock_wise=False)
        print("Case = %i, parsing NACA-%s ... " % (k0, naca[0]), end=" ")

    elif 'takeoff' in path:  # Fixed at Re=500 and AOA=5, adjust to height
        fname = path + file + '/all/'
        tmp = file.split('_')
        naca = [tmp[2], tmp[3]]
        Re = 500.0
        AoA = [5.0, 5.0]
        stagger = float(tmp[4][1:])
        gap = float(tmp[5][1:])
        height = float(tmp[6][1:])
        af_pos = rotate_tensor(torch.tensor([[0, -1 + height]]), theta=AoA[0], clock_wise=False)
        back_af_pos = af_pos + torch.tensor([[1 + stagger, gap]])
        af_pos = torch.tensor([[*af_pos[0]], [*back_af_pos[0]]])
        print("Case = %i, parsing NACA-%s ... " % (k0, naca[0] + '-' + naca[1]), end=" ")
    
    else:  # Two-Airfoil Cases (tandem cruise)
        fname = path + file + '/all/'
        tmp = file.split('_')
        if 'random' in path:  # Random Re and AOA
            naca = [tmp[1], tmp[2]]
            Re = float(tmp[3][2:])
            AoA = [float(tmp[4][3:]), float(tmp[5][4:])]
            stagger = float(tmp[6][1:])
            gap = float(tmp[7][1:])
        elif 'aoa5' in path:  # Fixed at Re=500 and AOA=5
            naca = [tmp[2], tmp[3]]
            Re = 500.0
            AoA = [5.0, 5.0]
            stagger = float(tmp[4][1:])
            gap = float(tmp[5][1:])
        else:  # Fixed at Re=500 and AOA=0
            naca = [tmp[2], tmp[3]]
            Re = 500.0
            AoA = [0.0, 0.0]
            stagger = float(tmp[4][1:])
            gap = float(tmp[5][1:])
        # Rotate by AOA at trailing edge coordinates [1,0]
        af_pos = rotate_tensor(torch.tensor([[0., 0]]), theta=AoA[0], clock_wise=False)
        back_af_pos = af_pos + torch.tensor([[1 + stagger, gap]])
        af_pos = torch.tensor([[*af_pos[0]], [*back_af_pos[0]]])
        print("Case = %i, parsing NACA-%s ... " % (k0, naca[0] + '-' + naca[1]), end=" ")
    
    mesh0 = fun_foam2mesh(fname) # Read the mesh and flow fields
    mesh0.G0 = fun_foam2graph(mesh0,) # Construct the graphs
    if 'wground_single_randomFields' in path:
        new_edges = fun_new_edges2(mesh0,)
        mesh0.G0.add_edges_from(new_edges)
        bdy_names = [b'Inlet', b'Outlet', b'Bottom', b'Top', b'Airfoil', b'oversetPatch']
    elif 'Three' in path:
        bdy_names = [b'Inlet', b'Outlet', b'Bottom', b'Top', b'Airfoil_f', b'Airfoil_b', b'Airfoil_c', b'oversetPatch']
    elif 'single' in path:
        bdy_names = [b'Inlet', b'Outlet', b'Bottom', b'Top', b'Airfoil']
    else:
        new_edges = fun_new_edges2(mesh0,) # Connect overset and background meshes
        mesh0.G0.add_edges_from(new_edges) 
        bdy_names = [b'Inlet', b'Outlet', b'Bottom', b'Top', b'Airfoil_f', b'Airfoil_b', b'oversetPatch']

    idx_tmp = [np.array([n0 for n0 in mesh0.boundary_cells(key)]) for key in bdy_names]
    boundary = torch.zeros(mesh0.C.shape[0], dtype=torch.uint8)
    for k1 in range(len(bdy_names)): boundary[idx_tmp[k1]] = k1 + 1
        
    # Remove the background mesh cells under the airfoils
    if xlim is None: condX = np.ones(mesh0.C.shape[0], dtype=bool)
    else: condX = (xlim[0] <= mesh0.C[:,0]) & (mesh0.C[:,0] <= xlim[1])
    if ylim is None: condY = np.ones(mesh0.C.shape[0], dtype=bool)
    else: condY = (ylim[0] <= mesh0.C[:,2]) & (mesh0.C[:,2] <= ylim[1])
    if 'wground_single_randomFields' in path:
        cond3 = (mesh0.cellTypes != 2)
    elif 'single' in path: cond3 = np.ones(mesh0.C.shape[0], dtype=bool)
    else: cond3 = (mesh0.cellTypes != 2)
    idx = condX & condY & cond3
    
    # Extract positions and create filtered graph
    pos = torch.tensor(mesh0.C[:, [0, 2]][idx], dtype=torch.float)
    tmp_graph = mesh0.G0.subgraph(np.where(idx)[0])
    # Relabel nodes and keep mapping for boundary reconstruction
    node_mapping = {old: new for new, old in enumerate(sorted(tmp_graph.nodes))}
    mesh0.G = nx.relabel_nodes(tmp_graph, node_mapping)

    # Rebuild boundary and zoneID aligned to new node indices
    new_boundary = torch.zeros(len(mesh0.G.nodes), dtype=torch.uint8)
    zoneID = torch.zeros(len(mesh0.G.nodes), dtype=torch.float)
    for old_idx, new_idx in node_mapping.items():
        new_boundary[new_idx] = boundary[old_idx]
        zoneID[new_idx] = mesh0.zoneID[old_idx]
   
    # Extract flow field variables (u, v, p)
    if 'wground_single_randomFields' in path:
        # For overset mesh cases, extract from graph nodes
        y_tmp = torch.tensor([list(mesh0.G.nodes[n0].values()) for n0 in sorted(mesh0.G.nodes)], dtype=torch.float)
        pos = y_tmp[:, :2].to(dtype=torch.float)
        y = y_tmp[:, 2:].to(dtype=torch.half)
    elif 'single' in path:
        # For single airfoil, directly extract from mesh arrays
        y = torch.tensor(np.vstack([mesh0.U[:, 0], mesh0.U[:, 2], mesh0.P]).T, dtype=torch.half)
    else:
        # For tandem airfoil, extract from sorted graph nodes after stitching meshes
        y_tmp = torch.tensor([list(mesh0.G.nodes[n0].values()) for n0 in sorted(mesh0.G.nodes)], dtype=torch.float)
        pos = y_tmp[:, :2].to(dtype=torch.float)
        y = y_tmp[:, 2:].to(dtype=torch.half)
    edges = torch.tensor(np.array(mesh0.G.edges).T)

    # Create PyTorch Geometric Data object
    data = Data(pos=pos, edge_index=edges, y=y, boundary=new_boundary, flowState=mesh0.flowState,
                af_pos=af_pos, NACA=naca, AoA=AoA, zoneID=zoneID)
    
    # Add configuration-specific attributes
    if 'wground_single_randomFields' in path:
        data.height = height
        data.hc_net = hc_net
    if 'Three' in path:
        data.stagger = stagger
        data.gap = gap
        data.stagger2 = stagger2
        data.gap2 = gap2
    elif 'raceCar_randomFields' in path:
        data.hc_net = hc_net
        data.resize = resize
        data.hcb_net = hcb_net
        data.scb_net = scb_net
        data.height = height
        data.gap = gap
        data.stagger = stagger
    if 'takeoff' in path:
        data.height = height
    
    return data, mesh0

##################################################################
# Main Data Extraction Script
##################################################################

# ============================================================================
# CONFIGURATION: Modify these parameters for your dataset
# ============================================================================
root_dir = './'
dataname = 'raceCar_randomFields/'  # Change this to your dataset directory
path = root_dir + dataname
pickle_name = 'raceCar_randomFields_orig.pickle'  # Output pickle filename

# Filter files to process (modify filter condition as needed)
fileList0 = os.listdir(path)
fileList = sorted([file for file in fileList0 if 'naca' in file])  # Example: filter for 'naca' in filename
# Alternative filters:
# fileList = sorted([file for file in fileList0 if 'naca' in file])

print('Number of cases = %i' % (len(fileList)))

# ============================================================================
# Extract data from OpenFOAM cases
# ============================================================================
vRunTime = []
Dataset0 = []
k0 = 0

for file in fileList[:]:
    start_time = default_timer()
    data, mesh0 = fun_mesh2data4(file, path,)
    
    # Compute geometry features: SAF (signed airfoil field) and DSDF (distance to surface)
    # These features are used as input to graph neural networks
    if 'Three' in dataname:
        bdy_af_bool = (data.boundary == 5) | (data.boundary == 6) | (data.boundary == 7)
    else:
        bdy_af_bool = (data.boundary == 5) | (data.boundary == 6)
    data.saf = getSV(data.pos, bdy_af_bool).to(dtype=torch.half)
    
    # Select appropriate DSDF function based on configuration
    if 'single' in dataname:
        data.dsdf = getDSDF(data, data.boundary == 5, theta_rot=torch.pi/4, 
                           theta_seg=torch.pi/4, inf=5).to(dtype=torch.half)
    elif 'Three' in dataname:
        data.dsdf = getDSDF3(data, theta_rot=torch.pi/4, theta_seg=torch.pi/4, 
                            inf=5).to(dtype=torch.half)
    else:
        data.dsdf = getDSDF2(data, theta_rot=torch.pi/4, theta_seg=torch.pi/4, 
                            inf=5).to(dtype=torch.half)
    
    Dataset0.append(data)
    vRunTime.append(default_timer() - start_time)
    print("runtime = %1.2f. " % (vRunTime[-1]))
    k0 += 1

print("Finished. Total Runtime = %1.2f seconds = %1.2f hours." % 
      (sum(vRunTime), sum(vRunTime)/3600))
print(Dataset0[-1])

# Save dataset to pickle file
torch.save(Dataset0, root_dir + pickle_name)
print('Saved to :', root_dir + pickle_name)

data = Dataset0[-1]
if 'Three' in dataname:
    bd_idx = (data.boundary == 5) | (data.boundary == 6) | (data.boundary == 7)
else:
    bd_idx = (data.boundary == 5) | (data.boundary == 6)
plt.figure()
fig, axs = plt.subplots(2,2, figsize=[12,8])
axs[0,0].scatter(data.pos[bd_idx,0], data.pos[bd_idx,1], s=1, c=data.y[bd_idx,0])
axs[0,0].axis(xmin=-0.2,xmax=5.2); axs[0,0].axis(ymin=-2,ymax=2)
axs[0,0].set_title('boundary')
axs[0,1].scatter(data.pos.cpu()[:,0], data.pos.cpu()[:,1], s=1, c=data.y[:,1])
axs[0,1].axis(xmin=-0.2,xmax=5.2); axs[0,1].axis(ymin=-2,ymax=2)
axs[0,1].set_title('v')
axs[1,0].scatter(data.pos.cpu()[:,0], data.pos.cpu()[:,1], s=1, c=data.y[:,2])
axs[1,0].axis(xmin=-0.2,xmax=5.2); axs[1,0].axis(ymin=-2,ymax=2)
axs[1,0].set_title('p')
axs[1,1].scatter(data.pos.cpu()[:,0], data.pos.cpu()[:,1], s=1, c=data.y[:,0])
axs[1,1].axis(xmin=-0.2,xmax=5.2); axs[1,1].axis(ymin=-2,ymax=2)
axs[1,1].set_title('u')
plt.plot(*data.af_pos[0],'ro')
if 'single' not in pickle_name: plt.plot(*data.af_pos[1],'bo'); plt.plot(*data.af_pos[2],'ko') 
#plt.xlim([-0.5,2.5]); plt.ylim([0,2]); #plt.colorbar()
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
plt.savefig(root_dir+'Ufield_threeTandem.jpg', dpi=300, bbox_inches='tight')

plt.figure()
node_x = data.edge_index[0, :]
node_y = data.edge_index[1, :]
assert len(node_x) == len(node_y)
edge_x, edge_y = [], []

for i in range(len(node_x)):
    # x1, y1 = data.x[node_x[i], 3], data.x[node_x[i], 4]
    # x2, y2 = data.x[node_y[i], 3], data.x[node_y[i], 4]
    x1, y1 = data.pos[node_x[i], 0], data.pos[node_x[i], 1]
    x2, y2 = data.pos[node_y[i], 0], data.pos[node_y[i], 1]
    edge_x.extend([x1, x2, None])  # Use None for line breaks in plot
    edge_y.extend([y1, y2, None])

# Plot using matplotlib (MUCH faster than nx.draw)
plt.figure(figsize=(15, 15))

# Plot edges as thin lines
plt.plot(edge_x, edge_y, 'black', alpha=0.5, linewidth=1)

# Plot nodes as small points
# plt.scatter(node_x, node_y, s=1, color='blue', alpha=0.7)

plt.xlim([-0.2, 5.2])
plt.ylim([-2, 2])

plt.title("Large Graph Visualization")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")

plt.savefig(root_dir+'Graph_edges_threeTandem.jpg', dpi=300, bbox_inches='tight')
plt.figure()
fig, axs = plt.subplots(2,2, figsize=[12,8])
axs[0,0].scatter(data.pos.cpu()[:,0], data.pos.cpu()[:,1], s=1, c=data.dsdf[:,0])
axs[0,0].axis(xmin=-0.2,xmax=5.2); axs[0,0].axis(ymin=-2,ymax=2)
axs[0,0].set_title('dsdf0')
axs[0,1].scatter(data.pos.cpu()[:,0], data.pos.cpu()[:,1], s=1, c=data.dsdf[:,1])
axs[0,1].axis(xmin=-0.2,xmax=5.2); axs[0,1].axis(ymin=-2,ymax=2)
axs[0,1].set_title('dsdf1')
axs[1,0].scatter(data.pos.cpu()[:,0], data.pos.cpu()[:,1], s=1, c=data.dsdf[:,2])
axs[1,0].axis(xmin=-0.2,xmax=5.2); axs[1,0].axis(ymin=-2,ymax=2)
axs[1,0].set_title('dsdf2')
axs[1,1].scatter(data.pos.cpu()[:,0], data.pos.cpu()[:,1], s=1, c=data.saf[:,0])
axs[1,1].axis(xmin=-0.2,xmax=5.2); axs[1,1].axis(ymin=-2,ymax=2)
axs[1,1].set_title('saf')
plt.plot(*data.af_pos[0],'ro')
if 'single' not in pickle_name: plt.plot(*data.af_pos[1],'bo'); plt.plot(*data.af_pos[2],'ko')
plt.xlim([-0.2,5.2]); plt.ylim([-2,2]); #plt.colorbar()
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
plt.savefig(root_dir+'dsdf_saf_threeTandem.jpg', dpi=300, bbox_inches='tight')

'''

