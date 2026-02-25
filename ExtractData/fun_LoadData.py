"""
OpenFOAM Data Loading and Mesh Processing

This module provides functions for reading OpenFOAM CFD results and converting
them to graph/mesh formats suitable for machine learning:
- fun_foam2mesh: Load OpenFOAM case (mesh, U, p, cellTypes, zoneID)
- fun_foam2graph: Build NetworkX graph from mesh with flow fields as node attributes
- fun_new_edges2: Connect overset and background meshes for tandem configurations

Author: W.X. Lim
Date: 2023-2025
"""

import os
import pickle
import numpy as np
import openfoamparser as Ofpp
import networkx as nx
import torch
from torch_geometric.data import Data

try:
    from FVdataset import GcomputeSDF, GcomputeSAF2, getDSDF
except ImportError:
    GcomputeSDF = GcomputeSAF2 = getDSDF = None  # Optional for aa_extract_openfoam_data


class Empty_Class():
    """Empty class placeholder for mesh objects."""
    pass
# -------------------------------------------------- #
# -------------------------------------------------- #
# NACA Airfoil Utilities
# -------------------------------------------------- #

def list_naca():
    """
    List all feasible NACA-4 designations for airfoil generation.
    Returns array of [m, p, t] combinations for NACA mptt series.
    """
    mNum, pNum, ttNum = (9, 9, 40) #Linearspaces of 0<m<9, 0<p<9, 1<t<40.

    NACA_all = np.array([[0,0,0]]) #zeros one to append t0
    for k in range(1, ttNum+1): #1<t<40
        m = 0; p = 0; t = int(k) #when m=0, p=0 for symmetrical airfoils
        NACA_input = [m,p,t]
        NACA_all = np.concatenate((NACA_all,np.array([NACA_input])),axis=0)

    for i in range(1, mNum+1): #1<m<9
        for j in range(1, pNum+1): #1<p<9
            for k in range(1, ttNum+1): #1<t<40
                m = int(i); p = int(j); t = int(k)
                NACA_input = [m,p,t]
                NACA_all = np.concatenate((NACA_all,np.array([NACA_input])),axis=0)
    NACA_all = np.delete(NACA_all, 0, axis=0) #delete first zeros one
    print('NACA_all.shape: ', NACA_all.shape); #Should be (3280,3)
    return NACA_all
# -------------------------------------------------- #
def read_pickle(airfoil, path="/mnt/sdb-data/old_pickle/"):
    os.chdir(path) # change to storage drive
    if airfoil[2] < 10: tmp="0"
    else: tmp = ""
    naca = str(airfoil[0])+str(airfoil[1])+tmp+str(airfoil[2])
    fname = path + 'NACA_'+str(airfoil[0])+'_'+str(airfoil[1])+'_'+str(airfoil[2])+'.pkl'
    with open(fname,'rb') as f:
        graph = pickle.load(f)

    keys = sorted(graph.dsdf[0].keys()) #keys in dSDF dictionaries
    nodes_data_x = [[graph.node_feat[n0]['x'], graph.node_feat[n0]['y'],
                     graph.sdf[n0],
                     (graph.saf[n0]-180)/104.3560]+[graph.dsdf[n0][k0] for k0 in keys]
                    for n0 in range(graph.num_nodes)]
    nodes_data_y = [[graph.node_feat[n0]['u'], graph.node_feat[n0]['v'],
                     graph.node_feat[n0]['p']] for n0 in range(graph.num_nodes)]

    x = torch.tensor(nodes_data_x, dtype=torch.float) #float
    y = torch.tensor(nodes_data_y, dtype=torch.float) #float
    pos = x[:,0:2].clone()
    edge_index = torch.tensor(graph.edges) #int64

    data = Data(x=x, y=y, pos=pos, edge_index=edge_index.t().contiguous(), NACA=naca)
    return data
# -------------------------------------------------- #
def save_pickle(ob, fname):
    with open (fname, 'wb') as f: 
        #Use the dump function to convert Python objects into binary object files
        pickle.dump(ob, f)
# -------------------------------------------------- #
def load_pickle(fname):
    with open (fname, 'rb') as f: 
        #Convert binary object to Python object
        ob = pickle.load(f) 
        return ob
# -------------------------------------------------- #
def splitdataset(_list, percent, seed = 1):
    """ returns a (percent, 100-percent) split of the dataset """
    N = len(_list)
    _cut = int(N*percent)
    np.random.seed(seed)
    indices = np.random.permutation(N)
    training_idx, test_idx = indices[:_cut], indices[_cut:]
    # print(training_idx)
    # training, test = _list[training_idx], _list[test_idx]
    training, test = [],[]
    for idx in training_idx:
        training.append(_list[idx])
    for idx in test_idx:
        test.append(_list[idx])
    return (training, test)
# -------------------------------------------------- #
# -------------------------------------------------- #
# OpenFOAM I/O
# -------------------------------------------------- #

def get_lastTime(fname):
    """Get the latest time directory name from an OpenFOAM case."""
    fileList = np.array(os.listdir(fname))
    timeList = [file.replace('.','',1).isdigit() for file in fileList]
    last = max(fileList[timeList].astype(float))
    if np.mod(last, 1)!=0:
        return str(last)
    else:
        return str(last.astype(int))
# -------------------------------------------------- #
def fun_edge_list(mesh):
    edge_list, node_ids = [], []
    for i,pt in enumerate(mesh.C):
        nbr_i = mesh.cell_neighbour_cells(i)
        for x in nbr_i:
            if x>=0:
                edge_list.append((i,x))
            node_ids.append(i)
    return edge_list, node_ids
# -------------------------------------------------- #
def readflowstate_old(path):
    _d = {}
    with open(path+"/0/flowState",'r') as f:
        for line in f.readlines():
            splits = line.split()
            if(len(splits) == 0): # empty line
                continue 
            key = splits[0]
            vals = " ".join(splits[1:])
            _d[key] = vals 
    return _d 
# -------------------------------------------------- #
def readflowstate(path):
    with open(path+"/0/flowState",'r') as f: 
        flowState = f.readlines()
    dict0 = {}
    for line in flowState:
        tmp = line.split('\n')[0].split(';')[0]
        if ('#calc' not in tmp) and ('$' not in tmp) and (tmp != ''):
            key = tmp.split()[0]
            dict0[key] = float(tmp.split()[1])
    # print('dict0: ',dict0)
    dict0['maxLength'] = 10 * 2;
    if 'raceCar' in path:
        dict0['nu'] = 1.461e-5
        dict0['Umag'] = dict0['Re'] * dict0['nu'] / dict0['c_eff']; #only for random dataset??
        dict0['y_l'] = 2e-5 * dict0['c_eff'];
    elif 'rand' in path:    
        dict0['nu'] = 1.461e-5
        dict0['Umag'] = dict0['Re'] * dict0['nu'] / dict0['c']; #only for random dataset??
        dict0['y_l'] = 2e-5 * dict0['c'];
    else:
        dict0['Umag'] = 0.146
        dict0['nu'] = dict0['Umag'] * dict0['c'] / dict0['Re'];
        dict0['y_l'] = 1e-3 * dict0['c']; 
    
    dict0['domain'] = dict0['maxLength']; # * dict0['c'];
    dict0['omegaInf'] = 2.0 * dict0['Umag'] / dict0['domain'];
    dict0['omega'] = dict0['omegaInf'];
    dict0['omega_wall'] = 60.0 * dict0['nu'] / (0.075 * dict0['y_l']**2);
    dict0['nuT'] = 1e-3 * dict0['nu'];
    dict0['Re_L'] = dict0['Umag'] * dict0['domain'] / dict0['nu'];
    dict0['kInf'] = dict0['nuT'] * dict0['omegaInf']; #1.0 * dict0['Umag']**2 / dict0['Re_L'];
    dict0['k'] = dict0['kInf'];
    dict0['radAOA'] = dict0['AOA'] * np.pi / 180;
    dict0['Ux'] = dict0['Umag'] * np.cos(dict0['radAOA']);
    dict0['Uz'] = dict0['Umag'] * np.sin(dict0['radAOA']);
    dict0['Uinlet'] = [dict0['Ux'], dict0['Uy'], dict0['Uz']];
    # dict0['liftD'] = [-1.0*np.sin(dict0['radAOA']), 0.0, np.cos(dict0['radAOA'])];
    # dict0['dragD'] = [np.cos(dict0['radAOA']), 0.0, np.sin(dict0['radAOA'])];
    return dict0
# -------------------------------------------------- #

def bound_box(mesh0, x_lim=None, y_lim=None, z_lim=None, offset=[0,0,0]):
    """
    This function creates a bounding box of the OpenFoam solution with
    an offset from the origion.
    """
    XY = mesh0.C
    XY = offset * np.ones([XY.shape[0],1]) + XY
    if x_lim is None: x_lim = [min(XY[:,0]), max(XY[:,0])]
    if y_lim is None: y_lim = [min(XY[:,1]), max(XY[:,1])]
    if z_lim is None: z_lim = [min(XY[:,2]), max(XY[:,2])]
    cond_x = (x_lim[0] <= XY[:,0]) & (XY[:,0] <= x_lim[1])
    cond_y = (y_lim[0] <= XY[:,1]) & (XY[:,1] <= y_lim[1])
    cond_z = (z_lim[0] <= XY[:,2]) & (XY[:,2] <= z_lim[1])
    idx = cond_x & cond_y & cond_z
    mesh = Empty_Class()
    mesh.C, mesh.U, mesh.P = XY[idx], mesh0.U[idx], mesh0.P[idx]
    return mesh
# -------------------------------------------------- #
def fun_stitch(src_mesh1, src_mesh2):
    dst_mesh = Empty_Class()
    dst_mesh.C = np.vstack([src_mesh1.C, src_mesh2.C])
    dst_mesh.P = np.hstack([src_mesh1.P, src_mesh2.P])
    dst_mesh.U = np.vstack([src_mesh1.U, src_mesh2.U])
    return dst_mesh
# ---------------------------------------------------------------------- #
def fun_interpolate(dst_mesh, src_mesh, interp_method="linear"):
    from scipy.interpolate import griddata, RBFInterpolator

    dst_mesh.C1, dst_mesh.P1, dst_mesh.U1 = src_mesh.C, src_mesh.P, src_mesh.U
    points = dst_mesh.C1[:,[0,2]]
    xi = dst_mesh.C[:,[0,2]]
    dst_mesh.U2 = np.zeros(dst_mesh.U.shape)
    dst_mesh.P2 = np.zeros(dst_mesh.P.shape)
    u_avg = dst_mesh.U1[:,0].mean()
    v_avg = dst_mesh.U1[:,2].mean()
    p_avg = dst_mesh.P1.mean()
    dst_mesh.U2[:,0] = griddata(points, dst_mesh.U1[:,0], xi, method=interp_method, fill_value=u_avg)
    dst_mesh.U2[:,2] = griddata(points, dst_mesh.U1[:,2], xi, method=interp_method, fill_value=v_avg)
    dst_mesh.P2 = griddata(points, dst_mesh.P1, xi, method=interp_method, fill_value=p_avg)
    #dst_mesh.U2 = RBFInterpolator(points, dst_mesh.U1, kernel=interp_method)(xi)
    #dst_mesh.P2 = RBFInterpolator(points, dst_mesh.P1, kernel=interp_method)(xi)
    return dst_mesh
# ---------------------------------------------------------------------- #
def fun_interpolate2(dst_mesh0, src_mesh, method='linear', fill_value=0):
    from scipy.interpolate import griddata
    points = src_mesh.C[:,[0,2]]
    xi = dst_mesh0.C[:,[0,2]]
    dst_mesh = Empty_Class()
    dst_mesh.U2 = np.zeros(dst_mesh0.U.shape)
    dst_mesh.P2 = np.zeros(dst_mesh0.P.shape)
    dst_mesh.U2[:,0] = griddata(points, src_mesh.U[:,0], xi, method=method, fill_value=fill_value)
    dst_mesh.U2[:,2] = griddata(points, src_mesh.U[:,2], xi, method=method, fill_value=fill_value)
    dst_mesh.P2 = griddata(points, src_mesh.P, xi, method=method, fill_value=fill_value)
    return dst_mesh
# ---------------------------------------------------------------------- #
def fun_interpolate3(data0, val0, data1, method='linear', fill_value=0):
    import torch
    from scipy.interpolate import griddata
    y_int  = torch.zeros(data1.y.shape)
    for k0 in range(data1.y.shape[1]):
        y_int[:,k0] = torch.tensor(griddata(data0.pos, val0[:,k0], data1.pos, method=method, fill_value=0))
    return y_int
# ---------------------------------------------------------------------- #
#sigmoid = lambda x, beta: 1/(1+np.exp(-beta*x))
def sigmoid(x,beta):
    return 1/(1+np.exp(-beta*x))
# ---------------------------------------------------------------------- #
def smoothstep(x):
    y = np.zeros(x.shape)
    y[x>=1] = 1
    idx = (0<=x) & (x<=1)
    y[idx] = (3*x[idx]**2) - (2*x[idx]**3)
    return y
# ---------------------------------------------------------------------- #
def fun_connect_overset(mesh, offset=None):
    """
    Connect overset boundary with background mesh.
    Remove the points (nodes) inside the oversetPatch boundary.
    """
    import networkx as nx
    cor_x, cor_y = mesh.C[:,0], mesh.C[:,2]
    
    # boundary nodes: airfoil and oversetPatch
    bdy_in = np.array( sorted([n0 for n0 in mesh.boundary_cells(b'Inlet')]) )
    bdy_os = np.array( sorted([n0 for n0 in mesh.boundary_cells(b'oversetPatch')]) )
    sdf = compute_SDF(mesh.G, bdy_in)
    val = np.array([sdf[n0] for n0 in sorted(mesh.G.nodes)])
    nOver = np.where(val == np.inf)[0] 
    nBack = np.where(val != np.inf)[0]
    
    # Find Coordinates of background mesh points inside oversetPatch
    if offset is None:
        nIn, nOut = fun_is_inside(cor_x, cor_y, nBack, bdy_os)
    else:
        x_avg, y_avg = np.mean(cor_x[bdy_os])-0.5, np.mean(cor_y[bdy_os])
        bdy_os1 = bdy_os[np.where(cor_y[bdy_os] > y_avg)[0]]
        bdy_os2 = bdy_os[np.where(cor_y[bdy_os] < y_avg)[0]]
        nIn1, nOut1 = fun_is_inside(cor_x, cor_y, nBack, bdy_os1)
        nIn2, nOut2 = fun_is_inside(cor_x, cor_y, nBack, bdy_os2)
        nIn, nOut = np.union1d(nIn1, nIn2), np.intersect1d(nOut1, nOut2)
    
    # Add new edges between overset and background mesh along the boundary
    new_edges = fun_new_edges(cor_x, cor_y, nOut, bdy_os)

    mesh.G.add_edges_from(new_edges)
    new_nodes = np.hstack([nOut, nOver])
    
    bdy_af = sorted([n0 for n0 in mesh.boundary_cells(b'Airfoil')]) 
    bdy_af_bool0 = np.zeros(mesh.G.number_of_nodes(), dtype=bool)
    bdy_af_bool0[bdy_af] = True
    
    #sdf1 = compute_SDF(mesh.G, bAirfoil)
    for n0 in sorted(mesh.G.nodes):
        mesh.G.nodes[n0]['airfoil_bool'] = bdy_af_bool0[n0]
        #mesh.G.nodes[n0]['sdf'] = sdf[n0]
    
    mesh_out = Empty_Class()
    #mesh_out.G = mesh.G.subgraph(new_nodes)
    tmp_graph = mesh.G.subgraph(new_nodes)
    mesh_out.G = nx.relabel.convert_node_labels_to_integers(tmp_graph)
    mesh_out.boundary = mesh.boundary
    mesh_out.boundary_cells = mesh.boundary_cells
    tmp = np.array([list(mesh_out.G.nodes[k1].values()) for k1 in sorted(mesh_out.G.nodes)])
    mesh_out.C = np.vstack([tmp[:,0], np.zeros(len(mesh_out.G.nodes)), tmp[:,1]]).T
    mesh_out.U = np.vstack([tmp[:,2], np.zeros(len(mesh_out.G.nodes)), tmp[:,3]]).T
    mesh_out.P = tmp[:,4]
    return mesh_out
#--------------------------------------------------#
def fun_foam2graph2(filepath, xlim=[-0.5,2], ylim=[-1.5,1.5]):
    import networkx as nx
    mesh = fun_foam2mesh(filepath)
    G0 = fun_foam2graph(mesh)
    
    return G0
# -------------------------------------------------- #
def fun_foam2graph(mesh):
    """
    Build NetworkX graph from OpenFOAM mesh with flow fields as node attributes.
    Node attributes: x, y (position), u, v, p (flow variables).
    """
    edges, nodes = fun_edge_list(mesh)
    G0 = nx.Graph(case=mesh.path.split("/")[-2])
    G0.add_edges_from(edges)
    for n0 in sorted(G0.nodes):
        G0.nodes[n0]['x'] = mesh.C[n0,0]
        G0.nodes[n0]['y'] = mesh.C[n0,2]
        G0.nodes[n0]['u'] = mesh.U[n0,0]
        G0.nodes[n0]['v'] = mesh.U[n0,2]
        G0.nodes[n0]['p'] = mesh.P[n0]
    return G0
# -------------------------------------------------- #
def fun_foam2mesh(filepath, readFlowState=True):
    """
    Load OpenFOAM mesh and flow fields.

    Args:
        filepath: Path to OpenFOAM case directory (e.g., 'case/all/')
        readFlowState: If True, parse flowState file for Re, AoA, etc.

    Returns:
        mesh: Object with .C (cell centers), .U (velocity), .P (pressure),
              .cellTypes (0=calculated, 1=interpolated, 2=hole),
              .zoneID (0=background, 1+=overset regions), .flowState (dict)
    """
    
    mesh = Ofpp.FoamMesh(filepath)
    mesh.path = filepath
    last = get_lastTime(filepath)
    if not os.path.isfile(os.path.join(filepath, last+'/C')):
        os.chdir(filepath)
        os.system('postProcess -func writeCellCentres > log_cellcentres.txt 2>&1')
        os.chdir("../")
    mesh.C = Ofpp.parse_internal_field(os.path.join(filepath, last+'/C'))
    mesh.U = Ofpp.parse_internal_field(os.path.join(filepath, last+'/U'))
    mesh.P = Ofpp.parse_internal_field(os.path.join(filepath, last+'/p'))
    if os.path.isfile(os.path.join(filepath, last+'/cellTypes')):
        mesh.cellTypes = Ofpp.parse_internal_field(os.path.join(filepath, last+'/cellTypes'))
    if os.path.isfile(os.path.join(filepath, last+'/zoneID')):
        mesh.zoneID = Ofpp.parse_internal_field(os.path.join(filepath, last+'/zoneID'))
    if readFlowState:
        mesh.flowState = readflowstate(mesh.path)
    return mesh
# -------------------------------------------------- #
def fun_foam2singlemesh(filepath, readFlowState=True):
    mesh = Ofpp.FoamMesh(filepath)
    mesh.path = filepath
    last = get_lastTime(filepath)
    if not os.path.isfile(os.path.join(filepath, last+'/C')):
        os.chdir(filepath)
        os.system('postProcess -func writeCellCentres > log_cellcentres.txt 2>&1')
        os.chdir("../")
    mesh.C = Ofpp.parse_internal_field(os.path.join(filepath, last+'/C'))
    mesh.U = Ofpp.parse_internal_field(os.path.join(filepath, last+'/U'))
    mesh.P = Ofpp.parse_internal_field(os.path.join(filepath, last+'/p'))
    #if os.path.isfile(os.path.join(filepath, last+'/cellTypes')):
    #    mesh.cellTypes = Ofpp.parse_internal_field(os.path.join(filepath, last+'/cellTypes'))
    #if os.path.isfile(os.path.join(filepath, last+'/zoneID')):
    #    mesh.zoneID = Ofpp.parse_internal_field(os.path.join(filepath, last+'/zoneID'))
    if readFlowState:
        mesh.flowState = readflowstate(mesh.path)
    return mesh
# -------------------------------------------------------------------------------
def read_data(filepath):    
    last = get_lastTime(filepath)
    # generate cell centers
    if not os.path.isfile(os.path.join(filepath, '0/C')):
        os.chdir(filepath)
        os.system('postProcess -func writeCellCentres > log_cellcentres.txt 2>&1')
        os.chdir("../")
    C = Ofpp.parse_internal_field(os.path.join(filepath, '0/C'))
    U = Ofpp.parse_internal_field(os.path.join(filepath, last+'/U'))
    P = Ofpp.parse_internal_field(os.path.join(filepath, last+'/p'))
    return C, U, P
# ---------------------------------------------------------------------- #
def fun_mesh2data2(mesh, naca=['0012', '0012'], af_pos=[[0, 0], [0, -1]], AoA=0, Re=1000, xlim=[-10, 10], ylim=[-10, 10]):
    """
    Convert mesh to PyTorch Geometric Data with geometry features.
    Requires FVdataset module (GcomputeSDF, GcomputeSAF2, getDSDF).
    """
    if GcomputeSDF is None or GcomputeSAF2 is None or getDSDF is None:
        raise ImportError("fun_mesh2data2 requires FVdataset module")
    
    idx1 = (xlim[0] <= mesh.C[:, 0]) & (mesh.C[:, 0] <= xlim[1])
    idx2 = (ylim[0] <= mesh.C[:,2]) & (mesh.C[:,2] <= ylim[1])
    idx3 = mesh.cellTypes != 2
    idx = idx1 & idx2 & idx3
    new_nodes = np.where(idx)[0]
    
    bdy_names = [b'Inlet', b'Outlet', b'Bottom', b'Top', b'Airfoil_f', b'Airfoil_b']
    idx_tmp = [np.array([n0 for n0 in mesh.boundary_cells(key)]) for key in bdy_names]
    boundary = torch.zeros(mesh.G0.number_of_nodes(), dtype=torch.uint8)
    for k0 in range(4,6):
        boundary[idx_tmp[k0]] = k0 + 1
        
    pos = torch.tensor(mesh.C[:,[0,2]][idx], dtype=torch.float)
    
    tmp_graph = mesh.G0.subgraph(new_nodes)
    mesh.G = nx.relabel.convert_node_labels_to_integers(tmp_graph)
    tmp = torch.tensor([list(mesh.G.nodes[n0].values()) for n0 in sorted(mesh.G.nodes)], dtype=torch.float)
    y = tmp[:,2:].to(dtype=torch.half)
    edges = torch.tensor( np.array(mesh.G.edges).T )
                         
    data = Data(pos=pos, edge_index=edges, y=y, af_pos=af_pos, NACA=naca, AoA=AoA, Re=Re)
    data.y_avg, data.y_std = torch.std_mean(data.y, dim=0,)
    
    data.boundary = boundary[new_nodes]
    bdy_af_bool = (boundary==5) | (boundary==6)
    bdy_af_bool = bdy_af_bool[idx]
    data.sdf = GcomputeSDF(pos, bdy_af_bool).unsqueeze(1).to(dtype=torch.half)
    data.saf = GcomputeSAF2(pos, bdy_af_bool).to(dtype=torch.half)
    data.dsdf = getDSDF(pos, bdy_af_bool, theta_rot=torch.pi/4, theta_seg=torch.pi/2, inf=10).T.to(dtype=torch.half)
    #data.pcdf = GcomputePCDF(data, inf_dist=10).to(dtype=torch.half)
    
    #data.y_int = torch.tensor( np.hstack( [mesh.U2[:,[0,2]][new_nodes], 
    #                                       mesh.P2[new_nodes].reshape(-1,1)]), dtype=torch.half)

    mesh.C, mesh.U, mesh.P = mesh.C[idx], mesh.U[idx], mesh.P[idx]
    return data, mesh
# ---------------------------------------------
def fun_new_edges2(mesh, delta=0.01):
    """
    Create edges connecting overset mesh cells to background mesh cells.
    Used for tandem airfoil configurations with overlapping grids.
    """
    import math
    vecX, vecY = mesh.C[:,0], mesh.C[:,2]
    
    bool_background = (mesh.zoneID==0) & (mesh.cellTypes!=2)
    idx_background = np.where(bool_background)[0]

    bool_overset = (mesh.zoneID>0)
    idx_overset = np.where(bool_overset)[0]
    pts_overset = mesh.C[bool_overset][:,[0,2]]

    new_edges = []
    for n0 in idx_background:
        corX, corY = mesh.C[n0,0], mesh.C[n0,2]
        condX = ((corX-delta) <=  pts_overset[:,0]) & (pts_overset[:,0] <= (corX+delta))
        condY = ((corY-delta) <=  pts_overset[:,1]) & (pts_overset[:,1] <= (corY+delta))
        cond = condX & condY
        idx_tmp = np.where(cond)[0]
        if np.size(idx_tmp) == 0:
            continue
        
        Nidx1 = idx_overset[idx_tmp]
        for n1 in range(len(idx_tmp)):
            idx = idx_overset[idx_tmp[n1]]
            tmp_dist = np.array([ math.dist([corX,corY], [vecX[idx],vecY[idx]]) for idx in Nidx1 ])
        idx_min = np.argmin(tmp_dist)
        new_edges.append(( n0, Nidx1[idx_min]))
    return new_edges
#--------------------------------------------------#
def shadow_polygon(pts_af1, pts_af2, xlim=[-1,5.5], ylim=[-2,1]):
    dist = torch.cdist(pts_af1, pts_af2, p=2)
    closest_idx = dist.argmin(dim=0)
    closest_pts = torch.tensor(np.array([ pts_af1[idx].numpy() for idx in closest_idx ]))

    tmp = closest_pts - pts_af2
    r = torch.norm(tmp, dim=1)
    theta = torch.atan(tmp[:,0]/tmp[:,1])
    theta = torch.nan_to_num(theta, nan=torch.pi/2)

    if torch.pi - (theta.max().item() - theta.min().item()) < 0.03:
        theta[theta<0] += 2*torch.pi
    idx_min, idx_max = theta.argmin(), theta.argmax()

    xlim = np.array([-1,3]); ylim = np.array([-2,1])

    x2, x1 = (pts_af2[[idx_min, idx_max],0].numpy(),closest_pts[[idx_min, idx_max],0].numpy())
    y2, y1 = (pts_af2[[idx_min, idx_max],1].numpy(),closest_pts[[idx_min, idx_max],1].numpy())
    m = (y2-y1)/(x2-x1) # m = (y2-y1)/(x2-x1)
    c = y1 - (m*x1) # y = mx + c

    pts_af = pts_af2[[idx_min, idx_max],:]
    x_intersect, y_intersect = [], [] 
    for k0 in range(2):
        for k1 in range(2):
            y_intersect.append( (xlim[k1], (m[k0]*xlim[k1]) + c[k0]) )
            x_intersect.append( ((ylim[k1] - c[k0]) / m[k0], ylim[k1]) )
        
    tmp = np.array(x_intersect + y_intersect)
    cond1 = (xlim[0] <= tmp[:,0]) & (tmp[:,0] <= xlim[1])
    cond2 = (ylim[0] <= tmp[:,1]) & (tmp[:,1] <= ylim[1])
    cond = cond1 & cond2
    pts_bdy = torch.tensor(tmp[cond], dtype=torch.float)
    intersect = []
    for k0 in range(2):
        dist1 = torch.cdist(pts_af[k0].unsqueeze(0), pts_bdy, p=2)
        idx1 = dist1.argmin().item()
        intersect.append(pts_bdy[idx1].numpy())
    intersect = np.array(intersect)

    polygon = []
    polygon.append( (pts_af2[idx_min,0].item(), pts_af2[idx_min,1].item()) )
    polygon.append( (intersect[0,0], intersect[0,1]) )
    if not( (intersect[0,0] == intersect[1,0]) | (intersect[0,1] == intersect[1,1]) ):
        polygon.append( (intersect[0,0], intersect[1,1]) )
    polygon.append( (intersect[1,0], intersect[1,1]) )
    polygon.append( (pts_af2[idx_max,0].item(), pts_af2[idx_max,1].item()) )
    return(polygon)
#--------------------------------------------------#
def GcomputePCDF(data, inf_dist=10):
    """
    Compute PCDF (Point Cloud Distance Field) for shadow regions.
    Requires FVdataset, naca, and pointInside modules.
    """
    if GcomputeSAF2 is None:
        raise ImportError("GcomputePCDF requires FVdataset module")
    from naca import naca4
    from pointInside import is_inside_sm
    from FVdataset import GcomputeSAF2
    nNodes = len(data.pos)

    pts_af1 = torch.tensor(np.array( naca4(data.NACA[0], 50,) ).T, dtype=torch.float)
    pts_af1[:,0] += data.af_pos[0][0]
    pts_af1[:,1] += data.af_pos[0][1]

    pts_af2 = torch.tensor(np.array( naca4(data.NACA[1], 50,) ).T, dtype=torch.float)
    pts_af2[:,0] += data.af_pos[1][0]
    pts_af2[:,1] += data.af_pos[1][1]

    polygon1 = shadow_polygon(pts_af1, pts_af2)
    flag_in1 = np.array([is_inside_sm (polygon1, (data.pos[n0,0], data.pos[n0,1]))
                         for n0 in range(nNodes) ], dtype=bool)

    polygon2 = shadow_polygon(pts_af2, pts_af1)
    flag_in2 = np.array([is_inside_sm (polygon2, (data.pos[n0,0], data.pos[n0,1]))
                         for n0 in range(nNodes) ], dtype=bool)

    pcdf1 = GcomputeSAF2(data.pos, data.boundary==5)
    pcdf2 = GcomputeSAF2(data.pos, data.boundary==6)
    pcdf1[flag_in1] = torch.sign(pcdf1[flag_in1]) * inf_dist
    pcdf2[flag_in2] = torch.sign(pcdf2[flag_in2]) * inf_dist
    pcdf = torch.cat((pcdf1,pcdf2), dim=1)
    return pcdf
#--------------------------------------------------#

def fun_mesh2data(file, path, AoA=0, Re=1000, xlim=None, ylim=None):
    """
    Read OpenFOAM tandem airfoil case into PyTorch Geometric Data format.
    """
    import numpy as np
    import networkx as nx
    import torch
    from torch_geometric.data import Data
    from fun_LoadData import fun_foam2mesh, fun_foam2graph, fun_new_edges2
    class Empty_Class(): pass
    
    # Read airfoils' types and positions
    naca = [file[8:12], file[13:17]]
    th_af1 = int(naca[0][-2:])/100 # thickness of 1st airfoil
    stagger = float(file.split('_')[-2].replace('S','')) # X-dist between 2 airfoils
    gap = float(file.split('_')[-1].replace('G','')) # Y-dist between 2 airfoils
    if stagger == 0:
        af_pos = np.array([[0,0], [stagger,gap]])
    else:
        af_pos = np.array([[0,0], [1+stagger,gap]])

    # Read the case files 
    fname = path+file+'/all/'
    mesh0 = Empty_Class()
    mesh0 = fun_foam2mesh(fname) # Read the mesh and flow fields
    mesh0.G0 = fun_foam2graph(mesh0,) # Construct the graphs
    new_edges = fun_new_edges2(mesh0,) # Connect overset and background meshes
    mesh0.G0.add_edges_from(new_edges) 
    
    # Remove the background mesh cells under the airfoils
    if xlim is not None:
        condX = (xlim[0] <= mesh0.C[:,0]) & (mesh0.C[:,0] <= xlim[1])
        condY = (ylim[0] <= mesh0.C[:,2]) & (mesh0.C[:,2] <= ylim[1])
        cond3 = (mesh0.cellTypes != 2)
        idx = condX & condY & cond3
    else:
        idx = (mesh0.cellTypes != 2)
    
    bdy_names = [b'Inlet', b'Outlet', b'Bottom', b'Top', b'Airfoil_f', b'Airfoil_b', b'oversetPatch']
    idx_tmp = [np.array([n0 for n0 in mesh0.boundary_cells(key)]) for key in bdy_names]
    boundary = torch.zeros(mesh0.G0.number_of_nodes(), dtype=torch.uint8)
    for k0 in range(len(bdy_names)):
        boundary[idx_tmp[k0]] = k0 + 1
   
    pos = torch.tensor(mesh0.C[:,[0,2]][idx], dtype=torch.float)
    
    tmp_graph = mesh0.G0.subgraph(np.where(idx)[0])
    mesh0.G = nx.relabel.convert_node_labels_to_integers(tmp_graph)
    tmp = torch.tensor([list(mesh0.G.nodes[n0].values()) for n0 in sorted(mesh0.G.nodes)], dtype=torch.float)
    y = tmp[:,2:].to(dtype=torch.half)
    edges = torch.tensor( np.array(mesh0.G.edges).T )
                         
    data = Data(pos=pos, edge_index=edges, y=y, boundary=boundary[idx], af_pos=af_pos, NACA=naca, AoA=AoA, Re=Re)
    data.y_avg, data.y_std = torch.std_mean(data.y, dim=0,)
    return data
#--------------------------------------------------#
def fun_mesh2data3(file, path, Re=1000, xlim=None, ylim=None):
    """
    Read OpenFOAM tandem airfoil case into PyTorch Geometric Data format.
    """
    import numpy as np
    import networkx as nx
    import torch
    from torch_geometric.data import Data
    from fun_LoadData import fun_foam2mesh, fun_foam2graph, fun_new_edges2
    class Empty_Class(): pass
    
    # Read airfoils' types and positions
    tmp = file.split('_')[-1].split('.')
    naca = [('{:0>4}'.format(tmp[0])), ('{:0>4}'.format(tmp[1]))]
    th_af1 = int(naca[0][-2:])/100 # thickness of 1st airfoil
    th_af2 = int(naca[1][-2:])/100 # thickness of 2nd airfoil
    stagger = float(file.split('S')[-1].split('G')[0][:-1]) # X-dist between 2 airfoils
    gap = float(file.split('G')[-1].split('AOA')[0][:-1]) # Y-dist between 2 airfoils
    AoA = float(file.split('AOA')[-1])
    if stagger == 0: af_pos = np.array([[0,0], [stagger,gap]])
    else:            af_pos = np.array([[0,0], [1+stagger,gap]])

    # Read the case files 
    fname = path+file+'/all/'
    mesh0 = Empty_Class()
    mesh0 = fun_foam2mesh(fname) # Read the mesh and flow fields
    mesh0.G0 = fun_foam2graph(mesh0,) # Construct the graphs
    new_edges = fun_new_edges2(mesh0,) # Connect overset and background meshes
    mesh0.G0.add_edges_from(new_edges) 
    
    # Remove the background mesh cells under the airfoils
    if xlim is not None:
        condX = (xlim[0] <= mesh0.C[:,0]) & (mesh0.C[:,0] <= xlim[1])
        condY = (ylim[0] <= mesh0.C[:,2]) & (mesh0.C[:,2] <= ylim[1])
        cond3 = (mesh0.cellTypes != 2)
        idx = condX & condY & cond3
    else:
        idx = (mesh0.cellTypes != 2)
    
    bdy_names = [b'Inlet', b'Outlet', b'Bottom', b'Top', b'Airfoil_f', b'Airfoil_b', b'oversetPatch']
    idx_tmp = [np.array([n0 for n0 in mesh0.boundary_cells(key)]) for key in bdy_names]
    boundary = torch.zeros(mesh0.G0.number_of_nodes(), dtype=torch.uint8)
    for k0 in range(len(bdy_names)):
        boundary[idx_tmp[k0]] = k0 + 1
   
    pos = torch.tensor(mesh0.C[:,[0,2]][idx], dtype=torch.float)
    
    tmp_graph = mesh0.G0.subgraph(np.where(idx)[0])
    mesh0.G = nx.relabel.convert_node_labels_to_integers(tmp_graph)
    tmp = torch.tensor([list(mesh0.G.nodes[n0].values()) for n0 in sorted(mesh0.G.nodes)], dtype=torch.float)
    y = tmp[:,2:].to(dtype=torch.half)
    edges = torch.tensor( np.array(mesh0.G.edges).T )
                         
    data = Data(pos=pos, edge_index=edges, y=y, boundary=boundary[idx], af_pos=af_pos, NACA=naca, AoA=AoA, Re=Re)
    data.y_avg, data.y_std = torch.std_mean(data.y, dim=0,)
    return data
#--------------------------------------------------#
