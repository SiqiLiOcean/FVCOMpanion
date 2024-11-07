import numpy as np
import netCDF4 as nc
import py2dm
import pyproj
import io
import esmpy

import matplotlib
import matplotlib.pyplot as plt

import cartopy
import cartopy.io.img_tiles as cimgt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes

from scipy.spatial import KDTree

from urllib.request import urlopen, Request
from PIL import Image

from T_esmpy import *
from bluedata import *

def file_exists(filepath):
    """
    Check if a file exists at the specified filepath.

    Parameters:
    - filepath (str): The path to the file to check.

    Returns:
    - bool: True if the file exists, False otherwise.

    Raises:
    - ValueError: If the file does not exist.
    """
    try:
        with open(filepath, 'r'):
            return True
    except FileNotFoundError:
        raise ValueError(f"File '{filepath}' does not exist.")


def read_2dm_quick(f2dm):
    """
    Read SMS 2dm file to get x, y, nv, h
    (This is a quick method)
    
    Parameters:
    - f2dm (str): Filepath to the SMS 2dm file.
    
    Returns:
    - x (np.ndarray): x-coordinate
    - y (np.ndarray): y-coordinate
    - nv (np.ndarray): ID of nodes around each cell
    - h (np.ndarray): depth (positive for water)
    """    
    file_exists(f2dm)

    x, y, h = [], [], []
    nv = []
    
    with open(f2dm, 'r') as file:
        for line in file:
            parts = line.split()
            if parts[0] == 'ND':
                x.append(float(parts[2]))
                y.append(float(parts[3]))
                h.append(float(parts[4]))
            elif parts[0] == 'E3T':
                nv.append([int(parts[2]), int(parts[3]), int(parts[4])])
    
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    h = np.array(h, dtype=float)
    nv = np.array(nv, dtype=int)

    return x, y, nv, h    


def read_2dm(f2dm):
    """
    Read SMS 2dm file to get x, y, nv, h, string
    
    Parameters:
    - f2dm (str): Filepath to the SMS 2dm file.
    
    Returns:
    - x (np.ndarray): x-coordinate
    - y (np.ndarray): y-coordinate
    - nv (np.ndarray): ID of nodes around each cell
    - h (np.ndarray): depth (positive for water)
    - string (list): node IDs of each string   
    """    
    file_exists(f2dm)
    
    with py2dm.Reader(f2dm) as mesh: 
        # Get the dimention length
        m = mesh.num_nodes
        n = mesh.num_elements
        l = mesh.num_node_strings
        
        # Initialize the variables
        x = np.zeros((m,), dtype=float)
        y = np.zeros((m,), dtype=float)
        h = np.zeros((m,), dtype=float)
        nv = np.zeros((n,3), dtype=int)

        method = 1
        if method == 1:
            # Read x, y, h
            nodes = np.array([(node.pos[0], node.pos[1], node.pos[2]) for node in mesh.nodes])
            x, y, h = nodes[:, 0], nodes[:, 1], nodes[:, 2]
            # Read nv
            nv = np.array([element.nodes for element in mesh.elements], dtype=int)
            # Read strings
            string = [node_string.nodes for node_string in mesh.node_strings]
        else:
            # Read x, y, h
            for i, node in enumerate(mesh.nodes, start=0):
                x[i] = node.pos[0]
                y[i] = node.pos[1]
                h[i] = node.pos[2]   
            # Read nv
            for j, element in enumerate(mesh.elements, start=0):
                nv[j,:] = element.nodes
            # Read strings
            string = []
            for l, node_string in enumerate(mesh.node_strings, start=0):
                string.append(node_string.nodes)

    return x, y, nv, h, string


def write_2dm(f2dm, x, y, nv, h=None, string=[]):
    """
    Write SMS 2dm file with x, y, nv, (h, string)
    
    Parameters:
    - f2dm (str): Filepath to the SMS 2dm file.
    - x (np.ndarray): x-coordinate
    - y (np.ndarray): y-coordinate
    - nv (np.ndarray): ID of nodes around each cell
    - h (np.ndarray): depth (positive for water)
    - string (list): node IDs of each string   
    """    
    # Set h = 0.0 by default
    if h is None:
        h = np.zeros_like(x)

    # Write the 2dm output
    with py2dm.Writer(f2dm) as mesh:
        # Write out the cell information
        for j in range(nv.shape[0]):
            mesh.element('E3T', -1, *nv[j,:])
        # Write out the node information
        for i in range(len(x)):
            mesh.node(-1, x[i], y[i], h[i])
        # Write out the string information
        for l in range(len(string)):
            mesh.node_string(*string[l])


def read_grd(fgrd):
    """
    Read FVCOM grd file to get x, y, nv
    
    Parameters:
    - fgrd (str): Filepath to the ASCII grd file.
    
    Returns:
    - x (np.ndarray): x-coordinate
    - y (np.ndarray): y-coordinate
    - nv (np.ndarray): ID of nodes around each cell
    """    
    file_exists(fgrd)

    with open(fgrd, 'r') as file:
        # Read the node and cell numbers
        node = int(file.readline().split('=')[1].strip())
        nele = int(file.readline().split('=')[1].strip())

        # Read nv (each line with 5 integers, extract columns 1-3)
        nv = np.zeros((nele, 3), dtype=int)
        for i in range(nele):
            line = file.readline().strip().split()
            nv[i] = [int(line[1]), int(line[2]), int(line[3])]

        # Read x, y (each line with 4 floats, extract columns 1-2)
        x = np.zeros((node,), dtype=float)
        y = np.zeros((node,), dtype=float)
        for i in range(node):
            line = file.readline().strip().split()
            x[i] = float(line[1])
            y[i] = float(line[2])

    print(f'Node Number: {len(x)}')
    print(f'Cell Number: {len(nv)}')
    print(f'X range: {np.min(x)} - {np.max(x)}')
    print(f'Y range: {np.min(y)} - {np.max(y)}')

    return x, y, nv


def read_dep(fdep):
    """
    Read FVCOM dep file to get x, y, h
    
    Parameters:
    - fdep (str): Filepath to the ASCII dep file.
    
    Returns:
    - x (np.ndarray): x-coordinate
    - y (np.ndarray): y-coordinate
    - h (np.ndarray): depth (positive for water)
    """    
    file_exists(fdep)

    with open(fdep, 'r') as file:
        # Read the node number
        node = int(file.readline().split('=')[1].strip())

        # Read x, y (each line with 3 floats, extract columns 0-2)
        x = np.zeros((node,), dtype=float)
        y = np.zeros((node,), dtype=float)
        h = np.zeros((node,), dtype=float)
        for i in range(node):
            line = file.readline().strip().split()
            x[i] = float(line[0])
            y[i] = float(line[1])
            h[i] = float(line[2])

    print(f'Depth range: {np.min(h)} - {np.max(h)}')
    
    return x, y, h


def read_cor(fcor):
    """
    Read FVCOM cor file to get x, y, lat
    
    Parameters:
    - fcor (str): Filepath to the ASCII cor file.
    
    Returns:
    - x (np.ndarray): x-coordinate
    - y (np.ndarray): y-coordinate
    - lat (np.ndarray): latitude (positive for the northern hemisphere)
    """    
    file_exists(fcor)

    with open(fcor, 'r') as file:
        # Read the node number
        node = int(file.readline().split('=')[1].strip())

        # Read x, y (each line with 3 floats, extract columns 0-2)
        x = np.zeros((node,), dtype=float)
        y = np.zeros((node,), dtype=float)
        lat = np.zeros((node,), dtype=float)
        for i in range(node):
            line = file.readline().strip().split()
            x[i] = float(line[0])
            y[i] = float(line[1])
            lat[i] = float(line[2])

    print(f'Latitude range: {np.min(lat)} - {np.max(lat)}')
    
    return x, y, lat


def read_obc(fobc):
    """
    Read FVCOM obc file to get obc_node, obc_type
    
    Parameters:
    - fobc (str): Filepath to the ASCII obc file.
    
    Returns:
    - obc_node (np.ndarray): obc node ID
    - obc_type (np.ndarray): obc node type
    """    
    file_exists(fobc)

    with open(fobc, 'r') as file:
        # Read the node number
        nobc = int(file.readline().split('=')[1].strip())

        obc_node = np.zeros((nobc,), dtype=int)
        obc_type = np.zeros((nobc,), dtype=int)
        if nobc > 0:
            # Read obc_node, obc_type (each line with 3 integers, extract columns 1-2)
            for i in range(nobc):
                line = file.readline().strip().split()
                obc_node[i] = int(line[1])
                obc_type[i] = int(line[2])

    print(f'OBC number: {len(obc_node)}')
    if len(obc_node) > 0:
        print(f'obc_node range: {np.min(obc_node)} - {np.max(obc_node)}')
        print(f'obc_type range: {np.min(obc_type)} - {np.max(obc_type)}')
    
    return obc_node, obc_type


def read_spg(fspg):
    """
    Read FVCOM spg file to get obc_node, obc_type
    
    Parameters:
    - fspg (str): Filepath to the ASCII spg file.
    
    Returns:
    - spg_node (np.ndarray): spg node ID
    - spg_R (np.ndarray): spg affecting influence radius
    - spg_F (np.ndarray): spg damping coefficient 
    """    
    file_exists(fspg)

    with open(fspg, 'r') as file:
        # Read the node number
        nspg = int(file.readline().split('=')[1].strip())

        spg_node = np.zeros((nspg,), dtype=int)
        spg_R = np.zeros((nspg,), dtype=float)
        spg_F = np.zeros((nspg,), dtype=float)
        if nspg > 0:
            # Read spg_node, spg_R, spg_F (each line with int, float, float, extract columns 0-2)
            for i in range(nspg):
                line = file.readline().strip().split()
                spg_node[i] = int(line[0])
                spg_R[i] = float(line[1])
                spg_F[i] = float(line[2])

    print(f'SPG number: {len(spg_node)}')
    if len(spg_node) > 0:
        print(f'spg_node range: {np.min(spg_node)} - {np.max(spg_node)}')
        print(f'spg_R range: {np.min(spg_R)} - {np.max(spg_R)}')
        print(f'spg_F range: {np.min(spg_F)} - {np.max(spg_F)}')
    
    return spg_node, spg_R, spg_F


def read_sigma(fsigma):
    """
    Read FVCOM sigma file to get sigma
    
    Parameters:
    - fsigma (str): Filepath to the ASCII sigma file.
    
    Returns:
    - sigma (dict): sigma dict
        KB: vertical level number
        TYPE: vertical coordinate type
            (GEOMETRIC)
        SIGMA_POWER: power of the parabolic function
            (TANH)
        DU: the upper water boundary thickness
        DL: the lower water bound ary thickness
            (GENERALIZED)
        DU: the upper water boundary thickness
        DL: the lower water boundary thickness
        MIN_DEPTH: the transition depth of the hybrid coordinate
        KU: layer number in the water column of DU
        KL: layer number in the water column of DL
        ZKU: thickness of each layer defined by KU (m)
        ZKL: thickness of each layer defined by KL (m)
    """    
    file_exists(fsigma)

    # Set the default values
    KB = np.empty((0,), dtype=int)
    TYPE = ''
    SIGMA_POWER = np.empty((0,), dtype=float)
    DU = np.empty((0,), dtype=float)
    DL = np.empty((0,), dtype=float)
    MIN_DEPTH = np.empty((0,), dtype=float)
    KU = np.empty((0,), dtype=int)
    KL = np.empty((0,), dtype=int)
    ZKU = np.empty((0,), dtype=float)
    ZKL = np.empty((0,), dtype=float)
    
    with open(fsigma, 'r') as file:
        for line in file:
            line = line.strip()
            # Skip lines starting with '!'
            if line.startswith('!'):
                continue
            # Extract values
            if line.startswith('NUMBER OF SIGMA LEVELS'):
                KB = int(line.split('=')[1].strip())
            elif line.startswith('SIGMA COORDINATE TYPE'):
                TYPE = line.split('=')[1].strip()
            elif line.startswith('SIGMA POWER'):
                SIGMA_POWER = float(line.split('=')[1].strip())
            elif line.startswith('DU'):
                DU = float(line.split('=')[1].strip())
            elif line.startswith('DL'):
                DL = float(line.split('=')[1].strip())
            elif line.startswith('MIN CONSTANT DEPTH'):
                MIN_DEPTH = float(line.split('=')[1].strip())
            elif line.startswith('KU'):
                KU = int(line.split('=')[1].strip())
            elif line.startswith('KL'):
                KL = int(line.split('=')[1].strip())
            elif line.startswith('ZKU'):
                values = line.split('=')[1].strip().split()
                ZKU = np.array([float(value) for value in values])
            elif line.startswith('ZKL'):
                values = line.split('=')[1].strip().split()
                ZKL = np.array([float(value) for value in values])
    
    if TYPE == 'UNIFORM':
        sigma = {'KB': KB, 'TYPE': TYPE}
    elif TYPE == 'GEOMETRIC':
        sigma = {'KB': KB, 'TYPE': TYPE, 'SIGMA_POWER': SIGMA_POWER}
    elif TYPE == 'TANH':
        sigma = {'KB': KB, 'TYPE': TYPE, 'DU': DU, 'DL': DL}
    elif TYPE == 'GENERALIZED':
        sigma = {
            'KB': KB, 
            'TYPE': TYPE, 
            'DU': DU, 
            'DL': DL,
            'MIN_DEPTH': MIN_DEPTH,
            'KU': KU,
            'KL': KL,
            'ZKU': ZKU,
            'ZKL': ZKL
        }
    else:
        raise ValueError(f"Unknown TYPE: {TYPE}")
        
    for key, value in sigma.items():
        print(f'{key}: {value}')

    return sigma


def write_grd(fgrd, x, y, nv, h=None, fmt='16.6f'):
    """
    Write FVCOM grd file with x, y, grd, (h)
    
    Parameters:
    - fgrd (str): Filepath to the ASCII grd file.
    - x (np.ndarray): x-coordinate
    - y (np.ndarray): y-coordinate
    - nv (np.ndarray): ID of nodes around each cell
    - h (np.ndarray): depth (positive for water)
    - fmt (str): output format
    """    
    # Ensure the arrays have the same length
    assert len(x) == len(y) 
    
    # Set h = 0.0 by default
    if h is None:
        h = np.zeros_like(x)

    # Define the format for the integers
    int_format = '{:8d}'

    # Open the file in write mode
    with open(fgrd, 'w') as file:
        # Write the length of the arrays in the first line
        file.write(f'Node Number = {len(x)}\n')
        file.write(f'Cell Number = {len(nv)}\n')

        # Write nv
        for i, row in enumerate(nv, start=1):
            formatted_row = ' '.join(int_format.format(num) for num in [i] + row.tolist() + [1])
            file.write(formatted_row + '\n')

        # Write x, y
        for i, (xi, yi, hi) in enumerate(zip(x, y, h), start=1):
            file.write(f'{int_format.format(i)} {xi.item():{fmt}} {yi.item():{fmt}} {hi.item():10.2f}\n')

            
def write_dep(fdep, x, y, h, fmt='16.6f'):
    """
    Write FVCOM dep file with x, y, h
    
    Parameters:
    - fdep (str): Filepath to the ASCII dep file.
    - x (np.ndarray): x-coordinate
    - y (np.ndarray): y-coordinate
    - h (np.ndarray): depth (positive for water)
    - fmt (str): output format
    """    
    # Ensure the arrays have the same length
    assert len(x) == len(y) == len(h)

    # Open the file in write mode
    with open(fdep, 'w') as file:
        # Write the length of the arrays in the first line
        file.write(f'Node Number = {len(x)}\n')

        # Write the elements of the arrays in subsequent lines
        for xi, yi, hi in zip(x, y, h):
            file.write(f'{xi.item():{fmt}} {yi.item():{fmt}} {hi.item():{fmt}}\n')

            
def write_cor(fcor, x, y, lat, fmt='16.6f'):
    """
    Write FVCOM cor file with x, y, lat
    
    Parameters:
    - fcor (str): Filepath to the ASCII cor file.
    - x (np.ndarray): x-coordinate
    - y (np.ndarray): y-coordinate
    - lat (np.ndarray): latitude (positive for the northern hemisphere)
    - fmt (str): output format
    """    
    # Ensure the arrays have the same length
    assert len(x) == len(y) == len(lat)

    # Open the file in write mode
    with open(fcor, 'w') as file:
        # Write the length of the arrays in the first line
        file.write(f'Node Number = {len(x)}\n')

        # Write the elements of the arrays in subsequent lines
        for xi, yi, lati in zip(x, y, lat):
            file.write(f'{xi.item():{fmt}} {yi.item():{fmt}} {lati.item():{fmt}}\n')

            
def write_obc(fobc, obc_node, obc_type):
    """
    Write FVCOM obc file with obc_node, obc_type
    
    Parameters:
    - fobc (str): Filepath to the ASCII obc file.
    - obc_node (np.ndarray): obc node ID
    - obc_type (np.ndarray): obc node type
    """    
    # Ensure the arrays have the same length
    assert len(obc_node) == len(obc_type)

    # Define the format for the integers
    int_format = '{:8d}'

    # Open the file in write mode
    with open(fobc, 'w') as file:
        # Write the length of the arrays in the first line
        file.write(f'OBC Node Number = {len(obc_node)}\n')

        # Write the elements of the arrays in subsequent lines
        if len(obc_node) > 0:
            for i, (iobc_node, iobc_type) in enumerate(zip(obc_node, obc_type), start=1):
                file.write(f'{int_format.format(i)} {int_format.format(iobc_node.item())} {int_format.format(iobc_type.item())}\n')

    
def write_spg(fspg, spg_node, spg_R, spg_F, fmt='16.6f'):
    """
    Write FVCOM spg file with spg_node, spg_R, spg_F
    
    Parameters:
    - fspg (str): Filepath to the ASCII spg file.
    - spg_node (np.ndarray): spg node ID
    - spg_R (np.ndarray): spg affecting influence radius
    - spg_F (np.ndarray): spg damping coefficient 
    """    
    # Ensure the arrays have the same length
    assert len(spg_node) == len(spg_R) == len(spg_F)

    # Define the format for the integers
    int_format = '{:8d}'

    # Open the file in write mode
    with open(fspg, 'w') as file:
        # Write the length of the arrays in the first line
        file.write(f'Sponge Node Number = {len(spg_node)}\n')
        
        # Write the elements of the arrays in subsequent lines
        if len(spg_node) > 0:
            for (ispg_node, ispg_R, ispg_F) in zip(spg_node, spg_R, spg_F):
                file.write(f'{int_format.format(ispg_node)} {ispg_R.item():{fmt}} {ispg_F.item():{fmt}}\n')

                
def write_sigma(fsigma, sigma):
    """
    Write FVCOM sigma file 
    
    Parameters:
    - fsigma (str): Filepath to the ASCII sigma file.
    - sigma (dict): sigma dict
        KB: vertical level number
        TYPE: vertical coordinate type
            (GEOMETRIC)
        SIGMA_POWER: power of the parabolic function
            (TANH)
        DU: the upper water boundary thickness
        DL: the lower water boundary thickness
            (GENERALIZED)
        DU: the upper water boundary thickness
        DL: the lower water boundary thickness
        MIN_DEPTH: the transition depth of the hybrid coordinate
        KU: layer number in the water column of DU
        KL: layer number in the water column of DL
        ZKU: thickness of each layer defined by KU (m)
        ZKL: thickness of each layer defined by KL (m)
    """    

    # Define the format for the integers
    int_format = '{:8d}'

    # Open the file in write mode
    with open(fsigma, 'w') as file:
        file.write(f'NUMBER OF SIGMA LEVELS = {sigma["KB"]}\n')
        file.write(f'SIGMA COORDINATE TYPE = {sigma["TYPE"]}\n')
        
        if sigma["TYPE"] == 'UNIFORM':
            file.write('\n')
        elif sigma["TYPE"] == 'GEOMETRIC':
            file.write(f'SIGMA POWER = {sigma["SIGMA_POWER"]}\n')
        elif sigma["TYPE"] == 'TANH':
            file.write(f'DU = {sigma["DU"]}\n')
            file.write(f'DL = {sigma["DL"]}\n')
        elif sigma["TYPE"] == 'GENERALIZED':
            file.write(f'DU = {sigma["DU"]}\n')
            file.write(f'DL = {sigma["DL"]}\n')
            file.write(f'MIN CONSTANT DEPTH = {sigma["MIN_DEPTH"]}\n')
            file.write(f'KU = {sigma["KU"]}\n')
            file.write(f'KL = {sigma["KL"]}\n')
            values = ' '.join(f'{value:.2f}' for value in sigma["ZKU"])
            file.write(f'ZKU = {values}\n')
            values = ' '.join(f'{value:.2f}' for value in sigma["ZKL"])
            file.write(f'ZKL = {values}\n')
        else:
            raise ValueError(f'Unknown TYPE: {sigma["TYPE"]}')


def nc_read_var(fnc, varname, *args, squeeze=True):
    """
    Read a variable or a subset of a variable from a NetCDF file.

    Parameters:
    fnc (str): Path to the NetCDF file.
    varname (str): Name of the variable to read.
    *args: Optional start, count, and stride arguments.
    squeeze (bool): Whether to squeeze single-dimensional entries from the shape of the array.

    Returns:
    numpy.ndarray: Data of the variable if it exists.
    None: If the variable does not exist or an error occurs.
    """
    start, end, stride = None, None, None

    if len(args) > 0:
        start = args[0][::-1]
    if len(args) > 1:
        end = args[1][::-1]
    if len(args) > 2:
        stride = args[2][::-1]

    try:
        with nc.Dataset(fnc, 'r') as ds:
            if varname in ds.variables:
                var = ds.variables[varname]
                var_shape = var.shape
                
                # Handle slicing if start, end, and stride are provided
                if start is None and end is None and stride is None:
                    data = var[:]
                else:
                    slices = []
                    for i, dim in enumerate(var_shape):
                        #s = start[i] if start and i < len(start) and start[i] != np.inf else None
                        s = start[i] - 1 if start and i < len(start) and start[i] != None else None
                        c = end[i] if end and i < len(end) and end[i] != None else None
                        e = None if s is None or c is None else s + c
                        st = stride[i] if stride and i < len(stride) else None
                        slices.append(slice(s, e, st))
                    data = var[tuple(slices)]

                data = np.transpose(data, range(len(np.shape(data))-1, -1, -1))

                # Squeeze the data to remove single-dimensional entries if squeeze option is True
                if squeeze:
                    data = np.squeeze(data)                

                return data
            else:
                print(f"Variable '{varname}' not found in the NetCDF file.")
                return None
    except Exception as e:
        print(f"Error reading variable '{varname}' from NetCDF file: {e}")
        return None


def nc_check_var(fnc, varname):
    """
    Check if a variable exists in a NetCDF file.

    Parameters:
    fnc (str): Path to the NetCDF file.
    varname (str): Name of the variable to check.

    Returns:
    bool: True if the variable exists, False otherwise.
    """
    try:
        with nc.Dataset(fnc, 'r') as ds:
            return varname in ds.variables
    except Exception as e:
        #print(f"Error: {e}")
        return False


def nc_info(fnc, *args):
    """
    Display information about the NetCDF file.

    Parameters:
    - fnc (str): Path to the NetCDF file.
    - args: Additional arguments to specify the type of information to display.
      Can be 'var' to show information of a specific variable or 
      'global' to show global attributes. If no additional argument is provided,
      shows dimensions and variables.
    """
    try:
        # Open the NetCDF file
        ds = nc.Dataset(fnc, 'r')
        
        if not args:
            # Print dimensions
            print("---- Dimensions ----")
            for dim_name, dim in ds.dimensions.items():
                print(f"{dim_name}: {len(dim)}")
            
            # Print variables
            print("---- Variables ----")
            for var_name, var in ds.variables.items():
                print(f"{var_name}: {var.dimensions[::-1]} {var.shape[::-1]}")
            print("\n")
            
        elif args[0] == 'global':
            # Print global attributes
            print("---- Global Attributes ----")
            for name in ds.ncattrs():
                print(f"{name}: {ds.getncattr(name)}")
            print("\n")
            
        else:
            # Print information of the specified variable
            var_name = args[0]
            if var_name in ds.variables:
                var = ds.variables[var_name]
                print(f"Variable: {var_name}")
                print(f"Dimensions: {var.dimensions[::-1]}")
                print(f"Shape: {var.shape[::-1]}")
                print(f"Attributes:")
                for attr_name in var.ncattrs():
                    print(f"    {attr_name}: {var.getncattr(attr_name)}")
                print("\n")
            else:
                print(f"Variable '{var_name}' not found in the NetCDF file.")
        
        # Close the dataset
        ds.close()
        
    except Exception as e:
        print(f"Error: {e}")


def calc_boundary(nv):
    """
    Calculate the boundary lines of a mesh and form closed polygons.

    Parameters:
    - nv: numpy.ndarray
        Array of shape (n_cells, 3) containing the node indices of each cell.

    Returns:
    - ordered_polygons: list of numpy.ndarray
        List of ordered polygons, where each polygon is represented as an array
        of node indices forming a closed loop.
    """
    # Devide each cell into three liens
    lines = np.vstack((nv[:, [0, 1]], nv[:, [0, 2]], nv[:, [1, 2]]))
    # Sort the ID in each line
    lines = np.sort(lines, axis=1)
    # Seperate lines into inner lines and boundary lines
    unique_lines, counts = np.unique(lines, axis=0, return_counts=True)
    #inner_lines = unique_lines[counts == 2]
    boundary_lines = unique_lines[counts == 1]
    
    # Create an adjacency list
    adjacency_list = {}
    for line in boundary_lines:
        if line[0] not in adjacency_list:
            adjacency_list[line[0]] = []
        if line[1] not in adjacency_list:
            adjacency_list[line[1]] = []
        adjacency_list[line[0]].append(line[1])
        adjacency_list[line[1]].append(line[0])

    # Find all connected components (polygons)
    visited = set()
    polygons = []
    for node in adjacency_list.keys():
        if node not in visited:
            component = []
            queue = [node]
            while queue:
                current_node = queue.pop(0)
                if current_node not in visited:
                    visited.add(current_node)
                    component.append(current_node)
                    for neighbor in adjacency_list[current_node]:
                        if neighbor not in visited:
                            queue.append(neighbor)
            polygons.append(component)

    # Order the lines of each polygon to form closed loops
    ordered_polygons = []
    for polygon in polygons:
        ordered_polygon = []
        start_node = polygon[0]
        current_node = start_node
        visited = set()

        while True:
            ordered_polygon.append(current_node)
            visited.add(current_node)
            neighbors = adjacency_list[current_node]
            next_node = None
            for neighbor in neighbors:
                if neighbor not in visited:
                    next_node = neighbor
                    break
            if next_node is None:
                break
            current_node = next_node
            if current_node == start_node:
                ordered_polygon.append(start_node)
                break

        # Ensure the loop is closed
        if ordered_polygon[0] != ordered_polygon[-1]:
            ordered_polygon.append(ordered_polygon[0])
        
        ordered_polygons.append(np.array(ordered_polygon))

    return ordered_polygons


def proj(src_epsg, tgt_epsg, *args):
    """
    Convert coordinates from one coordinate system to another.

    Parameters:
    - src_epsg: EPSG code number or 'earth' for the source coordinate system
    - tgt_epsg: EPSG code number or 'earth' for the target coordinate system
    - x: X coordinate in the source coordinate system
    - y: Y coordinate in the source coordinate system
    - z: Z coordinate in the source coordinate system

    Returns:
    - (x_tgt, y_tgt): Tuple of coordinates in the target coordinate system
    """
    # Handle 'earth' input
    if src_epsg == 'earth':
        src_epsg = '4326'
    if tgt_epsg == 'earth':
        tgt_epsg = '4326'

    src_epsg = f'epsg:{src_epsg}'
    tgt_epsg = f'epsg:{tgt_epsg}'

    # Create a Transformer object
    transformer = pyproj.Transformer.from_crs(src_epsg, tgt_epsg, always_xy=True)

    # Transform coordinates
    if len(args) == 2:
        x_tgt, y_tgt = transformer.transform(*args)
        return x_tgt, y_tgt
    elif len(args) == 3:
        x_tgt, y_tgt, h_tgt = transformer.transform(*args)
        return x_tgt, y_tgt, h_tgt


def calc_sigma(sigma):
    """
    Calculate the sigma levels for oceanographic models.

    Parameters:
    ----------
    h : array-like
        Array of depths or heights. Should be a 1-D array or list.
    sigma : dict
        Dictionary containing parameters for sigma level calculation. The dictionary should include:
        - 'nz': Number of sigma levels (integer).
        - 'type': Type of sigma level calculation. Options are 'UNIFORM', 'GEOMETRIC', 'TANH', or 'GENERALIZED' (string).
        - 'sigma_power': Power for GEOMETRIC type (float, optional, required if type is 'GEOMETRIC').
        - 'dl': Depth for TANH type (float, optional, required if type is 'TANH' or 'GENERALIZED').
        - 'du': Depth for TANH type (float, optional, required if type is 'TANH' or 'GENERALIZED').
        - 'min_const_depth': Minimum constant depth for GENERALIZED type (float, optional, required if type is 'GENERALIZED').
        - 'ku': Number of upper levels for GENERALIZED type (integer, optional, required if type is 'GENERALIZED').
        - 'kl': Number of lower levels for GENERALIZED type (integer, optional, required if type is 'GENERALIZED').
        - 'zku': List of depth increments for upper levels for GENERALIZED type (list or array, optional, required if type is 'GENERALIZED').
        - 'zkl': List of depth increments for lower levels for GENERALIZED type (list or array, optional, required if type is 'GENERALIZED').

    Returns:
    -------
    siglay : ndarray
        Sigma levels at the layer midpoints.
    siglev : ndarray
        Sigma levels at the vertical levels.
    deplay : ndarray
        Depth values at the layer midpoints.
    deplev : ndarray
        Depth values at the vertical levels.
    """
    kb = sigma['nz']
    kbm1 = kb - 1
    siglev = np.zeros((len(h), kb))
    h = np.array(h).reshape(-1)

    if sigma['type'].upper() == 'UNIFORM':
        for iz in range(kb):
            siglev[:, iz] = -(iz) / (kb - 1)

    elif sigma['type'].upper() == 'GEOMETRIC':
        if kb % 2 == 0:
            raise ValueError('When using GEOMETRIC, nz has to be odd.')
        for iz in range((kb + 1) // 2):
            siglev[:, iz] = -((iz) / ((kb + 1) / 2 - 1))**sigma['sigma_power'] / 2
        for iz in range((kb + 1) // 2, kb):
            siglev[:, iz] = -((kb - iz) / ((kb + 1) / 2 - 1))**sigma['sigma_power'] / 2 - 1

    elif sigma['type'].upper() == 'TANH':
        for iz in range(kbm1):
            x1 = sigma['dl'] + sigma['du']
            x1 = x1 * (kbm1 - iz) / kbm1
            x1 = x1 - sigma['dl']
            x1 = np.tanh(x1)
            x2 = np.tanh(sigma['dl'])
            x3 = x2 + np.tanh(sigma['du'])

            siglev[:, iz + 1] = (x1 + x2) / x3 - 1

    elif sigma['type'].upper() == 'GENERALIZED':
        for i in range(len(h)):
            if h[i] < sigma['min_const_depth']:
                DL2 = 0.001
                DU2 = 0.001
                for iz in range(kbm1):
                    x1 = DL2 + DU2
                    x1 = x1 * (kbm1 - iz) / kbm1
                    x1 = x1 - DL2
                    x1 = np.tanh(x1)
                    x2 = np.tanh(DL2)
                    x3 = x2 + np.tanh(DU2)

                    siglev[i, iz + 1] = (x1 + x2) / x3 - 1
            else:
                DR = (h[i] - sigma['du'] - sigma['dl']) / h[i] / (kb - sigma['ku'] - sigma['kl'] - 1)
                for iz in range(1, sigma['ku'] + 1):  # Upper
                    siglev[i, iz] = siglev[i, iz - 1] - sigma['zku'][iz - 1] / h[i]
                for iz in range(sigma['ku'] + 1, kb - sigma['kl']):  # Middle
                    siglev[i, iz] = siglev[i, iz - 1] - DR
                KK = 0
                for iz in range(kb - sigma['kl'], kb):  # Lower
                    KK += 1
                    siglev[i, iz] = siglev[i, iz - 1] - sigma['zkl'][KK - 1] / h[i]
    
    siglay = (siglev[:, :kbm1] + siglev[:, 1:]) / 2
    
    return siglay


def calc_nbve(nv):
    """
    Calculate the nbve (cells around nodes)
  
    Parameters:
    - nv: nodes around cells (index starting from 1)

    Returns:
    - nbve: cells around nodes (index starting from 1)
    """
    node = np.max(nv)
    MX_NBR_ELEM = 9

    nbve = np.zeros((node, MX_NBR_ELEM), dtype=int)

    for i in range(node):
        row = np.where(np.isin(nv, i+1))[0] + 1
        row = np.pad(row, (0, MX_NBR_ELEM-len(row)), mode='constant', constant_values=0)
        nbve[i, :] = row

    non_zero_columns = np.any(nbve!=0, axis=0)
    nbve = nbve[:, non_zero_columns]
    
    return nbve


def load_grid(*args, crs=None):
    """
    Load and process grid data from various file formats or directly from input arrays.

    Parameters:
    - *args: Variable length argument list.
        - If a single argument, it should be the file name (str) of the grid data.
        - If three arguments, they should be arrays (x, y, nv) representing node coordinates and connectivity.
        - If four arguments, they should be arrays (x, y, nv, h) including the bathymetry data.
    - crs: Coordinate reference system (optional). 
        EPSG code or 'earth' for converting coordinates to geographic (EPSG:4326).

    Returns:
    - grid_data: dict
        Dictionary containing the following keys:
        - 'node': int, number of nodes
        - 'nele': int, number of elements
        - 'x': numpy.ndarray, x coordinates of nodes
        - 'y': numpy.ndarray, y coordinates of nodes
        - 'nv': numpy.ndarray, node connectivity array
        - 'h': numpy.ndarray, bathymetry data
        - 'xc': numpy.ndarray, x coordinates of element centers
        - 'yc': numpy.ndarray, y coordinates of element centers
        - 'bdy': list of numpy.ndarray, boundary node indices forming closed loops
        - 'bdy_x': list of numpy.ndarray, x coordinates of boundary nodes
        - 'bdy_y': list of numpy.ndarray, y coordinates of boundary nodes
        - 'crs': Cartopy CRS, coordinate reference system used for plotting
    """    
    if len(args) < 3:
        fgrid = args[0]
        
        if fgrid.endswith('.nc'):
            x = nc_read_var(fgrid, 'x')
            y = nc_read_var(fgrid, 'y')
            nv = nc_read_var(fgrid, 'nv')
            #nv = np.transpose(nv, range(len(np.shape(nv))-1, -1, -1))
            if nc_check_var(fgrid, 'h'):
                h = nc_read_var(fgrid, 'h')
            else:
                h = x * 0.0
            if nc_check_var(fgrid, 'siglay'):
                siglay = nc_read_var(fgrid, 'siglay')
            else:
                siglay = x * 0.0

        elif fgrid.endswith('.dat'):
            x, y, nv = read_grd(fgrid)
            h = x * 0.0
            siglay = x * 0.0
        elif fgrid.endswith('.2dm'):
            x, y, nv, h = read_2dm_quick(fgrid)
            siglay = x * 0.0
        else:
            raise ValueError(f'Unknown input format: {fgrid}')
        
        if len(args) == 2:
            fsigma = args[1]
            sigma = read_sigma(fsigma)
            siglay = calc_sigma(sigma)
            
    elif len(args) == 3:
        x, y, nv = args
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        nv = np.array(nv, dtype=int)
        h = x * 0.0
        siglay = x * 0.0
    elif len(args) == 4:
        x, y, nv, h = args
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        nv = np.array(nv, dtype=int)
        h = np.array(h, dtype=float)
        siglay = x * 0.0
    elif len(args) == 5:
        x, y, nv, h, sigma = args
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        nv = np.array(nv, dtype=int)
        h = np.array(h, dtype=float)
        siglay = calc_sigma(sigma)
    else:
        raise ValueError(f'Wronge inputs. Use either (fin) or (x, y, nv)')

    if crs is not None:
        x, y = proj(crs, 'earth', x, y)

    node = len(x)
    nele = len(nv)
    kbm1 = int(np.size(siglay) / node)

    xc = np.mean(x[nv-1], axis=1)
    yc = np.mean(y[nv-1], axis=1)

    deplay = -siglay * np.reshape(h, (-1, 1))

    bdy = calc_boundary(nv)
    bdy_x = []
    bdy_y = []
    for ibdy in bdy:
        bdy_x.append(x[ibdy - 1])
        bdy_y.append(y[ibdy - 1])

    # Calculate NBVE
    nbve = calc_nbve(nv)

    if crs is not None:
        crs = ccrs.PlateCarree()
        
    return {'node': node, 'nele': nele, 'kbm1': kbm1,
            'x': x, 'y': y, 'nv': nv, 'h': h, 
            'xc': xc, 'yc': yc, 
            'siglay': siglay, 'deplay': deplay,
            'bdy': bdy, 'bdy_x': bdy_x, 'bdy_y': bdy_y,
            'nbve': nbve,
            'crs':crs}


def load_data(fnc, varname, *args, squeeze=True):
    """
    Load data from a NetCDF file.

    Parameters:
    - fnc: str
        File name of the NetCDF file.
    - varname: str
        Name of the variable to read from the NetCDF file.
    - *args: Variable length argument list.
        Additional arguments to pass to the `nc_read_var` function.
    - squeeze: bool, optional
        Whether to squeeze single-dimensional entries from the shape of the array (default is True).

    Returns:
    - data: numpy.ndarray
        The variable data read from the NetCDF file.
    """
    return nc_read_var(fnc, varname, *args, squeeze=squeeze)


def rarefy(*args):
    """
    Rarefy a set of points to ensure a minimum distance between selected points.

    Parameters:
    - *args: Variable length argument list.
        - If the first argument is a dictionary:
            - args[0]: dict
                Dictionary containing point coordinates.
            - args[1]: str
                Type of points to consider ('node' or 'cell').
            - args[2]: float
                Minimum distance (res) between selected points.
        - Otherwise:
            - args[0]: numpy.ndarray
                Array of x coordinates.
            - args[1]: numpy.ndarray
                Array of y coordinates.
            - args[2]: float
                Minimum distance (res) between selected points.

    Returns:
    - id: numpy.ndarray
        Array of indices of the selected points, starting from 1.
    """
    if type(args[0]) == dict:
        if args[1] == 'node':
            x = args[0]['x']
            y = args[0]['y']
        elif args[1] == 'cell':
            x = args[0]['xc']
            y = args[0]['yc']
        else:
            raise ValueError(f"Unknown type '{args[1]}'. Use node or cell")
        res = args[2]
    else:
        x = args[0]
        y = args[1]
        res = args[2]
    
    # Combine x and y into a single array of points
    points = np.column_stack((x, y))
    
    # Create a KDTree for efficient spatial searching
    tree = KDTree(points)
    
    # Initialize a list to store selected point indices
    id = []
    
    # Array to keep track of points that are already considered
    selected = np.zeros(len(points), dtype=bool)
    
    for i, point in enumerate(points):
        if not selected[i]:
            id.append(i + 1)    # Add 1 to make indices start from 1
            # Find all points within min_distance of this point
            indices = tree.query_ball_point(point, res)
            selected[indices] = True  # Mark these points as selected
            
    id = np.array(id)
    return id


def image_spoof(self, tile):
    """
    Reformat web requests from OSM for Cartopy.

    Parameters:
    - tile: tuple
        Tile information needed to fetch the image from the street map API.

    Returns:
    - img: PIL.Image.Image
        The image fetched from the street map API, reformatted for Cartopy.
    - extent: list
        The extent of the tile in the format required by Cartopy.
    - origin: str
        The origin of the tile, set to 'lower' for Cartopy.

    Description:
    This function reformats web requests from OpenStreetMap (OSM) for use with Cartopy.
    It is heavily based on code by Joshua Hrisko:
    https://makersportal.com/blog/2020/4/24/geographic-visualizations-in-python-with-cartopy

    Steps:
    1. Get the URL of the street map API.
    2. Start a request to the URL.
    3. Add a user agent to the request.
    4. Fetch the image data from the URL.
    5. Open the image with PIL and convert it to the desired format.
    6. Return the image, tile extent, and origin in the format required by Cartopy.
    """
    url = self._image_url(tile)                # get the url of the street map API
    req = Request(url)                         # start request
    req.add_header('User-agent','Anaconda 3')  # add user agent to request
    fh = urlopen(req) 
    im_data = io.BytesIO(fh.read())            # get image
    fh.close()                                 # close url
    img = Image.open(im_data)                  # open image with PIL
    img = img.convert(self.desired_tile_form)  # set image format
    return img, self.tileextent(tile), 'lower' # reformat for cartopy


def oms_image(style='satellite'):
    """
    Fetch and reformat an image from OpenStreetMap or satellite sources for Cartopy.

    Parameters:
    - style: str, optional
        The style of the map to fetch ('map' or 'satellite'). Defaults to 'satellite'.

    Returns:
    - img: object
        The image fetched from the specified source, reformatted for Cartopy.

    Description:
    This function fetches an image from OpenStreetMap or a satellite source based on the specified style.
    It modifies the `get_image` method of the relevant tile source class to use a custom `image_spoof` function
    for reformatted web requests.

    Steps:
    1. If `style` is 'map', use OpenStreetMap style and modify `get_image` for map spoofing.
    2. If `style` is 'satellite', use satellite style and modify `get_image` for satellite spoofing.
    3. If `style` is not recognized, print an error message.
    4. Return the fetched and reformatted image.
    
    Note:
    This function is based on modifications from:
    https://www.theurbanist.com.au/2021/03/plotting-openstreetmap-images-with-cartopy/
    """
    if style=='map':
        ## MAP STYLE
        cimgt.OSM.get_image = image_spoof # reformat web request for street map spoofing
        img = cimgt.OSM() # spoofed, downloaded street map
    elif style =='satellite':
        # SATELLITE STYLE
        cimgt.QuadtreeTiles.get_image = image_spoof # reformat web request for street map spoofing
        img = cimgt.QuadtreeTiles() # spoofed, downloaded street map
    else:
        print('no valid style')
    return img


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth's surface.

    Parameters:
    - lat1: float
        Latitude of the first point in degrees.
    - lon1: float
        Longitude of the first point in degrees.
    - lat2: float
        Latitude of the second point in degrees.
    - lon2: float
        Longitude of the second point in degrees.

    Returns:
    - distance: float
        The great-circle distance between the two points in meters.

    Description:
    This function uses the Haversine formula to calculate the shortest distance over the Earth's surface
    between two points specified by their latitude and longitude coordinates. The distance is computed
    in meters based on the Earth's radius.

    Formula:
    - Convert latitude and longitude from degrees to radians.
    - Apply the Haversine formula to compute the distance.
    """
    # Radius of the Earth in meters
    R = 6378137

    # Convert latitude and longitude from degrees to radians
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    # Haversine formula
    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c  # Output distance in meters
    return distance


def figure_scale(xlims, ylims):
    """
    Determine a scale factor for plotting based on the specified geographic limits.

    Parameters:
    - xlims: tuple of float
        Longitude limits of the plot area, in the format (lon1, lon2).
    - ylims: tuple of float
        Latitude limits of the plot area, in the format (lat1, lat2).

    Returns:
    - scale: int
        The scale factor for plotting, adjusted to ensure visibility of the plotted area.

    Description:
    This function calculates a scale factor for plotting based on the geographic limits of the plot area.
    It computes the distance in meters for the x and y dimensions of the plot area using the Haversine formula,
    then determines a suitable scale factor to ensure the plot is appropriately sized. The scale factor is capped
    at a maximum value to prevent excessively large scales.

    Steps:
    1. Calculate the x-length (distance between the bottom and top latitude at the middle longitude).
    2. Calculate the y-length (distance between the left and right longitude at the middle latitude).
    3. Determine the radius as half of the smaller of the x-length or y-length.
    4. Compute the scale factor based on the radius.
    5. Cap the scale factor to a maximum value if necessary.
    """
    lon1, lon2 = xlims
    lat1, lat2 = ylims
    
    # x-length: distance between bottom and top latitude at the middle longitude
    mid_lon = (lon1 + lon2) / 2
    x_length = haversine_distance(lat1, mid_lon, lat2, mid_lon)

    # y-length: distance between left and right longitude at the middle latitude
    mid_lat = (lat1 + lat2) / 2
    y_length = haversine_distance(mid_lat, lon1, mid_lat, lon2)

    radius = (x_length < y_length) and x_length or y_length
    radius = radius / 2
    scale = int(120/np.log(radius))
    scale = (scale<20) and scale or 19
    
    return scale


def limit(f):
    """
    Determine the geographic limits of a dataset.

    Parameters:
    - f: dict
        Dictionary containing 'x' and 'y' coordinates as numpy arrays.

    Returns:
    - xlims: list of float
        Longitude limits of the dataset, in the format [lon_min, lon_max].
    - ylims: list of float
        Latitude limits of the dataset, in the format [lat_min, lat_max].

    Description:
    This function calculates the geographic limits of a dataset by determining the minimum and maximum
    values of the x and y coordinates. The x coordinates represent longitudes and the y coordinates
    represent latitudes. The function returns these limits as lists for use in plotting or other analyses.
    """
    xlims = [np.min(f['x']), np.max(f['x'])]
    ylims = [np.min(f['y']), np.max(f['y'])]
    return xlims, ylims


def plane_limit(ax, *args):
    """
    Set the limits for the x and y axes of a plot based on input data or provided limits.

    Parameters:
    - ax: matplotlib.axes.Axes or cartopy.mpl.geoaxes.GeoAxes
        The axes object to set the limits for.
    - *args: Variable length argument list.
        - If one argument, it should be a dictionary with 'x' and 'y' coordinates to determine limits.
        - If two arguments, they should be lists or arrays specifying the x and y limits directly.

    Description:
    This function sets the limits for the x and y axes of a plot based on the provided data or limits.
    It handles both regular Matplotlib axes and Cartopy GeoAxes. The function also ensures that the aspect
    ratio of the plot is set to 'equal' for accurate representation of the data.

    Steps:
    1. Determine the x and y limits from the input arguments.
    2. Set the extent of Cartopy GeoAxes or the x and y limits of regular Matplotlib axes.
    3. Set the aspect ratio of the axes to 'equal'.
    """
    if len(args) == 1:
        xlims, ylims = limit(args[0])
    elif len(args) == 2:
        xlims, ylims = args
    
    if isinstance(ax, GeoAxes):
        ax.set_extent([*xlims, *ylims])
    else:
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
    ax.set_aspect('equal')


def plane_mesh(ax, f, id=None, *args, **kwargs):
    """
    Plot a triangular mesh on the provided axes.

    Parameters:
    - ax: matplotlib.axes.Axes
        The axes object to plot the mesh on.
    - f: dict
        Dictionary containing 'x', 'y', and 'nv' arrays:
            - 'x': Array of x coordinates.
            - 'y': Array of y coordinates.
            - 'nv': Array of node indices defining the mesh connectivity.
    - id: numpy.ndarray, optional
        Array of element indices to include in the plot. If None, all elements are included (default is None).
    - *args: Variable length argument list.
        Additional arguments to pass to `triplot`.
    - **kwargs: Keyword arguments.
        Additional keyword arguments to pass to `triplot`.

    Returns:
    - handle: matplotlib.lines.Line2D
        The handle to the plotted lines.

    Description:
    This function plots a triangular mesh using the `triplot` method on the specified axes. It allows
    for plotting a subset of elements based on the provided indices. If no indices are provided, all
    elements are plotted. The function returns the handle to the plotted lines, which can be used for
    further customization or reference.
    """
    if id is None:
        id = np.arange(f['nele']) + 1
    handle = ax.triplot(f['x'], f['y'], f['nv'][id-1,:]-1, 'k-', lw=0.1, *args, **kwargs)

    return handle


def plane_image(ax,f, var, mask=None, *args, **kwargs):
    """
    Plot a filled contour map on the provided axes based on variable data.

    Parameters:
    - ax: matplotlib.axes.Axes
        The axes object to plot the filled contour map on.
    - f: dict
        Dictionary containing 'x', 'y', and 'nv' arrays:
            - 'x': Array of x coordinates.
            - 'y': Array of y coordinates.
            - 'nv': Array of node indices defining the mesh connectivity.
    - var: numpy.ndarray
        Array of variable values to be contoured.
    - *args: Variable length argument list.
        Additional arguments to pass to `tricontourf`.
    - **kwargs: Keyword arguments.
        Additional keyword arguments to pass to `tricontourf`.

    Returns:
    - handle: matplotlib.tricontour.QuadContourSet
        The handle to the filled contour map.

    Description:
    This function plots a filled contour map on the 
    specified axes using the `tricontourf` method. It
    visualizes the values of a given variable (`var`) 
    over a triangular mesh defined by the coordinates
    and connectivity provided in the dictionary `f`. 
    The function returns the handle to the plotted
    contour map, which can be used for further 
    customization or reference.
    """

    if var.shape[0] == f['node']:
        var = var
    elif var.shape[0] == f['nele']:
        var = interp_cell2node(f, var)
    else:
        raise ValueError(f"Wrong length of the first dimension: {var.shape[0]}.")

    if mask is None:
        handle = ax.tricontourf(f['x'], f['y'], f['nv']-1, var, *args, **kwargs)
    else:
        handle = ax.tricontourf(f['x'], f['y'], f['nv']-1, var, mask=mask, *args, **kwargs)

    return handle


def plane_contour(ax,f, var, colors='k', *args, **kwargs):
    handle = ax.tricontour(f['x'], f['y'], f['nv']-1, var,  
                           colors=colors, 
                           *args, **kwargs)

    return handle


def plane_node(ax, f, id=None, color='k', marker='.', linestyle='', *args, **kwargs):
    """
    Plot a contour map on the provided axes based on variable data.

    Parameters:
    - ax: matplotlib.axes.Axes
        The axes object to plot the contour map on.
    - f: dict
        Dictionary containing 'x', 'y', and 'nv' arrays:
            - 'x': Array of x coordinates.
            - 'y': Array of y coordinates.
            - 'nv': Array of node indices defining the mesh connectivity.
    - var: numpy.ndarray
        Array of variable values to be contoured.
    - *args: Variable length argument list.
        Additional arguments to pass to `tricontour`.
    - **kwargs: Keyword arguments.
        Additional keyword arguments to pass to `tricontour`.

    Returns:
    - handle: matplotlib.tricontour.TriContourSet
        The handle to the contour map.

    Description:
    This function plots a contour map on the specified axes 
    using the `tricontour` method. It visualizes the values 
    of a given variable (`var`) over a triangular mesh 
    defined by the coordinates and connectivity provided in 
    the dictionary `f`. The function returns the handle to 
    the plotted contour map, which can be used for further 
    customization or reference.
    """
    if id is None:
        id = np.arange(f['node']) + 1
    if isinstance(id, range):
        id = np.array(id)
        
    handle = ax.plot(f['x'][id-1], f['y'][id-1], 
                     color=color, marker=marker, linestyle=linestyle,
                     *args, **kwargs)

    return handle


def plane_cell(ax, f, id=None, color='k', marker='.', linestyle='', *args, **kwargs):
    """
    Plot the cell centers on the provided axes.

    Parameters:
    - ax: matplotlib.axes.Axes
        The axes object to plot the cell centers on.
    - f: dict
        Dictionary containing 'xc' and 'yc' arrays:
            - 'xc': Array of x coordinates of the cell centers.
            - 'yc': Array of y coordinates of the cell centers.
            - 'nele': Number of cells.
    - id: numpy.ndarray, optional
        Array of cell indices to include in the plot. If None, all cells are included (default is None).
    - *args: Variable length argument list.
        Additional arguments to pass to `plot`.
    - **kwargs: Keyword arguments.
        Additional keyword arguments to pass to `plot`.

    Returns:
    - handle: matplotlib.lines.Line2D
        The handle to the plotted cell centers.

    Description:
    This function plots the centers of cells on the specified 
    axes using the `plot` method. It allows for plotting a 
    subset of cell centers based on the provided indices. If 
    no indices are provided, all cell centers are plotted. 
    The function returns the handle to the plotted points, 
    which can be used for further customization or reference.
    """
    if id is None:
        id = np.arange(f['nele']) + 1
    if isinstance(id, range):
        id = np.array(id)

    handle = ax.plot(f['xc'][id-1], f['yc'][id-1],
                     color=color, marker=marker, linestyle=linestyle,
                     *args, **kwargs)

    return handle


def plane_coast(ax, res='10m'):
    """
    Add coastal and geographic features to the provided axes.

    Parameters:
    - ax: cartopy.mpl.geoaxes.GeoAxes
        The Cartopy GeoAxes object to which the features will be added.
    - res: str, optional
        Resolution of the features to be added. Options include '10m', '50m', and '110m'. Default is '10m'.

    Description:
    This function adds various geographic features to the 
    specified GeoAxes using Cartopy. It includes land, lakes, 
    rivers, and state boundaries with a specified resolution. 
    The function customizes the appearance of these features, 
    such as line width, but does not add ocean or border 
    features by default.
    """
    ax.add_feature(cfeature.LAND)
    
    #ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAKES, lw=0.3)
    ax.add_feature(cfeature.RIVERS.with_scale(res), lw=0.3)
    
    #ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.STATES.with_scale(res), lw=0.3)


def plane_street(ax, scale=None, style='map'):
    """
    Add a street map or satellite image to the provided axes.

    Parameters:
    - ax: matplotlib.axes.Axes or cartopy.mpl.geoaxes.GeoAxes
        The axes object to which the street map or satellite image will be added.
    - scale: int, optional
        The scale of the image. If None, the scale is calculated based on the current extent of the axes.
    - style: str, optional
        The style of the image to add. Options include 'map' for street maps and 'satellite' for satellite imagery.
        Default is 'map'.

    Description:
    This function adds a street map or satellite image to the 
    specified axes using Cartopy. If the `scale` is not provided, 
    it is calculated based on the current extent of the axes. The 
    function retrieves the appropriate image based on the 
    specified style and adds it to the axes.
    """
    if scale is None:
        xlims = ax.get_extent()[0:2]
        ylims = ax.get_extent()[2:4]
        scale = figure_scale(xlims, ylims)

    img = oms_image(style=style)
    
    ax.add_image(img, scale)


def plane_boundary(ax, f, color='k', linestyle='-',*args, **kwargs):
    """
    Plot the boundaries of polygons on the provided axes.

    Parameters:
    - ax: matplotlib.axes.Axes or cartopy.mpl.geoaxes.GeoAxes
        The axes object to plot the boundaries on.
    - f: dict
        Dictionary containing 'bdy', 'bdy_x', and 'bdy_y' arrays:
            - 'bdy': List of boundary point indices for each polygon.
            - 'bdy_x': List of x coordinates for each boundary.
            - 'bdy_y': List of y coordinates for each boundary.
    - *args: Variable length argument list.
        Additional arguments to pass to `plot`.
    - **kwargs: Keyword arguments.
        Additional keyword arguments to pass to `plot`.

    Description:
    This function plots the boundaries of polygons on the specified 
    axes. It iterates through the list of boundary coordinates 
    provided in the dictionary `f` and plots each boundary using 
    the `plot` method. The function supports additional arguments 
    and keyword arguments for customization of the plot appearance.
    """
    for i in range(len(f['bdy'])):
        ax.plot(f['bdy_x'][i], f['bdy_y'][i], 
                color=color, linestyle=linestyle,
                *args, **kwargs)


def plane_vector(ax, f, u, v, id=None, res=None, *args, **kwargs):
    """
    Plot vector fields on the provided axes.

    Parameters:
    - ax: matplotlib.axes.Axes or cartopy.mpl.geoaxes.GeoAxes
        The axes object to plot the vector fields on.
    - f: dict
        Dictionary containing 'xc', 'yc', and 'nele' arrays:
            - 'xc': Array of x coordinates of the cell centers.
            - 'yc': Array of y coordinates of the cell centers.
            - 'nele': Number of cells.
    - u: numpy.ndarray
        Array of x components of the vectors.
    - v: numpy.ndarray
        Array of y components of the vectors.
    - id: numpy.ndarray, optional
        Array of cell indices to include in the plot. If None, all cells are included (default is None).
    - res: int, optional
        Minimum distance for vector reduction. If None, no reduction is applied (default is None).
    - *args: Variable length argument list.
        Additional arguments to pass to `quiver`.
    - **kwargs: Keyword arguments.
        Additional keyword arguments to pass to `quiver`.

    Returns:
    - handle: matplotlib.quiver.Quiver
        The handle to the plotted vector field.

    Description:
    This function plots vector fields on the specified axes using the 
    `quiver` method. It allows for optional reduction of vectors based 
    on a specified resolution, which reduces the number of vectors 
    plotted to avoid clutter. The function returns the handle to the 
    plotted vector field, which can be used for further customization
    or reference.
    """
    if res is not None:
        id = rarefy(f, 'cell', res)

    if id is None:
        id = np.arange(f['nele']) + 1

    px = f['xc'][id-1]
    py = f['yc'][id-1]
    pu = u[id-1]
    pv = v[id-1]
    
    handle = ax.quiver(px, py, pu, pv, *args, **kwargs)

    return handle


def legend_color(handle, label="", pos=None, *args, **kwargs):
    """
    Add a color legend to the current figure.

    Parameters:
    - handle: matplotlib.image.AxesImage or other color-mappable object
        The handle to the color-mappable object (e.g., an image or contour plot) for which the color legend is created.
    - label: str
        The label for the color legend.
    - pos: tuple, optional
        Position of the colorbar as a tuple (left, bottom, width, height) in normalized figure coordinates. If None,
        the colorbar is added to the default location (default is None).
    - *args: Variable length argument list.
        Additional arguments to pass to `colorbar`.
    - **kwargs: Keyword arguments.
        Additional keyword arguments to pass to `colorbar`.

    Returns:
    - None

    Description:
    This function adds a color legend to the current figure using the 
    `colorbar` method. The color legend is created based on the provided 
    color-mappable handle and is labeled with the specified label. If a 
    position is provided, the color legend is placed at that position; 
    otherwise, it is added to the default location. The function does
    not return a value.
    """
    if pos is not None:
        cax = plt.gcf().add_axes(pos)
    else:
        cax = None

    cb = plt.colorbar(handle, label=label, cax=cax, *args, **kwargs)

    return cb


def legend_vector(handle, x, y, u, label, *args, **kwargs):
    """
    Add a vector legend to the current figure.

    Parameters:
    - handle: matplotlib.quiver.Quiver
        The handle to the quiver object for which the vector legend is created.
    - x: float
        The x-coordinate of the location for the vector legend in figure coordinates.
    - y: float
        The y-coordinate of the location for the vector legend in figure coordinates.
    - u: float
        The magnitude of the vector for the legend.
    - label: str
        The label for the vector legend.
    - background: str, optional
        Background color for the legend text. If None, the background color is set to white (default is None).
    - *args: Variable length argument list.
        Additional arguments to pass to `quiverkey`.
    - **kwargs: Keyword arguments.
        Additional keyword arguments to pass to `quiverkey`.

    Returns:
    - hdl: matplotlib.quiver.QuiverKey
        The handle to the created vector legend.

    Description:
    This function adds a vector legend to the current figure using the `quiverkey` 
    method. The legend is placed at the specified (x, y) location and represents a 
    vector with the given magnitude `u`. The legend is labeled with the provided 
    label. If a background color is specified, it is applied to the legend text; 
    otherwise, the default background color is white. The function returns the 
    handle to the created vector legend, which can be used for further 
    customization.
    """
    hdl = plt.quiverkey(handle, x, y, u, label, coordinates='figure', *args, **kwargs)
    hdl.text.set_backgroundcolor('w')


def axis_geo(ax, dx=None, dy=None, xlabel=True, ylabel=True, xticklabel=True, yticklabel=True):
    """
    Customize the gridlines and axis labels for longitude and latitude on a Matplotlib `Axes` object.

    Parameters:
    - ax: matplotlib.axes.Axes
        The Matplotlib `Axes` object to which the gridlines and axis labels are added.
    - dx: float, optional
        The spacing between longitude gridlines. If None, default gridline spacing is used (default is None).
    - dy: float, optional
        The spacing between latitude gridlines. If None, default gridline spacing is used (default is None).

    Returns:
    - None

    Description:
    This function customizes gridlines and axis labels for longitude and latitude on the 
    specified Matplotlib `Axes` object. It sets the tick marks and labels based on the 
    provided spacing `dx` for longitude and `dy` for latitude. If custom spacing is not 
    provided, default spacing is used. The function also adjusts the axis labels based 
    on the direction of the tick values, ensuring proper representation of longitude and
    latitude in degrees East/West and North/South.
    """
    xlims = ax.get_extent()[0:2]
    ylims = ax.get_extent()[2:4]
    if dx is not None:
        x1 = np.floor(xlims[0] / dx) * dx
        x2 = np.ceil(xlims[1] / dx) * dx
        xticks = np.arange(x1, x2, dx)
        xticks = xticks[(xticks>=xlims[0]) & (xticks<=xlims[1])]
        ax.set_xticks(xticks)
    else:
        xticks = ax.get_xticks()
    if dy is not None:
        y1 = np.floor(ylims[0] / dy) * dy
        y2 = np.ceil(ylims[1] / dy) * dy 
        yticks = np.arange(y1, y2, dy)
        yticks = yticks[(yticks>=ylims[0]) & (yticks<=ylims[1])]
        ax.set_yticks(yticks)
    else:
        yticks = ax.get_yticks()

    if np.all(xticks <= 0):
        ax.set_xlabel('Longitude [$^\circ$W]')
        ax.set_xticklabels([label.get_text().replace('−', '') for label in ax.get_xticklabels()]);
    elif np.all(xticks >= 0):
        ax.set_xlabel('Longitude [$^\circ$E]')
    else:
        ax.set_xlabel('Longitude [$^\circ$]')
    if np.all(yticks <= 0):
        ax.set_ylabel('Latitude [$^\circ$S]')
        ax.set_yticklabels([label.get_text().replace('−', '') for label in ax.get_yticklabels()]);
    elif np.all(yticks >= 0):
        ax.set_ylabel('Latitude [$^\circ$N]')
    else:
        ax.set_ylabel('Latitude [$^\circ$]')

    if xlabel is not True:
        ax.set_xlabel('')
    if ylabel is not True:
        ax.set_ylabel('')
    if xticklabel is not True:
        ax.set_xticklabels([])
    if yticklabel is not True:
        ax.set_yticklabels([])

    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')


def axis_xticklabel_negate(ax):
    # Get the current xticks and labels
    xticks = ax.get_xticks()
    xtick_labels = ax.get_xticklabels()
    xlim = ax.get_xlim()
    
    # Create new labels based on the condition
    new_ticks = []
    new_labels = []
    for tick, label in zip(xticks, xtick_labels):
        if tick < xlim[0] or tick > xlim[1]:
            continue

        new_ticks.append(tick)
        
        text = label.get_text()
        if abs(tick) < 1e-8:
            new_labels.append(text)  # Keep 0 as it is
        else:
            if '−' in text:
                new_labels.append(text.replace('−', ''))  # Remove '−'
            else:
                new_labels.append(f'−{text}')  # Add '−'

    # Set the new labels
    ax.set_xticks(new_ticks)
    ax.set_xticklabels(new_labels)


def axis_yticklabel_negate(ax):
    # Get the current yticks and labels
    yticks = ax.get_yticks()
    ytick_labels = ax.get_yticklabels()
    ylim = ax.get_ylim()
    
    # Create new labels based on the condition
    new_ticks = []
    new_labels = []
    for tick, label in zip(yticks, ytick_labels):
        if tick < ylim[0] or tick > ylim[1]:
            continue

        new_ticks.append(tick)
        
        text = label.get_text()
        if abs(tick) < 1e-8:
            new_labels.append(text)  # Keep 0 as it is
        else:
            if '−' in text:
                new_labels.append(text.replace('−', ''))  # Remove '−'
            else:
                new_labels.append(f'−{text}')  # Add '−'

    # Set the new labels
    ax.set_yticks(new_ticks)
    ax.set_yticklabels(new_labels)


def plane_axis_street(ax, dx=None, dy=None, x0=-0.14, y0=-0.12, color='k', linestyle='--', linewidth=0.2):
    """
    Add gridlines and axis labels for longitude and latitude to a Cartopy `GeoAxes` object.

    Parameters:
    - ax: cartopy.mpl.geoaxes.GeoAxes
        The Cartopy `GeoAxes` object to which the gridlines and axis labels are added.
    - dx: float, optional
        The spacing between longitude gridlines. If None, no custom spacing is applied (default is None).
    - dy: float, optional
        The spacing between latitude gridlines. If None, no custom spacing is applied (default is None).
    - x0: float, optional
        The x-coordinate of the longitude axis label position in figure coordinates (default is -0.14).
    - y0: float, optional
        The y-coordinate of the latitude axis label position in figure coordinates (default is -0.12).
    - color: str, optional
        The color of the gridlines (default is 'k' for black).
    - linestyle: str, optional
        The style of the gridline lines (default is '--' for dashed lines).
    - linewidth: float, optional
        The width of the gridline lines (default is 0.2).

    Returns:
    - None

    Description:
    This function adds gridlines and axis labels for longitude and latitude to the specified Cartopy 
    `GeoAxes` object. The gridlines are customized based on the provided spacing `dx` and `dy` for 
    longitude and latitude, respectively. Longitude and latitude axis labels are added at specified 
    positions. The gridline appearance is controlled by the `color`, `linestyle`, and `linewidth` 
    parameters. If custom spacing is not provided, the default gridline spacing is used.
    """
    gl = ax.gridlines(draw_labels=True, color=color, linestyle=linestyle, linewidth=linewidth)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
    gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER

    xlims = ax.get_extent()[0:2]
    ylims = ax.get_extent()[2:4]
    if dx is not None:
        x1 = np.floor(xlims[0] / dx) * dx
        x2 = np.ceil(xlims[1] / dx) * dx
        xticks = np.arange(x1, x2, dx)
        xticks = xticks[(xticks>=xlims[0]) | (xticks<=xlims[1])]
        gl.xlocator = matplotlib.ticker.FixedLocator(xticks)
    if dy is not None:
        y1 = np.floor(ylims[0] / dy) * dy
        y2 = np.ceil(ylims[1] / dy) * dy 
        yticks = np.arange(y1, y2, dy)
        yticks = yticks[(yticks>=ylims[0]) | (yticks<=ylims[1])]
        gl.ylocator = matplotlib.ticker.FixedLocator(yticks)

    ax.text(0.5, y0, 'Longitude', ha='center', va='center', transform=ax.transAxes)
    ax.text(x0, 0.5, 'Latitude', ha='center', va='center', rotation='vertical', transform=ax.transAxes)
    

def fig_margin(figure_in, margin=None, margin_H=None, margin_V=None, *args):
    """
    Add margins to an image by removing white borders and adding specified margins.

    Parameters:
    - figure_in: str
        Path to the input image file.
    - margin: int, optional
        Default margin size (in pixels) to add around the image. If None, a default of 1 pixel is used (default is None).
    - margin_H: int, optional
        Horizontal margin size (in pixels) to add around the image. If None, the default margin is used (default is None).
    - margin_V: int, optional
        Vertical margin size (in pixels) to add around the image. If None, the default margin is used (default is None).
    - *args: tuple, optional
        Optional additional argument for specifying the output image path.

    Returns:
    - None

    Description:
    This function processes an image file by removing any white borders from the 
    edges and then adding specified margins around the cleaned image. The margins 
    can be specified separately for horizontal and vertical directions. If no 
    output path is provided, the function overwrites the input image with the 
    updated margins.
    """
    if len(args) == 0:
        figure_out = figure_in
    elif len(args) >0:
        figure_out = args[0]

    if margin is None:
        margin = 1
    if margin_H is None:
        margin_H = margin
    if margin_V is None:
        margin_V = margin

    # Load the image using PIL
    img = Image.open(figure_in)
    
    # Convert the image to RGB if it's not already
    img_rgb = img.convert('RGB')
    
    # Convert the image to a numpy array (RGB matrix)
    rgb_matrix = np.array(img_rgb)
    
    # Check if the first row is all white
    while True:
        if np.all(rgb_matrix[0, :, :] == [255, 255, 255]):
            rgb_matrix = rgb_matrix[1:, :, :]  # Remove the first row
        else:
            break
    # Check if the last row is all white
    while True:
        if np.all(rgb_matrix[-1, :, :] == [255, 255, 255]):
            rgb_matrix = rgb_matrix[:-1, :, :]  # Remove the last row
        else:
            break
    # Check if the first column is all white
    while True:
        if np.all(rgb_matrix[:, 0, :] == [255, 255, 255]):
            rgb_matrix = rgb_matrix[:, 1:, :]  # Remove the first column
        else:
            break
    # Check if the last column is all white
    while True:
        if np.all(rgb_matrix[:, -1, :] == [255, 255, 255]):
            rgb_matrix = rgb_matrix[:, :-1, :]  # Remove the last column
        else:
            break

    # Concatenate the new margin to the original matrix
    rgb_column = np.full((np.shape(rgb_matrix)[0], margin_H, 3), [255, 255, 255], dtype=np.uint8)
    rgb_matrix = np.concatenate([rgb_column, rgb_matrix, rgb_column], axis=1)
    rgb_row = np.full((margin_V, np.shape(rgb_matrix)[1], 3), [255, 255, 255], dtype=np.uint8)
    rgb_matrix = np.concatenate([rgb_row, rgb_matrix, rgb_row], axis=0)
    
    # Save the figure out
    img_cleaned = Image.fromarray(rgb_matrix, 'RGB')
    img_cleaned.save(figure_out)


def fig_save(fig, figure_out, dpi=1200, margin=None, margin_H=None, margin_V=None):
    """
    Save a figure to a file with optional margins and high resolution.

    Parameters:
    - fig: matplotlib.figure.Figure
        The Matplotlib figure object to be saved.
    - figure_out: str
        Path to the output image file.
    - dpi: int, optional
        The resolution of the saved image in dots per inch (DPI). Default is 1200.
    - margin: int, optional
        Default margin size (in pixels) to add around the image. If None, no margin is added (default is None).
    - margin_H: int, optional
        Horizontal margin size (in pixels) to add around the image. If None, the default margin is used (default is None).
    - margin_V: int, optional
        Vertical margin size (in pixels) to add around the image. If None, the default margin is used (default is None).

    Returns:
    - None

    Description:
    This function saves a Matplotlib figure to an image file with the specified resolution. After saving, it calls the 
    `fig_margin` function to add margins around the image, if specified. The margins can be set separately for horizontal 
    and vertical directions. The figure is saved in high resolution as specified by the `dpi` parameter.
    """
    fig.savefig(figure_out, dpi=dpi)
    
    fig_margin(figure_out, margin=margin, margin_H=margin_H, margin_V=margin_V)
    

def interp_1d(z1, z2, var1, list=None, upper=None, lower=None):

    n1, nz1 = np.shape(z1)
    if len(np.shape(z2)) == 1:
        z2 = np.tile(z2, (n1,1))
    n2, nz2 = np.shape(z2)
    n3, nz3 = np.shape(var1)

    if z1.shape != var1.shape:
        raise ValueError(f"The sizes of z1 ({z1.shape}) and var1 ({var1.shape}) are different.")
    if not (z1.shape[0] == z2.shape[0] == var1.shape[0]):
        raise ValueError(f"The node size of z1 ({z1.shape[0]}), z2 ({z2.shape[0]}) and var1 ({var1.shape[0]}) are different.")

    if list is None:
        list = np.arange(n1)  # Default list of nodes
    else:
        list = list - 1
        
    var2 = np.full((n2, nz2), np.nan)
    for i in list:
        var2[i,:] = np.interp(z2[i,:], z1[i,:], var1[i,:], left=upper, right=lower)

    return var2



def interp_2d(srcField, dstField, *args, regrid=None):
        
    if not isinstance(regrid, esmpy.api.regrid.Regrid):
        regrid = esmpy.Regrid(srcField, dstField,
                              regrid_method=esmpy.RegridMethod.BILINEAR,
                              unmapped_action=esmpy.UnmappedAction.IGNORE)

    if len(args) == 0:
        return regrid
    else:          
        srcField.data[...] = args[0]
        dstField = regrid(srcField, dstField)
        return dstField.data       


def find_node(f, px, py, k=1):
    points = np.vstack((f['x'], f['y'])).T
    tree = KDTree(points)

    query_points = np.vstack((px, py)).T
    distances, indices = tree.query(query_points, k=k)

    indices = indices + 1
    
    return distances, indices


def find_cell(f, px, py, k=1):
    points = np.vstack((f['xc'], f['yc'])).T
    tree = KDTree(points)

    query_points = np.vstack((px, py)).T
    distances, indices = tree.query(query_points, k=k)

    indices = indices + 1
    
    return distances, indices


def unique(data):
    unique_data = np.unique(data.flatten())
    
    return unique_data


def line_split(px, py, npoint=100):
    dist = np.sqrt((px[1:] - px[:-1])**2 + (py[1:] - py[:-1])**2)
    dist_sum = np.sum(dist)
    
    tran_x = [px[0]]
    tran_y = [py[0]]
    tran_d = [0.0]
    
    for i in range(1,len(px)):
        n = max(round(dist[i-1] / dist_sum * (npoint-1)), 1)
        dx = (px[i] - px[i-1]) / n
        dy = (py[i] - py[i-1]) / n
        
        for j in range(n):
            tran_x.append(tran_x[-1] + dx)
            tran_y.append(tran_y[-1] + dy)
            tran_d.append(tran_d[-1] + np.sqrt((tran_x[-1] - tran_x[-2])**2 + (tran_y[-1] - tran_y[-2])**2))
        
    tran_x = np.array(tran_x)
    tran_y = np.array(tran_y)
    return tran_x, tran_y, tran_d, len(tran_x)


def standard_depth(max_depth):
    '''
    Standard depth of HYCOM model
    https://www.hycom.org/dataserver/gofs-3pt0/analysis
    '''
    std0 = np.array([0, 10, 20, 30, 50, 75, 100,
                     125, 150, 200, 250, 300, 400, 500,
                     600, 700, 800, 900, 1000,
                     1250, 1500, 2000, 3000, 4000, 5000])
    std = std0[std0<=max_depth]
    if std[-1] != max_depth:
        std = np.append(std, max_depth)
    return std


def transect_topo_poly(x, z, max_depth):
    # Create the polygon points
    topo_x = np.concatenate((x, [x[-1]], [0], [x[0]]))
    topo_z = np.concatenate((z, [-max_depth], [-max_depth], [z[0]]))
    
    return [topo_x, topo_z]


def transect_data(f, tran_x, tran_y, std, var, max_dep=None, npoint=100, upper=None, lower=None):

    if max_dep is None:
        max_dep = np.max(std)
    
    # Increase the resolution of the transect
    px, py, dist, npoint = line_split(tran_x, tran_y, npoint=npoint)
    
    # Create the fields for the mesh and the transect
    srcField = create_TMSH(f['x'], f['y'], f['nv'])
    dstField = create_STRM(px, py)

    # ---> 1. Interpolation the data
    var2 = np.full((len(px),len(std)), np.nan)
    if var is not []:
        # Vertical interpolationvar
        var1 = interp_1d(f['deplay'], std, var, upper=upper, lower=lower)
        # Horizontal interpolation
        for iz in range(len(std)):
            var2[:,iz] = interp_2d(srcField, dstField, var1[:,iz])

    # ---> 2. Interpolation the bathymetry
    ph = interp_2d(srcField, dstField, f['h'])
    topo = transect_topo_poly(dist, -ph, max_dep)

    return {'n': npoint, 
            'd': dist, 'z': -std, 'v': var2,
            'px': px, 'py': py,
            'topo': topo}


def transect_image(ax, ft, *args, **kwargs):
    
    # Create the grid of the figure
    dd, zz = np.meshgrid(ft['d'], ft['z'], indexing='ij')

    # Draw the image
    handle = ax.contourf(dd, zz, ft['v'], *args, **kwargs)

    return handle


def transect_contour(ax, ft, colors='k', zorder=2, *args, **kwargs):
    
    # Create the grid of the figure
    dd, zz = np.meshgrid(ft['d'], ft['z'], indexing='ij')

    # Draw the image
    handle = ax.contour(dd, zz, ft['v'], colors=colors, *args, **kwargs)

    return handle


def transect_topo(ax, ft, color='tan', zorder=10, *args, **kwargs):
    
    # Draw the shaded topo 
    handle = ax.fill(ft['topo'][0], ft['topo'][1], color=color, zorder=zorder, *args, **kwargs)

    return handle

def interp_node2cell(f, var_node):
    """
    Interpolate node-based values to cell-based values by averaging nodes defined by nv.

    Parameters:
    f (dir): nv will be used.
    var_node (ndarray): A multi-dimensional array where the first dimension corresponds to nodes.

    Returns:
    var_cell (ndarray): A multi-dimensional array where the first dimension corresponds to cells,
                        and the remaining dimensions match the input.
    """
    nv = f['nv'] - 1
    
    # Extract the node values for each cell using np.take to handle multi-dimensional inputs.
    var_values = np.take(var_node, nv, axis=0)  # Shape: (N, 3, ...)

    # Average along the second axis (axis=1) which corresponds to the 3 nodes.
    var_cell = np.mean(var_values, axis=1)

    return var_cell


def interp_cell2node(f, var_cell0):

    # Add a fake column before the first one in the first dimension
    shape0 = var_cell0.shape
    shape = (f['nele']+1,) + shape0[1:]

    var_cell = np.full(shape, np.nan)
    var_cell[1:] = var_cell0

    # Extract the node values for each cell using np.take to handle multi-dimensional inputs.
    var_values = np.take(var_cell, f['nbve'], axis=0)  # Shape: (M, MX_NBR_ELEM, ...)

    # Average along the second axis (axis=1) which corresponds to the 3 nodes.
    var_node = np.nanmean(var_values, axis=1)

    return var_node


def calc_uv2current(u, v):
    spd = np.sqrt(u**2 + v**2)
    dir = np.degrees(np.arctan2(v, u))
    dir = np.mod(dir, 360.0)
    return spd, dir


def calc_current2uv(spd, dir):
    u = spd * np.cos(np.radians(dir))
    v = spd * np.sin(np.radians(dir))
    return u, v


def calc_uv2wind(u, v):
    spd = np.sqrt(u**2 + v**2)
    dir = 270 - np.degrees(np.arctan2(v, u))
    dir = np.mod(dir, 360)
    return spd, dir


def calc_wind2uv(spd, dir):
    u = spd * np.cos(np.radians(270 - dir))
    v = spd * np.sin(np.radians(270 - dir))
    return u, v


def interp_weight_ERA52FVCOM(GRID, TMSH, *args, **kwargs):

    return interp_weight_GRID2TMSH(GRID, TMSH, *args, **kwargs)


def interp_ERA52FVCOM(srcData, *args, **kwargs):

    return interp_GRID2TMSH(srcData, *args, **kwargs)


def forcing_ERA52FVCOM(f, fins, fout, start_time, end_time, interval_time=1, format='NETCDF3_CLASSIC', method='bilinear'):

    # Parameters
    K2C = -273.15
    eps = 0.622
    mjd = datetime(1858, 11, 17)

    # Create the output 
    out = nc.Dataset(fout, 'w', format=format)
    # Dimensions
    out.createDimension('node', f['node'])
    out.createDimension('nele', f['nele'])
    out.createDimension('three', 3)
    out.createDimension('DateStrLen', 19)
    out.createDimension('time', None)
    # Variables
    # x
    x = out.createVariable('x', 'f4', ('node',))
    x.long_name = "x coordinate"
    x[:] = f['x']
    # y
    y = out.createVariable('y', 'f4', ('node',))
    y.long_name = "y coordinate"
    y[:] = f['y']
    # nv
    nv = out.createVariable('nv', 'i4', ('three', 'nele'))
    nv.long_name = "nodes surrounding element"
    nv[:] = np.transpose(f['nv'])
    # time
    time = out.createVariable('time', 'f4', ('time',))
    time.long_name = "time"
    time.format = "modified julian day (MJD)"
    time.unit = "days since 1858-11-17 00:00:00"
    # Itime
    Itime = out.createVariable('Itime', 'i4', ('time',))
    Itime.format = "modified julian day (MJD)"
    Itime.unit = "days since 1858-11-17 00:00:00"
    # Itime2
    Itime2 = out.createVariable('Itime2', 'i4', ('time',))
    Itime2.format = "modified julian day (MJD)"
    Itime2.unit = "msec since 00:00:00"
    # Times
    Times = out.createVariable('Times', 'S1', ('time', 'DateStrLen'))
    Times.format = "yyyy-mm-dd_HH:MM:SS"
    Times.time_zone = "UTC"
    # uwind_speed
    uwind_speed = out.createVariable('uwind_speed', 'f4', ('time', 'nele'))
    uwind_speed.long_name = "eastward Wind Speed"
    uwind_speed.unit = "m/s"
    # vwind_speed
    vwind_speed = out.createVariable('vwind_speed', 'f4', ('time', 'nele'))
    vwind_speed.long_name = "northward wind stress"
    vwind_speed.unit = "m/s"
    # short_wave
    short_wave = out.createVariable('short_wave', 'f4', ('time', 'node'))
    short_wave.long_name = "shortwave radiation"
    short_wave.unit = "W/m2"
    # net_heat_flux
    net_heat_flux = out.createVariable('net_heat_flux', 'f4', ('time', 'node'))
    net_heat_flux.long_name = "net heat flux"
    net_heat_flux.unit = "W/m2"
    # air_pressure
    air_pressure = out.createVariable('air_pressure', 'f4', ('time', 'node'))
    air_pressure.long_name = "sea surface pressure"
    air_pressure.unit = "Pa"
    # precip
    precip = out.createVariable('precip', 'f4', ('time', 'node'))
    precip.long_name = "precipitation, negative for ocean losing water"
    precip.unit = "m/s"
    # evap
    evap = out.createVariable('evap', 'f4', ('time', 'node'))
    evap.long_name = "evaporation, negative for ocean losing water "
    evap.unit = "m/s"
    # SAT
    SAT = out.createVariable('SAT', 'f4', ('time', 'node'))
    SAT.long_name = "2m air temperature"
    SAT.unit = "degree C"
    # SPQ
    SPQ = out.createVariable('SPQ', 'f4', ('time', 'node'))
    SPQ.long_name = "2m specific humidity"
    SPQ.unit = "kg/kg"
    # cloud_cover
    cloud_cover = out.createVariable('cloud_cover', 'f4', ('time', 'node'))
    cloud_cover.long_name = "total cloud cover"
    cloud_cover.unit = "0-1"

    # Convert input times to datetime objects
    start = datetime.strptime(start_time, "%Y-%m-%d_%H")
    end = datetime.strptime(end_time, "%Y-%m-%d_%H")

    # Iterate over the time range in the given interval
    current_time = start
    it = 0
    while current_time <= end:
        # Generate the date and time strings for the request
        yyyy = f"{current_time.year:04d}"
        mm = f"{current_time.month:02d}"
        dd = f"{current_time.day:02d}"
        HH = f"{current_time.hour:02d}"
        MM = f"{current_time.minute:02d}"
        SS = f"{current_time.second:02d}"
        
        # Generate the input filename
        fin = fins
        fin = fin.replace('@yyyy', yyyy)
        fin = fin.replace('@mm', mm)
        fin = fin.replace('@dd', dd)
        fin = fin.replace('@HH', HH)
        fin = fin.replace('@MM', MM)
        fin = fin.replace('@SS', SS)
        print(fin)
        
        # Load grid
        if current_time == start:
            grid = load_grid_ERA5(fin)
            weight = interp_weight_ERA52FVCOM(grid, f, method=method)

        # Load data
        u10  = nc_read_var(fin, 'u10')   # 10m u wind                                m/s
        v10  = nc_read_var(fin, 'v10')   # 10m v wind                                m/s
        d2m  = nc_read_var(fin, 'd2m')   # 2m dewpoint temperature                   K
        t2m  = nc_read_var(fin, 't2m')   # 2m temperature                            K
        msl  = nc_read_var(fin, 'msl')   # mean sea level pressure                   Pa
        sp   = nc_read_var(fin, 'sp')    # surface pressure                          Pa
        ssr  = nc_read_var(fin, 'ssr')   # surface net shortwave/solar radiation     J/h/m2, downward
        str  = nc_read_var(fin, 'str')   # surface net longwave/thermal radiation    J/h/m2, upward
        slhf = nc_read_var(fin, 'slhf')  # surface latent heat flux                  J/h/m2, upward
        sshf = nc_read_var(fin, 'sshf')  # surface sensible heat flux                J/h/m2, upward
        tcc  = nc_read_var(fin, 'tcc')   # total cloud cover                         [0-1]
        e    = nc_read_var(fin, 'e')     # evaporation                               m/h
        tp   = nc_read_var(fin, 'tp')    # total precipitatin                        m/h
        # Note by Siqi Li
        # Although the latent, sensible heat fluxes and the longwave radiation are described as 'upward'
        # they seems to have the same defination on signs as that on the shortwave radiation.
        # Same thing occurs on the evaporation and precipitation

        # Calculation / Interpolation
        # Wind - nothing to do
        data_uwind_speed = interp_ERA52FVCOM(u10, weight, loc='cell')
        data_vwind_speed = interp_ERA52FVCOM(v10, weight, loc='cell')
        # Heat fluxes
        data_short_wave = interp_ERA52FVCOM(ssr/3600.0, weight)
        netheat = (ssr+str+slhf+sshf) / 3600.0
        data_net_heat_flux = interp_ERA52FVCOM(netheat, weight)
        # Sea level pressure
        data_air_pressure = interp_ERA52FVCOM(msl, weight)
        # Precipitation and evaporation
        data_precip = interp_ERA52FVCOM(tp/3600.0, weight)
        data_evap = interp_ERA52FVCOM(e/3600.0, weight)
        # Ice-related variables
        data_SAT = interp_ERA52FVCOM(t2m+K2C, weight)
        e2 = 6.112 * np.exp(17.67*(d2m+K2C)/(d2m+K2C+243.5))
        q2 = eps*e2 / (sp-(1-eps)*e2)
        data_SPQ = interp_ERA52FVCOM(q2, weight)
        data_cloud_cover = interp_ERA52FVCOM(np.clip(tcc, None, 1.0), weight)

        # Write data into output
        # time
        Itime[it] = (current_time - mjd).days
        Itime2[it] = (current_time - mjd).seconds * 1000.0
        time[it] = (current_time - mjd).days + (current_time - mjd).seconds/3600.0/24.0
        Times[it, :] = current_time.strftime("%Y-%m-%d_%H:%M:%S")
        # data
        uwind_speed[it, :] = data_uwind_speed
        vwind_speed[it, :] = data_vwind_speed
        short_wave[it, :] = data_short_wave
        net_heat_flux[it, :] = data_net_heat_flux
        air_pressure[it, :] = data_air_pressure
        precip[it, :] = data_precip
        evap[it, :] = data_evap
        SAT[it, :] = data_SAT
        SPQ[it, :] = data_SPQ
        cloud_cover[it, :] = data_cloud_cover

        # Increment the time by the specified interval
        current_time += timedelta(hours=interval_time)
        it += 1

    # Close the output
    out.close()
