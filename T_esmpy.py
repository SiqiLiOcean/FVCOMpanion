import numpy as np
import esmpy

def create_GRID(x, y, mask=None, coord='SPH_local', name='GRID'):

    xx, yy = np.meshgrid(x, y, indexing='ij')
    nx, ny = np.shape(xx)

    if mask is None:
        mask = np.zeros([nx,ny], dtype=int)
        
    if coord == 'SPH_local':
        GRID = esmpy.Grid(np.array([nx-1,ny-1]), staggerloc=esmpy.StaggerLoc.CORNER,
                          coord_sys=esmpy.CoordSys.SPH_DEG)
    elif coord == 'SPH_global':
        GRID = esmpy.Grid(np.array([nx,ny-1]), staggerloc=esmpy.StaggerLoc.CORNER,
                          coord_sys=esmpy.CoordSys.SPH_DEG, 
                          num_peri_dims=1, periodic_dim=0, pole_dim=1)
    elif coord == 'CART':
        GRID = esmpy.Grid(np.array([nx-1,ny-1]), staggerloc=[esmpy.StaggerLoc.CENTER, esmpy.StaggerLoc.CORNER],
                          coord_sys=esmpy.CoordSys.CART)
    else:
        raise ValueError(f"Unknown coord: ({coord})")
    
    gridX = GRID.get_coords(0, staggerloc=esmpy.StaggerLoc.CORNER)
    gridY = GRID.get_coords(1, staggerloc=esmpy.StaggerLoc.CORNER)
    gridX[:] = xx
    gridY[:] = yy

    Field = esmpy.Field(GRID, name=name, staggerloc=esmpy.StaggerLoc.CORNER)

    Field.data[...] = 1e20
    return Field


def create_TMSH(x, y, nv, coord='SPH', name='TMSH'):

    if coord == 'SPH':
        coord = esmpy.CoordSys.SPH_DEG
    elif coord == 'CART':
        coord = esmpy.CoordSys.CART
    else:
        raise ValueError(f"Unknown coord: ({coord})")
    
    # Create ESMF MESH
    TMSH = esmpy.Mesh(parametric_dim=2, spatial_dim=2, coord_sys=coord)
    
    # Set the nodes
    num_node = len(x) 
    nodeId = np.arange(num_node) + 1 
    nodeCoord = np.array([item for pair in zip(x, y) for item in pair])
    nodeOwner = np.zeros(num_node)
    
    # Set the elements
    num_elem = np.shape(nv)[0]
    elemId = np.arange(num_elem) + 1 
    elemType = np.full((num_elem,), esmpy.MeshElemType.TRI)
    elemConn = nv.reshape(-1) - 1
    
    # Add nodes and elements
    TMSH.add_nodes(num_node, nodeId, nodeCoord, nodeOwner) 
    TMSH.add_elements(num_elem, elemId, elemType, elemConn)

    Field = esmpy.Field(TMSH, name=name, meshloc=esmpy.MeshLoc.NODE)

    Field.data[...] = 1e20
    return Field


def create_STRM(x, y, mask=None, coord='SPH', name='field_locstream'):

    n = len(x)

    if mask is None:
        mask = np.zeros(n, dtype=int)

    if coord == 'SPH':
        coord = esmpy.CoordSys.SPH_DEG
    elif coord == 'CART':
        coord = esmpy.CoordSys.CART
    else:
        raise ValueError(f"Unknown coord: ({coord})")
        
    STRM = esmpy.LocStream(n, coord_sys=coord) 

    STRM["ESMF:Lon"] = x
    STRM["ESMF:Lat"] = y

    Field = esmpy.Field(STRM, name=name)
    
    return Field


def interp_weight_GRID2TMSH(GRID, TMSH, method='bilinear'):
    if method == 'bilinear':
        method = esmpy.RegridMethod.BILINEAR
    elif method == 'patch':
        method = esmpy.RegridMethod.PATH
        method = esmpy.RegridMethod.CONSERVE
    else:
        raise ValueError(f"Unknown coord: ({method})")

    # Create fields
    srcField = create_GRID(GRID['x'], GRID['y'], coord='SPH_local')
    dstField = create_TMSH(TMSH['x'], TMSH['y'], TMSH['nv'])

    weight = esmpy.Regrid(srcField, dstField, regrid_method=method)

    return weight


def interp_GRID2TMSH(srcData, *args, **kwargs):

    loc = kwargs.pop('loc', 'node')
    
    if len(args) > 1:
        weight = interp_weight_GRID2TMSH(*args, **kwargs)
    else:
        weight = args[0]
        
    srcField = weight.srcfield
    dstField = weight.dstfield

    srcField.data[...] = srcData
    dstField = weight(srcField, dstField)
    dstData = dstField.data * 1.0

    if loc == 'cell':
        nv = weight.dstfield.grid.element_conn.reshape(-1,3)
        dstData = np.mean(np.take(dstData, nv, axis=0), axis=1)

    return dstData

