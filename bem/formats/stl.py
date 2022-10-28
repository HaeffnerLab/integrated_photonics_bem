from collections import OrderedDict
import struct

import numpy as np

def unify_dup_points(o_points,o_triangles, tol = 1e-6):
    # define a point index dictionary to combine same points
    o_n = len(o_points)
    uni_dict = {}
    n_points = []
    for i in range(o_n):
        for j in uni_dict.keys():
            # if point is the same
            if np.linalg.norm(o_points[i]-o_points[j], 1) < tol * np.linalg.norm(o_points[i], 1):
                uni_dict[i] = uni_dict[j]
                break
        else:
            uni_dict[i] = len(n_points)
            n_points.append(o_points[i])

    n_points = np.array(n_points)

    n_triangles = np.vectorize(lambda x:uni_dict[x])(o_triangles)
    n_triangles = np.array(n_triangles,dtype = np.dtype(np.int32))

    return n_points, n_triangles


# check if it is right  handed
def correct_normals(nm,o_points,o_triangles):
    # check if left or right
    def _is_right(nm,a,b,c):
        n = np.cross(b-a, c-a)
        n /= np.sqrt(np.sum(np.square(n)))
        return np.allclose(n, nm, rtol=1e-3, atol=1e-10)
    o_n = len(o_points)
    uni_dict = {}
    n_triangles = []
    for tri in list(o_triangles):
        a = o_points[tri[0]]
        b = o_points[tri[1]]
        c = o_points[tri[2]]
        if _is_right(nm,a,b,c):
            n_triangles.append(tri)
        else:
            # if not right hand
            tri[1],tri[2] = tri[2],tri[1]
            n_triangles.append(tri)

    return o_points, np.array(o_triangles)



def read_stl(fil, ignore_attr_length=True):     # binary stl
    """reads binary STL in file "fil" with Inventor style
    (color hidden in attribute length), returns array of normals, array
    of vertices and list or attributes

    Inventor >= 2013 supports colors (bottom right in the export options
    dialog)
    
    The atribute short in binary STL is used by some tools to hide
    color information. The format is 15 bit RGB or BGR, MSB is validity
    (default color if MSB is 0) COLOR=\\xbf\\xbf\\xbf\\xff in the header
    is the overall color default color.
    """
    h = fil.read(80)    # ASCII header of length 80 bytes.
    #assert h.startswith(b"STLB")    # Check if it's a binary stl or raise an AssertionError.
    # n is the number of triangle faces.
    n, = struct.unpack("< I", fil.read(4))    # "< I" struct.unpack 4-byte unsigned int in little-endian.
    if ignore_attr_length:
        dtype = np.dtype([
            ("normal", "<f4", (3,)),
            ("triangle", "<f4", (3, 3)),
            ("attribute", "<u2")])
        data = np.fromfile(fil, dtype, n)
        normals, triangles = data["normal"], data["triangle"]
        attributes = data["attribute"]
    else: # actual STL binary format with variable length attribute bytes
        vectors, attributes = [], []
        while True:
            try:
                vectors.append(struct.unpack("< 12f", fil.read(48)))
                na, = struct.unpack("< H", fil.read(2))
                attributes.append(fil.read(na))    # attributes contains face color number from Inventor.
            except struct.error:
                break
        assert len(vectors) == n, (len(vectors), n)
        vectors = np.array(vectors)
        normals = vectors[:, :3]
        triangles = vectors[:, 3:].reshape(-1, 3, 3)
    return normals, triangles, attributes


def check_normals(normals, triangles):
    """verifies that given vertices are right-handed around normals"""
    a, b, c = triangles.transpose((1, 0, 2))
    n = np.cross(b-a, c-a)
    n /= np.sqrt(np.sum(np.square(n), axis=1))[:, None]
    assert np.allclose(n, normals, rtol=1e-3, atol=1e-10)

# wwc function
def partition_normals(normals,triangles,numbers=[],TOL=5e-6):
    """Partition points into different planes according to the normals in 
    one electrode, for 3D/2D meshing (Shewchuk's triangle C code is 2D meshing).
    Return points_numbers which has following format
    [(plane1_points,serial_numbers1),(plane2_points,serial_numbers2),...]
    plane_points -- array([ [x1,y1,z1],[x2,y2,z2],... [xn,yn,zn] ])
    serial_numbers -- array([ [0,1,2],[3,4,5],... [n-3,n-2,n-1] ])
    TOL: normal deviation tolerances
    """
    nm_unique = [normals[0]]      # normals[0] as the first nm
    points = [[]]    # Each sublist [] represents a plane
    points[-1].extend(triangles[0])
    for nm, tr in zip(normals[1:], triangles[1:]):
        add_plane = True
        # Search existing normals in nm_unique, in reversed order. (Faces in the same plane are often ajacent.)
        for ith in range(len(nm_unique)-1,-1,-1):   # ith normal -- ith plane
            diff = np.array(points[ith][-1]) - tr
            dot_abs = np.abs(nm_unique[ith] @ diff.T)
            if np.all(dot_abs < TOL):
                # Plane points aren't grouped by triangles -- [array([x1,y1,z1]),array([x2,y2,z2]), ...]
                points[ith].extend(tr)
                add_plane = False
                break
        # All normals don't coincide with the new nm, record it.
        if add_plane:
            nm_unique.append(nm)
            points.append([])
            points[-1].extend(tr)
    points_numbers = []
    for plane in points:
        plane_points = np.ascontiguousarray(plane,dtype=np.double)
        i = np.arange(0,plane_points.shape[0],3)    # shape[0] is the number of points in the plane.
        index_numbers = np.c_[i, i+1, i+2].astype(np.intc)
        points_numbers.append( (plane_points,index_numbers) )
    return points_numbers, np.array(nm_unique,dtype=np.double)


# hwh function
# partite by two standards:1.same normal 2. same interception 
# automatically unify same points
def partition_normals_interception(normals,triangles,TOL=1e-6, TOL2 = 1e-6, unify = True):
    # hwh_warning: this TOL2 must be adapted with scale
    """Partition points into different planes according to the normals in 
    one electrode, for 3D/2D meshing (Shewchuk's triangle C code is 2D meshing).
    Return points_numbers which has following format

    [(plane1_points,serial_numbers1),(plane2_points,serial_numbers2),...]

    plane_points -- array([ [x1,y1,z1],[x2,y2,z2],... [xn,yn,zn] ])
    serial_numbers -- array([ [0,1,2],[3,4,5],... [n-3,n-2,n-1] ])

    TOL: normal deviation tolerances
    """
    # compute interception
    interce = []
    for nm,tr in zip(normals,triangles):
        nm0 = np.dot(nm,tr[0])
        nm1 = np.dot(nm,tr[1])
        nm2 = np.dot(nm,tr[2])
        interce.append((nm0+nm1+nm2)/3)
    interceptions = np.array(interce)

    nm_unique = [normals[0]]      # normals[0] as the first nm
    ic_unique = [interceptions[0]]      # normals[0] as the first nm
    points = [[]]    # Each sublist [] represents a plane
    for nm, tr, ic in zip(normals, triangles,interceptions):
        add_plane = True
        # Search existing normals in nm_unique, in reversed order. (Faces in the same plane are often ajacent.)
        for ith in range(len(nm_unique)-1,-1,-1):   # ith normal -- ith plane
            bool_nm_1 = np.abs((nm @ nm_unique[ith]) - (np.linalg.norm(nm) * np.linalg.norm(nm_unique[ith]))) < TOL
            bool_ic_1 = np.abs(ic - ic_unique[ith]) <= TOL2 * np.abs(ic)
            bool1 = bool_nm_1 and bool_ic_1

            bool_nm_2 = np.abs((nm @ -nm_unique[ith]) - (np.linalg.norm(nm) * np.linalg.norm(nm_unique[ith]))) < TOL
            bool_ic_2 = np.abs(ic + ic_unique[ith]) <= TOL2 * np.abs(ic)
            bool2 = bool_nm_2 and bool_ic_2
            if bool1 or bool2:
                # Plane points aren't grouped by triangles -- [array([x1,y1,z1]),array([x2,y2,z2]), ...]
                points[ith].extend(tr)
                add_plane = False
                break
        # All normals don't coincide with the new nm, record it.
        if add_plane:
            nm_unique.append(nm)
            ic_unique.append(ic)
            points.append([])
            points[-1].extend(tr)
    points_numbers = []
    for plane,nm in zip(points,nm_unique):
        plane_points = np.ascontiguousarray(plane,dtype=np.double)
        i = np.arange(0,plane_points.shape[0],3)    # shape[0] is the number of points in the plane.
        index_numbers = np.c_[i, i+1, i+2].astype(np.intc)
        if unify:
            # unify duplicated points
            plane_points,index_numbers = unify_dup_points(plane_points,index_numbers) 
        # verify it is right handed
        plane_points,index_numbers = correct_normals(nm, plane_points, index_numbers)

        points_numbers.append( (plane_points,index_numbers) )

    # split into single connectivity branches
    #for plane,nm in

    return points_numbers, np.array(nm_unique,dtype=np.double)



# wwc function. Copied from cpy.py. This function only needs the coordinates of 3 points in
# triangles to split normals, but we alreadly have normals imported from stl file. So use 
# above partition_normals() is OK.  
def split_by_normal(stlmesh):
    """split curved faces into one face per triangle (aka split by
    normal, planarize). in place"""
    for name, faces in stlmesh.items():
        new_faces = []
        for points, triangles in faces:    # faces = [(points,triangles)]
            x = points[triangles, :]    # triangles are the index number of points.
            # For 3 points x0,x1,x2, cross product of vector x1-x0, x2-x0 produces the normal vector.
            normals = np.cross(x[:, 1]-x[:, 0], x[:, 2]-x[:, 0])
            normals /= np.sqrt(np.sum(np.square(normals), axis=1))[:, None]
            if np.allclose(normals, normals[0][None, :]):
                new_faces.append((points, triangles))
            else:
                for triangle in triangles:
                    new_faces.append((points[triangle, :],
                        np.arange(3, dtype=np.intc).reshape((1, 3))))
        stlmesh[name] = new_faces
    return stlmesh


def stl_to_mesh(normals, triangles, attributes, scale=None, rename=None, quiet=True, print_dropping = True):
    """generates a {name: [(points, triangles)]} mesh from the
    stl arrays. For further treatment, use e.g:

    >>> s = read_stl(open(filename, "rb"))
    >>> check_normals(*s[:2])
    >>> r = stl_to_mesh(*s)
    >>> del r["stl_0"] # delete the color-less faces
    >>> print r.keys()
    ['stl_18761', 'stl_12943']
    >>> r = split_by_normal(r)
    >>> m = Mesh.from_mesh(r)
    >>> m.triangulate("qQ")
    >>> m.to_vtk("test_stl_import")

    scale is usually chosen the typical length to scale the geometry. E.g., 40 nm.
    Note scale must have the same unit with STL file which is set unit in Inventor.
    
    rename can be a renaming dictionary mapping stl color numbers to
    electrode names. if None the color name is "stl_%i" %
    color_numer, else if the color is not found in rename,
    the electrode is dropped.
    """
    d = OrderedDict()    # an intermediate dict
    for a, nm, tr in zip(attributes, normals, triangles):
        d.setdefault(a, [[],[]])[0].append(nm) 
        d[a][1].append(tr)
        # d = {a:[nms, trs]}, a is face color number from Inventor.
        # nms = [array([x,y,z]),...] trs = [ array([[x1,y1,z1],[x2,y2,z2],[x3,y3,z3]]),...]
    o = OrderedDict()    # o = {name: [(points, triangles)]}
    for a, nm_tr in d.items():
        #nm_tr[0] are the normal vectors
        #nm_tr[1] are the vertex points
        nms, trs = np.array(nm_tr[0]), np.array(nm_tr[1])
        if scale:
            trs /= scale
        partition = False    # rename=None means we just want to see color, no need to partition normals.
        if rename:
            # At the first run you don't know color number a, this prints it out for the "rename" of next run.
            if a not in rename:
                if print_dropping:
                    print("dropping", a)
                continue
            # At the second run you provide the known rename to replace a.
            else:
                n = rename[a]    
                partition = True
        else:
            n = "stl_%i" % a    # default name
        if partition:    # Designed for 3D multiplane partition.  
            o[n], planes = partition_normals_interception(nms, trs)
            if not quiet:
                print("%i planes in electrode %s"%(len(planes),n))
                print("normals vectors:\n", planes)
        else:    # Robert's original 2D version.
            i = np.arange(0, trs.shape[0]*3, 3)
            # reshape flats triangle arrays, puts all points together not distinguishing different triangles.
            points = trs.reshape(-1, 3).astype(np.double)
            triangles = np.c_[i, i+1, i+2].astype(np.intc)    # "triangles" contains index number of points.
            # Note there is an additional list [ ] containing the tuple (points, triangles). Robert might 
            # leave this as a future interface for 3D meshing.
            o[n] = [(np.ascontiguousarray(points),    # n = o.keys(). Fastlap seems to require a continuous
                np.ascontiguousarray(triangles))]    # memory space, thus use ascontiguousarray().
    return o


if __name__ == "__main__":
    import sys
    r = read_stl(open(sys.argv[1], "rb"))
    m = stl_to_mesh(*r)
    print(m)