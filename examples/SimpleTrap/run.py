import numpy as np
from bem import Electrodes, Sphere, Mesh, Grid, Configuration, Result
from bem.formats import stl
from time import time
import ipyparallel as ipp

prefix = "SimpleTrap"

# load processed mesh from vtk file: prefix_mesh.vtk

mesh, _ = Mesh.from_vtk(prefix)

# grid to evalute potential and fields at
# Create a grid in unit of scaled length l. 
# Only choose the interested region (trap center) to save time.
s = 0.08 # step size relative to scale defined in mesh
Lx, Ly, Lz = 2.0, 2.0, 2.0   # in the unit of scaled length l
sx, sy, sz = s, s, s
# ni is grid point number, si is step size. Thus to fix size on i direction you need to fix ni*si.
nx, ny, nz = [2*np.ceil(L/2.0/s).astype('int')+1 for L in (Lx, Ly, Lz)]
print("Size/l:", Lx, Ly, Lz)
print("Step/l:", sx, sy, sz)
print("Shape (grid point numbers):", nx, ny, nz)
grid = Grid(center=(0, 0, 1.5), step=(sx, sy, sz), shape=(nx, ny, nz))
# Grid center (nx, ny ,nz)/2 is shifted to origin
print("Grid origin/l:", grid.get_origin()[0])
x, y, z = grid.to_xyz()
print('x coords: ', x)
print('y coords: ', y)
print('z coords: ', z)

# create job list
jobs = list(Configuration.select(mesh,"DC.*","RF"))

# Define calculation function.
def run_job(args):
    # job is Configuration instance.
    job, grid, prefix = args
    # refine twice adaptively with increasing number of triangles, min angle 25 deg.
    job.adapt_mesh(triangles=4e2, opts="q25Q")
    job.adapt_mesh(triangles=1e3, opts="q25Q")
    # solve for surface charges
    job.solve_singularities(num_mom=4, num_lev=3)
    # get potentials and fields
    result = job.simulate(grid, field=job.name=="RF", num_lev=2)    # For "RF", field=True computes the field.
    result.to_vtk(prefix)
    print("finished job %s" % job.name)
    return job.collect_charges()

# parallel computation
mycluster = ipp.Cluster()
mycluster.start_cluster_sync()
c = mycluster.connect_client_sync()
c.wait_for_engines()

t0 = time()
# Run a parallel map, executing the wrapper function on indices 0,...,n-1
lview = c.load_balanced_view()
# Cause execution on main process to wait while tasks sent to workers finish
lview.block = True 
asyncresult = lview.map_async(run_job, ((job, grid, prefix) for job in jobs))   # Run calculation in parallel
asyncresult.wait_interactive()
print("Computing time: %f s"%(time()-t0))