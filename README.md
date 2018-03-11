### Data 
`wget http://vision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip`
`wget http://modelnet.cs.princeton.edu/ModelNet40.zip`

### Preprocessing
- *.off polygon to *.binvox voxel tool --->>> 
    - https://www.patrickmin.com/binvox/
    - cmdline tool that can be run against *.off files and returns a *.binvox file which can be interpreted by the code below
    - The binvox_converter.py file does the work, it just needs to have the binvox executable in the same directory
    - MACOSX install:
        - `wget http://www.patrickmin.com/binvox/download.php?id=6`
    - Linux install:
        - `wget http://www.patrickmin.com/binvox/download.php?id=4`
- *.binvox reader --->>>
    - https://github.com/Ryanglambert/binvox-rw-py/blob/public/binvox_rw.py

- Resolution
    - http://vision.princeton.edu/projects/2014/3DShapeNets/paper.pdf
"To study 3D shape representation, we propose to represent
a geometric 3D shape as a probability distribution of
binary variables on a 3D voxel grid. Each 3D mesh is represented
as a binary tensor: 1 indicates the voxel is inside the
mesh surface, and 0 indicates the voxel is outside the mesh
(i.e., it is empty space). The grid size in our experiments is
30 × 30 × 30."
    - `./binvox -cb -d 30 sample.off`
    - this takes a couple of seconds
