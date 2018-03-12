## Download The Data 

```
cd 3d_model_retrieval/
wget http://vision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip -O .
wget http://modelnet.cs.princeton.edu/ModelNet40.zip -O .
```

    
**Install binvox binary**
    - MACOSX install:
        - `download http://www.patrickmin.com/binvox/download.php?id=6`
        - `chmod 755 binvox`
        - put into root of 3d_model_retrieval/
    - Linux install:
        - `download http://www.patrickmin.com/binvox/download.php?id=4`
        - `chmod 755 binvox`
        - put into root of 3d_model_retrieval/
        
**.off files*

Processing *.off files is done by a 3rd party tool made by Patrick Min.
    - https://www.patrickmin.com/binvox/

*Voxel Resolution*
    - http://vision.princeton.edu/projects/2014/3DShapeNets/paper.pdf
"To study 3D shape representation, we propose to represent
a geometric 3D shape as a probability distribution of
binary variables on a 3D voxel grid. Each 3D mesh is represented
as a binary tensor: 1 indicates the voxel is inside the
mesh surface, and 0 indicates the voxel is outside the mesh
(i.e., it is empty space). The grid size in our experiments is
30 × 30 × 30."
    - `./binvox -cb -e -c -d 30 sample.off`
        - -e  is important, with a lot of troubleshooting it was shown that not using this led to inconsistent voxelization :headache:
** *Viewing .binvox data for troubleshooting* **

`./viewvox <filename>.binvox`
        
## Convert all *.off files to *.binvox
`

## Load The Data in Python

```
from data import load_data
(x_train, y_train), (x_test, y_test), target_names = load_data('./ModelNet10')
```
