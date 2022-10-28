# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## TO DO:
- nothing at the moment

## 2022-10 Shuqi Xu

### Change
- Replace Mayavi with PyVista
- 3D visualization of the mesh supports 28 different colors. If the number of electrodes exceeds 28, duplicate colors will be used.
- A slider is added to highlight the chosen electrode with bold edges. This can be used to identify electrodes.
- Adjust the judgment condition of function ```partition_normals_interception``` inside stl.py file. Now it classify two triangles within the same plane but with opposite normals to the same face.
- Adjust the judgment condition tolerance inside stl.py file. It is now unitless.
- Can read mesh from vtk.
- Also install ```helper_functions``` and ```bemColors```. To import, run ```from helper_functions import helper_functions``` and ```from bem.bemColors_lib.bemColors import bemColors```
- Tested compatible with high performance cluster
- Improve README from Yi's bem-savio

## 2022-07 Wenhao He

### Change
- make this package compatible in Windows
    - main change triangle/triangulate.c and triangle/triangulate.h
    - for windows, we must change long into __int64
    - reference: 	https://blog.csdn.net/cuihaolong/article/details/72842026.  and   	https://www.796t.com/post/YThtcXk=.html
- 7/11/2022: tested compatible with latest vtk==9.0.3, latest mayavi==4.7.4, and old PyQt5==5.15.6 on python==3.9.12 on Linux(Ubuntu 20.04.3) , macOS(Monterey 12.1 intel) , HPC(CentOS Linux release 7.7.1908 (Core)) and Windows(Windows 10 pro 21H2).



## 2022-05 Wenhao He

### Change
- simulate overhang htrap(3D electrode): original code(especially triangulation part) behaves bad for 3D trap, so a lot of changes are made in Bem.
    - unify same points in mesh structure
    - counter clockwise triangle when passed into the package
    - originally, parallel planes will be classified as one plane, which leads to error for 3D cases. Fix now
    - now we can combine triangles first and cut it, so sharp triangles are eliminated
    - fix the error: exceed float precision .......
- a more convenient way to load color (explained in the document)
    - a set of standard color
    - more regular color names
- simplify the environment set up process, with one command: pip install .
    - setup.py, setup.cfg, pyproject.toml and MANIFEST.in
- compatible with python 3.9
    - solve the problem of multiprocessing
    - solve the environment for mayavi PyQt vtk
    - get around mayavi in our workflow
        - our code can run on HPC(CentOs)
        - saves a lot of effort to maintain the code in the future
- others
    - make unit problem clear(micronmeter or millimeter are both ok now)
    - split readme into readme+changelog+copying
    - bugs in ./fastlap
    - other small bugs

- add a document: ./examples/document

## 2022-03 Shuqi Xu

### Change
- fix bugs on partition normal
- fix declaration error on building process
- update documentation
- update cythonize version
- 3/1/2020: tested compatible with latest vtk==9.0.1 and latest mayavi==4.7.2 on python==3.6.12 on both Linux and macOS. Need more test on newer Python versions and on Windows.
- etc










