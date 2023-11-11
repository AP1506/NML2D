# NML2D
Normal mapping controls for pixel animation and images.

Currently not very user-friendly.

Prior Requirements:
- PIL python library
- numpy library

You can create an animation of a 2D image with an infinitely distant light source rotating around the object.
To configure the image used, update the image locations on line 157 in NormalMapExtract.py
Please note that because of an unresolved bug, the diffuse light map and normal map should be the same.

You can modify the number of frames on line 159 in NormalMapExtract.py.
