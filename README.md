# HOW TO USE
1. ``pip3 install requirements.txt``
2. Add some pictures of known people in the folder 'people' :
  a) Create a folder called by the name of the person
  b) Add pictures of it
3. Add some pictures of unknown people in the uk folder
4. ``python3 face_recoginiton.py``

# OPTIONS

``--known_people_folder`` : Insert custom known_people_folder.

``--unknown_people_folder`` : Insert custom unknown_people_folder.

``--cpus`` : Number of CPU cores to use in parallel (can speed up processing lots of images). -1 means "use all in system".

``--tolerance`` : Tolerance for face comparisons. Default is 0.6. Lower this if you get multiple matches for the same person.

``--show-distance`` : Output face distance. Useful for tweaking tolerance setting.

``--show-final`` : Show best result of recogintion. Use it to get an indicator of the accuracy of the results (lower is better).

  
  Script by @antoinebollengier and principally from https://github.com/ageitgey/face_recognition
