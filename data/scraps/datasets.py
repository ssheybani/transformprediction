# Manages construction of the dataset. 
# Handles data pipelining/staging areas, shuffling, reading from disk.

"""

 To do:

Pre-packaging verification:
 - Write a script that converts each JSON file into a Pandas DF. DONE
 Then into a numpy array. Then append other samples.
 - Do visualizations:
 	Quiver plot
 	Histogram

 	=> DONE

Assign numbers to the shapes.

Compute the CIELAB colors.


- Save the dataset as a collection of numpy arrays following the example of dSprites:
	- imgs:
		(shape x trajectory x image x height x width x channel))
	    (10 x 500 x 4 x 256x 256, 3, uint8) Images in RGB.
	- latents_values
		(10 x 500 x 4 x 16, float64) Values of the latent factors.
	- latents_classes
		(10 x 500 x 4 x 16, int64) Integer index of the latent factor values. 
	- metadata
		date
		description
		version
		'latents_names': 
			('color', 'shape', 'scale', 'orientation', 'posX', 'posY'), 
		'latents_possible_values': 
			{'rotX': array([0.        , 0.16110732, 0.32221463, 0.48332195, 0.64442926, ...
			rotY
			rotZ
			rotAxX
			rotAxY
			rotAxZ
			posR
			posA
			posE
			dPosX
			dPosY
			dPosZ
			shape: array([0., 1., 2., ...,9.])
			}
		'latents_sizes': array([ 1,  3,  6, 40, 32, 32])
		'author': 'lmatthey@google.com', 
		'title': 'dSprites dataset'

 - Copy and modify every function in nma_datasets.py
 	Ignore the few that don't seem useful.
 	To finish faster, leave some unimplemented, those that take a while but are not used immediately.

	Goal: having all of the utility functions needed for dealing with the dataset.
"""