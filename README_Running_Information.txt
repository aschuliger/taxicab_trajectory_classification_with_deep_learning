Taxicab Trajectory Classification using Deep Learning
Name: Ashley Schuliger

Running: The data you input into this model needs to be in the same
format as the data used for training (i.e. plate, longitude, latitude, time,
serving/seeking). Any files used for prediction then need to be put into a folder with
the name of your choosing. Then, in the run_job_eval.sh file, the folder "validation_data"
needs to be replaced with the folder name you chose. Once this is complete, you need
to create a "preprocessed_data" folder on your machine, and do not fill it. Then, run
"sbatch run_job_eval.py", and the resulting output will show the accuracy of the model's
prediction.