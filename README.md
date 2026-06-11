Okay, so this is not a proper readme file. However this contains the details of what and how you can use these codes.
Make sure you have downloaded the datasets from the drive link I have pasted next.
Drive link for SEED_VIG.mat: https://drive.google.com/file/d/15DxU3uSflzje76WK5w11mmvE6iLLAW09/view?usp=drive_link
Drive link for dataset.mat: https://drive.google.com/file/d/1r01Y8Y0YAyKYKbecAp48N1yGdkO7EybI/view?usp=drive_link
Note: These are the original work of the creators and I have no role in creating these datasets.
So, you need to use codes in this manner:
preprocessing->segmentation->models->neural operator->classifier head->neurodyn opnet->dataloader->train->predict fatigue->visualize results
Run the codes exactly in this manner.
You might need to change the location in these codes according to your location of datasets and codes etc.
Some codes give more files as output too, so you should be aware of that.
Training the model takes 1-1.5 hours, so its important to not do anything in your laptop at that point of time else the training might be delayed.(close other background apps(recommended))
So according to my model and preprocessing algorithm, you can see the stats in loso_results_summary.txt.
Rest if any doubt ask AI models by uploading these codes.
There were many other files that got created as I mentioned, but I am not uploading them.
A google colab link has been added too, there you can directly work on the code. There are slight differences in the code but they work the same.

Update:
I have completed the work and the final work has been uploaded. The name of the file is "NeuroDyn_V4_version3.ipynb". It is a jupyter file. Run it locally, just change the directory and make sure the datasets are in that directory.
Peace V
