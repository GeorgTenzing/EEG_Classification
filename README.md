# EEG_Classification


Main idea: we take 4 ssvep frequencies and map them to controlling a drone, maybe even in a race 
small ideas / todos:
- work on filtering such as melspectrogram fft or normalisations to get rid of filter 
- work on integrating it into drone control 
- work on live usage


Model ideas: 
# Ideas: big issue might be data, 
# random parameters: weighs in validation, lr, preproc 

TODOs:
- confusion matrix scale in percentage 
- find a way to retest models ur use them to evaluate data, aka load them from checkpoint 


after training, do testing, print class accuracies and confusion matrix and plot val results and save all in the folder with the metrics.csv, 
check on new data 

only take model prediction for drone control if probability is above 50 