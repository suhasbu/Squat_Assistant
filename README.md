## Squat Assistant


## Data Augmentation Ideas for Franco
- Sample more non-apex (but near-apex) frames to use as examples of bad squats

## Explore pose matching for Kim/Meher/Kathy
- Feature Engineering
  - Angle at knee
  - Angle at Hip
  - Difference in X between knee and ankle
  - Difference in Y between hip and knee
- Classification by Machine Learning 
  - Basic Neural Net
  - Random Forest
  - SVM
  - XGBoost
  - etc
- Classification by Matching Averages
  - Calculate Mean and Std.Deviation for each feature and classify as good or bad based on whether the test value falls wihtin n            std deviations of the mean. 

## Instruction to run
This is currently only a simple html-css-js project. Frameworks may be added when the simplicity becomes a bottleneck. 
To run, download the repo and just start a simple server 
```
python -m http.server
```

Go to ```localhost:8000``` on Chrome to see the site (as opposed to 0.0.0.0:8000) as camera on Chrome only runs on https or localhost. Then click on the debug card. That has eval running. 


