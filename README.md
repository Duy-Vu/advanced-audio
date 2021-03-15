# advanced-audio
Acoustic Scene Classification (ASC) is the task of classifying the location where the audio is recorded. Each audio corresponds to one target class (in total of 10 classes), which specifies which scenes the recordings take place. The 10 present locations are airport, shopping mall, metro station, street pedestrian, public square, street traffic, tram, bus, metro, park. There are no cases of multi-labeled data [1]. ASC was released in two subtasks A and B. This report works only in Subtask A, which consists of 64 hours of audio recorded in 10 different European cities using 9 (real and simulated) microphone devices. We employed a ResNet fusion model scheme that learns features from different frequency bins and merges the learnable features to predict the target class. Without any augmentation, but using data that are not originally included in the train segments of the cross-validation setup, in other words, 20,070 audio segments instead of 13965, the model was able to achieve the accuracy of around 77% on development test dataset. 


## Feature extraction
```python3 tools/feature_extraction_delta.py```

## Model training
2-ResNet
```python3 method.py -c settings_2resnet -j JOB_ID```

3-ResNet
```python3 method.py -c settings_3resnet -j JOB_ID```

