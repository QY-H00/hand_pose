# hand_pose

### Todo
1. Offline data processing
  - write an independent function, which inputs 2D keypoint, heatmap size and generates flux map and distance map. 

2. Validation indicators
  - we may use 2D distant (EPE) in the original images. In this case, we need a function to project our prediction back to the original image coordinate. I attached my function in the folder named for inference.

3. Store the model regularly during training

4. During the training of the network, try to explore the related papers. We can discuss the papers during the meeting.
  - One we may expore is to build the connection between mask and flux.

5. Once we have some initial results, we can further refine the pipeline/output maps or add more losses based on the analysis of the resutls.

