# How to 
## Data structure
- <repo_home>
  - data
    - images (contains the images)
      - a.png
      - b.png
    - labels (contains the binary segmentation results as bw images)
      - a.png
      - b.png

## Train
Run TrainScript.py to train on the dataset in ./data .
- checkpoints will be dumped to ./checkpoints
- tensorboard logs will be dumped to ./logs

## Eval
Run EvaluationScript.py to dump segmentation results for all images in the ./data folder.
Run "streamlit run ThresholdVisualizationTool.py" to get an interactive gui for result evaluation. 


