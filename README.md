# Document Recognition App
This app uses MaskRCNN, UNets, and YOLOv8 to detect documents. This is built using Pytorch and FastAPI.

## Instructions for running
### Cloning the repo
```git clone https://github.com/vaibagga/doc-recognition```

### Runnning the training script along with hyperaparameter tuning
```pip install -r requirements.txt``` <br>
```python src/train.py```

### Running FastAPI app
#### Downloading pretrained weights
<a href="https://drive.google.com/file/d/1_ZTU1KcnR7prQIKw0zUCFXdcORICzRb8/view?usp=drive_link"> Link for weights </a>
<br>

```pip install -r requirements.txt```

```python app.py```

#### Running
Go to <a href="http://0.0.0.0:8000/docs"> Swagger UI </a><br>
Run the ```predict``` API to run on a single API. <br>
Run the ``predict_on_video`` to run on a video.
