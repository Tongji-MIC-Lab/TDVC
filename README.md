
# Task-Driven Video Compression for Humans and Machines: Framework Design and Optimization

This is the source code for our paper [Task-Driven Video Compression for Humans and Machines: Framework Design and Optimization].

## Data preparation

### Process commands for the video compression backbone
#### Training data
1. Download the Vimeo90k dataset ([Download link](http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip)) .
2. Unzip the dataset into ./data/.
3. Run scripts in tools/process to generate
#### Validation data
1. Download the UVG dataset ([Download link](http://ultravideo.cs.tut.fi/#testsequences_x)).
2. Download the HEVC dataset ([Download link](ftp://ftp.tnt.uni-hannover.de/testsequences)).
3. Download the MCL\_JCV dataset ([Download link](http://mcl.usc.edu/mcl-jcv-dataset/)).
4. Runn scripts in tools/preprocess to generate I-frames.
### Process commands for the action recognition 
#### Training data & Validation data
1. Download the UCF-101 dataset ([Download link](https://www.crcv.ucf.edu/data/UCF101.php)) .
2. Runn scripts in app/VideoClassification/preprocess to generate compressed frames.

## Training
The video compression backbone is first trained as a pre-trained model for the overall framework.
### Training for the video compression backbone

```
cd tools
python train.py --cfg ../cfg/train.yaml
```

### Training for the overall framework 

```
cd app/VideoClassification
python train_cls.py --cfg cfg/compress.yaml
python train_sr.py --cfg cfg/ehc.yaml
```

## Testing
```
cd app/VideoClassification
python predict.py
python predict_sr.py
```