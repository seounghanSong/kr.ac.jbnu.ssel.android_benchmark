# **A comparative study on deep learning models for face detection task in mobile environment**

As mentioned in title, this project aims to compare the performance of variety deep learning models which made for face detection task within PC and Mobile environment both.

## Installing & checking out

First, clone this repository with below command.

```
git clone git@github.com:seounghanSong/kr.ac.jbnu.ssel.android_benchmark.git
```

Under this repository, there a separated branches that contains android implementation of each model.
And below table indicates the experiment result from PC environment.

#### *Only 20 instance have used!!

| Models(Alpha. order)   | Average Precision Rate | Average Speed(s) |
| -------------          | -------------          | -------------    |
| DSFD                   | 1.0000                 | 11.35295         |
| FaceBoxes              | 1.0000                 | 0.4185           |
| PyramidBox             | 0.9833                 | 9.51915          |
| RetinaFace_Resnet50    | 1.0000                 | 1.99225          |
| RetinaFace_MobilenetV1 | 1.0000                 | 0.1849           |


You can use below command to look around different models.

* Pytorch_DSFD[https://github.com/hukkelas/DSFD-Pytorch-Inference] - base repository

```
git checkout Pytorch_DSFD
```


* FaceBoxes[https://github.com/XiaXuehai/faceboxes] - base repository

```
git checkout Pytorch_FaceBoxes
```


* PyramidBox[https://github.com/EricZgw/PyramidBox] - base repository

```
git checkout Tensorflow_PyramidBox
```


* RetinaFace_Resnet50[https://github.com/hukkelas/DSFD-Pytorch-Inference] - base repository

```
git checkout Pytorch_RetinaFace_Resnet50
```


* RetinaFace_MobilenetV1[https://github.com/hukkelas/DSFD-Pytorch-Inference] - base repository

```
git checkout Pytorch_RetinaFace_MobilenetV1
```


## Deployment

This Project aim to deploy the face detection model (developed in PC) to the edge-device. Specifically speaking, the mobile phone, so Under this condition, **Android device (Samsung Galaxy S7 edge)** will be used.

## Built With

* [Android](https://developer.android.com/) - Target Mobile Environment
* [Gradle](https://gradle.org/) - Android Dependency Management
* [Pytorch Mobile](https://pytorch.org/mobile/home/) - Used to generate mobile runnable models
* [Tensorflow Lite](https://www.tensorflow.org/lite) - Used to generate mobile runnable models (not fully tested yet)

## Authors

* **Seounghan Song** - *Initial work* - [Github Repo](https://github.com/seounghanSong)


## Acknowledgments

* This project has been conducted for 10 days at **2020 Jeju ML (Machine Learning Camp)** from 08-17-2020 to 08-26-2020
