Video Object Self-Supervised Learning
=====================================================================================

[![Alt text](https://img.youtube.com/vi/R9dj6N_YDJU/0.jpg)](https://www.youtube.com/watch?v=R9dj6N_YDJU)

Visit our [Project Page](https://sites.google.com/view/ltnghia/research/video-self-annotation) for accessing the paper, and the pre-computed results.

We tested the code on python 3.7, PyTorch 1.2.0 and CUDA 10.1

## Installation

If you need to run Self-Annotation, please install 
[Annotation Interface](https://github.com/ltnghia/video-object-annotation-interface) and interfere the training process in [run.sh](run.sh).


## Prepare Data

```
mkdir -p -- CityScapes
cd CityScapes
mkdir -p -- val
cd val
mkdir -p -- Raw
cd Raw
```

Download [leftImg8bit_sequence_trainvaltest.zip (324GB)](https://www.cityscapes-dataset.com/downloads) and extract all sequences of val-set to CityScapes/val/Raw. 
For example, ./CityScapes/val/Raw/frankfurt/frankfurt_000000_000275_leftImg8bit.jpg

Download and extract [our pre-trained model](https://drive.google.com/file/d/10bqv7fUeUEdT1Q9T617QTcttit5EJi76/view?usp=sharing) to CityScapes/val/Initial_model

## Citations
Please consider citing this project in your publications if it helps your research:

```
@Inproceedings{ltnghia-WACV2020,
  Title          = {Toward Interactive Self-Annotation For Video Object Bounding Box: Recurrent Self-Learning And Hierarchical Annotation Based Framework},
  Author         = {Trung-Nghia Le and Akihiro Sugimoto and Shintaro Ono and Hiroshi Kawasaki},
  BookTitle      = {IEEE Winter Conference on Applications of Computer Vision},
  Year           = {2020},
}
```

## License

The code is released under the [Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License](https://creativecommons.org/licenses/by-nc-sa/3.0/), and used for academic purpose only.

## Contact

[Trung-Nghia Le](https://sites.google.com/view/ltnghia).

