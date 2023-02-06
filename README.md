# ShapleyFL: Robust Federated Learning Based on Shapley Value

Implementation of the FL paper : ShapleyFL: Robust Federated Learning Based on Shapley Value

**The image classification experiments** are produced on MNIST, Fashion MNIST and CIFAR10. Since the purpose of these experiments are to illustrate the effectiveness of the federated learning paradigm, only simple models CNN are used.

We study 5 popular data and model poisoning scenarios based on the Non-IID data setting

1. imbalanced data with long-tailed distribution
2. irrelevant data with open-set label noise
3. malicious clients with closed-set label noise
4. malicious clients with data noise
5. attacks with gradient poisoning

**The medical diagnosis experiments** are conducted on a realistic cross-silo healthcare dataset Fed-ISIC2019. The best-performing EfficientNets architecture is used as the central model. The code is based on Flamby( https://github.com/owkin/FLamby/tree/main/flamby/datasets). Please see its document for more details.

## Requirments

Install all the packages from requirements.txt

## Data

#### **Image Classification**

Download train and test datasets manually or they will be automatically downloaded from torchvision datasets.

#### **Medical Diagnosis**

We keep the orginal data creation and image preprocessing script from FLamby/fed_isic2019(https://github.com/owkin/FLamby/tree/main/flamby/datasets/fed_isic2019). Please see its document for more details.

## Running the experiments

#### **Image Classification**

E.g. Fashion Mnist Dataset with long-tailed data distribution noise on GPU no.1 

```shell
cd ImageClassification/src_opt
python AfedSV+.py --model=cnn --dataset=fmnist --epochs=100 --num_users=100 --frac=0.1 --gamma_sv=0.3 --noise=1 --gpu=1
```

**Option**

* ```--dataset:```  Default: 'cifar'. Options: 'fmnist',  'cifar'
* ```--model:```    Default: 'mlp'. Options: 'mlp', 'cnn'
* ```--gpu:```      Default: None (runs on CPU). Can also be set to the specific gpu id.
* ```--epochs:```   Number of rounds of training.
* ```--num_users:```Number of users. Default is 100.
* ```--frac:```     Fraction of users to be used for federated updates. Default is 0.1.
* ```--gamma_sv:```     SV update ratio.  Default is 0.3.
* ```--noise:```     Type of noise injected. Default is 0.
  * 0 - NonIID
  * 1 - Long-tailed distribution
  * 2 - Open-set label noise
  * 3 - Closed-set label noise
  * 4 - Data noise
  * 5 - Gradient poisoning
* ```--noiselevel:```     Level of noise injected. Default is 0.

**Medical Diagnosis**

```shell
cd MedicalDiagnosis
python AfedSV.py
```

## References

[1] FL framework(FedAvg) https://github.com/AshwinRJ/Federated-Learning-PyTorch

[2] Fed-ISIC2019 dataset https://github.com/owkin/FLamby/tree/main/flamby/datasets
