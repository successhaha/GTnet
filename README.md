# GTnet
Graph Embedding and Optimal Transport for Few-Shot Classification of Metal Surface Defect
<https://ieeexplore.ieee.org/document/9761830?source=authoralert>

Requirements

To install requirements:

           pip install -r requirements.txt

Dataset:

link：https://pan.baidu.com/s/14-x_blzNvtY7N5Ue1U2skw password：z584

Code：

link：https://pan.baidu.com/s/1H9ohxDf2qwxKkHgj9UQa2A password：aoi3 

Datasets split:
           
           Move the datafile to dataset/
           Run 'python write_dataset_filelist.py'
  
           


Training

To train the feature extractors in the paper, run this command:

           python train.py --dataset [miniImagenet/CUB] --method [S2M2_R/rotation] --model [WideResNet28_10/ResNet18] --train_aug

Evaluation

To evaluate my model on miniImageNet/CUB/cifar/cross, run:
For miniImageNet/CUB

           python save_plk.py

           python test.py



Hyperparameter setting

common setting:

           1-shot: k=10 kappa=9 beta=0.5 5-shot: k=4 kappa=1 beta=0.75

cross-domain setting1 :

           1-shot: k=10 kappa=1 beta=0.7 5-shot: k=4 kappa=1 beta=0.6

cross-domain setting2 :

           1-shot: k=10 kappa=1 beta=0.7 5-shot: k=4 kappa=1 beta=0.5

Contact the author e-mail：1900412@neu.edu.cn or 2878570391@qq.com
