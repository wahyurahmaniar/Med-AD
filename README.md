### Med-AD
Med-AD: Unsupervised Anomaly Detection for Multi-Modality Medical Images
Wahyu Rahmaniar and Kenji Suzuki (2024)


### Usage 
~~~
# python 3.8, torch==1.10.0, torchvision==0.11.1
pip install -r requirements.txt
python train.py --phase train or test --dataset_path ./medical--category brain --project_path ./results
python train.py --phase train --dataset_path ./medical--category brain --project_path ./results
