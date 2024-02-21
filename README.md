# iMOB-Demo
Neural network-assisted device modelling tool.
Rapid modelling of device compact model based on the nonlinear fitting capability of neural networks.
The tool provides a modelling interface with flexible and customizable model training scripts.

- [Publications](#publications)
- [Dependency](#dependency)
- [How to Install Python Dependency](#how-to-install-python-dependeny)
- [How to Run](#how-to-run)
- [Authors](#authors)
- [Features](#features)

# Publications
- [Lining Zhang](https://www.ece.pku.edu.cn/info/1045/2486.htm), [Mansun Chan](https://ece.hkust.edu.hk/mchan),
  "**Artificial neural network design for compact modeling of generic transistors**",
  Journal of Computational Electronics 16, 825¨C832 (2017), doi:10.1007/s10825-017-0984-9.

- Zhao Rong, Wu Dai, Ning Feng, Yuhang Yang, [Lining Zhang](https://www.ece.pku.edu.cn/info/1045/2486.htm), Zongwei Wang, Yimao Cai, and [Mansun Chan](https://ece.hkust.edu.hk/mchan),
  "**Generic Compact Modeling of Emerging Memories with Recurrent NARX Network**",
  IEEE Electron Device Letters, vol. 44, no. 8, pp. 1272-1275, Aug. 2023, doi: 10.1109/LED.2023.3290681.

- Wu Dai, Yu Li, Zhao Rong, Baokang Peng, [Lining Zhang](https://www.ece.pku.edu.cn/info/1045/2486.htm), Runsheng Wang, Ru Huang,
  "**Statistical Compact Modeling With Artificial Neural Networks**",
  IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, vol. 42, no. 12, pp. 5156-5160, Dec. 2023, doi: 10.1109/TCAD.2023.3285032.

- Wu Dai, Fangxing Zhang, Kaifeng Wang, Yu Li, Yukun Tang, Qianqian Huang, [Lining Zhang](https://www.ece.pku.edu.cn/info/1045/2486.htm), and Ru Huang,
  "**An Automatic Integration Network Approach for Generic Device Charge Modeling**",
  2022 IEEE 16th International Conference on Solid-State & Integrated Circuit Technology (ICSICT), Nangjing, China, 2022, pp. 1-3, doi: 10.1109/ICSICT55466.2022.9963215.

- Wu Dai, Yu Li, Baokang Peng, [Lining Zhang](https://www.ece.pku.edu.cn/info/1045/2486.htm), Runsheng Wang, and Ru Huang,
  "**Benchmarking Artificial Neural Network Models for Design Technology Co-optimization**",
  2023 International Symposium of Electronics Design Automation (ISEDA), Nanjing, China, 2023, pp. 423-427, doi: 10.1109/ISEDA59274.2023.10218514.

# Dependency
- [Python](https://www.python.org/) >= 3.5
- [Pytorch](https://pytorch.org/) >= 1.6
- [Numpy](https://numpy.org/) >= TBD
- [Matplotlib](https://matplotlib.org/) >= TBD
- [pandas](https://pandas.pydata.org/) >= TBD

# How to Install Python Dependency
Go to the root directory and then

```
pip install -r requirements.txt
```
# Working flow
Before running, make sure the python dependency packages have been installed.
Python IDE such as Pycharm is also recommended for running imob.
```
cd <installation directory>
python Run_Imob.py
```
The main interface of iMoB is displayed as follows:
<img src=/images/main_interface.png width=300>
After loading the example device current data(Data_sample.txt), the I-V curves will be shown:
<img src=/images/data_loading.png width=300>
By clicking the train Script buttom, the example training script can be loaded for demostration.
<img src=/images/model_training.png width=300>
The training process is displayed at the IDE terminal:
<img src=/images/training_process width=300>
# Authors
-Ying Ma and Yu Li, supervised by [Lining Zhang](https://www.ece.pku.edu.cn/info/1045/2486.htm), composed the demo version release.
