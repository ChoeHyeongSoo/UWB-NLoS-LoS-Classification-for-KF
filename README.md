# UWB-NLoS-LoS-Classification-for-KF
Research of NLoS probability calculation model based on UWB CIR using deep learning

[Abstract]  
-Classify the environment(LoS/NLoS) by analyzing CIR  
-The NLoS probability will be used as a parameter to update the process noise covariance in the KF

[Additional]  
-Try CNN+LSTM to improve accuracy

> Set all environment as follows:
1. [Reference Journal #1: Accurate Indoor Positioning for UWB-Based Personal Devices Using Deep Learning](https://ieeexplore.ieee.org/document/10054386)
2. [Reference Paper #2: UWB NLOS/LOS Classification Using Deep Learning Method](https://ieeexplore.ieee.org/document/9108193)

[DataSet]  
-Use data set and some function from https://github.com/ewine-project/UWB-LOS-NLOS-Data-Set.git  
-Get reference from https://github.com/JSYoon-git/UWB-LOS_NLOS-Classification-using-DL.git

---

## 1. Theory

<p align="center">
  <img src="figures/fig1_uwb_txrx.png" width="520" alt="UWB TX sends an impulse; RX receives a direct and a reflected path"><br>
  <sub><b>Fig. 1.</b> TX/RX form of a UWB-based communication system [1]</sub>
</p>

The transmitted impulse reaches the receiver as a CIR (Channel Impulse Response) distorted by the propagation environment. Environments are divided into **LoS** (e.g. open space) and **NLoS** (e.g. in a pocket or bag), which differ in signal distortion and delay. LoS measurements have small errors and can be trusted, while NLoS measurements are less reliable due to noise.

<p align="center">
  <img src="figures/fig2_ranging_correction.png" width="760" alt="Ranging measurement correction algorithm: DNN classifier feeding a weighted adaptive Kalman filter"><br>
  <sub><b>Fig. 2.</b> Ranging measurement correction algorithm [1]</sub>
</p>

The algorithm takes the extracted CIR as input:

1. The classifier distinguishes LoS/NLoS and provides an **NLoS probability**.
2. The process / measurement noise covariances **Q** / **R** are updated.
3. The error covariance and Kalman gain of the Kalman filter are updated.
4. A corrected estimate is computed, and the state error covariance **P** is updated for the next step.

<p align="center">
  <img src="figures/fig3_nlos_prob_kf.png" width="620" alt="Kalman filter steps with the NLoS probability updating Q and R"><br>
  <sub><b>Fig. 3.</b> Use of the NLoS probability in the Kalman filter [1]</sub>
</p>

**Operation in an NLoS environment**

- Measurement reliability drops, so **R** (measurement noise covariance) increases.
- As R increases, the Kalman gain **K** decreases.
- The filter trusts its prediction more and the noisy (distorted) measurement less.

## 2. Dataset

Each sample contains 15 measurement fields, the `NLOS` label (0 = LoS, 1 = NLoS), and **1016 CIR values** (`CIR0`–`CIR1015`). The CIR can be viewed as a 3D surface of (time, index, amplitude), projected onto (time, index), or sliced at one time step as (index, amplitude).

<p align="center">
  <img src="figures/fig4_cir_3d.png" width="520" alt="3D surface of CIR amplitude over time and index"><br>
  <sub><b>Fig. 4.</b> CIR data in a LoS case: 3D (time, index, amplitude) [1]</sub>
</p>

<table>
  <tr>
    <td align="center" width="50%"><img src="figures/fig5_cir_projection.png" alt="Heatmap of CIR amplitude over time and index"><br><sub><b>Fig. 5.</b> Projection on amplitude: 2D (time, index) [1]</sub></td>
    <td align="center" width="50%"><img src="figures/fig6_cir_slice.png" alt="Single CIR sample with a peak near index 750"><br><sub><b>Fig. 6.</b> Slice on time: 2D (index, amplitude)</sub></td>
  </tr>
</table>

## 3. Experimental Results

Data split: train / validation / test = 81 : 9 : 10 (stratified, `random_state=42`).

| Model | Architecture | Epochs | LR | Accuracy |
|---|---|:---:|:---:|:---:|
| DNN | FC 512 → 256 → 2 | 100 | 0.01 | 0.5000 |
| CNN | Conv1d(10, k4) → Conv1d(20, k5) → MaxPool → FC 128 | 10 | 0.01 | 0.7413 |
| CNN | 〃 | 10 | 0.001 | 0.8559 |
| **CNN + LSTM** | **Conv1d ×2 → BiLSTM(32, 2 layers) → FC 128** | **100** | **0.001** | **0.9616** |

### 3.1 DNN

<p align="center">
  <img src="figures/fig7_dnn_metrics.png" width="240" alt="DNN metrics: accuracy 0.5, precision 0.25, recall 0.5, F1 0.3333"><br>
  <sub><b>Fig. 7.</b> DNN evaluation metrics</sub>
</p>

The DNN was evaluated only as a simple baseline. Precision 0.25 with recall 0.5 indicates that it predicted a single class for every sample.

### 3.2 CNN

<p align="center">
  <img src="figures/fig8_cnn_log.png" width="760" alt="CNN training log over 10 epochs, test accuracy 0.8559"><br>
  <sub><b>Fig. 8.</b> CNN training log (batch size and epochs from [2], LR = 0.001)</sub>
</p>

<p align="center">
  <img src="figures/fig9_cnn_curve.png" width="560" alt="CNN loss and accuracy curve"><br>
  <sub><b>Fig. 9.</b> CNN training loss and accuracy</sub>
</p>

### 3.3 CNN + LSTM

<p align="center">
  <img src="figures/fig10_cnn_lstm_log.png" width="680" alt="CNN+LSTM training log, test accuracy 0.9616"><br>
  <sub><b>Fig. 10.</b> CNN + LSTM training log</sub>
</p>

<p align="center">
  <img src="figures/fig11_cnn_lstm_curve.png" width="560" alt="CNN+LSTM loss and accuracy curve"><br>
  <sub><b>Fig. 11.</b> CNN + LSTM training loss and accuracy</sub>
</p>

Combining a BiLSTM with the CNN to learn the temporal pattern of the CIR raised accuracy from **0.86 to 0.96**. The model with the best performance will be used to extract NLoS probabilities from DWM-1000 measurement data for Kalman filter correction.

## References

[1] [Accurate Indoor Positioning for UWB-Based Personal Devices Using Deep Learning](https://ieeexplore.ieee.org/document/10054386)  
[2] [UWB NLOS/LOS Classification Using Deep Learning Method](https://ieeexplore.ieee.org/document/9108193)  
[3] [ewine-project/UWB-LOS-NLOS-Data-Set](https://github.com/ewine-project/UWB-LOS-NLOS-Data-Set)  
[4] [JSYoon-git/UWB-LOS_NLOS-Classification-using-DL](https://github.com/JSYoon-git/UWB-LOS_NLOS-Classification-using-DL)
