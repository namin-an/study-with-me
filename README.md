# drone-nman

This repository is for summarizing papers read in the first semester (Spring 2022) of my graduate school, as well as ideas or insights that I have gotten while doing a research. You can test some functions related to the works by running the sample code below.



## Example run

```
python3 test.py --functions reinforcement_learning --evaluate True
```


## Papers

> Nicolas-Alonso *et al.* [*Sensors*, 2012]: Overall review of the SOTA BCIs of which steps include brain signal acquisition and enhancement method, mostly used feature extraction and classification algorithm, and their applications.

> Chen *et al.* [*PNAS*, 2015]: SSVEP-based BCI which achieves high spelling rate of 12 words per min. and surpasses other BCI spellers.

> Cho *et al.* [*IEEE Trans. Cybern.*, 2021]: Real-time classification model using MI-based EEG and EMG data for natural hand-grasp task.

> Kwon *et al.* [*IEEE Trans. Neural Netw. Learn. Syst.*, 2020]: Subject independent decoder of MI-based EEG data that outperforms the conventional CSP, CSSP, FBCSP, and BSSFO.

> Schroff *et al.* [*CVPR*, 2015]: The proposal of FaceNet, which maps feature representation of raw images into Euclidean space to measure face similarity using triplet loss.

> Palazzo *et al.* [*PAMI*, 2020]: Joint learning using VI-based EEG data as "anchor" and images as "positives (or negatives)" without relying on classification task.

> Spampinato *et al.* [*CVPR*, 2017]: The first image classifier using EEG data. 

> Schirrmeister *et al.* [*Hum. Brain. Mapp.*, 2017]: Decoder of EEG data using Deep ConvNet that does not depend on fixed priori frequencies like FBCSP.

> Ang *et al.* [*IJCNN*, 2008]: The advent of FBCSP which automatically applies CSP for non-overlapping filter banks which have different frequency ranges.

> Ang *et al.* [*Front. Neurosci.*, 2012]: One-versus-rest FBCSP yielding competitive performances in both BCI competition 4 datasets 2a and 2b.

> Rivet *et al.* [*IEEE Trans. Biomed. Eng.*, 2009]: xDAWN algorithm which can detect P300 visual evoked potentials in EEG by projecting the signal into the estimated P300 subspace and taking both signal and noise into account unlike PCA.

> Nguyen *et al.* [*NIPS*, 2016]: Synthesis of prefered input (e.g. an image) that highly activates certain neurons using DGN to learn which specific convolutional filter or unit has more impact than others. 

> Mnih *et al.* [*arXiv*, 2013, *Nature*, 2015]: Invention of DQN which uses experience replay to smooth out training and adds target DQN in learning process to generate Q-learning targets, which converges policy function values.

> Mnih *et al* [*ICML*, 2016]: RL using two distinct DNNs, namely, anchor network as generating probability of action with respect to the current feature vector under the state at time t and critic network, which produces the expected return under the current state.

> Bellemare *et al.* [*AAAI*, 2012]: New concept of contingenct regions in which future observation is solely determined by an agent's control but not by the environment itself.

> Hasselt *et al.* [*AAAI*, 2016]: Double Q-learning which solves the problem of overestimation in the recent DQN algorithm by decomposing the max operation in the target into action selection (argmax) and action evaluation (second weight parameter).

> Lawhern *et al.* [*J. Neural Eng.*, 2018]: EEGNet, a compact model that works well across P300 visual-evoked potentials, ERN, MRCP, and SMR compared to xDawn+RG (P300, ERN, and MRCP), DeepConvNet, ShallowConvNet, and FBCSP (SMR).

> Ancona *et al.* [*ICLR*, 2018]: Gradient-based attribution model that calculates how much gradients of neurons affect the prediction of the model.

> Shrikumar *et al.* [*PMLR*, 2017]: The proposal of DeepLIFT, which extracts the importance of features by comparing each neuron's activation with reference activation and assigns its difference to contribution score.

> Cecotti *et al.* [*PAMI*, 2011]: CNN classifiers to detect the presence of P300 in EEG and find the  optimal combinations of P300 responses for the target character to spell in the third BCI competition  dataset 2.

> Cecotti *et al.* [*IEEE Trans. Neural Netw. Learn. Syst.*, 2014]: The proposal of CNN of which layers correspond to supervised spatial filtering and classifier is MLP, which is trained based on maximization of AUC in order to accurately detect ERPs in EEG.

> Springenberg *et al.* [*ICLR*, 2015]: CNN composed of convolutional layers only that reaches SOTA performance on CIFAR-10, CIFAR-100, and ImageNet datasets.

> Srivastava *et al.* [*JMLR*, 2014]: Dropout to prevent overfitting problem in machine learning research.

> Chung *et al.* [*arXiv*, 2021]: Reinforcement aligner that helps agents make optimal duration prediction in order to produce enhanced synthesized audio from texts.

> Ko *et al.* [*IEEE Trans. Industr. Inform.*, 2022]: RL used for MI task-related signals selection and EEG classification.

> Ko *et al.* [*IEEE Comput. Intell. Mag.*, 2021]: EEG decoding model that learns feature representations of EEG signals with respect to different ranges of frequencies using a spectral convolution, three residually connected temporal separable convolutional layers, and spatial convolutional layers.

> Neftci *et al.* [*Nat. Mach. Intell.*, 2019]: Review of studies regarding connections between artificial and biological agent

> Kirkpatrick *et al.* [*Proc. Natl. Acad. Sci.*, 2017]: Implementation of EWC (elastic weight consolidation) into machine learning by training DNNs not to forget past tasks.

> Kaiser *et al.* [*Front. Neurosci.*, 2020]: SNN called DECOLLE (deep continuous local learning) that produces local error functions to mimic synaptic plasticity in biological neural networks.

> Nguyen *et al.* [*IEEE Trans. Cybern.*, 2020]: A survey related to multiagent deep RL (MADRL) which encourages further development in the future to create more robust and useful MADRL for real-world problems.

> Hausknecht *et al* [*AAAI*, 2015]: "Deep recurrent Q-learning for partially observable MDPs", where DQN uses more than four history frames for learning by replacing FC layer to LSTM.

> Lowe *et al* [*NIPS*, 2017]: MADDPG (multi-agent deep deterministic policy gradient) method that deals with non-stationarity problem with multiple actors trained with centralized critic.

> Stockl *et al.* [*Nat. Mach. Intell.*, 2021]: Development of FS (few-spike) neurons which can be replaced by ReLU or siLU activation function that considers timing of spikes and achieves similar image classification accuracy as the original models (ResNet and EfficientNet) on ImageNet2012 and CIFAR10 datasets.

> Ko *et al.* [*IEEE Comput. Intell. Mag.*, 2019]: Multimodal fuzzy fusion-based BCI system using Choquet and Sugeno integrals to better classify BCI commands.

> Shen *et al.* [*IEEE Trans. Fuzzy Syst.*, 2021]: Prediction and classification of multivariate long nonstationary time-series, such as EEG data, using high order fuzzy cognitive maps and 1D-CNN.

> Fromer *et al.* [*Nat. Comm.*, 2021]: Experimental proof that people consider expected reward AND efficacy of task performance to decide on how much effort they should exert.


## Datasets

|   Data   | # of channels | Sampling rate (Hz) | # of subjects | # of trials / subject | # of classes | Papers |
|:--------:|:-------------:|:------------------:|:-------------:|:---------------------:|:------------------:| :-----:|
| BCI Competition IV 2a dataset| 22 (EEG) + 3 (EOG) | 250 | 9 | 288 | 4 (left hand, right hand, foot or tongue) | Schirrmeister *et al.* [*Hum. Brain. Mapp.*, 2017], Lawhern *et al.* [*J. Neural Eng.*, 2018] |
| High Gamma dataset | 128 (EEG) | 60 - 100 | 14 | < 880 | 4 (left hand, right hand, foot or rest) | Schirrmeister *et al.* [*Hum. Brain. Mapp.*, 2017] |
| BCI Competition IV 2b dataset| 3 (EEG) + 3 (EOG) | 250 | 9 | < 400 | 2 (left hand or right hand) | Schirrmeister *et al.* [*Hum. Brain. Mapp.*, 2017] |
| Mixed Imagery dataset | 64 (EEG) | 60 - 100 | 4 | 675 (S1), 2172 (S2), 698 (S3), 464 (S4) | 4 (right hand and feet (motor) or mental and word (non-motor) | Schirrmeister *et al.* [*Hum. Brain. Mapp.*, 2017] |
| KU-MI dataset | 62 -> 20 (EEG) | 1000 | 54 | 50 (left hand or right hand) | 2 (left hand or right hand) | Lee *et al.* [*Gigascience*, 2019], Ko *et al.* [*IEEE Trans. Industr. Inform.*, 2022] |
| Drone dataset | 64 | 500 | 25 | 50 (lh, rh, both or feet) | 4 (lh, rh, both or feet) | Our dataset |
| BCI Competition III IVa dataset| 118 (EEG) | ? | 5 | 280 | 2 (right hand and right foot) | Ang *et al.* [*IJCNN*, 2008] |

- KU-MI dataset is made for subject-independent scenarios.


## Insights

- Different types of competitive behaviors of organisms

- Experimental conditions from excellent papers:

|                    Paper                 | # of public datasets | # of neural networks | # of comparisons with previous works |
| :--------------------------------------- | :------------------: | :------------------: | :-----------------------------: |
| Chen et al. [*Nat. Mach. Intell.*, 2020] | 2 | 4 | 1 |
| Palazzo et al. [*PAMI*, 2020]            | 1 | 4 |   |
| Warnat-Herresthal et al. [*Nature*, 2021]| 7 | 1 |   |
| Baek et al. [*Nat. Commun.*, 2021]       | 6 | 1 |   |
| Mnih et al. [*Nature*, 2015]             | 49 | 1 | 2 |
| Hasselt et al. [*AAAI*, 2016]             | 57 | 1 | 1 |


## Main Idea

- Competitive and continuous reinforcement learning to better control motor imagery


## Miscallaneous ideas

**Brain and cognitive science**

- Is high-gamma related to motor-imagery the most among all the other frequency bands such as alpha, beta and gamma? (Palazzo *et al.* [*PAMI*, 2020], hand-movements in Schirrmeister *et al.* [*Hum. Brain. Mapp.*, 2017]) But aren't gamma rhythms affected by EMG or EOG so they are less likely to be used in EEG-based BCI system? (Nicolas-Alonso *et al.* [*Sensors*, 2012]) More recent study (Cho *et al.* [*IEEE Trans. Cybern.*, 2021]) uses both EMG and EEG to effectively learn EEG signals, though.

- Does the order matter in Butterworth filter? (Second-order in Palazzo *et al.* [*PAMI*, 2020] and third-order used in Schirrmeister *et al.* [*Hum. Brain. Mapp.*, 2017])

- What type of preprocessing method is mostly used? Band-pass filter? Envelope? (Schirrmeister *et al.* [*Hum. Brain. Mapp.*, 2017])

- Is it true that low-frequency signls (~ 4Hz) are usually related to eye-movements? (Schirrmeister *et al.* [*Hum. Brain. Mapp.*, 2017])

- Why do they use log-variance of the spatially filtered signals per frequency band and for each spatial filter? (Schirrmeister *et al.* [*Hum. Brain. Mapp.*, 2017])

- Is cropping strategy the only way to think of to improve Deep ConvNet?, i.e., is there any other way other than data-augmentation to reduce overfitting in order to improve performance of EEG decoding model? Why does larger crop lead to better decoding accuracy? (Schirrmeister *et al.* [*Hum. Brain. Mapp.*, 2017])

- Why is single-trial mentioned so often? (Chen *et al.* [*PNAS*, 2015], Schirrmeister *et al.* [*Hum. Brain. Mapp.*, 2017])

- Does online session mean real-time decoding process of brain signals for each human-subject with calibration based on offline training session? (Chen *et al.* [*PNAS*, 2015], Schirrmeister *et al.* [*Hum. Brain. Mapp.*, 2017], Cho *et al.* [*IEEE Trans. Cybern.*, 2021], Chen *et al.* [*PNAS*, 2015], Kwon *et al.* [*IEEE Trans. Neural Netw. Learn. Syst.*, 2020])

- Why do MRCP and SMR have similar neural responses? Is it because their features are oscillatory, which means that they use signals from specific EEG frequency bands and are generally asynchronous? On the other hand, ERP-based BCIs are used to detect high amplitude and low frequency given the external stimulus. (Lawhern *et al.* [*J. Neural Eng.*, 2018])

- C3 and C4 are commonly used channels for motor imagery classification task, since neural responses to motor actions (SMRs) are observed strongest compared to other channels. (Lawhern *et al.* [*J. Neural Eng.*, 2018])

- Why is it that time-frequency analysis important? (Lawhern *et al.* [*J. Neural Eng.*, 2018])

- DeepConvNet, shallow ConvNet, EEGNet, and MSNN are commonly used EEG decoding models. (Ko *et al.* [*IEEE Trans. Industr. Inform.*, 2022])

- Short-term memory is stored in hippocampus, where it goes to prefrontal cortex when it becomes long-term memory. (Neftci *et al.* [*Nat. Mach. Intell.*, 2019])

- Amygdala and striatum interact within same pathway of brain. The former is stimulated by emotion such as intimidation. The latter usually comes after the former. (Neftci *et al.* [*Nat. Mach. Intell.*, 2019])

- Neuron is spiked when the electric current caused by ions in cell is summed up and it reaches certain threshold. EPSP (excitatory postsynaptic potential) is potential that makes the postsynaptic neuron more likely to fire an action potential, whereas ISIP is the opposite. It is important to keep firing threhold on average to ensure optimal firing rates, and it can be applied to loss function in DL. (Kaiser *et al.* [*Front. Neurosci.*, 2020], Neftci *et al.* [*Nat. Mach. Intell.*, 2019])

- Attaching random readout to spiking neurons for each layer helps DNNs to learn task-relevant features. Spiking neuron datasets are produced usually by Poisson encoding. (Kaiser *et al.* [*Front. Neurosci.*, 2020])

- CSP is successful in detecting ERD and ERS, which are increasing trend of localized neural rhythmic activities caused by both actual and imagined motor activities. (Ang *et al.* [*IJCNN*, 2008])

- The brain region that controls is in the middle of the vertex. (Ang *et al.* [*IJCNN*, 2008])

- It is best to use MIBIF (Mutual Information based Best Individual Feature) algorithm combined with NBPW (Naive Bayes Parzen Window), FLD (Fisher Linear Discriminant) or SVM (Support Vector Machine) in the proposed FBCSP algorithm compared to SBCSP (sub-band common spatial pattern) and CSP. (Ang *et al.* [*IJCNN*, 2008])

- They used EEG data as a case study to validate the effectiveness of the proposed method, which is multivariate (channel diversity) long nonstationary time series forecasting. (Shen *et al.* [*IEEE Trans. Fuzzy Syst.*, 2021])

- Fuzzy cognitive maps have components of nodes (channels or variables in general) and weights that connect those nodes to depict casual relations among them. Their model considers the relationship among channels from the past. In more specific terms, they update weights for each node and combine all of them to output the weight matrix of the HFCM (high order fuzzy cognitiv maps). (Shen *et al.* [*IEEE Trans. Fuzzy Syst.*, 2021])
<br />

**Artificial intelligence**

- Why is ELU more fast learning than RELU? (Schirrmeister *et al.* [*Hum. Brain. Mapp.*, 2017]) Can it be explained in mathematical terms?

- Why is it that sometimes shallow CNNs are better than deep CNNs? Is it possibly due to different characteristics for different data? (Schirrmeister *et al.* [*Hum. Brain. Mapp.*, 2017])

- Two main issues of RL: 1. Labeled reward data is sparse, noisy, and delayed, so it is hard to learn. 2. Different from the usual data distribution of DNN samples, which is i.i.d., it is not okay to assume a fixed distribution for RL. (Mnih et al. [*Nature*, 2015])

- What other variants of Q-learning can there be instead of combining stochastic minibatch update with experience replay memory which can increase variances of data distribution? (Mnih et al. [*Nature*, 2015])

- Why is it that agent cannot differentiate different magnitude of rewards? The paper clipped rewards in order to match learning rate across multiple games and to limit the scale of error derivates. (Mnih et al. [*Nature*, 2015]) Can we change values of rewards that can also have an impact on agents' actions?

- The average of the maximum value of discounted reward (Q, action-value function) less noisy than the average reward, so it gives smoother visual representations (Mnih et al. [*Nature*, 2015]). It's probably because of the definition of Q-value in deep Q-learning, which is the maximum of target Q-values across all possible actions. This overestimation problem can be solved by decomposing it into action selection and evaluation. (Hasselt *et al.* [*AAAI*, 2016])

- They did not tune hyperparameters due to high computational cost but showed that the best performance can be achieved with the hyperparameters that they have chosen (Mnih et al. [*Nature*, 2015]) Double DQN (Hasselt *et al.* [*AAAI*, 2016]) also shares same parameters as DQN (Mnih et al. [*Nature*, 2015]) and achievees better results.

- What is the meaning of *generating* target values while training in RL? It is to *train* target DQN to have true values, apart from main DQN, so that Q-learning algorithm is model-free? (Mnih et al. ([*Nature*, 2015]) The assumption might be that if networks are well-trained, they should converge in terms of expected rewards.

- Overestimation happens when estimates are guaranteed to be higher than or equal to true values in which the lower bound is always positive. (Hasselt *et al.* [*AAAI*, 2016])

- Machine learning method is better when there is not much of a data as in SMR dataset (288 trials / subject). Also they used elastic net regression (ENR) which combines lasso and ridge regression to solve overfitting problem in training data. Redundant features extracted from 9 filter banks and 4 CSP filters are eliminated when being classified into 4 classes using ENR. (Lawhern *et al.* [*J. Neural Eng.*, 2018])

- Adagrad decays the learning rate in an inverse proportion to their updating time using gradient of cost function (G) with respect to weights. RMSProp considers exponential moving average of the gradient of cost function (1 - gamma) and G(i-1) (gamma). AdaDelta replaces learning rate parameter with D, which gives updates to differences of weights. Lastly, the most commonly used Adam is a combination of momentum and RMSProp. (Kam [Regularization for Deep Learning Course, 2022])