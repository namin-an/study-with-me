
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
