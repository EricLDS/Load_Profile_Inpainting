# Load_Profile_Inpainting

GAN-based method to restore missing data and estimate baseline of CVR events.

Abstract: This paper introduces a Generative Adversarial Nets (GAN) based, Load Profile Inpainting Network (Load-PIN) for restoring missing load data segments and estimating the baseline for a demand response event. The inputs are time series load databefore and after the inpainting period together with explanatory variables (e.g., weather data). We propose a Generator structure consisting of a coarse network and a fine-tuning network. The coarse network provides an initial estimation of the data segment in the inpainting period. The fine-tuning network consists of selfattention blocks and gated convolution layers for adjusting the initial estimations. Loss functions are specially designed for the fine-tuning and the discriminator networks to enhance both the point-to-point accuracy and realisticness of the results. We test the Load-PIN on three real-world data sets for two applications: patching missing data and deriving baselines of conservation voltage reduction (CVR) events. We benchmark the performance of Load-PIN with five existing deep-learning methods. Our simulation results show that, compared with the state-of-the-art methods, Load-PIN can handle varying-length missing data events and achieve 15-30% accuracy improvement.

If you are using the code, please cite this paper:

Li, Yiyan, Lidong Song, Yi Hu, Hanpyo Lee, Di Wu, P. J. Rehm, and Ning Lu. "Load Profile Inpainting for Missing Load Data Restoration and Baseline Estimation." IEEE Transactions on Smart Grid (2023).
