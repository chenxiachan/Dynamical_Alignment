# Dynamical Alignment

This repository contains the official code and experimental results for the paper "Dynamical Alignment: A Principle for Adaptive Neural Computation". We propose that a neural network's computational capabilities are not solely determined by its static architecture but can be dynamically sculpted by the temporal dynamics of its input signals.


<img width="2034" height="948" alt="GA_" src="https://github.com/user-attachments/assets/aefe4445-9770-4bdb-9444-319342757c3e" />


Our contributions are:

  * **A Novel Principle for Computation**: We introduce **Dynamical Alignment**, a unified principle demonstrating that a fixed neural network can be steered into fundamentally different computational modes—from energy-efficient sparse coding to high-performance dense coding—by controlling the temporal dynamics of its input signals.
  * **A Bimodal Optimization Landscape**: We uncover a bimodal performance landscape governed by the rate of phase space volume change ($\\Sigma\\lambda\_i$) of the input dynamics, which challenges the traditional "edge of chaos" view.
  * **Universal and Scalable Advantages**: We validate this principle across diverse tasks, showing that it not only closes the performance gap between SNNs and ANNs in deep learning but also provides an emergent mechanism for key cognitive functions like the exploitation-exploration trade-off in reinforcement learning and feature binding.

We provide all code necessary to reproduce our results and invite the community to explore this new paradigm for adaptive and efficient AI.

### Repository Structure

Our repository is organized to directly correspond with the experiments and figures presented in the paper, ensuring full reproducibility and ease of navigation.

```
D:.
├── 0_Fig1                                  # Section 2.2: Experimental Validation of Dynamical Alignment
│   ├── 0_Fig1                              # Fig 1: Baseline SNNs vs. Lorenz-SNN on MNIST
│   │   ├── 0_code
│   │   └── 1_Result
│   └── 1_Addtional_Experiment              # Controlled experiment on UMAP preprocessing (code and results)
│       ├── 0_Code                          
│       └── 1_Result
├── 1_Fig2                                  # Section 2.2: Network-in bridging on CIFAR-10
├── 2_Fig3&4                                # Section 3: Dissecting the Dynamical Correlates of Encoding Efficiency
│   ├── 0_Code                              # Code for attractor comparison and parameter search
│   └── 1_Result                            # Results from grid search and Lyapunov analysis
├── 3_Fig5                                  # Section 4: Critical Phase Transitions and Bimodal Optimization
│   ├── 0_Code                              # Code for analyzing the parameterized mixed oscillator system
│   └── 1_Result                            # Results demonstrating bimodal optimization and phase transitions
├── 4_Fig6                                  # Section 4.2: Mechanisms of Bimodal Neural Computation
│   ├── 0_Code                              # Code for analyzing internal network dynamics and robustness
│   └── 1_D4                                # Code and saved models for theoretical analysis support from Appendix D.4
└── 5_Fig7&8&9                              # Section 5: Universality and Scalability of Dynamical Alignment
    ├── 1_TinyImagenet                      # Section 5.1: Scalability in deep learning (ResNet-18)
    ├── 2_Reinforcement_Learning            # Section 5.2: Strategic Advantages in dynamic decision-making (CartPole)
    │   ├── 0_results                       # Detailed results for 30 independent RL runs
    │   └── 1_videos                        # Videos of agent behaviors in different dynamical modes
    └── 3_Feature_Binding                   # Section 5.3: Mechanistic Advantages in a cognitive task
```

### Setup

To set up the environment and run the experiments, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone [your_repo_url]
    cd [your_repo_name]
    ```

2.  **Create and activate the Conda environment:**

    ```bash
    conda create --name da_snn python=3.8
    conda activate da_snn
    ```

### Running the Experiments

Each main experiment is located in a corresponding folder and is self-contained. For example, to run the parameter optimization experiments from Section 3, you would run the code in the `2_Fig3&4/0_Code` directory.

The code is designed to be modular and easy to follow. You can modify the parameters within the main scripts (e.g., `Config` classes) to explore different settings and reproduce the various results presented in our paper.

### Results and Analysis

The `1_Result` subdirectories contain the generated plots, data tables, and statistical summaries for each experiment, allowing for easy verification of the paper's findings. We have also included additional control experiments to further validate our core claims.

This repository serves as a complete and transparent resource for our research, and we hope it inspires further work in the field of dynamic and adaptive neural computation.
