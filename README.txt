AnyonLearner is a computational framework that utilizes PyTorch to discover and analyze Unitary Fusion Categories (the algebraic structures underlying Levin-Wen models and TQFT). It is part of a paper to be released (Obstructions to Unitary Hamiltonians in Levin-Wen Models).         

There are two parts: 
1.  Direct Optimization: Solving the Pentagon and Hexagon equations using gradient descent.
2.  Machine Learning: Training a neural network to approximate the Frobenius-Perron theorem, predicting Quantum Dimensions ($d_i$) directly from Fusion Rules ($N_{ijk}$).

Structure: 

Codes/1_Simple_Learner/: Standalone scripts for solving specific fusion rings and testing edge cases.

Codes/2_With_Training/: The full Machine Learning pipeline. Generates datasets, trains the neural network, and evaluates performance.

Example_Training_Output_Rank_4/`: Sample artifacts (plots, reports, and trained models) demonstrating a successful run on Rank-4 theories.

data_repo/: Reference data from https://github.com/JCBridgeman/smallRankUnitaryFusionData.

Installation:
```bash
git clone https://github.com/StoneStoney/AnyonLearner.git
cd AnyonLearner
pip install -r requirements.txt

