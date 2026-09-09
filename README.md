Brief introduction of the trust game

MIE(Multilevel Interactive Equilibrium) can be empirically tested by computer simulations. 

Here we consider a trust game experiment between two AI agents. In this gameplay, Agent 1 (Investor) and Agent 2 (Trustee) are
modeled by two gated recurrent neural networks (RNNs), which aim to maximize their reward or minimize the loss (Figure A).
To introduce non-stationarity, we assume that Investor has two policies A and B) and switch its strategy twice during the 
repeated game (A → B and then B → A). As a result, the Trustee needs to adapt its policy based on the loss or reward in the 
feedback. 

Traditionally, the behavioral equilibrium is measured via the observed outcome (reward, loss), which can be studied within 
the MARL framework using model- free actor-critic RL (Figure B). Here, we generalize the concept by studying neural and 
cognitive equilibria, at both within-level and cross-level (Figure C). In our experiments, we characterize the RNN’s 
population activity via principal component analysis (PCA) and identify fixed- point attractors that are related to 
pre-specified policies (see animation demo in this file), and measure the minimum distance of population activity to the 
fixed points in the neural space. Alternatively, we calculate the dominant canonical correlation (CC) between population 
activities of two RNNs in the shared neural subspace. Additionally, the agent RNN output predicts the opponent’s action of 
the next round, the prediction error between the predicted and actual action can be used as an empirical criterion for the 
cognitive equilibrium. When neural, cognitive, and behavioral equilibria are simultaneously satisfied, MIE arises (Figure C).

As a possible application demonstration, we envision that acute stress can affect trust priors and regulate social
decision-making. In computer simulations, we modify the learning-rate parameter of Trustee agent as a function of stress 
level while keeping the learning-rate of Investor agent as a constant. We further compare the convergence time-to-equilibrium
under different relative stress level (range: 0-1; 0 being no stress). As noted, the stress level positively correlate with 
the delay to equilibria across levels, but the MIE achieves the highest Pearson’s correlation and the lowest P-value (Figure
D).

Collectively, these experiments provide a proof-of-concept to study MIE in social interactions and its relationship to 
mental health (e.g., stress, anxiety, depression). A systematical study of MAGT and MIE for precision psychiatry as well 
as detailed discussions will be reported elsewhere. 

The figure is shown as below.
<img width="1562" height="1278" alt="image" src="https://github.com/user-attachments/assets/6784ef37-48c7-4ee7-a7d4-3b19b9b4d3e9" />
