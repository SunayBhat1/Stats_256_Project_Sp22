#!/usr/bin/env python
# coding: utf-8

# ## **Conclusion and New Directions**
# 
# 
# 
# ### Summary of Estimators
# 
# * **AIPW** - It is a weighting based estimator that improves IPTW by fully utilizing information about both the treatment assignment and the outcome. It is a combination of IPTW and a weighted average of the outcome regression estimators.
# * **TMLE** - It incorporates a targeting step that optimizes the bias-variance tradeoff for the targeted estimator, i.e., ATE. It obtains initial outcome estimates via outcome modeling and propensity scores via treatment modeling, respectively. These initial outcome estimates are then updated to reduce the bias of confounding, which generates the targeted predicted outcome values.
# * **DML** - It utilizes predictive power of advanced ML algorithms in estimating heterogeneous treatment effects when all potential confounders are observed and are also high-dimensional.
# 
# 
# ### Doubly Robust Methods: Applications
# 
# * Molecular Epidemiology - Meng et al. (2021) applied efficient estimators like AIPW and TMLE to estimate average treatment effects under various scenarios of mis-specification.
# * Social Sciences - Knaus (2020) showed the efficiency of DML for the evaluation of programs of the Swiss Active Labour Market Policy.
# * Medical Sciences - Rose et al. (2020) proposed the use of TMLE for the evaluation of the comparative effectiveness of drug-eluting coronary stents.
# 
# 
# ### Potential Future Works (Tan et al. (2022))
# 
# * It is recommended to do a variable selection first, followed by using SuperLearner to model PS and outcomes. After that TMLE can be applied for estimating ATE.
# * The use of ML algorithms like random forest and neural networks can be used to remove treatment predictors for variable selection.
# * Soft variable selection strategies can be used where the variable selection is conducted without requiring any modeling on the outcome regression, and thus provides robustness against mis-specification.
# 
# 

# 
