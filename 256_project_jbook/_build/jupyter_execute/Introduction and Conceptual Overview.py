#!/usr/bin/env python
# coding: utf-8

# ## What does "doubly robust" mean? 
# 
# Doubly robust methods estimate two models:
# - an *outcome model*
# $$\mu_d(X_i) = E(Y_i \mid D_i = d, X_i)$$
# - and a *exposure model* (or treament model or propensity score):
# $$\pi(X_i) = E(D_i \mid X_i)$$
# 
# where $\mu_d(\cdot)$ is the model of control or treatment $D_i = d=\{0, 1\}$, $X_i$ is a vector of covariates for unit $i = 1, \ldots, N$ for treatment (1) and control (0), $Y_i$ is the outcome, and $\pi(\cdot)$ is the exposure model. The covariates included in $X_i$ can be different for the two models. 
# 
# An estimator is called "doubly robust" if it achieves consistent estimation of the ATE (or whatever estimand we're interested in) as long as *at least one* of these two models is consistently estimated. This means that the outcome model can be completely misspecified, but as long as the exposure model is correct, our estimation of the ATE will be consistent. This also means that the exposure model can be completely wrong, as along as the outcome model is correct.  
# 

# ### Origins of Doubly Robust Methods
# 
# According to Bang and Robins (2005), doubly robust methods have their origins in missing data models. Robins, Rotnitzky, and Zhao (1994) and Rotnitzky, Robins, and Scharfstein (1998) developed augmented orthogonal inverse probability-weighted (AIPW) estimators in missing data models, and Scharfstein, Rotnitzky, and Robins (1999) showed that AIPW was doubly robust and extended to causal inference.  
# 
# But Kang and Schafer (2007) argue that doubly robust methods are older. They cite work by Cassel, Särndal, and Wretman (1976), who proposed “generalized regression estimators” for population means from surveys where sampling weights must be estimated.  
# 
# Arguably, doubly robust methods go back even further than this. The form of doubly robust methods is similar to residual-on-residual regression, which dates back to Frisch, Waugh, and Lovell (1933) famous FWL theorem:
# $$\beta_D = \frac{\text{Cov}(\tilde Y_i, \tilde D_i)}{\text{Var}(\tilde D_i)}$$
# where $\tilde D_i$ is the residual part of $D_i$ after regressing it on $X_i$, and $\tilde Y_i$ is the residual part of $Y_i$ after regressing it on $X_i$. This formulation writes the regression coefficient as composed of an outcome model ($\tilde Y_i$) and exposure model ($\tilde D_i$), the two models used in doubly robust estimators.  
# 
# There are also links between doubly robust methods and matching with regression adjustment. This work goes back to at least Rubin (1973), who suggested that regression adjustment in matched data produces less biased estimates that either matching (exposure adjustment) or regression (outcome adjustment) do by themselves. 

# ### Assumptions
# 
# Most doubly robust methods require almost all of the standard assumptions necessary formost methods that depend on selection on observables. Although some doubly robust methods relax one or two of these, the six standard assumptions are:
# 1. Consistency
# 2. Positivity/overlap
# 3. One version of treatment
# 4. No interference
# 5. IID observations
# 6. Conditional ignorability: $\{Y_{i0}, Y_{i1}\} \perp \!\!\! \perp D_i \mid X_i$
# 
# Special attention should be paid to Assumption 6: doubly robust methods will not work if we do not measure an important confounder that affects both treatment and exposure. But notably, the doubly robust methods covered in this tutorial make no functional form assumptions. Most use flexible machine learning algorithms to estimate both the outcome and exposure models, with regularization (often through cross-fitting) to avoid overfitting.  
# 
# If these six assumptions are met, and we use the right estimator, we get double robustness: consistent estimation if either treatment or outcome model correct.
# 

# ### A simple demonstration
# 
# To demonstrate double robustness, this section presents one of the simpler doubly robust estimators: augmented inverse probability weights (AIPW). The following is adapted from Chapter 12 of Matheus Facure Alves’s (2021) *[Causal Inference for the Brave and True](https://matheusfacure.github.io/python-causality-handbook/landing-page.html)*.  
# 
# We can write this estimator as follows:
# $$\begin{aligned}
# \widehat{ATE} = &\frac{1}{N} \sum_{i=1}^N \left( \frac{D_i(Y_i - \hat \mu_1 (X_i))}{\hat \pi (X_i)} + \hat \mu_1(X_i) \right) \\
# &- \frac{1}{N} \sum_{i=1}^N \left( \frac{(1-D_i)(Y_i - \hat \mu_0 (X_i))}{1-\hat \pi(X_i)} + \hat \mu_0(X_i) \right)
# \end{aligned}$$
# 
# For each individual in the sample, this estimator calculates two quantities:
# - The treated potential outcome
# $$\hat Y_{1i} = \frac{D_i(Y_i - \hat \mu_1 (X_i))}{\hat \pi (X_i)} + \hat \mu_1(X_i)$$
# 
# - The control potential outcome
# $$\hat Y_{0i} = \frac{(1-D_i)(Y_i - \hat \mu_0 (X_i))}{1-\hat \pi(X_i)} + \hat \mu_0(X_i)$$
# 
# Let's focus on the treated model:
# $$\hat Y_{1i} = \frac{D_i(Y_i - \hat \mu_1 (X_i))}{\hat \pi (X_i)} + \hat \mu_1(X_i)$$
# 
# First, assume that the outcome model $\mu_1(X_i)$ is *correctly* specified and the exposure model $\pi(X_i)$ is *incorreclty* specified. Let's also assume (for now) that we're dealing with a treated unit, i.e. $D_i = 1$. Then
# $$\hat \mu_1 (X_i) = Y_i$$
# and hence
# $$\hat Y_{1i} = \frac{D_i(0)}{\hat \pi (X_i)} + \hat \mu_1(X_i) = \hat \mu_1(X_i).$$
# So the model relies *only* on the outcome model! The incorrectly specified exposure model completely disappears from the equation. If we're dealing with a control unit ($D_i=0$), we get the same result:
# $$\hat Y_{1i} = \frac{0(Y_i - \hat \mu_1 (X_i))}{\hat \pi (X_i)} + \hat \mu_1(X_i) = \hat \mu_1(X_i).$$
# 
# Now, what if the *exposure* model $\pi(X_i)$ is correctly specified and the outcome model $\mu_1(X)$ is incorrect? First, we rewrite the estimator for the treated outcome:
# 
# $$\begin{aligned}
# \hat Y_{1i}& = \frac{D_i(Y_i - \hat \mu_1 (X_i))}{\hat \pi (X_i)} + \hat \mu_1(X_i) \\
# &= \frac{D_iY_i}{\hat \pi (X_i)} - \frac{D_i\hat \mu_1 (X_i)}{\hat \pi (X_i)} + \frac{\hat \pi (X_i)\hat \mu_1(X_i)}{\hat \pi (X_i)} \\
# & = \frac{D_iY_i}{\hat \pi (X_i)} - \left( \frac{D_i - \hat \pi(X_i)}{\hat \pi (X_i)}\right) \hat \mu_1(X_i). &&(*)
# \end{aligned}$$
# 
# Since the exposure model is correclty specified, we have $D_i = \hat \pi(X_i)$ on average, so
# $$E[D_i - \hat \pi(X_i)] = 0.$$
# This means that the second term in equation $(*)$ is 0, so
# $$E[\hat Y_{1i}]= E \left [ \frac{D_iY_i}{\hat \pi (X_i)}\right].$$
# 
# This shows that when the exposure model is correct, then the estimator depends *only* on the exposure model. We can make similar arguments for the control model $\hat Y_{0i}$.  
# 
# This demonstration shows that this estimator achieves double robustness: the estimator is robust to misspecification of either the exposure or the outcome model (but not both!).

# #### References
# 
# Bang, H., & Robins, J. M. (2005). Doubly Robust Estimation in Missing Data and Causal Inference Models. *Biometrics*, 61(4), 962–973. https://doi.org/10.1111/j.1541-0420.2005.00377.x  
# 
# CASSEL, C. M., SÄRNDAL, C. E., & WRETMAN, J. H. (1976). Some results on generalized difference estimation and generalized regression estimation for finite populations. *Biometrika*, 63(3), 615–620. https://doi.org/10.1093/biomet/63.3.615  
# 
# Frisch, R., & Waugh, F. V. (1933). Partial Time Regressions as Compared with Individual Trends. *Econometrica*, 1(4), 387–401. https://doi.org/10.2307/1907330  
# 
# Kang, J. D. Y., & Schafer, J. L. (2007). Demystifying Double Robustness: A Comparison of Alternative Strategies for Estimating a Population Mean from Incomplete Data. *Statistical Science*, 22(4), 523–539. https://doi.org/10.1214/07-STS227  
# 
# Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). Estimation of Regression Coefficients When Some Regressors are not Always Observed. *Journal of the American Statistical Association*, 89(427), 846–866. https://doi.org/10.1080/01621459.1994.10476818  
# 
# Rotnitzky, A., Robins, J. M., & Scharfstein, D. O. (1998). Semiparametric Regression for Repeated Outcomes with Nonignorable Nonresponse. *Journal of the American Statistical Association*, 93(444), 1321–1339. https://doi.org/10.2307/2670049  
# 
# Rubin, D. B. (1973). The Use of Matched Sampling and Regression Adjustment to Remove Bias in Observational Studies. *Biometrics*, 29(1), 185–203. https://doi.org/10.2307/2529685  
# 
# Scharfstein, D. O., Rotnitzky, A., & Robins, J. M. (1999). Adjusting for Nonignorable Drop-Out Using Semiparametric Nonresponse Models. *Journal of the American Statistical Association*, 94(448), 1096–1120. https://doi.org/10.1080/01621459.1999.10473862
# 

# 
