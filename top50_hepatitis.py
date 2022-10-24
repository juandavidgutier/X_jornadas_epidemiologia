# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 19:04:08 2021

@author: juand
"""

#version modules
#pip install dowhy==0.6
#econml==0.13
#pandas==1.4.2
#numpy==1.21
#sklearn
#plotnine







# importing required libraries
import os, warnings, random
import dowhy
import econml
from dowhy import CausalModel
import pandas as pd
import numpy as np
import econml
from econml.dml import DML
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingClassifier
from plotnine import ggplot, aes, geom_line, geom_ribbon, ggtitle, labs





# Set seeds to make the results more reproducible
def seed_everything(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed = 123
seed_everything(seed)
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: '%.2f' % x)



######
#import data
data = pd.read_csv("C:/Users/RPC/Desktop/X_jornadas_epidemiologia-main/top50.csv", encoding='latin-1') 

                           
#NeutralNina
data_NeutralNina = data[['excess_cases1', 'NeutralNina', 'qbo', 'wpac', 'zwnd', 'NBI']] 
data_NeutralNina = data_NeutralNina.dropna()


Y = data_NeutralNina.excess_cases1.to_numpy()
T = data_NeutralNina.NeutralNina.to_numpy()
W = data_NeutralNina[['zwnd', 'wpac', 'qbo']].to_numpy().reshape(-1, 3)
X = data_NeutralNina[['NBI']].to_numpy()


reg = lambda: GradientBoostingClassifier() 


#Step 1: Causal mechanism
model_hep=CausalModel(
        data = data_NeutralNina,
        treatment=['NeutralNina'],
        outcome=['excess_cases1'],
        graph= """graph[directed 1 node[id "NeutralNina" label "NeutralNina"]
                    node[id "excess_cases1" label "excess_cases1"]
                    node[id "wpac" label "wpac"]
                    node[id "qbo" label "qbo"]
                    node[id "zwnd" label "zwnd"]
                    node[id "NBI" label "NBI"]
                    
                    
                    edge[source "wpac" target "NeutralNina"]
                    edge[source "wpac" target "excess_cases1"]
                    
                    edge[source "qbo" target "NeutralNina"]
                    edge[source "qbo" target "excess_cases1"]
                    
                    edge[source "zwnd" target "NeutralNina"]
                    edge[source "zwnd" target "excess_cases1"]
                    
                    edge[source "NBI" target "excess_cases1"]
                    
                    
                    edge[source "wpac" target "qbo"]
                    edge[source "wpac" target "zwnd"]
                    edge[source "zwnd" target "qbo"]
                    edge[source "zwnd" target "wpac"]
                    edge[source "qbo" target "wpac"]
                    edge[source "qbo" target "zwnd"]
                    
                                     
                    edge[source "NeutralNina" target "excess_cases1"]]"""
                    )

#view model 
model_hep.view_model()      

#Step 2: Identifying effects
identified_estimand_NeutralNina = model_hep.identify_effect(proceed_when_unidentifiable=False)
print(identified_estimand_NeutralNina)

#Step 3: Estimation of the effect 
estimate_NeutralNina = DML(model_y=reg(), model_t=reg(), model_final=LassoCV(), discrete_treatment=True,
                   featurizer=PolynomialFeatures(degree=3),
                   linear_first_stages=False, cv=3, random_state=123)

estimate_NeutralNina = estimate_NeutralNina.dowhy

# fit the model
estimate_NeutralNina.fit(Y=Y, T=T, X=X, W=W, inference='bootstrap')

# predict effect for each sample X
estimate_NeutralNina.effect(X)

# ate
ate_NeutralNina = estimate_NeutralNina.ate(X) 
print(ate_NeutralNina) 

# confidence interval of ate
ci_NeutralNina = estimate_NeutralNina.ate_interval(X) 
print(ci_NeutralNina) 


#CATE
#range of NBI
min_NBI = 0.3 
max_NBI = 44
delta = (max_NBI - min_NBI) / 100
X_test = np.arange(min_NBI, max_NBI + delta - 0.001, delta).reshape(-1, 1)

# Calculate marginal treatment effects
treatment_effects = estimate_NeutralNina.const_marginal_effect(X_test)

# Calculate default (95%) marginal confidence intervals for the test data
te_upper, te_lower = estimate_NeutralNina.const_marginal_effect_interval(X_test)

estimate2_nina = DML(model_y=reg(), model_t=reg(), model_final=LassoCV(), discrete_treatment=True,
                   featurizer=PolynomialFeatures(degree=3),
                   linear_first_stages=False, cv=3, random_state=123)

estimate2_nina.fit(Y=Y, T=T, X=X, inference="bootstrap")

treatment_effects2 = estimate2_nina.effect(X_test)
te_lower2, te_upper2 = estimate2_nina.effect_interval(X_test)

#plot 

(
ggplot(aes(x=X_test.flatten(), y=treatment_effects2)) 
  + geom_line() 
  + geom_ribbon(aes(ymin = te_lower2, ymax = te_upper2), alpha = .1)
  + labs(x='NBI', y='Effect of Neutral vs La Niña on excess of cases of hepatitis A')
  + ggtitle("CATE by NBI")
)  





#NeutralNino
data_NeutralNino = data[['excess_cases1', 'NeutralNino', 'qbo', 'wpac', 'zwnd', 'NBI']] 
data_NeutralNino = data_NeutralNino.dropna()


Y = data_NeutralNino.excess_cases1.to_numpy()
T = data_NeutralNino.NeutralNino.to_numpy()
W = data_NeutralNino[['zwnd', 'wpac', 'qbo']].to_numpy().reshape(-1, 3)
X = data_NeutralNino[['NBI']].to_numpy()



#Step 1: Causal mechanism
model_hep=CausalModel(
        data = data_NeutralNino,
        treatment=['NeutralNino'],
        outcome=['excess_cases1'],
        graph= """graph[directed 1 node[id "NeutralNino" label "NeutralNino"]
                    node[id "excess_cases1" label "excess_cases1"]
                    node[id "wpac" label "wpac"]
                    node[id "qbo" label "qbo"]
                    node[id "zwnd" label "zwnd"]
                    node[id "NBI" label "NBI"]
                    
                    
                    edge[source "wpac" target "NeutralNino"]
                    edge[source "wpac" target "excess_cases1"]
                    
                    edge[source "qbo" target "NeutralNino"]
                    edge[source "qbo" target "excess_cases1"]
                    
                    edge[source "zwnd" target "NeutralNino"]
                    edge[source "zwnd" target "excess_cases1"]
                    
                    edge[source "NBI" target "excess_cases1"]
                    
                    
                    edge[source "wpac" target "qbo"]
                    edge[source "wpac" target "zwnd"]
                    edge[source "zwnd" target "qbo"]
                    edge[source "zwnd" target "wpac"]
                    edge[source "qbo" target "wpac"]
                    edge[source "qbo" target "zwnd"]
                    
                                     
                    edge[source "NeutralNino" target "excess_cases1"]]"""
                    )

#view model 
model_hep.view_model()      

#Step 2: Identifying effects
identified_estimand_NeutralNino = model_hep.identify_effect(proceed_when_unidentifiable=False)
print(identified_estimand_NeutralNino)

#Step 3: Estimation of the effect 
estimate_NeutralNino = DML(model_y=reg(), model_t=reg(), model_final=LassoCV(), discrete_treatment=True,
                           featurizer=PolynomialFeatures(degree=3),
                           linear_first_stages=False, cv=3, random_state=123)

estimate_NeutralNino = estimate_NeutralNino.dowhy

# fit the model
estimate_NeutralNino.fit(Y=Y, T=T, X=X, W=W, inference='bootstrap') 

# predict effect for each sample X
estimate_NeutralNino.effect(X)

# ate
ate_NeutralNino = estimate_NeutralNino.ate(X) 
print(ate_NeutralNino) 

# confidence interval of ate
ci_NeutralNino = estimate_NeutralNino.ate_interval(X) 
print(ci_NeutralNino) 



#CATE

# Calculate marginal treatment effects
treatment_effects = estimate_NeutralNino.const_marginal_effect(X_test)

# Calculate default (95%) marginal confidence intervals for the test data
te_upper, te_lower = estimate_NeutralNino.const_marginal_effect_interval(X_test)

estimate2_nino = DML(model_y=reg(), model_t=reg(), model_final=LassoCV(), discrete_treatment=True,
                           featurizer=PolynomialFeatures(degree=3),
                           linear_first_stages=False, cv=3, random_state=123)

estimate2_nino.fit(Y=Y, T=T, X=X, W=W, inference="bootstrap")

treatment_effects2 = estimate2_nino.effect(X_test)
te_lower2, te_upper2 = estimate2_nino.effect_interval(X_test)

#plot 

(
ggplot(aes(x=X_test.flatten(), y=treatment_effects2)) 
  + geom_line() 
  + geom_ribbon(aes(ymin = te_lower2, ymax = te_upper2), alpha = .1)
  + labs(x='NBI', y='Effect of Nuetral vs El Niño on excess of cases of hepatitis A')
  + ggtitle("CATE by NBI")
)  



#NinoNina

data_NinoNina = data[['excess_cases1', 'NinoNina', 'qbo', 'wpac', 'zwnd', 'NBI']] 
data_NinoNina = data_NinoNina.dropna()


Y = data_NinoNina.excess_cases1.to_numpy()
T = data_NinoNina.NinoNina.to_numpy()
W = data_NinoNina[['zwnd', 'wpac', 'qbo']].to_numpy().reshape(-1, 3)
X = data_NinoNina[['NBI']].to_numpy()



#Step 1: Causal mechanism
model_hep=CausalModel(
        data = data_NinoNina,
        treatment=['NinoNina'],
        outcome=['excess_cases1'],
        graph= """graph[directed 1 node[id "NinoNina" label "NinoNina"]
                    node[id "excess_cases1" label "excess_cases1"]
                    node[id "wpac" label "wpac"]
                    node[id "qbo" label "qbo"]
                    node[id "zwnd" label "zwnd"]
                    node[id "NBI" label "NBI"]
                    
                    
                    edge[source "wpac" target "NinoNina"]
                    edge[source "wpac" target "excess_cases1"]
                    
                    edge[source "qbo" target "NinoNina"]
                    edge[source "qbo" target "excess_cases1"]
                    
                    edge[source "zwnd" target "NinoNina"]
                    edge[source "zwnd" target "excess_cases1"]
                    
                    edge[source "NBI" target "excess_cases1"]
                    
                    
                    edge[source "wpac" target "qbo"]
                    edge[source "wpac" target "zwnd"]
                    edge[source "zwnd" target "qbo"]
                    edge[source "zwnd" target "wpac"]
                    edge[source "qbo" target "wpac"]
                    edge[source "qbo" target "zwnd"]
                    
                                     
                    edge[source "NinoNina" target "excess_cases1"]]"""
                    )

#view model 
model_hep.view_model()      

#Step 2: Identifying effects
identified_estimand_NinoNina = model_hep.identify_effect(proceed_when_unidentifiable=False)
print(identified_estimand_NinoNina)

#Step 3: Estimation of the effect 
estimate_NinoNina = DML(model_y=reg(), model_t=reg(), model_final=LassoCV(), discrete_treatment=True,
                        featurizer=PolynomialFeatures(degree=3),
                        linear_first_stages=False, cv=3, random_state=123)

estimate_NinoNina = estimate_NinoNina.dowhy

# fit the model
estimate_NinoNina.fit(Y=Y, T=T, X=X, W=W, inference='bootstrap')

# predict effect for each sample X
estimate_NinoNina.effect(X)

# ate
ate_NinoNina = estimate_NinoNina.ate(X) 
print(ate_NinoNina) 

# confidence interval of ate
ci_NinoNina = estimate_NinoNina.ate_interval(X) 
print(ci_NinoNina) 



#CATE

# Calculate marginal treatment effects
treatment_effects = estimate_NinoNina.const_marginal_effect(X_test)

# Calculate default (95%) marginal confidence intervals for the test data
te_upper, te_lower = estimate_NinoNina.const_marginal_effect_interval(X_test)

estimate2_nino_nina = DML(model_y=reg(), model_t=reg(), model_final=LassoCV(), discrete_treatment=True,
                          featurizer=PolynomialFeatures(degree=3),
                          linear_first_stages=False, cv=3, random_state=123)

estimate2_nino_nina.fit(Y=Y, T=T, X=X, inference="bootstrap")

treatment_effects2 = estimate2_nino_nina.effect(X_test)
te_lower2, te_upper2 = estimate2_nino_nina.effect_interval(X_test)

#plot 

(
ggplot(aes(x=X_test.flatten(), y=treatment_effects2)) 
  + geom_line() 
  + geom_ribbon(aes(ymin = te_lower2, ymax = te_upper2), alpha = .1)
  + labs(x='NBI', y='Effect of El Niño vs La Niña on excess of cases of hepatitis A')
  + ggtitle("CATE by NBI")
)  






#BUT IS THE ESTIMATION UNBIASED?? sensitivity tests are useful for it
#try with Neutral vs El Niño quasy-experiment
data_NeutralNino = data[['excess_cases1', 'NeutralNino', 'qbo', 'wpac', 'zwnd', 'NBI']] 
data_NeutralNino = data_NeutralNino.dropna()


Y = data_NeutralNino.excess_cases1.to_numpy()
T = data_NeutralNino.NeutralNino.to_numpy()
W = data_NeutralNino[['zwnd', 'wpac', 'qbo']].to_numpy().reshape(-1, 3)
X = data_NeutralNino[['NBI']].to_numpy()



#Step 1: Causal mechanism
model_hep=CausalModel(
        data = data_NeutralNino,
        treatment=['NeutralNino'],
        outcome=['excess_cases1'],
        graph= """graph[directed 1 node[id "NeutralNino" label "NeutralNino"]
                    node[id "excess_cases1" label "excess_cases1"]
                    node[id "wpac" label "wpac"]
                    node[id "qbo" label "qbo"]
                    node[id "zwnd" label "zwnd"]
                    node[id "NBI" label "NBI"]
                    
                    
                    edge[source "wpac" target "NeutralNino"]
                    edge[source "wpac" target "excess_cases1"]
                    
                    edge[source "qbo" target "NeutralNino"]
                    edge[source "qbo" target "excess_cases1"]
                    
                    edge[source "zwnd" target "NeutralNino"]
                    edge[source "zwnd" target "excess_cases1"]
                    
                    edge[source "NBI" target "excess_cases1"]
                    
                    
                    edge[source "wpac" target "qbo"]
                    edge[source "wpac" target "zwnd"]
                    edge[source "zwnd" target "qbo"]
                    edge[source "zwnd" target "wpac"]
                    edge[source "qbo" target "wpac"]
                    edge[source "qbo" target "zwnd"]
                    
                                     
                    edge[source "NeutralNino" target "excess_cases1"]]"""
                    )

#view model 
model_hep.view_model()      

#Step 2: Identifying effects
identified_estimand_NeutralNino = model_hep.identify_effect(proceed_when_unidentifiable=False)
print(identified_estimand_NeutralNino)

#Step 3: Estimation of the effect 
estimate_NeutralNino = DML(model_y=reg(), model_t=reg(), model_final=LassoCV(), discrete_treatment=True,
                           featurizer=PolynomialFeatures(degree=3),
                           linear_first_stages=False, cv=3, random_state=123)

estimate_NeutralNino = estimate_NeutralNino.dowhy

# fit the model
estimate_NeutralNino.fit(Y=Y, T=T, X=X, W=W, inference='bootstrap') 

# predict effect for each sample X
estimate_NeutralNino.effect(X)

# ate
ate_NeutralNino = estimate_NeutralNino.ate(X) 
print(ate_NeutralNino) 

# confidence interval of ate
ci_NeutralNino = estimate_NeutralNino.ate_interval(X) 
print(ci_NeutralNino) 



#robustness tests
#with random common cause
random_nino = estimate_NeutralNino.refute_estimate(method_name="random_common_cause", random_state=123)
print(random_nino)

##with add unobserved common cause
unobserved_nino = estimate_NeutralNino.refute_estimate(method_name="add_unobserved_common_cause", effect_strength_on_treatment=0.005, 
                                           effect_strength_on_outcome=0.005, confounders_effect_on_outcome="binary_flip")
print(unobserved_nino)

#with replace a random subset of the data
subset_nino = estimate_NeutralNino.refute_estimate(method_name="data_subset_refuter", subset_fraction=0.8, num_simulations=3)
print(subset_nino)

#with placebo 
placebo_nino = estimate_NeutralNino.refute_estimate(method_name="placebo_treatment_refuter", placebo_type="permute", num_simulations=3)
print(placebo_nino)

























