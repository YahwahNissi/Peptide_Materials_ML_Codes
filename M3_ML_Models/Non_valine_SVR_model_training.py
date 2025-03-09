#!/usr/bin/env python
# coding: utf-8

# #### Import SVM

# In[13]:


import sys


# In[14]:


sys.path.append('/shared/ml_code')


# In[15]:


from Regression import SVM


# #### Import libraries

# In[16]:


import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error


# In[ ]:





# In[17]:


data = pd.read_csv('/shared/peptide/final_model_training/final_model_train_data.csv',index_col=0)


# In[21]:


data =data.loc[~data['pep'].str.contains('V')]
data= data.sample(frac=1, random_state=37)


# In[22]:


data['pep'].loc[data['pep'].duplicated()]


# In[ ]:





# In[23]:


data


# In[ ]:





# In[24]:


#Xcols_= ['norm_AP', 'norm_logP','norm_abs_logP', 'norm_bscore', 'net_chg', 'abs_chg', 'patterning']
#Xcols = (data.columns[data.columns.str.contains("fp_")]).union(Xcols_)


# In[25]:


Xcols_= ['norm_logP','norm_abs_logP', 'norm_bscore', 'net_chg', 'abs_chg', 'patterning']
Xcols = (data.columns[data.columns.str.contains("fp_")]).union(Xcols_)


# In[26]:


X = data[Xcols]


# In[27]:


y = data['IR_score']


# In[ ]:





# In[ ]:





# ### svr code

# In[28]:


svr = SVM(n_cv =10,normalize=True)


# In[29]:


svr.print_params()
print('──' * 40)


# In[31]:


svr.build_svm_hyperparams(X,y,C=[0.1,0.5,1,3,5,7,8,9,9.5,10,10.5,11,12,15,20,50,100], epsilon=[0,0.001,0.01,0.03,0.05,0.075,0.1,0.2,0.3,0.4,.55])


# In[ ]:





# In[32]:


print('model parameters :\n')
svr.print_params()
print('──' * 40)


# In[ ]:





# In[33]:


print('──' * 40)
print('initial model cv errors\n')
(svr.fit(X,y,return_cv=True, verbose=1))
print('──' * 40)


# In[26]:


svr.verbose_plot=0


# In[27]:


print('\n')
print('\n')
print('──' * 40)
print('rfe without hyp opt :\n')


# In[28]:


sel_features, rfe_results = svr.rfe(X, y,
                                        nrfe_steps = 89,
                                        optimize_hp=False,
                                        verbose_plot=True)


# In[29]:


print('──' * 40)
print('\n')


# In[30]:


print('rfe w/o hyp optimization RMSE CV errors:\n')
print((rfe_results['rmse_cv']))
print('\n')
print('minimum rfe error w/o hyp opt:',min(rfe_results['rmse_cv']))
print('──' * 40)


# In[44]:


print('\n')
print('\n')
print('──' * 40)
print('──' * 40)
print('rfe with hyp opt :\n')


# In[ ]:


sel_features2, rfe_results2 = svr.rfe(X, y,
                                        nrfe_steps = 89,
                                        optimize_hp=True,
                                        verbose_plot=True)


# In[ ]:


print('──' * 40)
print('\n')


# In[ ]:


print('rfe with hyp optimization RMSE CV errors:\n')
print((rfe_results2['rmse_cv']))
print('\n')
print('minimum rfe error with hyp opt:',min(rfe_results2['rmse_cv']))
print('──' * 40)


# In[ ]:





# In[82]:


l_init =len(Xcols)


# In[84]:


savefig = 1

fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5,4), sharex=True, sharey=True)

plt.plot(np.flip(np.arange(l_init - len(rfe_results['rmse_cv']),l_init)), rfe_results['rmse_cv'], label='w/o hyp optimize')
plt.plot(np.flip(np.arange(l_init - len(rfe_results2['rmse_cv']),l_init)), rfe_results2['rmse_cv'],label='w hyp optimize')
#ax1.text(0.93, 0.77, "features: high + mol. level fp", transform=ax1.transAxes, ha='right', fontsize=8)
#ax1.text(0.93, 0.70, "normalize: true", transform=ax1.transAxes, ha='right', fontsize=8)

plt.xlabel("# features",fontsize=14)
plt.ylabel("RMSE",fontsize=14)
plt.title("SVR",fontsize=14)
plt.legend()

if savefig:
    plt.savefig('/shared/peptide/final_model_training/plots/non_val_svr_v2',bbox_inches='tight',dpi=300)


# In[ ]:





# In[85]:


print('\n')
print('──' * 40)


# #### save selected features

# In[ ]:


f_dict = {
    'without_hy_opt': list(sel_features),
    'with_hy_opt': list(sel_features2)
}


# In[ ]:


import json
with open("non_val_svr_sel_features_v2.json", "w") as f:
    json.dump(f_dict , f) 


# In[ ]:





# In[87]:


print('──' * 40)
print('──' * 40)
print('sel_features:\n')
print(sel_features)


# In[88]:


print('──' * 40)
print('──' * 40)
print('sel_features2:\n')
print(sel_features2)


# In[89]:


svr.verbose_plot = False


# In[90]:


svr.print_params()


# In[ ]:





# In[ ]:





# In[91]:


print('\n')
print('──' * 40)
print('model rfe w/o hyp opt:\n')


# In[92]:


print(np.asarray(svr.fit_svm(X[sel_features],y,return_cv=True, verbose=False)))


# In[93]:


print('──' * 40)
print('──' * 40)
print('\n')
print('model rfe w hyp opt:\n')


# In[ ]:


print(np.asarray(svr.fit_svm(X[sel_features2],y,return_cv=True, verbose=False)))


# In[ ]:





# In[ ]:





# In[ ]:


print('──' * 40)
print('──' * 40)
print('final model :\n')


# In[97]:


print(svr.ml_model)


# In[98]:


print('──' * 40)
print('──' * 40)


# #### save model

# In[31]:


with open('/shared/peptide/final_model_training/models/non_val_svr_v2','wb') as f: #file name
    pickle.dump(svr.ml_model,f)


# In[ ]:




