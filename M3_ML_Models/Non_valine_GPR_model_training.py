#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[17]:


import sys


# In[18]:


sys.path.append('/shared/ml_code')


# In[19]:


from Regression import GPR, LR


# In[20]:


import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error


# In[ ]:





# In[21]:


data = pd.read_csv('/shared/peptide/final_model_training/final_model_train_data.csv',index_col=0)
data =data.loc[~data['pep'].str.contains('V')]
data= data.sample(frac=1, random_state=37)


# In[22]:


len(data)


# In[ ]:





# In[23]:


data


# In[ ]:





# In[24]:


data['pep'].loc[data['pep'].duplicated()]


# In[ ]:





# In[25]:


Xcols_= ['norm_logP','norm_abs_logP', 'norm_bscore', 'net_chg', 'abs_chg', 'patterning']
Xcols = (data.columns[data.columns.str.contains("fp_")]).union(Xcols_)


# In[26]:


X = data[Xcols]


# In[27]:


y = data['IR_score']


# In[ ]:





# In[ ]:





# In[ ]:





# ### GPR code check

# In[28]:


gpr_obj = GPR(normalize=True, n_cv=10)
print('──' * 40)
print('model parameters :\n')
gpr_obj.print_params()
print('──' * 40)


# In[29]:


gpr_obj.build_gpr_hyperparams(X,y, noise_param=[.1,.15,.2,.25,.3,.35,.4,.45,.5])


# In[30]:


print('──' * 40)
print('initial model cv errors\n')
print(gpr_obj.fit_gpr(X,y,return_cv=True))
print('──' * 40)


# In[31]:


gpr_obj.ml_model.named_steps['regressor']


# In[ ]:





# In[32]:


print('\n')
print('\n')
print('──' * 40)
print('rfe without hyp opt :\n')


# In[ ]:


sel_features, rfe_results = gpr_obj.rfe(X, y,
                                        nrfe_steps = 89,
                                        optimize_hp=False,
                                        verbose_plot=True)


# In[67]:


print('──' * 40)


# ### Feature selection with parameter optimization for each CV run

# In[ ]:





# In[71]:


print('\n')
print('rfe with hyp opt : \n')


# In[ ]:





# In[21]:


sel_features2, rfe_results2 = gpr_obj.rfe(X, y,
                                        nrfe_steps = 89,
                                        optimize_hp=True,
                                        verbose_plot=True)


# In[72]:


print('──' * 40)


# In[73]:


l_init = len(Xcols)


# In[ ]:





# In[27]:


savefig = 1

fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5,4), sharex=True, sharey=True)

plt.plot(np.flip(np.arange(l_init - len(rfe_results['rmse_cv']),l_init)), rfe_results['rmse_cv'], label='w/o hyp optimize')
plt.plot(np.flip(np.arange(l_init - len(rfe_results2['rmse_cv']),l_init)), rfe_results2['rmse_cv'],label='w hyp optimize')
#ax1.text(0.93, 0.77, "features: high_mol level fp", transform=ax1.transAxes, ha='right', fontsize=8)
#ax1.text(0.93, 0.70, "normalize: true", transform=ax1.transAxes, ha='right', fontsize=8)

plt.xlabel("# features",fontsize=14)
plt.ylabel("RMSE",fontsize=14)
plt.title("Gaussian Process Regression",fontsize=14)
plt.legend()

if savefig:
    plt.savefig('/shared/peptide/final_model_training/plots/nvdata_gpr_v2',bbox_inches='tight',dpi=300)


# In[ ]:





# In[ ]:





# In[76]:


print('\n')
print('──' * 40)


# #### save selected features

# In[30]:


f_dict = {
    'without_hy_opt': list(sel_features),
    'with_hy_opt': list(sel_features2)
}


# In[31]:


import json
with open("nvdata_gpr_v2_sel_features.json", "w") as f:
    json.dump(f_dict , f) 


# In[ ]:





# In[ ]:





# In[86]:


print('\n')
print('──' * 40)
print('rfe w/o hyp opt cv_rmse errors: \n')


# In[ ]:





# In[32]:


print(gpr_obj.fit_gpr(X[sel_features],y,return_cv=True))


# In[79]:


print('──' * 40)


# In[ ]:





# In[81]:


print('\n')
print('──' * 40)
print('\n')
print('rfe with hyp opt cv_rmse errors: \n')


# In[ ]:





# In[33]:


print(gpr_obj.fit_gpr(X[sel_features2],y,return_cv=True))


# In[ ]:


print('──' * 40)


# In[ ]:





# #### model building

# In[38]:


gpr_obj.print_params()


# In[ ]:


print('──' * 40)
print('final model :\n')


# In[39]:


print(gpr_obj.ml_model) # This model can be used for further prediction


# In[82]:


print('──' * 40)


# #### save model

# In[40]:


with open('/shared/peptide/final_model_training/models/nvdata_gpr_v2','wb') as f: #file name
    pickle.dump(gpr_obj.ml_model,f)


# #### load model

# In[41]:


#with open('/shared/peptide/version_1/valine_refined_codes/models/gpr_high_level_fp_norm_false', 'rb') as f:
    #model = pickle.load(f)


# #### RMSE CV Train

# In[43]:


#pred_y = model.predict(X[sel_features2])


# In[44]:


#import numpy as np
#from sklearn.metrics import mean_squared_error

savefig =0

fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5,4))
ax1.scatter(y, pred_y)
rmse_train = np.sqrt(mean_squared_error(y,pred_y))

ax1.text(0.9, 0.18, "features: high_mol level fp", transform=ax1.transAxes, ha='right', fontsize=10)
ax1.text(0.9, 0.13, "normalize: false", transform=ax1.transAxes, ha='right', fontsize=10)
ax1.text(0.9, 0.07, "RMSE Train: %.2f"%rmse_train,transform=ax1.transAxes, ha='right',fontsize=12)


ax1.plot([0,5], [0,5], '--k')

ax1.set_xlabel('IR_Score | True y', fontsize=14)    
ax1.set_ylabel('ML IR_Score | Pred y', fontsize=14)

plt.axhline(y=1, color='r', linestyle='--')
plt.axvline(x=1, color='r', linestyle='--')


plt.title('Gaussian process regression')
plt.tight_layout()

plt.show()

if savefig:
    plt.savefig('/shared/peptide/version_1/valine_refined_codes/plots/gpr_high_mol_level_fp_norm_false_train',bbox_inches='tight',dpi=300)
# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




