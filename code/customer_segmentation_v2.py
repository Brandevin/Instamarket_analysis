#%%

import pickle
from matplotlib.colors import Colormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
from sklearn.cluster import MiniBatchKMeans,FeatureAgglomeration,KMeans,Birch,DBSCAN,MeanShift
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
recalculate_customer_department_pctgs=1
recalculate_metrics_kmeans=1
recalculate_metrics_gaussian=1
from collections import Counter
import random
random.seed(1)

#%%
orders_users=pd.read_csv('../data/orders.csv')[['user_id','order_id']]

unique_users=len(pd.unique(orders_users['user_id']))
order_products=pd.read_csv('../data/order_products__prior.csv')[['order_id','product_id']]
reassigned_prod_department=pd.read_csv('../processed_data/reassigned_products.csv',index_col=0)
departments=pd.read_csv('../data/departments.csv')

dict_order_users=dict(zip(orders_users['order_id'],orders_users['user_id']))
reassigned_prod_department_dict=dict(zip(reassigned_prod_department['product_id'],reassigned_prod_department['new_department_id']))
dict_departments=dict(zip(departments['department_id'],departments['department']))

#%%

order_products['user_id']=list(map(lambda x:dict_order_users[x],order_products['order_id']))

order_products['department_id']=list(map(lambda x:reassigned_prod_department_dict[x],order_products['product_id']))

order_products['department_name']=list(map(lambda x: dict_departments[x],order_products['department_id']))
# %%
order_products[['order_id','user_id','product_id','department_id','department_name']]
# %%
if recalculate_customer_department_pctgs:
    products_by_user_department=order_products.groupby(['user_id', 'department_name']).agg({'order_id': 'count'})

    products_by_user_department_pctg = products_by_user_department.groupby(level=0).apply(lambda x:100 * x / float(x.sum()))

    products_by_user_department_pctg=products_by_user_department_pctg.rename(columns={'order_id':'pctg_of_user_products'})

    pctg_table=pd.pivot_table(products_by_user_department_pctg.reset_index(),values='pctg_of_user_products',columns='department_name',index='user_id')

    #filtering only customers with more than 20 items bought
    products_bought_by_customer=order_products.groupby('user_id').count()[['order_id']].rename(columns={'order_id':'products_bought'}).reset_index()
    customers_filtered=products_bought_by_customer[products_bought_by_customer['products_bought']>20]['user_id']
    pctg_table=pctg_table.loc[customers_filtered,:]


    pctg_table.to_csv('../processed_data/customer_department_pctg.csv')
pctg_table=pd.read_csv('../processed_data/customer_department_pctg.csv')
unique_users_after_filter=len(pd.unique(pctg_table['user_id']))

pctg_table=pctg_table.fillna(0).set_index('user_id')
X=pctg_table


#%%
pca = PCA()
x_pca = pca.fit_transform(X)
del X
data_reduced=pd.DataFrame(x_pca)

explained_variance = pca.explained_variance_ratio_

#%%
fig,ax1=plt.subplots(1,1,figsize=(12,8))
ax1.bar(np.arange(len(explained_variance)),explained_variance*100)
ax1.set_xticks(np.arange(len(explained_variance)))
ax2=ax1.twinx()
ax2.plot(np.arange(len(explained_variance)),np.cumsum(explained_variance)*100,color='r')
ax1.tick_params(axis='y', labelcolor='b')
ax2.tick_params(axis='y', labelcolor='r')
ax1.set_ylabel('Variance explained by component',color='b')
ax2.set_ylabel('Cumulative variance explained',color='r')
plt.show()
#%%
data_pred=data_reduced[np.arange(0,10)]

#%%
model = MiniBatchKMeans(n_clusters=5)
model.fit(data_pred)
model_labels=model.labels_
yhat=model.predict(data_pred)

data_reduced['pred']=yhat
data=pctg_table.copy()
data['pred']=yhat

clusters = pd.unique(yhat)
for cluster in clusters:
	# get row indexes for samples with this cluster
	subset = data[data['pred'] == cluster]
	# create scatter of these samples
	plt.scatter(subset['beverages'], subset['produce'],s=0.1,label=cluster)
plt.xlabel('% beverages')
plt.ylabel('% produce')
plt.title('Kmeans clustering with 5 groups, as seen using two of the percentage variables')
plt.show()
for cluster in clusters:
	# get row indexes for samples with this cluster
	subset = data_reduced[data_reduced['pred'] == cluster]
	# create scatter of these samples
	plt.scatter(subset[0], subset[1],s=0.1)
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.title('Kmeans clustering with 5 groups, as seen using first two principal components')
plt.show()

#%%
kmeans_kwargs = {"init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,}
if recalculate_metrics_kmeans:

    #n_init - > Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.
    sse=[]
    silhouette_coefficients = []
    db_scores=[]
    ch_scores=[]
    #cluster_algorithm
    for k in range(2,15):
        print(k)
        model = KMeans(n_clusters=k,**kmeans_kwargs)
        model.fit(data_pred)
        model_labels=model.labels_
        sse.append(model.inertia_)
        score = silhouette_score(data_pred, model_labels)
        silhouette_coefficients.append(score)
        db_scores.append(davies_bouldin_score(data_pred,model_labels))
        ch_scores.append(calinski_harabasz_score(data_pred,model_labels))
    ################

    metrics_kmeans=pd.DataFrame(zip(*[list(range(2,15)),sse,silhouette_coefficients,db_scores,ch_scores]),columns=['clusters','sse','silhouette','davies_bouldin','calinski_harabasz_score'])
    metrics_kmeans.to_excel('../processed_data/metrics_kmeans.xlsx')
else:
    metrics_kmeans=pd.read_excel('../processed_data/metrics_kmeans.xlsx',index_col=0)

#%%
#%%

# Plotting metrics for different number of clusters used. Try to identify which number of clusters is the best
for measure in ['sse','silhouette','davies_bouldin','calinski_harabasz_score']:
    metric=metrics_kmeans[measure]
    metricdiff = [(metric[n-1]-metric[n])/(metric[n]-metric[n+1]) for n in range(1,len(metric)-1)]
    fig,ax1=plt.subplots(figsize=(12,8))
    p1,=ax1.plot(metrics_kmeans['clusters'], metric,label=measure)
    ax1.set_xticks(metrics_kmeans['clusters'])
    ax1.set_xlabel("Number of Clusters")
    ax1.tick_params(axis='y', colors=p1.get_color())

    ax1.set_ylabel(measure,color='blue')
    ax1.spines["left"].set_edgecolor(p1.get_color())

    if measure in ['davies_bouldin','silhouette']:continue
    ax2=ax1.twinx()
    p2,=ax2.plot(metrics_kmeans['clusters'][1:-1], metricdiff,'red',label='Delta J - {}'.format(measure))
    lines = [p1, p2]
    ax1.legend(lines, [l.get_label() for l in lines])


    ax2.set_ylabel('Delta J - {}'.format(measure),color='red')
    ax2.tick_params(axis='y', colors=p2.get_color())
    ax2.spines["right"].set_edgecolor(p2.get_color())

#%%

#Using model with ideal number of clusters, which were chosen based on metrics plots
model = KMeans(n_clusters=6,**kmeans_kwargs)
model.fit(data_pred)
model_labels=model.labels_
#building a dictionary associating customers to cluster group
customer_segmentation_dict=dict(zip(list(pctg_table.index),model_labels))
#assigning customers to groups
order_products['customer_group']=list(map(lambda x: customer_segmentation_dict[x] if x in customer_segmentation_dict else -1,order_products['user_id']))

def plot_customers_by_group(model_labels,model_name):
    '''
    Plots the total number of items bought by customer cluster.

    Input:
        model_labels - list of cluster numbers, each item is equivalent to a given customer.
        model_name - name of the clustering alogrithm used
    '''

    groups=[-1]+list(Counter(model_labels).keys())

    #Plotting number of customers that belong to each cluster
    count_by_group=[unique_users-unique_users_after_filter]+list(Counter(model_labels).values())
    plt.figure(figsize=(12,8))
    plt.bar(groups,count_by_group)
    plt.xlabel('Group')
    plt.ylabel('Number of customers in group')
    plt.title('Number of customers by groups obtained through {}'.format(model_name))

plot_customers_by_group(model_labels, 'KMeans')

def plot_items_by_dep_and_customer_group():
    '''
    Plots the percentage of items by department bought by each customer group. Helps in identifying

    Uses the current prediction, present in the "customer_group" column of the order_products table
    '''

    # Getting items bought by department, grouped by customer cluster. Plot how much participation each cluster got in each department
    customer_group_product_count = (order_products.groupby(['department_name','customer_group'])['order_id'].count()/order_products.groupby(['department_name'])['order_id'].count()).reset_index()

    pivot_table_kmeans=pd.pivot_table(customer_group_product_count,index='department_name',columns='customer_group',values='order_id')
    pivot_table_kmeans=pivot_table_kmeans*100
    pivot_table_kmeans.plot(kind='bar',stacked=True,figsize=(12,8))
    plt.title('Percentage of department items bought by customer group')
    plt.ylabel('% bought by group')

plot_items_by_dep_and_customer_group()

#%%
def get_probabilities_given_group():
    '''
    Given customer groups predicted by the clustering algorithm (Present in the "customer_group" column of the order_products table):
    1) Plots the a posteriori probability that a given item belongs to a department, given that the customer which bought the item belongs to cluster ___
    2) Returns the table with the a posteriori probability of the item belong to a customer, given that the customer which bought the item belongs to cluster ___.
    '''
    # Getting items bought by customer cluster, grouped by department.
    customer_group_product_count = (order_products.groupby(['customer_group','department_name'])['order_id'].count()/order_products.groupby(['customer_group'])['order_id'].count()).reset_index()

    #getting items bought by department as a percentage of total items. This will later be added to the table
    total_pctg_by_dep=(order_products.groupby(['department_name'])['order_id'].count()/order_products['order_id'].count()).reset_index()
    total_row=dict(zip(total_pctg_by_dep['department_name'],total_pctg_by_dep['order_id']))
    total_row['customer_group']='Any'

    #getting likelihood of an item given it belongs to a given group
    pivot_table_by_customer_group=pd.pivot_table(customer_group_product_count,index='customer_group',columns='department_name',values='order_id').reset_index().rename_axis(None,axis=1)

    #appending a row represent likelihood with no "a priori" condition
    pivot_table_by_customer_group=pivot_table_by_customer_group.append(total_row,ignore_index=True)
    pivot_table_by_customer_group=pivot_table_by_customer_group.set_index('customer_group')
    #showing table
    pivot_table_by_customer_group
    #%%
    #Ploting probability a posteriori that an item belongs to a department, given the customer cluster to which the customer belongs.
    columns=pivot_table_by_customer_group.columns
    my_colors = ['red', 'green', 'blue', 'orange','magenta','yellow','grey','saddlebrown','black']
    fig,axs=plt.subplots(7,3,figsize=(12,18))
    axs=axs.reshape(-1)
    i=0
    for col in columns:
        y_values=pivot_table_by_customer_group[col]
        axs[i].bar(np.arange(0,len(y_values)),100*y_values,color=my_colors)
        axs[i].set_xticks(np.arange(0,len(y_values)))

        axs[i].set_xticklabels(pivot_table_by_customer_group.index,rotation=90)
        axs[i].set_title(col)
        axs[i].set_ylabel('Probability (%)')
        axs[i].set_xlabel('Customer group')

        i+=1
    while i<21:
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['left'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].set_visible(False)
        axs[i].get_xaxis().tick_bottom()
        axs[i].get_yaxis().tick_left()
        i+=1
    plt.suptitle('Probability (%) that a random item selected from customer basket\n is from department, given that customer belongs to group "__"',size=20)
    plt.tight_layout()
    return pivot_table_by_customer_group
get_probabilities_given_group()
# %%
#%%

#repeat entire process for gaussian mixture
model = GaussianMixture(n_components=5)
model.fit(data_pred)

yhat=model.predict(data_pred)

data_reduced['pred']=yhat
data=pctg_table.copy()
data['pred']=yhat

clusters = pd.unique(yhat)
for cluster in clusters:
	# get row indexes for samples with this cluster
	subset = data[data['pred'] == cluster]
	# create scatter of these samples
	plt.scatter(subset['beverages'], subset['produce'],s=0.1)
plt.xlabel('% beverages')
plt.ylabel('% produce')
plt.title('Gaussian mixture clustering with 5 groups, as seen using two of the percentage variables')
plt.show()
for cluster in clusters:
	# get row indexes for samples with this cluster
	subset = data_reduced[data_reduced['pred'] == cluster]
	# create scatter of these samples
	plt.scatter(subset[0], subset[1],s=0.1)
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.title('Gaussian mixture clustering with 5 groups, as seen using first two principal components')
plt.show()

#%%
if recalculate_metrics_gaussian:
    silhouette_coefficients = []
    log_likelihoods=[]
    db_scores=[]
    ch_scores=[]
    aic=[]
    bic=[]
    #cluster_algorithm
    for k in range(2,15):
        print(k)
        model = GaussianMixture(n_components=k)
        model.fit(data_pred)
        log_likelihoods.append(model.lower_bound_)
        model_labels=model.predict(data_pred)
        aic.append(model.aic(data_pred))
        bic.append(model.bic(data_pred))
        score = silhouette_score(data_pred, model_labels)
        silhouette_coefficients.append(score)
        db_scores.append(davies_bouldin_score(data_pred,model_labels))
        ch_scores.append(calinski_harabasz_score(data_pred,model_labels))
    metrics_gaussian_mix=pd.DataFrame(zip(*[list(range(2,15)),log_likelihoods,silhouette_coefficients,db_scores,ch_scores]),columns=['clusters','log_likelihoods','silhouette','davies_bouldin','calinski_harabasz_score'])
    metrics_gaussian_mix.to_excel('../processed_data/metrics_gaussian_mix.xlsx')
else:
    metrics_gaussian_mix=pd.read_excel('../processed_data/metrics_gaussian_mix.xlsx',index_col=0)

# %%
# Plotting metrics for different number of clusters used. Try to identify which number of clusters is the best
for measure in ['log_likelihoods','silhouette','davies_bouldin','calinski_harabasz_score']:
    metric=metrics_gaussian_mix[measure]
    metricdiff = [(metric[n-1]-metric[n])/(metric[n]-metric[n+1]) for n in range(1,len(metric)-1)]
    fig,ax1=plt.subplots(figsize=(12,8))
    p1,=ax1.plot(metrics_gaussian_mix['clusters'], metric,label=measure)
    ax1.set_xticks(metrics_gaussian_mix['clusters'])
    ax1.set_xlabel("Number of Clusters")
    ax1.tick_params(axis='y', colors=p1.get_color())

    ax1.set_ylabel(measure,color='blue')
    ax1.spines["left"].set_edgecolor(p1.get_color())

    if measure in ['davies_bouldin','silhouette']:continue
    ax2=ax1.twinx()
    p2,=ax2.plot(metrics_gaussian_mix['clusters'][1:-1], metricdiff,'red',label='Delta J - {}'.format(measure))
    lines = [p1, p2]
    ax1.legend(lines, [l.get_label() for l in lines])


    ax2.set_ylabel('Delta J - {}'.format(measure),color='red')
    ax2.tick_params(axis='y', colors=p2.get_color())
    ax2.spines["right"].set_edgecolor(p2.get_color())
# %%
#Using model with ideal number of clusters, which were chosen based on metrics plots

model = GaussianMixture(n_components=5)
model.fit(data_pred)
model_labels=model.predict(data_pred)

customer_segmentation_dict_gaussian=dict(zip(list(pctg_table.index),model_labels))
order_products['customer_group']=list(map(lambda x: customer_segmentation_dict_gaussian[x] if x in customer_segmentation_dict_gaussian else -1,order_products['user_id']))


plot_customers_by_group(model_labels,'Gaussian Mixture')
#%%
plot_items_by_dep_and_customer_group()
#%%
get_probabilities_given_group()

# %%
# %%
