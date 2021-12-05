#%%
from math import trunc
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering,AgglomerativeClustering, AffinityPropagation, OPTICS
from scipy.sparse import csr, csr_matrix
from sklearn.decomposition import TruncatedSVD,SparsePCA
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.gridspec as gridspec
import squarify
import itertools
import time
import pickle
import sys
import sys

#local_vars = list(locals().items())
#for var, obj in local_vars:
#    print(var, sys.getsizeof(obj))
#%%

pd.set_option('display.max_rows', 134)

def plot_dendrogram(model, **kwargs):
    '''
    Given a clustering model, plots a dendrogram.
    '''
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def cosine_similariy(v1,v2):
    '''
    Calculates cosine similarity between two arrays of values. (v1*v2)/(mod(v1)*mod(v2))

    vector1 - array with length N
    vector2 - array with length N

    '''
    mod1=np.sqrt((v1*v1).sum())
    mod2=np.sqrt((v2*v2).sum())
    dot_product=(v1*v2).sum()
    return dot_product/(mod1*mod2)
def invert_dictionary(dict_in):
    '''
    Inverts keys and values of dictionary
    '''
    return {v: k for k, v in dict_in.items()}
#Boolean variables to inform if there is a need to recalculate a section (or just import data that was calculated in this section previously so that there is no need to spend time recalculating)
recalculate_group_order_aisle=0
recalculate_cosine_similarity=0
recalculate_aisle_assignment=0
recalculate_group_order_department=0
recalculate_cosine_similarity_dept=0
recalculate_dept_product_dict=1
#%%

#reading csvs
orders=pd.read_csv('../data/order_products__prior.csv')[['order_id','product_id']]

products=pd.read_csv('../data/products.csv')
aisles=pd.read_csv('../data/aisles.csv')
aisles_id=aisles['aisle_id']
departments=pd.read_csv('../data/departments.csv')

#getting unique aisles
unique_aisles_dep=products[['aisle_id','department_id']].drop_duplicates()
unique_aisles_dep=unique_aisles_dep.merge(aisles,on='aisle_id').merge(departments,on='department_id').sort_values(['department_id','aisle_id'])
#getting aisle and department relation
aisle_name_with_dep_id=unique_aisles_dep.sort_values(by='aisle_id')
aisle_name_with_dep_id=aisle_name_with_dep_id.apply(lambda x:x['aisle']+'('+str(x['department_id'])+')',axis=1)


#dictionary with aisles and aisles id
aisles_dict=dict(zip(aisles['aisle_id'],aisles['aisle']))
departments_dict=dict(zip(unique_aisles_dep['department_id'],unique_aisles_dep['department']))
product_dict=dict(zip(products['product_name'],products['product_id']))
inverted_product_dict=invert_dictionary(product_dict)
aisle_dept_dict=dict(zip(unique_aisles_dep['aisle_id'],unique_aisles_dep['department_id']))



#merging orders, departments and aisles
complete_orders=orders.merge(products,on='product_id',how='left').merge(aisles,on='aisle_id').merge(departments,on='department_id')
#count number of items bought by aisle and calculate pctg
bought_by_aisle=complete_orders.groupby('aisle').count()[['order_id']].rename(columns={'order_id':'total_bought'}).sort_values(by='total_bought',ascending=False)
bought_by_aisle['%total']=bought_by_aisle['total_bought']/bought_by_aisle['total_bought'].sum()*100
bought_by_aisle['name']=bought_by_aisle.index
bought_by_aisle['name+pctg']=bought_by_aisle.apply(lambda x:x['name']+'\n'+'{:.1f}%'.format(x['%total']),axis=1)

#count number of items bought by department and calculate pctg
bought_by_department=complete_orders.groupby('department').count()[['order_id']].rename(columns={'order_id':'total_bought'}).sort_values(by='total_bought',ascending=False)
bought_by_department['%total']=bought_by_department['total_bought']/bought_by_department['total_bought'].sum()*100
bought_by_department['name']=bought_by_department.index
bought_by_department['name+pctg']=bought_by_department.apply(lambda x:x['name']+'\n'+'{:.1f}%'.format(x['%total']),axis=1)

#%%
#Plot pctg of items bought by department using a tree map 
plt.figure(figsize=(12,8))
num_labels_in_legend = 6
legends=list(bought_by_department['name+pctg'])
ax=squarify.plot(sizes=bought_by_department['total_bought'], label=legends[:-num_labels_in_legend], alpha=.8 , color=plt.cm.plasma(np.linspace(0, 1, len(legends))), text_kwargs={'color': 'white', 'size': 10,'rotation':30},ec='black',norm_x=144, norm_y=89)
plt.axis('off')
ax.invert_xaxis()
ax.set_aspect('equal')
ax.legend(handles=ax.containers[0][:-num_labels_in_legend - 1:-1], labels=legends[:-num_labels_in_legend - 1:-1],fontsize=8,handlelength=1, handleheight=1)
plt.title('Tree map of number of products bought by department')
plt.show()
#%%
#Plot pctg of items bought by aisle using a tree map 

plt.figure(figsize=(12,8))
num_labels_in_legend = 110
legends=list(bought_by_aisle['name+pctg'])
ax=squarify.plot(sizes=bought_by_aisle['total_bought'], label=legends[:-num_labels_in_legend], alpha=.8 , color=plt.cm.plasma(np.linspace(0, 1, len(legends))), text_kwargs={'color': 'white', 'size': 8,'rotation':45},ec='black',norm_x=144, norm_y=89)
plt.axis('off')
ax.invert_xaxis()
ax.set_aspect('equal')
plt.title('Tree map of number of products bought by aisle')

plt.show()

# %%
#Get all combinations of orders and aisles. This will later be used to correlate each aisle according to the orders in which they appeared together
if recalculate_group_order_aisle:
    #merging products with orders
    orders_merged=orders.merge(products,on='product_id',how='left')
    del orders

    #Counting number of products of each transaction, separated into different aisles
    count_per_order=orders_merged.groupby(['order_id','aisle_id']).count()[['product_id']].reset_index().rename(columns={'product_id':'product'})
    count_per_order['product']=1
    del orders_merged

    #Transforming each transaction into a feature. 
    count_per_order=count_per_order.pivot_table(values='product',index='aisle_id',columns='order_id')

    count_per_order.to_hdf('..//processed_data/grouped_by_order_aisle.h5','main',mode='w',complib='blosc',complevel=9)
else:
    count_per_order=pd.read_hdf('..//processed_data/grouped_by_order_aisle.h5','main',mode='r')
count_per_order

# %%
#Calculate cosine similarity between each aisle
if recalculate_cosine_similarity:
    #iterate through each aisle number and calculate the cosine similarity with all other aisles 
    n_isles=len(count_per_order.index)
    cosine_similarity_matrix=np.zeros((n_isles+1,n_isles+1))
    for aisle_id_i in np.arange(1,n_isles+1):
        for aisle_id_j in np.arange(aisle_id_i,n_isles+1):
            cosine_similarity_matrix[aisle_id_i][aisle_id_j]=cosine_similariy(count_per_order.loc[aisle_id_i,:],count_per_order.loc[aisle_id_j,:])
            cosine_similarity_matrix[aisle_id_j][aisle_id_i]=cosine_similariy(count_per_order.loc[aisle_id_i,:],count_per_order.loc[aisle_id_j,:])

    #disconsidering index 0, as there was no aisle with index 0
    cosine_similarity_matrix=cosine_similarity_matrix[1:,1:]

    #saving data for later retrieval
    with open('../processed_data/cosine_similarity_matrix.npy', 'wb') as f:
        np.save(f, cosine_similarity_matrix)
else:
    #loading data processed in a previous run of the program
    with open('../processed_data/cosine_similarity_matrix.npy', 'rb') as f:
        cosine_similarity_matrix=np.load(f)
#%%
#show cosine similarity matrix
pd.DataFrame(cosine_similarity_matrix)
# %%

#using the precomputed cosine matrix and using complete linkage to group items together, plot the hierarchical structure using a dendogram
clustering2=AgglomerativeClustering(n_clusters=10,affinity='precomputed',linkage='complete').fit(1-cosine_similarity_matrix)

plt.figure(figsize=(8,15))
gspec= gridspec.GridSpec(80,10)

left_ax= plt.subplot(gspec[:,:6])
right_ax=plt.subplot(gspec[:,6:])

#calculate linkage matrix using wards method
linkage_matrix = linkage(1-cosine_similarity_matrix, "ward")
dend=dendrogram(linkage_matrix,truncate_mode='level',orientation='right',labels=list(aisle_name_with_dep_id),color_threshold=1.5,ax=left_ax)
left_ax.spines['right'].set_visible(False)
left_ax.spines['top'].set_visible(False)
left_ax.spines['left'].set_visible(False)
left_ax.spines['bottom'].set_visible(False)
left_ax.tick_params(axis='y', which='major', labelsize=7)
left_ax.set_xlabel('Closeness measure')
left_ax.xaxis.grid(True,linestyle='--',alpha=0.4)
left_ax.set_title('Hierarchical clustering with ward linkage')
cell_text = []
for row in range(len(departments)):
    cell_text.append(departments.iloc[row])
right_ax.table(cellText=cell_text, colLabels=departments.columns, loc='center')
right_ax.annotate('Original department division',(0,0.68),fontsize=12)
plt.axis('off')
plt.tight_layout()
plt.savefig('clustering_aisles.jpg',dpi=200)
del cosine_similarity_matrix,linkage_matrix
#%%
##1. Group kosher and indian food with seafood and dried vegetables and sea food
##2. Vegan section: vegan+tofu
##3. Junk food section: drinks, snacks, ice cream, cakes,
##4. Move instant foods to frozen prepepared meals
##5. Create condiments/spices/seasoning section 
##6. Put bread near other breakfast related items


#why not use purely what was calculated:
##1. Red section does not make sense and is counter intuitive: pets, household, dessserts and first aid personal care are grouped together
##2. Personal care was split into two different sections
# %%
del count_per_order

#####PART 2 - ASSIGNING OTHERS AND MISSING TO OTHER SECTION#######
if recalculate_aisle_assignment:
    #%%
    #remaking aisles so that each item in aisle "other" or "missing" is considered as a separate aisle
    aisles=complete_orders['aisle']
    products=complete_orders['product_name']
    aisle_ids=complete_orders['aisle_id']
    combined_data=np.transpose([aisles,products,aisle_ids])
    del aisles,products,aisle_ids

    remade_aisles=list(map(lambda x:x[1] if x[2] in [6,100] else x[0],combined_data))

    complete_orders['remade_aisles']=remade_aisles
    #%%
    #Obtaining all aisles and order_ids that happened together
    count_per_order2=complete_orders.groupby(['order_id','remade_aisles']).count()[['product_id']].reset_index().rename(columns={'product_id':'product'})
    count_per_order2['product']=1


    #creating new dictionary considering products as aisles. Keep original aisles (aisles 1 to 21) identification. In the dictionary: Aisle_name as key, Aisle_id as value
    aisles_custom=list(set(pd.unique(count_per_order2['remade_aisles']))-set(aisles_dict.values()))
    aisles_custom.sort()
    aisles_dict_custom=dict(zip(aisles_custom,np.arange(len(aisles_dict)+1,len(aisles_dict)+1+len(aisles_custom))))

    inverted_aisles_dict=invert_dictionary(aisles_dict)

    aisles_dict_unified={**inverted_aisles_dict, **aisles_dict_custom}

    inverted_aisles_dict_unified={v: k for k, v in aisles_dict_unified.items()}

    #Adding remade aisle ids to dataframe
    count_per_order2['aisle_id_custom']=list(map(lambda x:aisles_dict_unified[x],count_per_order2['remade_aisles']))

    #%%

    #creating a dictionary. Each key is a transaction. Each value is the list of aisles in the transaction
    orders_aisles_list=np.transpose([list(count_per_order2['order_id']),list(count_per_order2['aisle_id_custom'])])
    dict_transactions={}
    for order_id in count_per_order2['order_id']:
        dict_transactions[order_id]=[]
    for item in orders_aisles_list:
        dict_transactions[item[0]]+=[item[1]]

    del count_per_order2
    #total number of new aisles (original aisles+products with no aisle assignment)
    number_unique_aisles_custom=len(aisles_dict_unified)

    #For each transaction, check which aisles appeared together to form a matrix that will be used to calculate the cosine similarity matrix

    common_appearances_matrix=np.zeros((number_unique_aisles_custom,number_unique_aisles_custom))
    total_count=np.zeros(number_unique_aisles_custom)

    #iterating through transactions
    for transaction in dict_transactions:

        aisles_this_trans=np.array(dict_transactions[transaction])
        #we are only interested in combinations of nonmain_aisles (from "missing" and "other" groups) and main aisles (frozen, bakery,etc)
        main_aisles_transactions=aisles_this_trans[aisles_this_trans<=len(aisles_dict)]
        nonmain_aisles_transactions=aisles_this_trans[aisles_this_trans>len(aisles_dict)]

        #add to respective index in 1D array each time an aisle appeared in a transaction
        for aisle_custom_id in aisles_this_trans:
            total_count[aisle_custom_id-1]+=1
            common_appearances_matrix[aisle_custom_id-1,aisle_custom_id-1]+=1

        #add to respective index in DD array each time two aisaislesles appeared simultaneously in a transaction

        for combination in list(itertools.product(nonmain_aisles_transactions,main_aisles_transactions)):
            common_appearances_matrix[combination[0]-1,combination[1]-1]+=1
            common_appearances_matrix[combination[1]-1,combination[0]-1]+=1
    del dict_transactions
    #calculating cosine similarity between each aisle
    cosine_similarity_matrix=common_appearances_matrix.copy()
    for i in np.arange(0,len(total_count)):
        for j in np.arange(0,len(total_count)):
            if i==j:
                cosine_similarity_matrix[i][j]=1
            else:
                cosine_similarity_matrix[i][j]=cosine_similarity_matrix[i][j]/(np.sqrt(total_count[i])*np.sqrt(total_count[j]))
    del common_appearances_matrix

    #iterate through each custom aisle (items that had no aisle assignment) and get original aisle (21 original aisles) that has the highest similarity to the product.
    #The product is then assigned to this section.
    aisle_assignment=[]
    for index in np.arange(len(aisles_dict),len(aisles_dict)+len(aisles_custom)):
        cosine_sim_this_custom_aisle=cosine_similarity_matrix[index]

        max_value=0
        count=0
        for cosine_similarity in cosine_sim_this_custom_aisle:
            if cosine_similarity>max_value and cosine_similarity!=1:
                max_value=cosine_similarity
                index_selected=count
            count+=1
        aisle_assignment+=[[inverted_aisles_dict_unified[index+1], inverted_aisles_dict_unified[index_selected+1],max_value]]
    del cosine_similarity_matrix
    aisle_assignment=pd.DataFrame(aisle_assignment,columns=['Product','Aisle Assigned','Cosine Similarity'])

    #Add extra information to the aisle assignment dataframe: number of transactions the product appeared on
    item_count=complete_orders[complete_orders['department_id'].apply(lambda x: x in [2,21])].groupby(['product_name','aisle']).count()[['order_id']].reset_index().rename(columns={'order_id':'number of transactions that had item','product_name':'Product','aisle':'aisle_origin'})

    del complete_orders

    aisle_assignment=aisle_assignment.merge(item_count,on='Product').sort_values(by='Cosine Similarity',ascending=False).reset_index(drop=True)

    aisle_assignment.to_csv('../processed_data/aisle_assignment_missing.csv')
    with open('../processed_data/aisles_dict_unified.pkl','wb') as fp:
        pickle.dump(aisles_dict_unified, fp)
else:
    aisle_assignment=pd.read_csv('../processed_data/aisle_assignment_missing.csv',index_col=0)
    with open('../processed_data/aisles_dict_unified.pkl','rb') as fp:
        aisles_dict_unified=pickle.load(fp)

# %%

pd.set_option('display.max_rows', 1805)

aisle_assignment
#%%
#####PART 3 - Clustering departments

#making dictionary with equivalency between product id and new aisle to which they are assigned
aisle_assignment['product_id']=aisle_assignment['Product'].apply(lambda x:product_dict[x])


aisle_assignment['aisle_id']=aisle_assignment['Aisle Assigned'].apply(lambda x:aisles_dict_unified[x])

new_aisle_assignment_dict=dict(zip(aisle_assignment['product_id'],aisle_assignment['aisle_id']))

#
orders=pd.read_csv('../data/order_products__prior.csv')[['order_id','product_id']]
products=pd.read_csv('../data/products.csv')
aisles=pd.read_csv('../data/aisles.csv')
departments=pd.read_csv('../data/departments.csv')

products['new_aisle_id']=products.apply(lambda x:new_aisle_assignment_dict[x['product_id']] if x['product_id'] in new_aisle_assignment_dict else x['aisle_id'],axis=1)
products['new_department_id']=products['new_aisle_id'].apply(lambda x:aisle_dept_dict[x])
products.to_csv('../processed_data/reassigned_products.csv')
if recalculate_group_order_department:
    #merging products with orders
    orders_merged=orders.merge(products,on='product_id',how='left')
    del orders

    #Counting number of products of each transaction, separated into different aisles
    count_per_order=orders_merged.groupby(['order_id','new_department_id']).count()[['product_id']].reset_index().rename(columns={'product_id':'product'})
    count_per_order['product']=1
    del orders_merged

    #Transforming each transaction into a feature. 
    count_per_order=count_per_order.pivot_table(values='product',index='new_department_id',columns='order_id')

    count_per_order.to_hdf('..//processed_data/grouped_by_order_department.h5','main',mode='w',complib='blosc',complevel=9)
else:
    count_per_order=pd.read_hdf('..//processed_data/grouped_by_order_department.h5','main',mode='r')
count_per_order
# %%
# %%
#Calculate cosine similarity between each aisle
if recalculate_cosine_similarity_dept:
    #iterate through each aisle number and calculate the cosine similarity with all other aisles 
    n_depts=len(count_per_order.index)
    cosine_similarity_matrix=np.zeros((n_depts+2,n_depts+2))
    for dept_id_i in np.arange(1,n_depts+2):
        for dept_id_j in np.arange(dept_id_i,n_depts+2):
            if dept_id_i==2 or dept_id_j==2:continue
            cosine_similarity_matrix[dept_id_i][dept_id_j]=cosine_similariy(count_per_order.loc[dept_id_i,:],count_per_order.loc[dept_id_j,:])
            cosine_similarity_matrix[dept_id_j][dept_id_i]=cosine_similariy(count_per_order.loc[dept_id_i,:],count_per_order.loc[dept_id_j,:])

    #disconsidering index 0, as there was no aisle with index 0
    cosine_similarity_matrix=cosine_similarity_matrix[1:,1:]

    #saving data for later retrieval
    with open('../processed_data/cosine_similarity_matrix_dept.npy', 'wb') as f:
        np.save(f, cosine_similarity_matrix)
else:
    #loading data processed in a previous run of the program
    with open('../processed_data/cosine_similarity_matrix_dept.npy', 'rb') as f:
        cosine_similarity_matrix=np.load(f)
# %%
#deleting second item, as it includes "other" department, which does not exist anymore
cosine_similarity_matrix=np.delete(np.delete(cosine_similarity_matrix,1,0),1,1)
#%%
#using the precomputed cosine matrix and using complete linkage to group items together, plot the hierarchical structure using a dendogram
clustering2=AgglomerativeClustering(n_clusters=10,affinity='precomputed',linkage='complete').fit(1-cosine_similarity_matrix)

plt.figure(figsize=(8,15))
gspec= gridspec.GridSpec(80,10)

left_ax= plt.subplot(gspec[:,:6])
right_ax=plt.subplot(gspec[:,6:])

labels=list(departments_dict.values())
labels=[labels[0]]+labels[2:-1]

#calculate linkage matrix using wards method
linkage_matrix = linkage(1-cosine_similarity_matrix, "ward")
dend=dendrogram(linkage_matrix,truncate_mode='level',orientation='right',labels=labels,color_threshold=1,ax=left_ax)
left_ax.spines['right'].set_visible(False)
left_ax.spines['top'].set_visible(False)
left_ax.spines['left'].set_visible(False)
left_ax.spines['bottom'].set_visible(False)
left_ax.tick_params(axis='y', which='major', labelsize=7)
left_ax.set_xlabel('Closeness measure')
left_ax.xaxis.grid(True,linestyle='--',alpha=0.4)
left_ax.set_title('Hierarchical clustering with ward linkage')
cell_text = []
for row in range(len(departments)):
    cell_text.append(departments.iloc[row])
plt.axis('off')
plt.tight_layout()
plt.savefig('clustering_depts.jpg',dpi=200)
#%%
del count_per_order
#%%
section_distance={1:2,
2:np.nan,
3:2,
4:1,
5:3,
6:4,
7:1,
8:4,
9:3,
10:4,
11:3,
12:3,
13:2,
14:2,
15:2,
16:1,
17:3,
18:3,
19:1,
20:2,
21:np.nan}
#%%
#write here about the layout of the supermarket
#%%
# 4 - Getting items to display in front based on how many times they happened, as well as how much uncorrelated to their department they are.

# %%
orders=pd.read_csv('../data/order_products__prior.csv')[['order_id','product_id']]
products=pd.read_csv('../processed_data/reassigned_products.csv',index_col=0)
orders_merged=orders.merge(products,on='product_id',how='left')
del orders,products
new_dept_assignment_dict=dict(zip(orders_merged['product_id'],orders_merged['new_department_id']))

unique_products_dep=orders_merged[['product_id','department_id']].drop_duplicates()
product_dept_dict=dict(zip(unique_products_dep['product_id'],unique_products_dep['department_id']))

if recalculate_dept_product_dict:
    count_per_order3=orders_merged.groupby(['order_id','new_department_id']).count()[['product_id']].reset_index().rename(columns={'product_id':'product'})

    orders_dept_list=np.transpose([list(count_per_order3['order_id']),list(count_per_order3['new_department_id']),list(count_per_order3['product'])])
    dict_transactions_dept={}
    for order_id in count_per_order3['order_id']:
        dict_transactions_dept[order_id]={}
        dict_transactions_dept[order_id]['departments']=[]
        dict_transactions_dept[order_id]['n_items']=[]
  
    for item in orders_dept_list:
        dict_transactions_dept[item[0]]['departments']+=[item[1]]
        dict_transactions_dept[item[0]]['n_items']+=[item[2]]

    dict_product_departments={}
    for prod_id in pd.unique(orders_merged['product_id']):
        dict_product_departments[prod_id]={}
        dict_product_departments[prod_id]['occurrences_with_dep']=np.zeros(len(departments)+1)
        dict_product_departments[prod_id]['occurrences']=0
        dict_product_departments[prod_id]['occurrences_alone_in_section']=0
        dict_product_departments[prod_id]['department']=product_dept_dict[prod_id]



    for index,row in orders_merged.iterrows():
        if index%100000==0:
            print(index//100000,end=', ')

        dict_product_departments[row['product_id']]['occurrences']+=1
        department_product=row['new_department_id']

        # if item is the only item of a department that appears in an order, add 1 to "occurrences_alone_in_section"
        if department_product in dict_transactions_dept[row['order_id']]['departments']:
            index_in_list=dict_transactions_dept[row['order_id']]['departments'].index(department_product)
            if dict_transactions_dept[row['order_id']]['n_items'][index_in_list]==1:
                dict_product_departments[row['product_id']]['occurrences_alone_in_section']+=1
            
        item_length=len(dict_transactions_dept[row['order_id']]['departments'])
        for i in np.arange(item_length):
            dept=dict_transactions_dept[row['order_id']]['departments'][i]
            n_items=dict_transactions_dept[row['order_id']]['n_items'][i]
            #dict_product_departments[row['product_id']]['occurrences_with_dep'][dept]+=n_items

            #skip in case the product is the only one of the department
            if dept == department_product and n_items==1:continue

            dict_product_departments[row['product_id']]['occurrences_with_dep'][dept]+=1


    with open('../processed_data/dict_product_departments.pkl','wb') as fp:
        pickle.dump(dict_product_departments, fp)
else:
    with open('../processed_data/dict_product_departments.pkl','rb') as fp:
        dict_product_departments=pickle.load(fp)
#%%


# %%
#counting number of times dept appears
occurences_by_dep={}
for i in np.arange(len(departments)+1):
    occurences_by_dep[i]=0

for dept_id in orders_merged['new_department_id']:
    occurences_by_dep[dept_id]+=1
#%%
#getting product id to dept id assignment
# %%

# %%
product_info=[]
for product_id in new_dept_assignment_dict:

    dept_id_of_product=new_dept_assignment_dict[product_id]

    total_occur=dict_product_departments[product_id]['occurrences']
    occurences_other_depts=dict_product_departments[product_id]['occurrences_with_dep']

    occurences_unique_in_dept=dict_product_departments[product_id]['occurrences_alone_in_section']
    ptcg_unique_in_section=occurences_unique_in_dept/total_occur*100

    #occurences_other_depts[dept_id_of_product]-=total_occur

    cosimilarity_array=[]
    for dept_id in np.arange(len(occurences_other_depts)):
        if occurences_by_dep[dept_id]==0:
            cosimilarity_array+=[np.nan]
            continue
        cosimilarity_array+=[occurences_other_depts[dept_id]/np.sqrt(occurences_by_dep[dept_id]*total_occur)]
    max_cosimilarity=np.nanmax(cosimilarity_array)
    median_cosimilarity=np.nanmedian(cosimilarity_array)
    cosimilarity_its_own_dept=cosimilarity_array[dept_id_of_product]

    product_info+=[[product_id,dept_id_of_product,total_occur,cosimilarity_its_own_dept,median_cosimilarity,max_cosimilarity,ptcg_unique_in_section]]
# %%
data_cosimilarity=pd.DataFrame(product_info,columns=['product_id','dept_id','total_occurrences','cosimilarity_with_own_dept','median_cosimilarity','max_cosimilarity','pctg_unique_item_of_section'])
data_cosimilarity['prod_name']=data_cosimilarity['product_id'].apply(lambda x:inverted_product_dict[x])
data_cosimilarity['dept_name']=data_cosimilarity['dept_id'].apply(lambda x:departments_dict[x])
# %%
data_cosimilarity['ratio_itself_to_median_cosimilarity']=data_cosimilarity['cosimilarity_with_own_dept']/data_cosimilarity['median_cosimilarity']
# %%
data_cosimilarity['rank_occurrences']=data_cosimilarity['total_occurrences'].rank(ascending=False)
data_cosimilarity['rank_itself_to_median']=data_cosimilarity['ratio_itself_to_median_cosimilarity'].rank(ascending=True)
data_cosimilarity['rank_unique_in_dept']=data_cosimilarity['pctg_unique_item_of_section'].rank(ascending=False)

# %%
data_cosimilarity['final_rank']=data_cosimilarity['rank_occurrences']+data_cosimilarity['rank_unique_in_dept']+data_cosimilarity['rank_itself_to_median']
#%%
data_cosimilarity['final_rank_modified']=data_cosimilarity['final_rank']/data_cosimilarity['dept_id'].apply(lambda x:section_distance[x])

# %%
data_cosimilarity[data_cosimilarity['total_occurrences']>100].sort_values('final_rank').head(100)[['prod_name','dept_name','total_occurrences','pctg_unique_item_of_section','cosimilarity_with_own_dept','median_cosimilarity','ratio_itself_to_median_cosimilarity','rank_occurrences','rank_itself_to_median','rank_unique_in_dept','final_rank']]
# %%

# %%
