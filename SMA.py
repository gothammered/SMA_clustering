from tqdm import tqdm
from esda.moran import Moran
import libpysal.weights.set_operations as Wsets
from libpysal.weights import Queen, KNN
import seaborn
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

########################################################################################################################
# Load data into memory
print('loading SMA_gdf...')
SMA_gdf = gpd.read_file('./data/SMA/SMA_boundary_2020.shp')

print('loading SMA_landuse_gdf...')
SMA_landuse_gdf = gpd.read_file('./data/SMA/LandUse_Actual/SMA_LandUse_Actual.shp')

print('Coordinate system for SMA_gdf : {0}'.format(SMA_gdf.crs))
SMA_gdf.info()

print('Coordinate system for SMA_landuse_gdf : {0}'.format(SMA_landuse_gdf.crs))
SMA_landuse_gdf.info()

print('projecting SMA_landuse_gdf to new coordinate system...')
SMA_landuse_gdf.to_crs(5179, inplace=True)

# plot data for check
print('plotting...')
ax = SMA_landuse_gdf.plot(color='red')
SMA_gdf.plot(ax=ax, color='green', alpha=0.5)
plt.title('SMA_overlay')
plt.savefig('./result/SMA_overlay.png', dpi=300)
plt.show()

# intersect two data
print('Running intersect...\nThis takes time please be patient...')
SMA_intersect = gpd.overlay(SMA_gdf, SMA_landuse_gdf, how='intersection')
SMA_intersect.info()
# SMA_intersect.plot()
# plt.title('SMA_intersect')
# plt.savefig('./result/SMA_intersect.png', dpi=300)
# plt.show()

# calculate area (km2)
print('calculating area...')
SMA_intersect['AREA_intersect'] = pd.to_numeric(SMA_intersect['geometry'].area)/1000000

# calculate proportion for each area
print('calculating proportion of each area...')
SMA_intersect['AREA_proportion'] = SMA_intersect['AREA_intersect'] / SMA_intersect['AREA']

# calculate proportion of each UCB per TOT_REG_CD
print('calculating pivot table...')
SMA_pivot = pd.pivot_table(SMA_intersect, index='TOT_REG_CD', columns='UCB', values='AREA_proportion', aggfunc='sum')
SMA_pivot.fillna(0, inplace=True)
SMA_pivot.info()
print(SMA_pivot.describe())

# merge data
print('loading population and employment data...')
SMA_popemp_df = pd.read_csv('./data/SMA/SMA_DATA.csv', dtype={'TOT_REG_CD': object})

print('merging data...')
SMA_intersect = SMA_intersect.dissolve(by='TOT_REG_CD')
SMA_intersect = SMA_intersect.merge(SMA_popemp_df, on='TOT_REG_CD', how='left')
SMA_intersect = SMA_intersect.merge(SMA_pivot, on='TOT_REG_CD', how='left')

SMA_intersect.info()
print(SMA_intersect.describe())
SMA_intersect.plot()
plt.title('SMA_MERGED')
plt.savefig('./result/SMA_MERGED.png', dpi=300)
plt.show()

########################################################################################################################
# plot basic geodemographic maps
cluster_var = ['pop', 'companyCount', 'employeeCount', 'popDen', 'compDen', 'empDen']
cluster_var_core = ['popDen', 'empDen']
cluster_var_wLU = ['popDen', 'empDen', '1110', '1120', '1210', '1220', '2110', '2120', '2210', '2220', '2230',
                       '2310', '2320', '2330', '2340', '3110', '3120', '3130', '3140', '3210', '3220', '3230', '3240',
                       '3310', '3320', '3410', '3420', '3430', '3440', '3510', '3520', '3530', '3540', '3550', '4110',
                       '4120', '4210', '4310', '4320', '4410']
cluster_val = [3, 4, 5]
DBSCAN_val = [2500, 5000, 7500, 10000]

f, axs = plt.subplots(nrows=2, ncols=3)

axs = axs.flatten()  # make the axes accessible with single indexing

for i, col in enumerate(tqdm(cluster_var, desc='plotting basic geodemographic maps... ')):
    ax = axs[i]  # select the axis where the map will go
    SMA_intersect.plot(column=col, ax=ax, scheme='Quantiles', linewidth=0, cmap='RdPu')
    ax.set_axis_off()  # Remove axis clutter
    ax.set_title(col)  # Set the axis title to the name of variable being plotted

plt.savefig('./result/SMA_descriptive.png', dpi=300)
plt.show()  # Display the figure

########################################################################################################################
# plot Moran's I
print("calculating Moran's I...")
w = Queen.from_dataframe(SMA_intersect)
mi_results = [Moran(SMA_intersect[variable], w) for variable in cluster_var]
table = pd.DataFrame([(variable, res.I, res.p_sim) for variable, res in zip(cluster_var, mi_results)],
                     columns=['Variable', "Moran's I", 'p-value']).set_index('Variable')
print(table)

print("plotting Moran's I...")
seaborn.pairplot(SMA_intersect[cluster_var], kind='reg', diag_kind='kde')
plt.savefig('./result/SMA_Moran.png', dpi=300)

########################################################################################################################
# calculate KMeans
for v in tqdm(cluster_val, desc='Calculating KMeans '):
    KMeans_model = KMeans(n_clusters=v)
    KMeans_cls = KMeans_model.fit(SMA_intersect[cluster_var_core])
    SMA_intersect['KM{0}cls'.format(v)] = KMeans_cls.labels_

# plot categorization result (KMeans)
for KM in tqdm(['KM3cls', 'KM4cls', 'KM5cls'], desc='Plotting KMeans categorization result '):
    tidy_db = SMA_intersect.set_index(KM)
    tidy_db = tidy_db[cluster_var_core]
    tidy_db = tidy_db.stack()
    tidy_db = tidy_db.reset_index()
    tidy_db = tidy_db.rename(columns={'level_1': 'Attribute', 0: 'Values'})
    tidy_db.head()
    facets = seaborn.FacetGrid(data=tidy_db, col='Attribute', hue=KM, sharey=False, sharex=False, aspect=2, col_wrap=3).add_legend()
    _ = facets.map(seaborn.kdeplot, 'Values', shade=True)
    plt.legend(title='Category')
    plt.savefig('./result/SMA_{0}.png'.format(KM), dpi=300)
    plt.show()

# plot on map (KMeans)
for KM in tqdm(['KM3cls', 'KM4cls', 'KM5cls'], desc='Plotting KMeans categorization on map '):
    f, ax = plt.subplots(1, figsize=(10, 10))
    SMA_intersect.plot(column=KM, categorical=True, legend=True, linewidth=0, ax=ax)
    ax.set_axis_off()
    plt.axis('equal')
    plt.title('Geodemographic Clusters (K-Means, k={0})'.format(KM[2]))
    plt.savefig('./result/SMA_{0}_map.png'.format(KM), dpi=300)
    plt.show()

########################################################################################################################
# calculate KMeans with landuse data
for v in tqdm(cluster_val, desc='Calculating KMeans with landuse data '):
    KMeans_model = KMeans(n_clusters=v)
    KMeans_cls = KMeans_model.fit(SMA_intersect[cluster_var_wLU])
    SMA_intersect['KM{0}clswLU'.format(v)] = KMeans_cls.labels_

# plot categorization result (KMeans with landuse data)
for KM in tqdm(['KM3clswLU', 'KM4clswLU', 'KM5clswLU'], desc='Plotting KMeans with landuse data categorization result '):
    tidy_db = SMA_intersect.set_index(KM)
    tidy_db = tidy_db[cluster_var_wLU]
    tidy_db = tidy_db.stack()
    tidy_db = tidy_db.reset_index()
    tidy_db = tidy_db.rename(columns={'level_1': 'Attribute', 0: 'Values'})
    tidy_db.head()
    facets = seaborn.FacetGrid(data=tidy_db, col='Attribute', hue=KM, sharey=False, sharex=False, aspect=2, col_wrap=3).add_legend()
    _ = facets.map(seaborn.kdeplot, 'Values', shade=True)
    plt.legend(title='Category')
    plt.savefig('./result/SMA_{0}.png'.format(KM), dpi=300)
    plt.show()

# plot on map (KMeans)
for KM in tqdm(['KM3clswLU', 'KM4clswLU', 'KM5clswLU'], desc='Plotting KMeans categorization on map '):
    f, ax = plt.subplots(1, figsize=(10, 10))
    SMA_intersect.plot(column=KM, categorical=True, legend=True, linewidth=0, ax=ax)
    ax.set_axis_off()
    plt.axis('equal')
    plt.title('Geodemographic Clusters (K-Means with landuse, k={0})'.format(KM[2]))
    plt.savefig('./result/SMA_{0}_map.png'.format(KM), dpi=300)
    plt.show()

########################################################################################################################
# calculate Agglomerative Clustering with spatial constraint
for v in tqdm(cluster_val, desc='Calculating Agglomerative Clustering '):
    AggCls_model = AgglomerativeClustering(linkage='ward', connectivity=w.sparse, n_clusters=v)
    AggCls_model.fit(SMA_intersect[cluster_var_core])
    SMA_intersect['Agg{0}cls'.format(v)] = AggCls_model.labels_
    SMA_intersect.info()

# plot categorization result (Agglomerative Clustering)
for KM in tqdm(['Agg3cls', 'Agg4cls', 'Agg5cls'], desc='Plotting Agglomerative Clustering result '):
    tidy_db = SMA_intersect.set_index(KM)
    tidy_db = tidy_db[cluster_var_core]
    tidy_db = tidy_db.stack()
    tidy_db = tidy_db.reset_index()
    tidy_db = tidy_db.rename(columns={'level_1': 'Attribute', 0: 'Values'})
    tidy_db.head()
    facets = seaborn.FacetGrid(data=tidy_db, col='Attribute', hue=KM, sharey=False, sharex=False, aspect=2, col_wrap=3).add_legend()
    _ = facets.map(seaborn.kdeplot, 'Values', shade=True)
    plt.legend(title='Category')
    plt.savefig('./result/SMA_{0}.png'.format(KM), dpi=300)
    plt.show()

# plot on map (Agglomerative Clustering)
for KM in tqdm(['Agg3cls', 'Agg4cls', 'Agg5cls'], desc='Plotting Agglomerative Clustering result on maps '):
    f, ax = plt.subplots(1, figsize=(10, 10))
    SMA_intersect.plot(column=KM, categorical=True, legend=True, linewidth=0, ax=ax)
    ax.set_axis_off()
    plt.axis('equal')
    plt.title('Geodemographic Clusters (Agglomerative Clustering, k={0})'.format(KM[3]))
    plt.savefig('./result/SMA_{0}_map.png'.format(KM), dpi=300)
    plt.show()

########################################################################################################################
# calculate Agglomerative Clustering with spatial constraint and landuse Data
for v in tqdm(cluster_val, desc='Calculating Agglomerative Clustering with landuse data '):
    AggCls_model = AgglomerativeClustering(linkage='ward', connectivity=w.sparse, n_clusters=v)
    AggCls_model.fit(SMA_intersect[cluster_var_wLU])
    SMA_intersect['Agg{0}clswLU'.format(v)] = AggCls_model.labels_

# plot categorization result (Agglomerative Clustering)
for KM in tqdm(['Agg3clswLU', 'Agg4clswLU', 'Agg5clswLU'], desc='Plotting Agglomerative Clustering with LandUse result '):
    tidy_db = SMA_intersect.set_index(KM)
    tidy_db = tidy_db[cluster_var_wLU]
    tidy_db = tidy_db.stack()
    tidy_db = tidy_db.reset_index()
    tidy_db = tidy_db.rename(columns={'level_1': 'Attribute', 0: 'Values'})
    tidy_db.head()
    facets = seaborn.FacetGrid(data=tidy_db, col='Attribute', hue=KM, sharey=False, sharex=False, aspect=2, col_wrap=3).add_legend()
    _ = facets.map(seaborn.kdeplot, 'Values', shade=True)
    plt.legend(title='Category')
    plt.savefig('./result/SMA_{0}.png'.format(KM), dpi=300)
    plt.show()

# plot on map (Agglomerative Clustering)
for KM in tqdm(['Agg3clswLU', 'Agg4clswLU', 'Agg5clswLU'], desc='Plotting Agglomerative Clustering result on maps '):
    f, ax = plt.subplots(1, figsize=(10, 10))
    SMA_intersect.plot(column=KM, categorical=True, legend=True, linewidth=0, ax=ax)
    ax.set_axis_off()
    plt.axis('equal')
    plt.title('Geodemographic Clusters (Agglomerative Clustering with landuse, k={0})'.format(KM[3]))
    plt.savefig('./result/SMA_{0}_map.png'.format(KM), dpi=300)
    plt.show()

########################################################################################################################
# calculate DBSCAN
for value in tqdm(DBSCAN_val, desc='Calculating DBSCAN '):
    DBSCAN_model = DBSCAN(eps=value)
    DBSCAN_model.fit(SMA_intersect[cluster_var_core])
    SMA_intersect['DB{0}'.format(value)] = DBSCAN_model.labels_

# Plot categorization result (DBSCAN)
for value in tqdm(DBSCAN_val, desc='Plotting DBSCAN result '):
    target_index = 'DB{0}'.format(value)
    tidy_db = SMA_intersect.set_index(target_index)
    tidy_db = tidy_db[cluster_var_core]
    tidy_db = tidy_db.stack()
    tidy_db = tidy_db.reset_index()
    tidy_db = tidy_db.rename(columns={'level_1': 'Attribute', 0: 'Values'})
    tidy_db.head()
    facets = seaborn.FacetGrid(data=tidy_db, col='Attribute', hue=target_index, sharey=False, sharex=False, aspect=2,
                               col_wrap=2).add_legend()
    facets.map(seaborn.kdeplot, 'Values', shade=True)
    plt.legend(title='Category')
    plt.savefig('./result/SMA_{0}.png'.format(target_index), dpi=300)
    plt.show()

# plot on map (DBSCAN)
for value in tqdm(DBSCAN_val, desc='Plotting DBSCAN result on map '):
    target_index = 'DB{0}'.format(value)
    f, ax = plt.subplots(1, figsize=(10, 10))
    SMA_intersect.plot(column=target_index, categorical=True, legend=True, linewidth=0, ax=ax)
    ax.set_axis_off()
    plt.axis('equal')
    plt.title('Geodemographic Clusters ({0})'.format(target_index))
    plt.savefig('./result/SMA_{0}_map.png'.format(target_index), dpi=300)
    plt.show()

########################################################################################################################
# calculate DBSCAN with landuse data
for value in tqdm(DBSCAN_val, desc='Calculating DBSCAN with landuse data '):
    DBSCAN_model = DBSCAN(eps=value)
    DBSCAN_model.fit(SMA_intersect[cluster_var_wLU])
    SMA_intersect['DBLU{0}'.format(value)] = DBSCAN_model.labels_

# Plot categorization result (DBSCAN)
for value in tqdm(DBSCAN_val, desc='Plotting DBSCAN with landuse data categorization result '):
    target_index = 'DBLU{0}'.format(value)
    tidy_db = SMA_intersect.set_index(target_index)
    tidy_db = tidy_db[cluster_var_wLU]
    tidy_db = tidy_db.stack()
    tidy_db = tidy_db.reset_index()
    tidy_db = tidy_db.rename(columns={'level_1': 'Attribute', 0: 'Values'})
    tidy_db.head()
    facets = seaborn.FacetGrid(data=tidy_db, col='Attribute', hue=target_index, sharey=False, sharex=False, aspect=2,
                               col_wrap=3).add_legend()
    facets.map(seaborn.kdeplot, 'Values', shade=True)
    plt.legend(title='Category')
    plt.savefig('./result/SMA_{0}.png'.format(target_index), dpi=300)
    plt.show()

# plot on map (DBSCAN)
for value in tqdm(DBSCAN_val, desc='Plotting DBSCAN with landuse data categorization result on map '):
    target_index = 'DBLU{0}'.format(value)
    f, ax = plt.subplots(1, figsize=(10, 10))
    SMA_intersect.plot(column=target_index, categorical=True, legend=True, linewidth=0, ax=ax)
    ax.set_axis_off()
    plt.axis('equal')
    plt.title('Geodemographic Clusters ({0})'.format(target_index))
    plt.savefig('./result/SMA_{0}_map.png'.format(target_index), dpi=300)
    plt.show()

########################################################################################################################
# calculate Gaussian Mixture
for v in tqdm(cluster_val, desc='Calculating Gaussian Mixture '):
    gmm = GaussianMixture(n_components=v, random_state=0)
    gmm.fit(SMA_intersect[cluster_var_core])
    SMA_intersect['GM{0}cls'.format(v)] = gmm.predict(SMA_intersect[cluster_var_core])

# Plot categorization result (Gaussian Mixture)
for value in tqdm(cluster_val, desc='Plotting Gaussian Mixture result '):
    target_index = 'GM{0}cls'.format(value)
    tidy_db = SMA_intersect.set_index(target_index)
    tidy_db = tidy_db[cluster_var_core]
    tidy_db = tidy_db.stack()
    tidy_db = tidy_db.reset_index()
    tidy_db = tidy_db.rename(columns={'level_1': 'Attribute', 0: 'Values'})
    tidy_db.head()
    facets = seaborn.FacetGrid(data=tidy_db, col='Attribute', hue=target_index, sharey=False, sharex=False, aspect=2,
                               col_wrap=2).add_legend()
    facets.map(seaborn.kdeplot, 'Values', shade=True)
    plt.legend(title='Category')
    plt.savefig('./result/SMA_{0}.png'.format(target_index), dpi=300)
    plt.show()

# plot on map (Gaussian Mixture)
for value in tqdm(cluster_val, desc='Plotting Gaussian Mixture result on map '):
    target_index = 'GM{0}cls'.format(value)
    f, ax = plt.subplots(1, figsize=(10, 10))
    SMA_intersect.plot(column=target_index, categorical=True, legend=True, linewidth=0, ax=ax)
    ax.set_axis_off()
    plt.axis('equal')
    plt.title('Geodemographic Clusters ({0})'.format(target_index))
    plt.savefig('./result/SMA_{0}_map.png'.format(target_index), dpi=300)
    plt.show()

########################################################################################################################
# calculate Gaussian Mixture with landuse data
for v in tqdm(cluster_val, desc='Calculating Gaussian Mixture '):
    gmm = GaussianMixture(n_components=v, random_state=0)
    gmm.fit(SMA_intersect[cluster_var_wLU])
    SMA_intersect['GM{0}clswLU'.format(v)] = gmm.predict(SMA_intersect[cluster_var_wLU])

# Plot categorization result (Gaussian Mixture with landuse)
for value in tqdm(cluster_val, desc='Plotting Gaussian Mixture result '):
    target_index = 'GM{0}clswLU'.format(value)
    tidy_db = SMA_intersect.set_index(target_index)
    tidy_db = tidy_db[cluster_var_wLU]
    tidy_db = tidy_db.stack()
    tidy_db = tidy_db.reset_index()
    tidy_db = tidy_db.rename(columns={'level_1': 'Attribute', 0: 'Values'})
    tidy_db.head()
    facets = seaborn.FacetGrid(data=tidy_db, col='Attribute', hue=target_index, sharey=False, sharex=False, aspect=2,
                               col_wrap=3).add_legend()
    facets.map(seaborn.kdeplot, 'Values', shade=True)
    plt.legend(title='Category')
    plt.savefig('./result/SMA_{0}.png'.format(target_index), dpi=300)
    plt.show()

# plot on map (Gaussian Mixture with landuse)
for value in tqdm(cluster_val, desc='Plotting Gaussian Mixture with landuse result on map '):
    target_index = 'GM{0}clswLU'.format(value)
    f, ax = plt.subplots(1, figsize=(10, 10))
    SMA_intersect.plot(column=target_index, categorical=True, legend=True, linewidth=0, ax=ax)
    ax.set_axis_off()
    plt.axis('equal')
    plt.title('Geodemographic Clusters ({0})'.format(target_index))
    plt.savefig('./result/SMA_{0}_map.png'.format(target_index), dpi=300)
    plt.show()

# save data to shp file
print('saving...')
SMA_intersect.to_file('./data/SMA/SMA_result.shp')
