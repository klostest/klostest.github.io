---
layout: post
title:  "Mapping the global potential of natural reforestation projects using ground observations, remote sensing, and machine learning"
date:   2023-08-20
permalink: /reforestation/
post_description: Creation of a global model for natural forest regeneration rates, with a Google Earth Engine app demonstrated for Africa.
post_image: "/assets/images/2023-08-20-reforestation/fig_8.webp"
category: Conceptual
---

By Stephen Klosterman, Margaux Masson-Forsythe, Trevor F Keenan, Brookie P Guzder-Williams, Anika Staccone, Pedro Ribeiro Piffer, and M. Joseph Hughes

# Introduction
Ecological restoration projects often require investment to get activities up and running. In order to create carbon finance opportunities for forest growth and conservation projects, it’s necessary to be able to predict the accumulation, or avoided emission in the case of prevented deforestation, of carbon in woody biomass. This is in addition to trying to understand the likely changes in a wide array of other ecosystem properties, e.g. plant and animal species composition and water quality. In order to create carbon accumulation predictions, a common approach is to devote individual attention and research effort to projects in specific locations, which may be scattered across the globe. It would therefore be convenient to have a locally accurate and global map of growth rates, or other parameter values of interest, for fast “prospecting” work of determining ecosystem restoration opportunities. Here we describe methods to create such a map, derived from a machine learning model trained on data from a previously published literature review. We then demonstrate the [implementation of the map for Africa in a Google Earth Engine app](https://ee-steve-klosterman.projects.earthengine.app/view/africa-unr-biomass).

# Data and methods
We used a [recently published dataset](https://zenodo.org/record/3983644#.Y7hoXezMJ80) of forest stand biomass measurements, ages, and geographic locations (Cook-Patton et al. 2020) to train a machine learning model to predict a parameter of the commonly used Chapman-Richards (CR) growth function.

After cleaning the data of outliers and unrealistic observations similar to what was done in the original publication, we were left with about 2000 observations, shown here on a global map with symbol size proportional to the number of observations per site:

![png](/assets/images/2023-08-20-reforestation/fig_1.webp)

The observations were spread across 390 sites. Most sites (64%) just have one measurement, while there is one site that has 274 measurements.

Cook-Patton et al. (2020) used these data in combination with other inventory data from the U.S. and Sweden to create a global map of carbon accumulation rates. However, that work assumed a linear carbon accumulation model, which has more biologically-realistic alternatives. Here we show how to create a global parameter layer for a curve fit equation (the CR function). Our approach is similar to the work of Chazdon et al. (2016), creating maps of parameter values for the Michaelis-Menten equation in the Latin American tropics. However, here, instead of restricting the curve fit parameter model to linear combinations of environmental covariates, we used the curve fit parameter as the response variable of a statistical learning method (XGBoost), in order to capture any nonlinear or interactive behavior among features. We chose to depict growth with the CR equation because it is a flexible curve that can take on sigmoidal or logarithmic shapes, and is a standard approach for a simple yet biologically realistic tree growth model in the forestry industry (Bukoski et al., 2022):‍

![png](/assets/images/2023-08-20-reforestation/fig_2.webp)

where y is the biomass at time t, y_max is the upper limit of biomass as the forest matures, b controls the amount of biomass at time 0, m is a shape parameter, and k is the growth parameter we are estimating. In this work we assume b = 1, limiting biomass to zero at time zero, and m = ⅔, as done in similar work (Bukoski et al., 2022). This leaves just y, y_max, t, and k as unknowns. Here are a few example CR curves with m = ⅔, b = 1, y_max = 1, and different values of k:‍

![png](/assets/images/2023-08-20-reforestation/fig_3.webp)

The k parameter clearly affects the time needed to achieve levels approaching the maximum potential biomass:

![png](/assets/images/2023-08-20-reforestation/fig_4.webp)

We took y_max values for the relevant locations from the potential biomass map of Walker et al. (2022). The maps made available along with this publication include predictions of both current day, as well as maximum potential biomass across the world, under a range of assumptions and conditions. In particular we used the Base_Pot_AGB_MgCha_500m.tif map available [here](https://doi.org/10.7910/DVN/DSDDQK). With an estimate of maximum biomass for each measurement site, we substituted the paired values of biomass and forest stand age from Cook-Patton et al. (2020) for y and t, respectively, and calculated values for k for every measurement in that dataset. Here is the distribution of k values we obtained, indicating a wide range with a long right tail:‍

![png](/assets/images/2023-08-20-reforestation/fig_5.webp)

W‍hen interpreting these data it’s important to note that because of the tree growth data source we used, our model will be relevant to naturally regenerating forests. Other literature review efforts have been recently completed for monospecies plantations (Bukoski et al. 2022), and are underway for agroforestry systems ( [Cook-Patton et al.](https://www.nature.org/en-us/about-us/who-we-are/our-people/susan-cook-patton/) in prep) and projects planting diverse tree species ( [Werden et al.](https://lwerden.mystrikingly.com/) in prep ). A similar, related effort involves econometric comparison of the relative effectiveness of natural regeneration and monocultures for mitigating climate change ([Busch et al.](https://www.jonahbusch.com/) in prep).

We used 61 spatially explicit features for our model, including [biome](https://developers.google.com/earth-engine/datasets/catalog/RESOLVE_ECOREGIONS_2017) (12 features after one-hot-encoding), soil properties from the [SoilGrids project](https://www.isric.org/news/soilgrids-data-now-available-google-earth-engine) (11 features), monthly climate data from Terraclimate averaged over the period 1960–1990 (14 features) to match the time period of the [Bioclim](https://developers.google.com/earth-engine/datasets/catalog/WORLDCLIM_V1_BIO) data (19 features), and the terrain features [elevation, slope, aspect, and hillshade](https://developers.google.com/earth-engine/apidocs/ee-terrain-products) (4 features). These features are similar to those used in other studies that apply machine learning to map carbon accumulation rates (Cook-Patton et al., 2020) or other relatively time-invariant ecosystem properties, such as maximum potential biomass at a location (Walker et al., 2022).

We used [XGBoost](https://xgboost.readthedocs.io/en/stable/) to build several regression models for k and explore the relationship between the response variable and our list of potential features. To aid in interpretability, one of our goals was to choose a model with as few features as needed to achieve good performance. We used [SHAP](https://shap.readthedocs.io/en/latest/index.html) (SHapley Additive exPlanation) values to determine feature importance and found substantial variability in model performance and selected features depending on which data was chosen for training versus testing. Therefore, we used 10-fold cross-validation for all model development as opposed to setting aside a single test set. We used [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html) so that measurements from a single site were not split between folds; in other words no fold had data from the same site in both the training and testing data, to reduce the effect of spatial autocorrelation in model assessment.

1. *Pruning correlated predictors:* To begin selecting features, we initially ranked all features on mean absolute SHAP value. To do this, we trained a model for each of the 10 training folds, then calculated the mean absolute SHAP values for the corresponding validation folds to see how important the features were when used for prediction outside the training sample. We summed the mean absolute SHAPs for each feature across all 10 folds and sorted them in descending order for a “ranked choice voting” feature selection procedure. Starting with the feature at the top of the list and proceeding sequentially, we discarded all predictors with Pearson’s r > 0.8 that were lower on the list to reduce multicollinearity and begin the process of pruning the feature set. This step had a negligible impact on final model performance, and reduced the feature set to 41 features.

2. *Training separate models per fold and combining insights:* Using this smaller feature set, we performed backwards selection for each of the 10 folds individually: at each iteration, we ranked the features on mean absolute SHAP value on the validation set. From this ranking, we determined the model with the least number of features, where the RMSE (root mean squared error) of the biomass estimate was within 1 Mg AGB/ha (aboveground biomass per hectare) of the lowest RMSE, roughly a 1% difference — in other words the simplest model that was nearly “as good” as the best model. As a result of this step, the best models for different folds had different features and different numbers of features. To combine insights across folds, we performed a similar ranked choice vote to the previous step to determine which features should be considered for the final model, and their ordering (27 features).

3. *Using a common feature set for each fold for final feature selection:* Again using a backwards selection procedure, we examined the mean RMSE and R² across the validation sets for the 10 folds, but with the same set of features for each fold. We found that model validation performance was generally noisy across folds, highlighting the need for additional data gathering to develop a more robust model. We also found that models with relatively few features had nearly as good validation scores as models with more features.

![png](/assets/images/2023-08-20-reforestation/fig_6.webp)

We chose a model with 5 features (highest validation R², second lowest RMSE by a narrow margin). Results of the final backwards selection (step 3) are shown above, while SHAP values for the final model, collected across validation sets, are shown below. SHAP summary plots like the one shown here, also known as “beeswarm” plots, have a row for each feature, and in each row a point for every prediction in the dataset (here the validation data). The points are colored by the feature value, the vertical offset indicates data density at that SHAP value, and the x-axis coordinate indicates the SHAP value, or the signed impact on model prediction for that sample (Lundberg and Lee, 2017). SHAP summary plots show whether high or low values of the features result in positive or negative impacts on model predictions, and their magnitude.

![png](/assets/images/2023-08-20-reforestation/fig_7.webp)

‍In order to implement this model to create a wall-to-wall map of predicted k values, we exported CSVs (Comma Separated Values files) of spatially tiled predictors, perform model inference, and then export and merge the results into a single geotiff (shown in the next section).

# Discussion and conclusions

By comparison to related research efforts, our model appears to have similar performance. The average (± standard error) cross-validation R² is 0.485 ± 0.04, while the R² for a 20% held-out test set from a model for linear carbon accumulation rates using these and additional data for training was 0.445 (Cook-Patton et al., 2020).

Of the five features in the model, the most important feature is isothermality, which represents the magnitude of the day-night temperature swings in an area, relative to the summer-winter temperature swings. The values range from 0 to 100, where the day-night range is expressed as a percentage of the annual range. This relationship between diurnal and annual temperature is suggested to affect species distribution and be useful for “tropical, insular, and maritime environments” (O’Donnell and Ignizio, 2012). It may be an indicator of how readily a given environment supports tree growth, where larger feature values (larger day-night swings relative to annual) lead to higher predictions of k (faster growth relative to maximum, as seen in the SHAP value plot above). Isothermality is relatively large close to the equator, where annual temperature changes are minimal, and so also proxies for warmer temperatures generally. From the remaining features, the SHAP values tell clear stories that trees are predicted to grow more quickly toward their mature biomass in less windy places, and for the most part with more soil moisture, both physically and biologically sensible model behaviors. The relationships to the Palmer Drought Severity Index and soil carbon are more complex and interactive and we won’t go into their details here.

We took the trained model and performed inference over Africa at 1 km spatial resolution, creating a continental map of k values. Plugging these values into the CR formula, along with potential biomass y_max, we are able to produce maps for AGB at any time after the beginning of forest regeneration. Our [Google Earth Engine app](https://ee-steve-klosterman.projects.earthengine.app/view/africa-unr-biomass) shows biomass at 10, 20, and 30 years after the start of growth; biomass at 30 years is shown below. This data product can be used for nearly instantaneous projection of natural forest regeneration, at high spatial resolution and over large geographic scales, enabling quick prospecting of carbon projects.

![png](/assets/images/2023-08-20-reforestation/fig_8.webp)

‍While we chose to use ground-based data to create our model of biomass accumulation for this work, we are also investing research effort across the team into other possible approaches. These include statistical modeling driven more by remote sensing observations than ground observations, as well as process-based models. Remote sensing of metrics such as canopy height and biomass is done using LiDAR (Light Detection And Ranging) or SAR (Synthetic Aperture Radar) technologies, which typically don’t have a multitemporal component by which one can track biomass growth trajectories over years and decades. Academic researchers and companies are building machine learning models with multitemporal remote sensing data (e.g. Landsat) as features, as a way to effectively extrapolate LiDAR measurements in space (e.g. Walker et al. 2020) and, more recently, time (efforts of [CTREEs](https://ctrees.org/), [Chloris Geospatial](https://www.chloris.earth/), and [Sylvera](https://www.sylvera.com/blog/mapping-forest-structure-across-the-landscape)). This opens up great possibilities, keeping in mind the limitations and difficulties of assessing tree size from space, and the ground validation required. Process-based modeling will provide another valuable perspective, especially with its capabilities to model details relating to biodiversity and other ecosystem services (see e.g. Fisher and Koven, 2020) and efforts to combine it with remote sensing to provide benchmarking data (Ma et al. 2022).

To continue the work described here, we plan to create maps of uncertainty in model prediction, using multiple models trained on bootstrapped samples of the training data, as well as a Monte Carlo approach to account for uncertainty in maximum potential biomass. We plan to explore and compare a diverse array of modeling techniques, that leverage multiple ground-based datasets, in future work.

# References
[Bukoski, J.J., Cook-Patton, S.C., Melikov, C. et al. Rates and drivers of aboveground carbon accumulation in global monoculture plantation forests. Nat Commun 13, 4206 (2022).](https://doi.org/10.1038/s41467-022-31380-7)

[Chazdon, R.L., Broadbent, E.N., Rozendaal, D.M.A. et al. Carbon sequestration potential of second-growth forest regeneration in the Latin American tropics. Science Advances 2, 5 (2016).](https://doi.org/10.1126/sciadv.1501639)

[Cook-Patton, S.C., Leavitt, S.M., Gibbs, D. et al. Mapping carbon accumulation potential from global natural forest regrowth. Nature 585, 545–550 (2020).](https://doi.org/10.1038/s41586-020-2686-x)

[Fisher, R.A., and Koven, C.D. Perspectives on the Future of Land Surface Models and the Challenges of Representing Complex Terrestrial Systems. J Adv Mod Earth Sys 12, 4 (2020).](https://doi.org/10.1029/2018MS001453)

[Lundberg, S.M. and Lee, Su-In. A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems 30 (2017).](https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf)

[Ma, L., Hurtt, G., Ott, L. et al. Global evaluation of the Ecosystem Demography model (ED v3.0). Geosci Model Dev 15, 1971–1994 (2022).](https://doi.org/10.5194/gmd-15-1971-2022)

[O’Donnell, M.S., and Ignizio, D.A. Bioclimatic predictors for supporting ecological applications in the conterminous United States: U.S. Geological Survey Data Series 691(2012).](https://pubs.usgs.gov/ds/691/ds691.pdf)

[Walker, W.S., Gorelik, S.R., Cook-Patton, S.C. et al.The global potential for increased storage of carbon on land. Proc Natl Acad Sci USA 119, 23 (2022).](https://doi.org/10.1073/pnas.2111312119)

