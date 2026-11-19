# Data dictionary

## 1. Purpose and sources

This document describes every field used by the residential valuation platform.
It covers three layers:

1. **Base data** - the 81 columns of the Kaggle *House Prices - Advanced
   Regression Techniques* competition, which contains 1460 historical
   sales in Ames, Iowa with 79 explanatory attributes.
2. **Synthetic contextual data** - 21 fields generated in Python with
   documented business logic. These fields are produced for every property, including
   the competition test records, and are never derived from the sale price.
3. **Engineered features** - 18 derived variables constructed during
   feature engineering, with their formula, unit and business rationale.

The target variable is `SalePrice`, expressed in US dollars and modelled on the
natural-log scale as `log_price = ln(1 + SalePrice)`.

## 2. Data quality conventions

| Convention | Meaning | Example |
|---|---|---|
| Structural absence | The property does not have the feature | A house without a basement has no basement quality |
| Genuine missingness | The value is unknown and must be imputed | Lot frontage, recorded for 1,201 of 1,460 training properties |
| Zero quantity | The feature exists conceptually but has zero extent | Garage area of a house without a garage |

Every cleaning rule and its before/after impact is recorded in
`artifacts/metrics/cleaning_report.json`.

## 3. Base data fields

The table below lists every raw column, its type, role, missingness and the
definition taken from the competition documentation.

| Field         | Type        | Role       |   Missing in train |   Missing in test | Valid range or levels              | Definition                                                             |
|:--------------|:------------|:-----------|-------------------:|------------------:|:-----------------------------------|:-----------------------------------------------------------------------|
| Id            | numeric     | identifier |                  0 |                 0 | 1 to 1460                          | See Kaggle data_description.txt.                                       |
| MSSubClass    | numeric     | feature    |                  0 |                 0 | 20 to 190                          | Identifies the type of dwelling involved in the sale.                  |
| MSZoning      | categorical | feature    |                  0 |                 4 | RL, RM, C (all), FV                | Identifies the general zoning classification of the sale.              |
| LotFrontage   | numeric     | feature    |                259 |               227 | 21 to 313                          | Linear feet of street connected to property                            |
| LotArea       | numeric     | feature    |                  0 |                 0 | 1300 to 2.152e+05                  | Lot size in square feet                                                |
| Street        | categorical | feature    |                  0 |                 0 | Pave, Grvl                         | Type of road access to property                                        |
| Alley         | categorical | feature    |               1369 |              1352 | Grvl, Pave                         | Type of alley access to property                                       |
| LotShape      | categorical | feature    |                  0 |                 0 | Reg, IR1, IR2, IR3                 | General shape of property                                              |
| LandContour   | categorical | feature    |                  0 |                 0 | Lvl, Bnk, Low, HLS                 | Flatness of the property                                               |
| Utilities     | categorical | feature    |                  0 |                 2 | AllPub, NoSeWa                     | Type of utilities available                                            |
| LotConfig     | categorical | feature    |                  0 |                 0 | Inside, FR2, Corner, CulDSac       | Lot configuration                                                      |
| LandSlope     | categorical | feature    |                  0 |                 0 | Gtl, Mod, Sev                      | Slope of property                                                      |
| Neighborhood  | categorical | feature    |                  0 |                 0 | CollgCr, Veenker, Crawfor, NoRidge | Physical locations within Ames city limits                             |
| Condition1    | categorical | feature    |                  0 |                 0 | Norm, Feedr, PosN, Artery          | Proximity to various conditions                                        |
| Condition2    | categorical | feature    |                  0 |                 0 | Norm, Artery, RRNn, Feedr          | Proximity to various conditions (if more than one is present)          |
| BldgType      | categorical | feature    |                  0 |                 0 | 1Fam, 2fmCon, Duplex, TwnhsE       | Type of dwelling                                                       |
| HouseStyle    | categorical | feature    |                  0 |                 0 | 2Story, 1Story, 1.5Fin, 1.5Unf     | Style of dwelling                                                      |
| OverallQual   | numeric     | feature    |                  0 |                 0 | 1 to 10                            | Rates the overall material and finish of the house                     |
| OverallCond   | numeric     | feature    |                  0 |                 0 | 1 to 9                             | Rates the overall condition of the house                               |
| YearBuilt     | numeric     | feature    |                  0 |                 0 | 1872 to 2010                       | Original construction date                                             |
| YearRemodAdd  | numeric     | feature    |                  0 |                 0 | 1950 to 2010                       | Remodel date (same as construction date if no remodeling or additions) |
| RoofStyle     | categorical | feature    |                  0 |                 0 | Gable, Hip, Gambrel, Mansard       | Type of roof                                                           |
| RoofMatl      | categorical | feature    |                  0 |                 0 | CompShg, WdShngl, Metal, WdShake   | Roof material                                                          |
| Exterior1st   | categorical | feature    |                  0 |                 1 | VinylSd, MetalSd, Wd Sdng, HdBoard | Exterior covering on house                                             |
| Exterior2nd   | categorical | feature    |                  0 |                 1 | VinylSd, MetalSd, Wd Shng, HdBoard | Exterior covering on house (if more than one material)                 |
| MasVnrType    | categorical | feature    |                872 |               894 | BrkFace, Stone, BrkCmn             | Masonry veneer type                                                    |
| MasVnrArea    | numeric     | feature    |                  8 |                15 | 0 to 1600                          | Masonry veneer area in square feet                                     |
| ExterQual     | categorical | feature    |                  0 |                 0 | Gd, TA, Ex, Fa                     | Evaluates the quality of the material on the exterior                  |
| ExterCond     | categorical | feature    |                  0 |                 0 | TA, Gd, Fa, Po                     | Evaluates the present condition of the material on the exterior        |
| Foundation    | categorical | feature    |                  0 |                 0 | PConc, CBlock, BrkTil, Wood        | Type of foundation                                                     |
| BsmtQual      | categorical | feature    |                 37 |                44 | Gd, TA, Ex, Fa                     | Evaluates the height of the basement                                   |
| BsmtCond      | categorical | feature    |                 37 |                45 | TA, Gd, Fa, Po                     | Evaluates the general condition of the basement                        |
| BsmtExposure  | categorical | feature    |                 38 |                44 | No, Gd, Mn, Av                     | Refers to walkout or garden level walls                                |
| BsmtFinType1  | categorical | feature    |                 37 |                42 | GLQ, ALQ, Unf, Rec                 | Rating of basement finished area                                       |
| BsmtFinSF1    | numeric     | feature    |                  0 |                 1 | 0 to 5644                          | Type 1 finished square feet                                            |
| BsmtFinType2  | categorical | feature    |                 38 |                42 | Unf, BLQ, ALQ, Rec                 | Rating of basement finished area (if multiple types)                   |
| BsmtFinSF2    | numeric     | feature    |                  0 |                 1 | 0 to 1474                          | Type 2 finished square feet                                            |
| BsmtUnfSF     | numeric     | feature    |                  0 |                 1 | 0 to 2336                          | Unfinished square feet of basement area                                |
| TotalBsmtSF   | numeric     | feature    |                  0 |                 1 | 0 to 6110                          | Total square feet of basement area                                     |
| Heating       | categorical | feature    |                  0 |                 0 | GasA, GasW, Grav, Wall             | Type of heating                                                        |
| HeatingQC     | categorical | feature    |                  0 |                 0 | Ex, Gd, TA, Fa                     | Heating quality and condition                                          |
| CentralAir    | categorical | feature    |                  0 |                 0 | Y, N                               | Central air conditioning                                               |
| Electrical    | categorical | feature    |                  1 |                 0 | SBrkr, FuseF, FuseA, FuseP         | Electrical system                                                      |
| 1stFlrSF      | numeric     | feature    |                  0 |                 0 | 334 to 4692                        | First Floor square feet                                                |
| 2ndFlrSF      | numeric     | feature    |                  0 |                 0 | 0 to 2065                          | Second floor square feet                                               |
| LowQualFinSF  | numeric     | feature    |                  0 |                 0 | 0 to 572                           | Low quality finished square feet (all floors)                          |
| GrLivArea     | numeric     | feature    |                  0 |                 0 | 334 to 5642                        | Above grade (ground) living area square feet                           |
| BsmtFullBath  | numeric     | feature    |                  0 |                 2 | 0 to 3                             | Basement full bathrooms                                                |
| BsmtHalfBath  | numeric     | feature    |                  0 |                 2 | 0 to 2                             | Basement half bathrooms                                                |
| FullBath      | numeric     | feature    |                  0 |                 0 | 0 to 3                             | Full bathrooms above grade                                             |
| HalfBath      | numeric     | feature    |                  0 |                 0 | 0 to 2                             | Half baths above grade                                                 |
| BedroomAbvGr  | numeric     | feature    |                  0 |                 0 | 0 to 8                             | See Kaggle data_description.txt.                                       |
| KitchenAbvGr  | numeric     | feature    |                  0 |                 0 | 0 to 3                             | See Kaggle data_description.txt.                                       |
| KitchenQual   | categorical | feature    |                  0 |                 1 | Gd, TA, Ex, Fa                     | Kitchen quality                                                        |
| TotRmsAbvGrd  | numeric     | feature    |                  0 |                 0 | 2 to 14                            | Total rooms above grade (does not include bathrooms)                   |
| Functional    | categorical | feature    |                  0 |                 2 | Typ, Min1, Maj1, Min2              | Home functionality (Assume typical unless deductions are warranted)    |
| Fireplaces    | numeric     | feature    |                  0 |                 0 | 0 to 3                             | Number of fireplaces                                                   |
| FireplaceQu   | categorical | feature    |                690 |               730 | TA, Gd, Fa, Ex                     | Fireplace quality                                                      |
| GarageType    | categorical | feature    |                 81 |                76 | Attchd, Detchd, BuiltIn, CarPort   | Garage location                                                        |
| GarageYrBlt   | numeric     | feature    |                 81 |                78 | 1900 to 2010                       | Year garage was built                                                  |
| GarageFinish  | categorical | feature    |                 81 |                78 | RFn, Unf, Fin                      | Interior finish of the garage                                          |
| GarageCars    | numeric     | feature    |                  0 |                 1 | 0 to 4                             | Size of garage in car capacity                                         |
| GarageArea    | numeric     | feature    |                  0 |                 1 | 0 to 1418                          | Size of garage in square feet                                          |
| GarageQual    | categorical | feature    |                 81 |                78 | TA, Fa, Gd, Ex                     | Garage quality                                                         |
| GarageCond    | categorical | feature    |                 81 |                78 | TA, Fa, Gd, Po                     | Garage condition                                                       |
| PavedDrive    | categorical | feature    |                  0 |                 0 | Y, N, P                            | Paved driveway                                                         |
| WoodDeckSF    | numeric     | feature    |                  0 |                 0 | 0 to 857                           | Wood deck area in square feet                                          |
| OpenPorchSF   | numeric     | feature    |                  0 |                 0 | 0 to 547                           | Open porch area in square feet                                         |
| EnclosedPorch | numeric     | feature    |                  0 |                 0 | 0 to 552                           | Enclosed porch area in square feet                                     |
| 3SsnPorch     | numeric     | feature    |                  0 |                 0 | 0 to 508                           | Three season porch area in square feet                                 |
| ScreenPorch   | numeric     | feature    |                  0 |                 0 | 0 to 480                           | Screen porch area in square feet                                       |
| PoolArea      | numeric     | feature    |                  0 |                 0 | 0 to 738                           | Pool area in square feet                                               |
| PoolQC        | categorical | feature    |               1453 |              1456 | Ex, Fa, Gd                         | Pool quality                                                           |
| Fence         | categorical | feature    |               1179 |              1169 | MnPrv, GdWo, GdPrv, MnWw           | Fence quality                                                          |
| MiscFeature   | categorical | feature    |               1406 |              1408 | Shed, Gar2, Othr, TenC             | Miscellaneous feature not covered in other categories                  |
| MiscVal       | numeric     | feature    |                  0 |                 0 | 0 to 1.55e+04                      | $Value of miscellaneous feature                                        |
| MoSold        | numeric     | feature    |                  0 |                 0 | 1 to 12                            | Month Sold (MM)                                                        |
| YrSold        | numeric     | feature    |                  0 |                 0 | 2006 to 2010                       | Year Sold (YYYY)                                                       |
| SaleType      | categorical | feature    |                  0 |                 1 | WD, New, COD, ConLD                | Type of sale                                                           |
| SaleCondition | categorical | feature    |                  0 |                 0 | Normal, Abnorml, Partial, AdjLand  | Condition of sale                                                      |
| SalePrice     | numeric     | target     |                  0 |                 0 | 3.49e+04 to 7.55e+05               | See Kaggle data_description.txt.                                       |

## 4. Synthetic contextual fields

Each synthetic field is documented with its data type, unit, valid range, the exact
generation logic and the business assumption behind it. Generation is deterministic
for a given seed and input order, and the generator is seeded with 42.

| Field                  | Type          | Unit                          | Valid range                     | Generation logic                                                                                                                                                                                                                                           | Business assumption                                                           |
|:-----------------------|:--------------|:------------------------------|:--------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------|
| neigh_lat              | float         | decimal degrees               | 41.99 - 42.07                   | Neighbourhood centroid latitude (documented table) plus a jitter drawn from N(0, 0.0015) to represent intra-neighbourhood position.                                                                                                                        | Property coordinates are available from the listing geocoder.                 |
| neigh_lon              | float         | decimal degrees               | -93.67 - -93.60                 | Neighbourhood centroid longitude plus N(0, 0.0015) jitter.                                                                                                                                                                                                 | Same as above.                                                                |
| dist_school_km         | float         | km                            | 0.10 - 8.00                     | Log-normal draw with median exp(1.90 - 0.22 * tier) and sigma 0.35, clipped. School accessibility improves with the neighbourhood tier.                                                                                                                    | Distance to the nearest public elementary school, straight line.              |
| dist_hospital_km       | float         | km                            | 0.30 - 20.00                    | Log-normal median exp(5.00 - 0.55 * tier), sigma 0.30, clipped.                                                                                                                                                                                            | Distance to the nearest general hospital.                                     |
| dist_transit_km        | float         | km                            | 0.20 - 15.00                    | Log-normal median exp(3.20 - 0.42 * tier), sigma 0.33, clipped.                                                                                                                                                                                            | Distance to the nearest bus stop with a 15-minute service frequency.          |
| dist_park_km           | float         | km                            | 0.10 - 10.00                    | Log-normal median exp(1.60 - 0.15 * tier), sigma 0.30, clipped.                                                                                                                                                                                            | Distance to the nearest public park of at least 1 hectare.                    |
| school_rating          | integer       | score 1-10                    | 1 - 10                          | clip(round(3.2 + 1.15 * tier + N(0, 0.9)), 1, 10).                                                                                                                                                                                                         | State accountability rating rescaled to a ten-point scale.                    |
| walk_score             | integer       | score 0-100                   | 0 - 100                         | clip(round(18 + 10.5 * tier + N(0, 8)), 0, 100).                                                                                                                                                                                                           | Walk Score-style index of everyday errands on foot.                           |
| crime_index            | integer       | index 0-100 (higher is worse) | 0 - 100                         | clip(round(78 - 10.5 * tier + N(0, 7)), 0, 100).                                                                                                                                                                                                           | Inverse of a standardised property-crime rate at the census block level.      |
| amenity_density_1_5km  | integer       | count                         | 0 - 40                          | Poisson draw with mean 2 + 1.6 * tier.                                                                                                                                                                                                                     | Number of retail, education and health amenities within 1.5 km.               |
| park_area_ha           | float         | hectares                      | 0.05 - 60.00                    | Log-normal median exp(2.0 + 0.8 * tier), sigma 0.6, clipped.                                                                                                                                                                                               | Aggregate green space within the neighbourhood catchment.                     |
| amenity_score          | float         | index 0-100                   | 0 - 100                         | Weighted composite of inverse distances (school 0.25, hospital 0.15, transit 0.20), school rating (0.15), walk score (0.15) and inverse crime (0.10); z-scored with fixed reference moments and mapped linearly to 0-100.                                  | A single interpretable accessibility index for agents and buyers.             |
| renovation_flag        | integer (0/1) | flag                          | 0 - 1                           | 1 when YearRemodAdd > YearBuilt, otherwise 0.                                                                                                                                                                                                              | The county assessor records the last remodelling permit year.                 |
| renovation_scope       | category      | label                         | None / Minor / Moderate / Major | None when the property was never remodelled. Otherwise the scope is Major when the remodelling lag is at least 20 years, Moderate when the lag is at least 8 years, and Minor otherwise.                                                                   | Larger lags and higher quality imply a deeper intervention.                   |
| days_on_market         | integer       | days                          | 1 - 365                         | Log-normal with mean 16 + 7.5 * (5 - tier) + 4 * max(0, 8 - OverallQual), multiplied by a month-of-year factor (0.80 in June, 1.25 in January), the interest-rate ratio (rate / 5.5)^0.9 and the supply ratio (months_supply / 6.5)^0.5; clipped to 1-365. | Slow markets, weaker locations and winter listings stay on the market longer. |
| listing_date           | date          | yyyy-mm-dd                    | 2005-01 - 2011-12               | Sale month (day 15) minus days_on_market.                                                                                                                                                                                                                  | A listing is published at least one day before the transaction closes.        |
| mortgage_rate_pct      | float         | percent per annum             | 4.30 - 7.10                     | Monthly path interpolated from documented anchors (6.40% in 2006-01, 6.80% in 2006-07, 5.00% in 2009-06, 4.90% in 2010-12) plus AR(1) noise with sigma 0.05.                                                                                               | 30-year fixed conforming mortgage rate; approximates FRED MORTGAGE30US.       |
| hpi_index              | float         | index (2006-01 = 100)         | 94 - 109                        | Monthly path interpolated from documented anchors plus AR(1) noise with sigma 0.6.                                                                                                                                                                         | Local house-price index, calibrated to the mild Ames cycle of 2006-2011.      |
| hpi_yoy_pct            | float         | percent                       | -12 - 12                        | Year-over-year percentage change of hpi_index.                                                                                                                                                                                                             | Derived from the simulated index.                                             |
| months_supply          | float         | months                        | 3.8 - 9.5                       | Monthly path interpolated from documented anchors plus AR(1) noise with sigma 0.15.                                                                                                                                                                        | Inventory divided by monthly absorption rate.                                 |
| local_unemployment_pct | float         | percent                       | 3.7 - 6.3                       | Monthly path interpolated from documented anchors plus AR(1) noise with sigma 0.05.                                                                                                                                                                        | Story County unemployment rate, approximates BLS LAUS.                        |

## 5. Engineered features

Derived features are computed by `revaluation.data.features.engineer_features`.
The function is row-wise, which guarantees that the same transformation can be
applied to a single listing at inference time.

| Field                   | Formula                                                                   | Unit                 | Rationale                                                                            |
|:------------------------|:--------------------------------------------------------------------------|:---------------------|:-------------------------------------------------------------------------------------|
| total_sf                | total_bsmt_sf + first_flr_sf + second_flr_sf                              | sqft                 | Single measure of usable floor area across basement and living levels.               |
| total_bath              | full_bath + 0.5 * half_bath + bsmt_full_bath + 0.5 * bsmt_half_bath       | bathroom equivalents | Weights half bathrooms at one half, the conventional listing convention.             |
| total_porch_sf          | sum of deck, open, enclosed, three-season and screened porch areas        | sqft                 | Aggregates outdoor amenity space into one interpretable quantity.                    |
| age_at_sale             | yr_sold - year_built                                                      | years                | Depreciation and vintage effect; replaces the raw construction year in modelling.    |
| remod_age               | yr_sold - year_remod_add                                                  | years                | Time since the last permitted remodelling.                                           |
| is_renovated            | 1 if year_remod_add > year_built else 0                                   | flag                 | Identifies properties with a recorded renovation.                                    |
| renovation_scope_code   | ordinal mapping of renovation_scope: None=0, Minor=1, Moderate=2, Major=3 | ordinal              | Captures renovation depth for models that cannot use the categorical label directly. |
| has_pool                | 1 if pool_area > 0 else 0                                                 | flag                 | Rare amenity with a non-linear price effect.                                         |
| has_garage              | 1 if garage_area > 0 or garage_cars > 0 else 0                            | flag                 | Separates absence of a garage from a small garage.                                   |
| has_fireplace           | 1 if fireplaces > 0 else 0                                                | flag                 | Separates absence of a fireplace from one fireplace.                                 |
| has_second_floor        | 1 if second_flr_sf > 0 else 0                                             | flag                 | Distinguishes single-storey from two-storey stock.                                   |
| liv_area_ratio          | gr_liv_area / lot_area                                                    | ratio                | Development intensity of the parcel.                                                 |
| rooms_per_1000sf        | tot_rms_abv_grd / (total_sf / 1000)                                       | rooms per 1,000 sqft | Room density, a proxy for compartmentalisation.                                      |
| avg_room_size           | gr_liv_area / tot_rms_abv_grd                                             | sqft per room        | Spaciousness of the layout.                                                          |
| garage_area_ratio       | garage_area / lot_area                                                    | ratio                | Share of the parcel devoted to the garage.                                           |
| quality_condition_index | overall_qual * overall_cond                                               | index 1-100          | Joint material quality and maintenance condition.                                    |
| price_per_sqm           | SalePrice / (gr_liv_area * 0.092903)                                      | USD per sqm          | Analysis and reporting only; excluded from model inputs because it uses the target.  |
| log_price               | ln(1 + SalePrice)                                                         | log USD              | Modelling target; reduces skewness from 1.88 to 0.12.                                |

## 6. Cleaning rules

| Field                   | Rule                                            | Treatment                                               |
|:------------------------|:------------------------------------------------|:--------------------------------------------------------|
| alley                   | Missing means the feature is absent             | Impute the literal 'None'                               |
| bsmt_qual               | Missing means the feature is absent             | Impute the literal 'None'                               |
| bsmt_cond               | Missing means the feature is absent             | Impute the literal 'None'                               |
| bsmt_exposure           | Missing means the feature is absent             | Impute the literal 'None'                               |
| bsmt_fin_type1          | Missing means the feature is absent             | Impute the literal 'None'                               |
| bsmt_fin_type2          | Missing means the feature is absent             | Impute the literal 'None'                               |
| fireplace_qu            | Missing means the feature is absent             | Impute the literal 'None'                               |
| garage_type             | Missing means the feature is absent             | Impute the literal 'None'                               |
| garage_finish           | Missing means the feature is absent             | Impute the literal 'None'                               |
| garage_qual             | Missing means the feature is absent             | Impute the literal 'None'                               |
| garage_cond             | Missing means the feature is absent             | Impute the literal 'None'                               |
| pool_qc                 | Missing means the feature is absent             | Impute the literal 'None'                               |
| fence                   | Missing means the feature is absent             | Impute the literal 'None'                               |
| misc_feature            | Missing means the feature is absent             | Impute the literal 'None'                               |
| mas_vnr_type            | Missing means the feature is absent             | Impute the literal 'None'                               |
| mas_vnr_area            | Missing means a zero quantity                   | Impute 0                                                |
| bsmt_fin_sf1            | Missing means a zero quantity                   | Impute 0                                                |
| bsmt_fin_sf2            | Missing means a zero quantity                   | Impute 0                                                |
| bsmt_unf_sf             | Missing means a zero quantity                   | Impute 0                                                |
| total_bsmt_sf           | Missing means a zero quantity                   | Impute 0                                                |
| bsmt_full_bath          | Missing means a zero quantity                   | Impute 0                                                |
| bsmt_half_bath          | Missing means a zero quantity                   | Impute 0                                                |
| garage_cars             | Missing means a zero quantity                   | Impute 0                                                |
| garage_area             | Missing means a zero quantity                   | Impute 0                                                |
| electrical              | Unstructured missingness                        | Impute the training mode                                |
| kitchen_qual            | Unstructured missingness                        | Impute the training mode                                |
| exterior1st             | Unstructured missingness                        | Impute the training mode                                |
| exterior2nd             | Unstructured missingness                        | Impute the training mode                                |
| sale_type               | Unstructured missingness                        | Impute the training mode                                |
| functional              | Unstructured missingness                        | Impute the training mode                                |
| utilities               | Unstructured missingness                        | Impute the training mode                                |
| ms_zoning               | Unstructured missingness                        | Impute the training mode                                |
| lot_frontage            | Unstructured missingness (259 training records) | Impute the neighbourhood median, then the global median |
| garage_yr_blt           | No garage or missing permit year                | Construction year, adjusted by the median garage lag    |
| gr_liv_area / SalePrice | GrLivArea > 4000 and SalePrice < 300000         | Remove from the training set (2 records)                |
| lot_area                | LotArea > 100000                                | Flag only; no removal                                   |

## 7. Leakage controls

The following fields are available in the analytical table but are deliberately
excluded from the model inputs:

| Field | Reason |
|---|---|
| `SalePrice`, `log_price`, `price_per_sqm` | Direct functions of the target |
| `sale_condition`, `sale_type` | Observed only after the transaction closes |
| `days_on_market`, `listing_date` | Marketing outcome, unknown at listing time |
| `yr_sold`, `mo_sold` | Replaced by the macro-financial context of the sale month |
| `id`, `source_file`, `sale_date` | Identifiers and bookkeeping |

## 8. Versioning

The dictionary is generated by `scripts/07_export_data_dictionary.py`. Any change
to the synthetic specification or to the feature engineering code must be followed
by a regeneration of this document and of `data_dictionary.xlsx`.
