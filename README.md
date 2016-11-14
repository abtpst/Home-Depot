# Home-Depot

Determining product search relevance in Kaggle Home Depot challenge

https://www.kaggle.com/c/home-depot-product-search-relevance

Please look at the description of the data to get a sense of what the training and test data looks like.
Since we are predicting relevance scores instead of classes, we will be using `Regressors` instead of `Classifiers`

### Main Idea

1. Use the following regressors, with the aim of minimizing the `Root Mean Square Error`

      - `RandomForestRegressor`
      - `GradientBoostingRegressor`
      - `ExtraTreesRegressor`

2. The hard part in this challenge will be to determine the features for any regressor we use. We will use some tried and tested metrics and text processing tricks to figure out `relevant` features.

3. As always, we will use our old friend `GridSearchCV` for getting the optimal values of parameters for initializing the above regressors.

4. Lastly, we will experiment with `AdaBoostRegressor` to see if we can get better results for any regressor.
 
### Set up

1. Anaconda for python 3.5
2. Required packages can be found in requirements.txt

### Layout

1. `src` folder contains all of the packages
2. `resources` folder contains
    1. `data` :-> This should have four folders. `train` for training data, `test` for test data, `dframes` for optionally storing DataFrames during processing and `params` for storing he optimal values of parameters calculated by `GridSearchCV`.
    2. `pickled` :-> For storing pickled objects, if needed.
    3. `results` :-> Store prediction results here.
             
### Packages

***hub***

This has a single module named `Primary`, which is our driver module

***plots***

This has a single module named `Explore`, which can be used for looking into the data via `seaborn`

***preproc***

This is where most of the magic happens. There are several modules for preparing features

1. `Attribute_Features` :-> For generating features by looking at product attributes
2. `Text_Proc_Features` :-> For generating features after some text processing of attribute features
3. `Distance_Metric_Features` :-> For adding features related to various distance metrics
4. `Similarity_Features` :-> For adding features for similarity metrics
5. `Helper_Tools` :-> Some methods to aid in text processing and other common tasks
6. `Slicing_Aides` :-> Helper methods for slicing data frames

***predict***

This has four modules

1. `Random_Forest_Regressor` :-> Predict using `RandomForestRegressor`
2. `Gradient_Boosting_Regressor` :-> Predict using `GradientBoostingRegressor`
3. `Extra_Trees_Regressor` :-> Predict using `ExtraTreesRegressor` 
4. `Ada_B` :-> Predict using `AdaBoostRegressor` 

###Approach

Lets take a look at the data we are dealing with

If we look at the `train.csv`, a row looks like

	"id","product_uid","product_title",                     "search_term",  "relevance"
	 2,   100001,      "Simpson Strong-Tie 12-Gauge Angle", "angle bracket", 3
	 
If we look at the `product_descriptions.csv`, a row looks like

	"product_uid", "product_description"
	 100001,        "Not only do angles make joints stronger, they also provide more consistent, straight corners. Simpson Strong-Tie offers a wide variety of angles in various sizes and thicknesses to handle light-duty jobs or projects where a structural connection is needed. Some can be bent (skewed) to match the project. For outdoor projects or those where moisture is present, use our ZMAX zinc-coated connectors, which provide extra resistance against corrosion (look for a ""Z"" at the end of the model number).Versatile connector for various 90 connections and home repair projectsStronger than angled nailing or screw fastening aloneHelp ensure joints are consistently straight and strongDimensions: 3 in. x 3 in. x 1-1/2 in.Made from 12-Gauge steelGalvanized for extra corrosion resistanceInstall with 10d common nails or #9 x 1-1/2 in. Strong-Drive SD screws"

Lastly, if we look at `attributes.csv`, a row looks like

	"product_uid", "name",      "value"
	 100001,       "Bullet01",  "Versatile connector for various 90° connections and home repair projects"

So, we have to somehow combine the `search_term`,`product_title` and `product_description` to determine how relevant the `search_term` is. 

The `attributes.csv` is very interesting. Essentially, it has a lot of information that can be mined out. At first glance it might look like we have everything we need in the `product_descriptions.csv`, but it will become clear that we can use the attributes to our advantage.

**Key Idea**

### Flow

The common steps for prediction are 


and specifically

1. Run `Random_Forest_Regressor` for predicting using `RandomForestRegressor`
2. Run `Gradient_Boosting_Regressor` for predicting using `GradientBoostingRegressor`
3. Run `Extra_Trees_Regressor` for predicting using `ExtraTreesRegressor` 
4. Run `Ada_B` for predicting using `AdaBoostRegressor` 

Please follow the well documented code.  

### Results 
