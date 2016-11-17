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

	"product_uid",    "product_title",                     "search_term",  "relevance"
     100001,          "Simpson Strong-Tie 12-Gauge Angle", "angle bracket", 3
	 
If we look at the `product_descriptions.csv`, a row looks like

	"product_uid", "product_description"
	 100001,        "Not only do angles make joints stronger, they also provide more consistent, straight corners. Simpson Strong-Tie offers a wide variety of angles in various sizes and thicknesses to handle light-duty jobs or projects where a structural connection is needed. Some can be bent (skewed) to match the project. For outdoor projects or those where moisture is present, use our ZMAX zinc-coated connectors, which provide extra resistance against corrosion (look for a ""Z"" at the end of the model number).Versatile connector for various 90 connections and home repair projectsStronger than angled nailing or screw fastening aloneHelp ensure joints are consistently straight and strongDimensions: 3 in. x 3 in. x 1-1/2 in.Made from 12-Gauge steelGalvanized for extra corrosion resistanceInstall with 10d common nails or #9 x 1-1/2 in. Strong-Drive SD screws"

Lastly, if we look at `attributes.csv`, a row looks like

	"product_uid", "name",      "value"
	 100001,       "Bullet01",  "Versatile connector for various 90° connections and home repair projects"

So, we have to somehow combine the `search_term`,`product_title` and `product_description` to determine how relevant the `search_term` is. 

The `attributes.csv` is very interesting. Essentially, it has a lot of information that can be mined out. At first glance it might look like we have everything we need in the `product_descriptions.csv`, but it will become clear that we can use the attributes to our advantage.

**Key Idea**

So, given the above data, we have to design features to feed our regressors. Here are a few things we can look at

1. What are the common terms between `search_term`, `product_title`, and `product_description`? 
2. Should we stem the text? Should we have a spell check?
3. What are some of the ways to measure similarity and distance between two pieces of text?
4. Lastly, it turns out that `attributes.csv` can be quite useful. Recall that we had 3 columns

	"product_uid", "name",      "value"
	
From this point on, when i say `attribute` of a product, i will be referring to the value in the `name` column for that product. The value in the `value` column, is the value of that `attribute`. For instance, look at the following row

	"product_uid", "name",         "value"
	 100001         MFG Brand Name  Simpson Strong-Tie
   
this shows that for product having the product_uid as 100001, the MFG Brand Name is Simpson Strong Tie

So, lets take a look and see the top 30 `attributes` by count

	df_attr = pd.read_csv('/path/to/attributes.csv', encoding='ISO-8859-1')
	df_attr['name'].value_counts()[:30]
	
gives us 

	Bullet02                       86248
	Bullet03                       86226
	MFG Brand Name                 86220
	Bullet04                       86174
	Bullet01                       85940
	Product Width (in.)            61137
	Bullet05                       60528
	Product Height (in.)           54698
	Product Depth (in.)            53652
	Product Weight (lb.)           45175
	Bullet06                       44901
	Color Family                   41508
	Bullet07                       34349
	Material                       31499
	Color/Finish                   28540
	Bullet08                       26645
	Certifications and Listings    24583
	Bullet09                       20567
	Assembled Height (in.)         18299
	Assembled Width (in.)          18263
	Assembled Depth (in.)          18198
	Product Length (in.)           16705
	Bullet10                       14763
	Indoor/Outdoor                 12939
	Bullet11                       11784
	Commercial / Residential        9530
	Bullet12                        8795
	ENERGY STAR Certified           8420
	Hardware Included               7462
	Package Quantity                6904

So looks like the above attributes are quite common. We will be using some of them to create our features.

### Flow

We start in the **hub** package where `Primary.py` is our driver module. First, we will generate the features. Note that we create all of these features per product. I have divided this into 4 steps. Before performing any of these steps, the combined data frame looks like this



1. `Attribute_Features.generate_attribute_features()`

These are mostly derived from the `attributes.csv`. 

    - Create `brand name` as a feature. Look for `MFG Brand Name` as an attribute
    - Create `bullet` as a feature by combining all of the `Bullet` attributes
    - Create `bullet_count` as a feature by counting the number of bullets a product has, if any
    - Identify `color` as a feature by looking at any attribute values that have `color` as a substring
    - Identify `material` as a feature by looking at any attribute values that have `material` as a substring
    - Whether the product is available for commercial or residential use can be a feature. We create this by looking at all attribute values  that have `commercial / residential` as a substring
    - Whether the product is available for indoor or outdoor use can be a feature. We create this by looking at all attribute values that have `indoor / outdoor` as a substring
    - Whether the product is Energy Star Certified can be another feature. Create it by checking if any attribute for a product has `energy star certified` as a substring and whether the value of this attribute if `Yes`
    
Please follow the well documented code to see how this is implemented.

2. `Text_Proc_Features.generate_text_proc_features()`

Here's how the next set of features are generated 

    - Creating `len_` features denoting the length of each field that has text
    - Stemming each of the fields that have text
    - Spell check for `search_terms`. There are a lot of typos in the search terms. Here, i have used Google's *did you mean* suggestions to correct the spellings. This gives dramatic improvements in results
    - Count how many times the `search_term` appears in `product_title` and `product_description`. Each of these would be a feature.
    - Determine if the last word of the `search_term` appears in `product_title`. Each of these would be a feature.
    - Determine if the first word of the `search_term` appears in `product_title`. Each of these would be a feature.
    - Count the number of words in `search_title` that are common with `product_title`, `product_description` and `brand`
    - Ratio of the above counts over total words in `search_term` will also be features
    - Brand names would need to be encoded into numeric values
    - We will have some boolean features indicating whether a product has `color` and `material` as attributes

3. `Distance_Metric_Features.generate_distance_metric_features()`

In this step, we will calculate cosine similarity using `CountVectorizer` and `TfidfVectorizer`. Also, we will use Truncated Singular Value Decomposition for dimensionality reduction

	- Using `CountVectorizer`, learn a vocabulary dictionary of all tokens in `search_term`, `product_title`, `product_description` and `bullet`
	- Using `TfidfVectorizer`, learn vocabulary and idf of the words in `search_term`, `product_title`, `product_description` and `bullet`
	- For `product_title`,`product_description` and `bullet`, add cosine similarity between **count** vectors of `search_term` and these columns as new features
	- For `product_title`,`product_description` and `bullet`, add cosine similarity between **tfidf** vectors of `search_term` and these columns as new features
	- For `product_title`,`product_description` and `bullet`, perform truncated singular value decomposition for dimensionality reduction. Each truncated svd will be a feature
    
4. `Similarity_Features.generate_similarity_features()`


and specifically

1. Run `Random_Forest_Regressor` for predicting using `RandomForestRegressor`
2. Run `Gradient_Boosting_Regressor` for predicting using `GradientBoostingRegressor`
3. Run `Extra_Trees_Regressor` for predicting using `ExtraTreesRegressor` 
4. Run `Ada_B` for predicting using `AdaBoostRegressor` 

Please follow the well documented code.  

### Results 
