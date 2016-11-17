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

We start in the **hub** package where `Primary.py` is our driver module. First, we will generate the features. Note that we create all of these features per product. We need to do this for all of our training and test data minus the relevance scores as those would be separated out as an independent data frame. 

Hence, we merge `train.csv`, `attributes.csv` and `product_descriptions.csv` and call it `combined_data_frame`. The idea is that at then end of all of the transformations, each column of this data frame would serve as a feature. 

Before any of the transformations, here are the columns of the `combined_data_frame`

- id
- product_uid
- product_title
- search_term
- product_description
- name
- value

I have divided the feature creation into 4 parts. Please follow the well documented code to see how this is implemented. Here is a brief summary

1 `Attribute_Features.generate_attribute_features()`

These are mostly derived from the `attributes.csv`. 
- Create `brand name` as a feature. Look for `MFG Brand Name` as an attribute
- Create `bullet` as a feature by combining all of the `Bullet` attributes
- Create `bullet_count` as a feature by counting the number of bullets a product has, if any
- Identify `color` as a feature by looking at any attribute values that have `color` as a substring
- Identify `material` as a feature by looking at any attribute values that have `material` as a substring
- Whether the product is available for commercial or residential use can be a feature. We create this by looking at all attribute values  that have `commercial / residential` as a substring
- Whether the product is available for indoor or outdoor use can be a feature. We create this by looking at all attribute values that have `indoor / outdoor` as a substring
- Whether the product is Energy Star Certified can be another feature. Create it by checking if any attribute for a product has `energy star certified` as a substring and whether the value of this attribute if `Yes`

So we added 10 new columns to `combined_data_frame` as

- brand
- bullet
- bullet_count
- color
- material
- flag_commercial
- flag_residential
- flag_indoor
- flag_outdoor
- flag_estar

2 `Text_Proc_Features.generate_text_proc_features()`

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

So we added 41 new columns to `combined_data_frame` as

- ratio_search_term_in_product_description
- flag_search_term_in_brand
- num_search_term_in_brand
- ratio_search_term_in_brand
- flag_search_term_in_bullet
- num_search_term_in_bullet
- ratio_search_term_in_bullet
- 0th_word_in_product_title
- 1th_word_in_product_title
- 2th_word_in_product_title
- 3th_word_in_product_title
- 4th_word_in_product_title
- 5th_word_in_product_title
- 6th_word_in_product_title
- 7th_word_in_product_title
- 8th_word_in_product_title
- 9th_word_in_product_title
- 0th_word_in_product_description
- 1th_word_in_product_description
- 2th_word_in_product_description
- 3th_word_in_product_description
- 4th_word_in_product_description
- 5th_word_in_product_description
- 6th_word_in_product_description
- 7th_word_in_product_description
- 8th_word_in_product_description
- 9th_word_in_product_description
- 0th_word_in_bullet
- 1th_word_in_bullet
- 2th_word_in_bullet
- 3th_word_in_bullet
- 4th_word_in_bullet
- 5th_word_in_bullet
- 6th_word_in_bullet
- 7th_word_in_bullet
- 8th_word_in_bullet
- 9th_word_in_bullet
- brand_encoded
- flag_attr_has_material
- flag_attr_has_color
- flag_has_attr

3 `Distance_Metric_Features.generate_distance_metric_features()`

In this step, we will calculate cosine similarity using `CountVectorizer` and `TfidfVectorizer`. Also, we will use Truncated Singular Value Decomposition for dimensionality reduction
- Using `CountVectorizer`, learn a vocabulary dictionary of all tokens in `search_term`, `product_title`, `product_description` and `bullet`
- Using `TfidfVectorizer`, learn vocabulary and idf of the words in `search_term`, `product_title`, `product_description` and `bullet`
- For `product_title`,`product_description` and `bullet`, add cosine similarity between **count** vectors of `search_term` and these columns as new features
- For `product_title`,`product_description` and `bullet`, add cosine similarity between **tfidf** vectors of `search_term` and these columns as new features
- For `product_title`,`product_description` and `bullet`, perform truncated singular value decomposition for dimensionality reduction. Each truncated svd will be a feature

So we added 86 new columns to `combined_data_frame` as

- cv_cos_sim_search_term_product_title
- tiv_cos_sim_search_term_product_title
- cv_cos_sim_search_term_product_description
- tiv_cos_sim_search_term_product_description
- cv_cos_sim_search_term_bullet
- tiv_cos_sim_search_term_bullet
- search_term_bow_tsvd_0
- search_term_bow_tsvd_1
- search_term_bow_tsvd_2
- search_term_bow_tsvd_3
- search_term_bow_tsvd_4
- search_term_bow_tsvd_5
- search_term_bow_tsvd_6
- search_term_bow_tsvd_7
- search_term_bow_tsvd_8
- search_term_bow_tsvd_9
- search_term_tfidf_tsvd_0
- search_term_tfidf_tsvd_1
- search_term_tfidf_tsvd_2
- search_term_tfidf_tsvd_3
- search_term_tfidf_tsvd_4
- search_term_tfidf_tsvd_5
- search_term_tfidf_tsvd_6
- search_term_tfidf_tsvd_7
- search_term_tfidf_tsvd_8
- search_term_tfidf_tsvd_9
- product_title_bow_tsvd_0
- product_title_bow_tsvd_1
- product_title_bow_tsvd_2
- product_title_bow_tsvd_3
- product_title_bow_tsvd_4
- product_title_bow_tsvd_5
- product_title_bow_tsvd_6
- product_title_bow_tsvd_7
- product_title_bow_tsvd_8
- product_title_bow_tsvd_9
- product_title_tfidf_tsvd_0
- product_title_tfidf_tsvd_1
- product_title_tfidf_tsvd_2
- product_title_tfidf_tsvd_3
- product_title_tfidf_tsvd_4
- product_title_tfidf_tsvd_5
- product_title_tfidf_tsvd_6
- product_title_tfidf_tsvd_7
- product_title_tfidf_tsvd_8
- product_title_tfidf_tsvd_9
- product_description_bow_tsvd_0
- product_description_bow_tsvd_1
- product_description_bow_tsvd_2
- product_description_bow_tsvd_3
- product_description_bow_tsvd_4
- product_description_bow_tsvd_5
- product_description_bow_tsvd_6
- product_description_bow_tsvd_7
- product_description_bow_tsvd_8
- product_description_bow_tsvd_9
- product_description_tfidf_tsvd_0
- product_description_tfidf_tsvd_1
- product_description_tfidf_tsvd_2
- product_description_tfidf_tsvd_3
- product_description_tfidf_tsvd_4
- product_description_tfidf_tsvd_5
- product_description_tfidf_tsvd_6
- product_description_tfidf_tsvd_7
- product_description_tfidf_tsvd_8
- product_description_tfidf_tsvd_9
- bullet_bow_tsvd_0
- bullet_bow_tsvd_1
- bullet_bow_tsvd_2
- bullet_bow_tsvd_3
- bullet_bow_tsvd_4
- bullet_bow_tsvd_5
- bullet_bow_tsvd_6
- bullet_bow_tsvd_7
- bullet_bow_tsvd_8
- bullet_bow_tsvd_9
- bullet_tfidf_tsvd_0
- bullet_tfidf_tsvd_1
- bullet_tfidf_tsvd_2
- bullet_tfidf_tsvd_3
- bullet_tfidf_tsvd_4
- bullet_tfidf_tsvd_5
- bullet_tfidf_tsvd_6
- bullet_tfidf_tsvd_7
- bullet_tfidf_tsvd_8
- bullet_tfidf_tsvd_9

4 `Similarity_Features.generate_similarity_features()`

Finally, we will calculate **edit distance** and **jaccard similarity** between `search_term` and certain other fields and add those as features

- Find out the jaccard similarity between `search_term` and each of `product_title`,`product_description`,`brand`,`bullet`. Add those as features

- Find out the min and avg edit distance between `search_term` and each of `product_title`,`product_description`. Add those as features   

So we added 8 new columns to `combined_data_frame` as
    
- jaccard_search_term_product_title
- jaccard_search_term_product_description
- jaccard_search_term_brand
- jaccard_search_term_bullet
- edit_dist_search_term_product_title_min
- edit_dist_search_term_product_title_avg
- edit_dist_search_term_product_description_min
- edit_dist_search_term_product_description_avg

Now all of the features have been created and out `combined_data_frame` is almost ready for training our regressors. As a last step we will drop the following 10 columns as they have text values and are not suitable for regression

- attr
- search_term
- product_title
- product_description
- brand
- bullet
- color
- material    
- tokens_search_term
- tokens_product_title
- tokens_product_description
- tokens_brand
- tokens_bullet

and so we end up with **163** features.

#### Predicting

1. Run `Random_Forest_Regressor` for predicting using `RandomForestRegressor`
2. Run `Gradient_Boosting_Regressor` for predicting using `GradientBoostingRegressor`
3. Run `Extra_Trees_Regressor` for predicting using `ExtraTreesRegressor` 
4. Run `Ada_B` for predicting using `AdaBoostRegressor` 

Please follow the well documented code.  

### Results 

The top ranked score for this competition is **0.43192**

1. `Random_Forest_Regressor` using `{"max_features": 100, "n_estimators": 500}` gets me a score of **0.47138**
2. `Gradient_Boosting_Regressor` using `{"max_features": 100, "n_estimators": 500, "subsample": 0.8}` gets me a score of **0.46829**
3. `Extra_Trees_Regressor` using `{"verbose": 100, "max_features": 100, "n_estimators": 500}` gets me a score of **0.47586** 
4. `AdaBoostRegressor` does not give much improvement over the above scores. Need to investigate how to better use this.