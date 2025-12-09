Quantitative Methods: Conservation 350: Final Project

1) Introduction and Paper Selection:

Forest restoration research frequently relies on biodiversity indicators, especially bird communities to evaluate habitat recovery. The paper titled “Active Restoration Fosters Better Recovery of Tropical Rainforest Birds Than Natural Regeneration in Degraded Forest Fragments” uses ecological statistics (particularly rarefaction and jackknife estimates) to assess differences across restoration treatments. They chose to use rarefaction curves since there were many differences in sampling conditions across the different sites and this curve allowed for fair comparison of species richness while also controlling sampling effort. The jackknife estimates were used to accommodate for cryptic or shy species that could have gone undetected. 

I extend on the paper’s ideas by applying a Random Forest machine-learning classifier to predict Site_type (Benchmark Rainforest, Active Restoration, Natural Regeneration) based on field survey observations. This method is relevant to class since it uses scikit-learn pipelines, train/test splits, and model evaluation. 

2) Data Collection and Preprocessing

2.1. Raw Data Documentation

The dataset final_project_data.csv contains bird observations from multiple sites with each row representing a single detection event. 


2.2. Data Cleaning

Steps performed (using pandas)
Checked for missing or null values in every column
Verified that species names were consistent (no duplicates caused by spelling/spacing)
Ensured numeric columns were correctly typed (StartTime, Number, Distance etc.)
Confirmed that Site_type had three classes with adequate representation

2.3. Data Visualization

Four bar graphs were created to visualize the initial dataset. The first graph represented the number of species in unrestored sites if there were more than 50 individuals, the second was the number of species in unrestored sites if less than 20 individuals. The third represented the number of species in restored sites if more than 50 individuals, and the fourth was the number of species in restored sites if less than 20 individuals. The graphs were split in this way to make the visualization easier since the dataset was too large to fit onto one graph.

Pie graphs were then used to visualize the model output and results. The first pie chart represented the proportion of bird survey records that came from each site type (most surveys came from Benchmark with 37%, 29.6% came from unrestored, and 33.4% came from restored. Another two pie charts were created to show the distribution of species in unrestored sites if there were more than 50 individuals, and to show the distribution of species in restored sites if there were more than 50 individuals present.

Furthermore, stacked bar graphs were created. In the first graph titled: “Stacked Bar Chart of Species Count by site (Grouped ‘<20’ into ‘Other’)”, the species are ranked from most abundant to least abundant. For example, the Nilgiri Flowerpecker, White-cheeked Barbet, and Greenish Warbler have the highest number of counts across all three site types. The bars were also each divided by site condition, allowing us to compare how species are distributed across the different habitat conditions. 

The second stacked bar graph titled: “Species Overlap Across Sites (Grouped ‘<20’ as ‘Other’)”, is much more detailed. Instead of showing totals by site type like the previous graph, it breaks down the counts by species for every individual site. Therefore, each bar is representative of 1 species, and each color is a count for a specific site. This is a useful graph, however, it is very visually overwhelming, making patterns particularly difficult to see and isolate. 

The third graph titled: “Species Abundance by Site Type (Grouped <20 as ‘other’)” shows species abundance based on total abundance across the three sites. This graph shows that the Benchmark (blue) dominates for many species - indicating that forest sites do indeed support the highest total bird abundance. It also shows that the Restored sites (orange) host intermediate bird abundance, meaning that the restoration is working and species are slowly recovering. As expected, Unrestored sites (green) have the smallest segments since they have lower habitat complexity and resource availability. 
2.4) Data Setup for Modeling

Input (X): 
Species: categorical
Habitat Structural Features (HSF): categorical
StartTime, TimeSeg, Number, Distance: numeric

Output (Y):
Site_type 

3) Modeling and Analysis

3.1. Choice of Model

A Random Forest Classifier was chosen. This is essentially a model made up of many decision trees, where each tree makes a prediction, and the forest determines the final output. This model handles mixed data types (categorical and numerical) to compare nonlinear relationships. It is also robust to outliers which is important since there were a few rare outlier species in the dataset. This model is also ideal since normalization is not necessary and it requires minimal preprocessing. 

3.2. Train/Test Split

An 80/20 split was used with stratify=y:

train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


Stratification ensures equal representation of AR, NR, and BM in both sets. Without stratification, some classes may have been underrepresented or missing in the training set. 

The 80/20 split was chosen since 80% for training provides enough data for the model to learn patterns, and 20% for testing provides a representative hold-out set to evaluate the model’s performance on unseen data. This is also important to prevent overfitting to a dominant class, making the evaluation metrics (accuracy, precision, recall) reliable and interpretable. 

3.3. Model training and Testing 

The model was trained using: 
	rf_model.fit(X_train, y_train)

Performance was evaluated using: 
Accuracy 
Precision
Recall
Confusion matrix
Classification report

The overall accuracy of my model was 0.45 (45%). This means that my model correctly predicts the site type almost half of the time. This is a lower accuracy than desired. Likely reasons for the lower accuracy is because AR (Restored) and NR (Unrestored) sites shared many bird species making it more difficult to distinguish. There was also quite a bit of overlap in habitat structure (HSF) or abundance. High species diversity in some areas also led to many species being classified as rare (‘noisy’). 

Furthermore, below is a summarized analysis for precision, recall, and F-1 Score (combining precision and recall for when classes are imbalanced): 

Benchmark Rainforest (BM)
Precision = 0.51 (51% of predications labeled BM were actually BM)
Recall = 0.57 (model correctly identified 57% of actual BM sites)
BM sites are generally easier to detect since they have very distinct rainforest species. 

Restored (AR)
Precision = 0.42 (42%)
Recall = 0.42 (42%)
This is a moderate performance
There was some overlap with BM and NR sites which made it harder to separate and more confusing.

Unrestored (NR)
Precision = 0.40 (40%)
Recall = 0.34 (34%)
This is the lowest performance
These sites were the most difficult to correctly identify. This is potentially due to some pioneer or generalist species also occurring in AR sites which created misclassification. 

3.4. Model Interpretation:

Several rainforest specialist species strongly predicted BM sites. This is because specialist species cannot tolerate degradation and need the mature forest with tall tree canopies, complex structures, and high tree diversity. Open-Country species predicted NR sites since Natural Regeneration often begins with open, sunny, grassy habitat that is dominated by pioneer species that need little tree-canopy height, and simple structure. Actively Restored Sites generally had a mixture of rainforest specialists and generalists. This is because these forest structures are still developing (these sites habitat quality do improve more quickly than NR sites). Abundance is also important to take into consideration since a higher abundance generally indicates better habitat quality, and a lower abundance is an indicator of a stressed or simplified habitat. 

In Summary: 

BM: dense canopy =  forest-dependent species 
NR: pioneers and generalists = open-country indicators
AR: transition community = mixture of species

[This supports my initial thinking since I hypothesized that an environment with greater species richness and abundance is going to be more complex and less degraded than an environment where there is low species richness and abundance. I was thinking about Shannon Diversity which looks at both species richness and abundance (Higher Shannon Diversity means higher biodiversity).] 


4)  Model Tuning and Comparison

4.1. Tuning

After initial training, the Random Forest model was tuned to improve predictive performance. Hyperparameter tuning was performed using cross-validation, which evaluates the model on multiple subsets of the training data to estimate its expected performance.

Note: The cross-validation accuracy is lower than the final test accuracy. This is expected because CV evaluates the model on different splits of the training data and is a more conservative estimate. It also prevents overfitting to the training data, especially in high-dimensional categorical datasets like bird species observations.

Confusion Matrix:

[[413  84  56]
 [108 314  77]
 [129  94 218]]


The confusion matrix (showing misidentified classes) confirms this since BM sites remain the easiest to identify, with 75% of true BM sites correctly classified. AR sites show moderate performance (63.3% recall), reflecting their transitional species composition. NR sites remain the most challenging (49% recall), though performance has improved compared to the initial model. Misclassification primarily occurs between AR and NR sites, likely due to overlapping pioneer and generalist species. 

Overall, hyperparameter tuning substantially improved predictive performance, particularly for Restored and unrestored sites. Misclassifications remain concentrated in transitional habitats, highlighting ecological overlap, but the model now more reliably distinguishes between mature rainforest, restored, and unrestored sites.

4.2. Comparison

To evaluate the effectiveness of different machine learning approaches for predicting site type, several models were trained and tested using the same dataset. The overall test accuracies for each model were: 


This indicates that logistic regression has the highest predictive accuracies (45%), suggesting linear relationships between features and site type. Gradient boosting (42%) captures non-linearities but it is slightly affected by overlapping and noisy species. SVM had a lower performance at 40% which is also likely due to class overlap and high dimensionality. The Random Forest, while showing the lowest CV accuracy at 35%, improved substantially on the final test set to 63% accuracy, demonstrating that cross-validation provides a conservative estimate and that the tuned Random Forest generalizes well to unseen data. Overall, simpler models may initially appear to outperform more complex ones in cross-validation when the dataset is noisy, but after tuning, the Random Forest provides the best final predictive performance. Additionally, Random Forest offers valuable ecological insight through feature importance analysis, allowing identification of which species and habitat factors are most influential in predicting site type.

5) Discussion and Reflection 

The results of this project provide valuable insights into how bird communities reflect forest restoration status. The Random Forest model, particularly after hyperparameter tuning, was able to distinguish Benchmark Rainforest, Actively Restored, and Unrestored sites with reasonable accuracy, achieving 63% on the test set. Benchmark sites were consistently the easiest to classify, likely due to their distinct assemblages of forest-dependent species and dense canopy structures. Restored and Unrestored sites exhibited more overlap in species composition, leading to moderate and lower classification performance. These patterns support the original hypothesis that environments with higher species richness and abundance, as reflected by Shannon Diversity, tend to be more complex and less degraded. The model’s feature importance further highlighted that certain rainforest specialists, pioneer species, and habitat structural characteristics strongly influence site type, offering ecological validation and interpretability that extends beyond simple predictive accuracy.
Reflecting on the project process, several strengths are evident. The data cleaning, visualization, and stratified train-test split ensured that the model was trained on reliable and representative data, while the use of Random Forests provided both robustness to outliers and insight into feature importance. Hyperparameter tuning significantly improved predictive performance, illustrating the value of iterative model optimization. However, challenges included dealing with noisy and overlapping species data, high-dimensional categorical features, and interpreting lower cross-validation performance compared to the final test set. Additionally, while simpler models like Logistic Regression achieved higher CV accuracy, they offered less ecological interpretability compared to the tuned Random Forest. Lessons learned include the importance of carefully preprocessing ecological data, the need to balance model complexity with interpretability, and the usefulness of visualizing both species distributions and model outputs to contextualize predictions. Overall, this project demonstrated how machine learning can complement traditional ecological analyses to evaluate restoration outcomes, while also highlighting the challenges inherent in modeling complex biological communities.


See PDF document for a more detailed description (Could not add tables to ReadMe)

