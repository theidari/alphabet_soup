<p align="center">
<img src="https://github.com/theidari/alphabet_soup/blob/main/assets/header.png" width=900px>
</p>
<h3>Overview</h3>
<p align="justify">Alphabet Soup, a nonprofit foundation, is seeking assistance in identifying the most promising funding applicants using a data-driven approach. To achieve this goal, a binary classifier will be developed using a provided dataset that contains information on over 34,000 organizations that have previously received funding from Alphabet Soup. The <b>objective</b> is to predict the likelihood of success for applicants who receive funding from the foundation.</P>
<h4>Why this project?</h4>
<hr>
<h3>2. Results</h3>
<h4>1.2. Data Preprocessing</h4>
<p align="justify">
<h5>1.1.2. select taget variable and features</h5>
➤ The target variable in this model was set to the 'IS_SUCCESSFUL' column.<br>
➤ Features: 
<ol>
<li align="justify"><b>initial attempt:</b> The feature set excluded the identification columns 'EIN' and 'NAME', while the remaining columns were used as features.</li>
<li align="justify"><b>pre-optimization attempt:</b> During pre-optimization attempts, the 'EIN', 'NAME', and 'SPECIAL_CONSIDERATIONS' columns were dropped from the feature set. The 'EIN' column, which contains a unique identification number for each organization, was deemed irrelevant to the model since it does not provide any predictive power. The 'NAME' column, which contains the name of the organization, is also unlikely to be a significant predictor of the target variable. Finally, the 'SPECIAL_CONSIDERATIONS' column was removed because it only contains two unique values, which may not provide significant information to the model. Also, The feature set excluded the 'ASK_AMT' and 'STATUS' columns, as they may not significantly contribute to predicting the target variable. However, before removing these columns, exploratory data analysis should be conducted to investigate if there is any correlation between 'ASK_AMT' or "STATUS" and 'IS_SUCCESSFUL'. To guide the decision-making process, a condition such as <code>df['column'].corr(df['IS_SUCCESSFUL']) < 0.1</code> can be used. </li>
<li align="justify"><b>Optimization attempt:</b> The "NAME" column was added back to the feature set based on a particular condition. Although including the "NAME" column may introduce bias into the modeling process since it serves as an identification column, a criterion was established to mitigate this potential bias by grouping the names into just over 100 categories.</li>
</ol>
<h5>2.1.2. binning</h5>
<ul align="justify">
<li>In an <b><ins>initial</ins></b> and <b><ins>pre-optimization</ins></b> stage, the 'APPLICATION_TYPE' and 'CLASSIFICATION' features had 17 and 71 distinct values, respectively, while the other features had less than 7 unique values. To simplify the data and facilitate analysis and modeling, a new category called 'other' was created for data values below 500 and 800 in the respective columns.</li>
<li>Furthermore, during the <b><ins>optimization</ins></b> attempt, a new category labeled as 'other' was introduced for data values below 100 in the 'NAME' column.</li>
</ul>
</p>
<h4>2.2. Data Preprocessing</h4>







different feature sets were used to understand their effects on the loss and accuracy.

What variable(s) are the target(s) for your model?
What variable(s) are the features for your model?
What variable(s) should be removed from the input data because they are neither targets nor features?
Compiling, Training, and Evaluating the Model

How many neurons, layers, and activation functions did you select for your neural network model, and why?
Were you able to achieve the target model performance?
What steps did you take in your attempts to increase model performance?
