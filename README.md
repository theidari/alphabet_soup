<p align="center">
<img src="https://github.com/theidari/alphabet_soup/blob/main/assets/header.png" width=900px>
</p>
<h3>1. Overview</h3>
<p align="justify">Alphabet Soup, a nonprofit foundation, is seeking assistance in identifying the most promising funding applicants using a data-driven approach. To achieve this goal, a binary classifier will be developed using a provided dataset that contains information on over 34,000 organizations that have previously received funding from Alphabet Soup. The <b>objective</b> is to predict the likelihood of success for applicants who receive funding from the foundation.</P>
<hr>
<h3>2. Results</h3>
<h4>1.2. Data Preprocessing</h4>
<h5>1.1.2. select taget variable and features</h5>
<ul>
<li>The target variable in this model was set to the 'IS_SUCCESSFUL' column.</li>
<li>Features: 
<ul>
<li align="justify"><b>initial attempt:</b> The feature set excluded the identification columns 'EIN' and 'NAME', while the remaining columns were used as features.</li>
<li align="justify"><b>pre-optimization attempt:</b> During pre-optimization attempts, the 'EIN', 'NAME', and 'SPECIAL_CONSIDERATIONS' columns were dropped from the feature set. The 'EIN' column, which contains a unique identification number for each organization, was deemed irrelevant to the model since it does not provide any predictive power. The 'NAME' column, which contains the name of the organization, is also unlikely to be a significant predictor of the target variable. Finally, the 'SPECIAL_CONSIDERATIONS' column was removed because it only contains two unique values, which may not provide significant information to the model. Also, The feature set excluded the 'ASK_AMT' and 'STATUS' columns, as they may not significantly contribute to predicting the target variable. However, before removing these columns, exploratory data analysis should be conducted to investigate if there is any correlation between 'ASK_AMT' or "STATUS" and 'IS_SUCCESSFUL'. To guide the decision-making process, a condition such as <code>df['column'].corr(df['IS_SUCCESSFUL']) < 0.1</code> can be used. </li>
<li align="justify"><b>Optimization attempt:</b> The "NAME" column was added back to the feature set based on a particular condition. Although including the "NAME" column may introduce bias into the modeling process since it serves as an identification column, a criterion was established to mitigate this potential bias by grouping the names into just over 100 categories.</li>
</ul>
</li>
</ul>
<h5>2.1.2. binning</h5>
<ul align="justify">
<li>In an <b><ins>initial</ins></b> and <b><ins>pre-optimization</ins></b> stage, the 'APPLICATION_TYPE' and 'CLASSIFICATION' features had 17 and 71 distinct values, respectively, while the other features had less than 7 unique values. To simplify the data and facilitate analysis and modeling, a new category called 'other' was created for data values below 500 and 800 in the respective columns.</li>
<li>Furthermore, during the <b><ins>optimization</ins></b> attempt, a new category labeled as 'other' was introduced for data values below 100 in the 'NAME' column.</li>
</ul>

<h4>2.2. Compiling, Training, and Evaluating the Model</h4>
<ul align="justify">
<li align="justify"><b><ins>Initial attempt:</ins></b> This model comprises of two hidden layers, one with 80 neurons and the other with 30 neurons. The activation function used for the hidden layers was "relu", and "sigmoid" was used for the output layer. The analysis was conducted over 100 epochs. <i><ins>The initial model did not meet the desired performance target of 75%, with an accuracy of only 72.90% and a loss of 56.02%.</ins></i></li><br>
  
<li align="justify"><b><ins>Pre-optimization and Optimization attempt:</ins></b> In order to determine the optimal number of layers using the "wider and deeper" approach and identify the best number of epochs using the "callback stop" method, Keras Tuner, a hyperparameter tuning library, was utilized during the experiments. Based on the results, the optimal number of epochs was found to be 35, and the activation functions were set to ['relu', 'tanh', 'sigmoid']. The first layer was set between 1 and 320, while other layers were set between 1 and 120. The output layer employed the "sigmoid" activation function. Additionally, various feature sets were tested to assess their impact on the loss and accuracy of the model.
<ul>
<li><i><ins>The pre-optimization model achieved a maximum accuracy of 72.94% and a loss of 18.71%. This model was build using:</ins></i> <code>{'activation': 'relu', 'first_units': 301, 'num_layers': 5, 'units_0': 81, 'units_1': 36, 'units_2': 41, 'units_3': 11, 'units_4': 116, 'units_5': 76, 'units_6': 81, 'units_7': 46, 'tuner/epochs': 35, 'tuner/initial_epoch': 0, 'tuner/bracket': 0, 'tuner/round': 0}</code><br>it means that the optimization process focuses on reducing the model's loss without affecting its accuracy. In other words, the model is already performing well in terms of accuracy, but it can still be optimized to reduce the loss further.</li>



</li>
</ul>
</li>
 
</ul>

<hr>
<h3>2. Summary</h3>
