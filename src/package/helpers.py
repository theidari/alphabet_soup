# --------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------ All libraries, variables and functions are defined in this fil ----------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------

# main dependencies and setup
from sklearn.model_selection import train_test_split # to get training and test sets
from sklearn.preprocessing import StandardScaler # to removes the mean and scales each feature/variable to unit variance

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

import plotly.graph_objects as go # plotting
from plotly.subplots import make_subplots # subplotting

# package dependencies and setup
from alphabet_soup.src.package.constants import * # Constants

# --------------------------------------------------------------------------------------------------------------------------------------------
# functions
# calculation and cleaning ___________________________________________________________________________________________________________________
def binning(df, param, value):
    # Look at parameter value counts for binning
    before_counts = df[param].value_counts()
    
    # Choose a cutoff value and create a list of parameter to be replaced
    types_to_replace = list(before_counts[before_counts < value].index)
    
    # Replace in dataframe
    for data in types_to_replace:
        df[param] = df[param].replace(data,"Other")
        
    # Check to make sure binning was successful
    after_counts = df[param].value_counts()
    return print(f"{H_LINE} Value Count before binning:{H_LINE}{before_counts}{H_LINE}Value Count after binning:{H_LINE}{after_counts}")

# Compile, Train and Evaluate the Model ______________________________________________________________________________________________________
def as_model_func(layers, input_features, act_func):
  as_model = tf.keras.models.Sequential()

  for i, layer in enumerate(layers):
    # first hidden layer
    if i==0:
      as_model.add(tf.keras.layers.Dense(units=layers[0],
                                         input_dim=input_features, activation=act_func))

    # other hidden layer
    else:
      as_model.add(tf.keras.layers.Dense(units=layers[i], activation=act_func))

  # Output layer
  as_model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
  print(as_model.summary())

  # Compile the model
  as_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

  return as_model

# early stop and epoch calculation ___________________________________________________________________________________________________________
def epoch_func(layers, act_func, batches, epochs_est, features=[input_features, X_train_scaled, y_train, X_test_scaled, y_test]):
	early_stopping = callbacks.EarlyStopping(
	    min_delta=0.001, # minimium amount of change to count as an improvement
	    patience=20, # how many epochs to wait before stopping
	    restore_best_weights=True)
	
	model = tf.keras.models.Sequential()
	for i, layer in enumerate(layers):
		# first hidden layer
		if i==0:
			model.add(tf.keras.layers.Dense(units=layers[0],
                                         input_dim=input_features, activation=act_func))
		# other hidden layer
		else:
			model.add(tf.keras.layers.Dense(units=layers[i], activation=act_func))
			
	# Output layer
	model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
	print(model.summary())
	
	# Compile the model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	history = model.fit(
	    X_train_scaled, y_train,
	    validation_data=(X_test_scaled, y_test),
	    batch_size=batches,
	    epochs=epochs_est,
	    callbacks=[early_stopping], # put your callbacks in a list
	    verbose=0,  # turn off training log)
		
	history_df = pd.DataFrame(history.history)
	history_df=history_df[['loss', 'val_loss']]
	# plotting
	accuracy_loss (history_df)
	print("Minimum validation loss: {}".format(history_df['val_loss'].min()))
		
	# Evaluate the model using the test data
	model_loss, model_accuracy = model.evaluate(X_test_scaled,y_test,verbose=2)
	print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

# building model _____________________________________________________________________________________________________________________________
def build_model(hp):
    input_features = get_input_features()	
    nn_model = tf.keras.models.Sequential()

    # Allow keras tuner to decide which activation function to use in hidden layers
    activation = hp.Choice('activation',['relu','sigmoid'])
    
    # Allow kerastuner to decide number of neurons in first layer
    nn_model.add(tf.keras.layers.Dense(units=hp.Int('first_units', min_value=1, max_value= 80, step=5),
                                       activation=activation, input_dim=input_features))

    # Allow keras tuner to decide number of hidden layers and neurons in hidden layers
    for i in range(hp.Int('num_layers', 1, 4)):
        nn_model.add(tf.keras.layers.Dense(units=hp.Int('units ' + str(i),
            min_value=1,
            max_value=30,
            step=5),
            activation=activation))
			
    # Output layer
    nn_model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

    # Compile the model
    nn_model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])
    
    return nn_model


# Plotting function
# accuracy vs. loss plot _____________________________________________________________________________________________________________________
def accuracy_loss (df):
    # Create a list of traces for each column in the DataFrame
    traces = []
    for i, col in enumerate(df.columns):
        trace = go.Scatter(x=df.index,
                           y=df[col],
                           name=col.capitalize(),
                           mode='lines',
                            line=dict(color=TWOSET[i%len(TWOSET)]))
        traces.append(trace)
        
    # Create the layout
    layout = go.Layout(title=dict(text="Model Accuracy And Loss",
                                  font=dict(size= 24, color= 'black', family= "Times New Roman"),
                                  x=0.5,
                                  y=0.9),
                       width=600,
                       height=600,
                       legend=dict(yanchor="middle",
                           y=0.5,
                           xanchor="right",
                           x=0.99,
                           bgcolor= '#f7f7f7',
                           font=dict(color='black')),
                       xaxis=dict(title='Epoch',
                                  color= 'black',
                                  showline=True,
                                  linewidth=1,
                                  linecolor='black',
                                  mirror=True), 
                       yaxis=dict(title='Amount',
                                  color= 'black',
                                  showline=True,
                                  linewidth=1,
                                  linecolor='black',
                                  mirror=True),
                       plot_bgcolor='#f7f7f7',
                       paper_bgcolor="#ffffff")

    # Create the figure
    fig = go.Figure(data=traces, layout=layout)
    # Show the figure
    fig.show()
	
print(f"☑ helpers is imporetd")
# --------------------------------------------------------------------------------------------------------------------------------------------
