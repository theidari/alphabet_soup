# --------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------ All libraries, variables and functions are defined in this fil ----------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------

# main dependencies and setup
from sklearn.model_selection import train_test_split # to get training and test sets
from sklearn.preprocessing import StandardScaler # to removes the mean and scales each feature/variable to unit variance

import pandas as pd
import tensorflow as tf
from tensorflow import keras

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
                            line=dict(color=TWOSET[i%len(TWOSET)]),                          )
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
	
# --------------------------------------------------------------------------------------------------------------------------------------------
	
# km function and scatter plot
def scatter_cluster(n, df, columns):
    km = KMeans(n_clusters = n, n_init = 25, random_state = 1234)
    km.fit(df)
    cluster_centers = pd.DataFrame(km.cluster_centers_, columns=df.columns)
    # Create the trace for the data points
    trace_points = go.Scatter(
        x=df[columns[0]],
        y=df[columns[1]],
        mode='markers',
        name='Coins',
        marker=dict(
            size=7.5,
            color=km.labels_,
            colorscale=SEVENSET,
            opacity=0.9,
            line=dict(
                width=1,
                color='black'
            )
        ),
        text=df.index  # Set the hover text to the index value
    )

    # Create the trace for the centroid points
    trace_centroids = go.Scatter(
        x=cluster_centers[columns[0]],
        y=cluster_centers[columns[1]],
        mode='markers',
        name='Cluster Centers',
        marker=dict(
            size=30,
            color=cluster_centers.index,
            colorscale=SEVENSET,
            symbol='circle',
            opacity=0.3,
            line=dict(
                width=1,
                color='black'
            )
        ),
        text=[f"Centroid {i}" for i in range(len(cluster_centers))]  # Set the hover text to "Centroid {i}"
    )

    # Define the layout of the plot
    layout = go.Layout(
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor= '#f7f7f7',
            font=dict(color='black', size=14)

    ),
        width=700,
        height=700,
        title=dict(text="clustering with number of clusters "+str(n),
                  font=dict(size= 20, color= 'black'),
                  x=0.5,
                  y=0.91),
        xaxis=dict(title='Price Change Percentage 24h',
                  showline=True,
            linewidth=0.5,
            linecolor='black',
            mirror=True,
                  color= 'black',
                   gridcolor='white'),
        yaxis=dict(title='Price Change Percentage 7d',
                   showline=True,
                   linewidth=0.5,
                   linecolor='black',
                   mirror=True,
                   color= 'black',
                   gridcolor='white'),
        hovermode='closest',
        plot_bgcolor='#ffffff',
        paper_bgcolor="#f7f7f7"
    )

    # Create the figure object and add the traces to it
    fig = go.Figure(data=[trace_points, trace_centroids], layout=layout)

    # Show the figure
    fig.show()
    
print(f"☑ helpers is imporetd")
