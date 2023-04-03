# --------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------ All libraries, variables and functions are defined in this fil ----------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------

# main dependencies and setup
from sklearn.model_selection import train_test_split # to get training and test sets
from sklearn.preprocessing import StandardScaler # to removes the mean and scales each feature/variable to unit variance

import pandas as pd
import tensorflow as tf

import plotly.graph_objects as go # plotting
from plotly.subplots import make_subplots # subplotting

# package dependencies and setup
from alphabet_soup.src.package.constants import * # Constants

# --------------------------------------------------------------------------------------------------------------------------------------------
# functions

# Plotting function
def line (df, chart_title):
    # Create a list of traces for each column in the DataFrame
    traces = []
    for i, col in enumerate(df.columns):
        col_name = col.split("_")
        trace = go.Scatter(x=df.index,
                           y=df[col],
                           name=col_name[-1],
                           mode='lines',
                           line=dict(color=SEVENSET[i%len(SEVENSET)]),
                          )
        traces.append(trace)
        
    # Create the layout
    layout = go.Layout(title=dict(text=chart_title,
                                  font=dict(size= 24, color= 'black', family= "Times New Roman"),
                                  x=0.5,
                                  y=0.9),
                       width=1200,
                       height=600,
                       legend=dict(
                           yanchor="top",
                           y=0.99,
                           xanchor="left",
                           x=0.01,
                           bgcolor= '#f7f7f7',
                           font=dict(color='black')),
                       xaxis=dict(title='Crypto',
                                  color= 'black',
                                  showline=True,
                                  linewidth=1,
                                  linecolor='black',
                                  mirror=True), 
                       yaxis=dict(title='Price Change (%)',
                                  color= 'black',
                                  showline=True,
                                  linewidth=1,
                                  linecolor='black',
                                  mirror=True),
                       plot_bgcolor='#f7f7f7',
                       paper_bgcolor="#f7f7f7")

    # Create the figure
    fig = go.Figure(data=traces, layout=layout)
    # Show the figure
    fig.show()

def histogram (df, bins, location):
    # Set the figure size
    plt.figure(figsize=(FIG_HEIGHT, FIG_WIDTH))

    #Plot the Clusters
    ax = sns.scatterplot(data = df_market_scaled,
                         x = 'price_change_percentage_24h',
                         y = 'price_change_percentage_7d', 
                         hue = km.labels_, 
                         palette = 'colorblind', 
                         alpha = 0.8, 
                         s = 150,
                         legend = False)

    #Plot the Centroids
    ax = sns.scatterplot(data = cluster_centers, 
                         x = 'price_change_percentage_24h',
                         y = 'price_change_percentage_7d', 
                         hue = cluster_centers.index, 
                         palette = 'colorblind', 
                         s = 600,
                         marker = 'D',
                         ec = 'black', 
                         legend = False)

    # Add Centroid Labels
    for i in range(len(cluster_centers)):
                   plt.text(x = cluster_centers.price_change_percentage_24h[i], 
                            y = cluster_centers.price_change_percentage_7d[i],
                            s = i, 
                            horizontalalignment='center',
                            verticalalignment='center',
                            size = 15,
                            weight = 'bold',
                            color = 'white')
            
def score_plot(methods):
    words = [s.name for s in methods]
    subplots_data = [] 
    for i, data in enumerate(methods):
        line_trace = go.Scatter(x=data.index, y=data, mode='lines',
                                line=dict(color=SEVENSET[i%len(SEVENSET)]))
        marker_trace = go.Scatter(x=data.index, y=data, mode='markers', marker=dict(size=10,color=SEVENSET[i%len(SEVENSET)]),
                                  hovertemplate=" number of clusters (k): <b>%{x}</b><br>"+"score: <b>%{y}</b><br>"+"<extra></extra>")
        plot_trace = [line_trace, marker_trace]
        subplots_data.append(plot_trace)


    layout = go.Layout(width=1300,
        height=500,
        plot_bgcolor='#f7f7f7',
        paper_bgcolor="#f7f7f7")

    fig = make_subplots(rows=1, cols=len(methods), horizontal_spacing=0.1, shared_xaxes=True)

    for i, data in enumerate(subplots_data):
        for trace in data:
            fig.add_trace(trace, row=1, col=i+1)
            fig.update_xaxes(tickfont=dict(size= 14, family='calibri', color='black' ),
                             showline=True, linewidth=0.5, linecolor='black', mirror=True, row=1, col=i+1)
            fig.update_yaxes(title=dict(text=words[i].capitalize()+" Score",
                                        font=dict(size= 18, color= 'black', family= "Calibri")),
                             tickfont=dict(size= 14, family='calibri', color='black' ),
                             showline=True, linewidth=0.5, linecolor='black', mirror=True, row=1, col=i+1)

    fig.update_xaxes(title=dict(text="Number of Clusters (k)",
                                font=dict(size= 18, color= 'black', family= "Calibri")), row=1, col=2)

    # Update the layout of the figure
    fig.update_layout(layout, showlegend=False) 

    fig.show()        
# -------------------------------------------------------------------------------------------------------/
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
