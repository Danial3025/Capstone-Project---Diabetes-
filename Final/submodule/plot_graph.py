import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib 

# Load the dataset
df = pd.read_csv(r"C:\Users\Acer\Desktop\Data-Science\capstone\application\data\diabetes.csv")

def plot_graph(data):
    # Heatmap of correlations
    fig1, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(data.corr(), annot=True, ax=ax)

    # Pairplot of selected features
    fig2 = sns.pairplot(data, vars=['Pregnancies', 'Glucose', 'BloodPressure', 
                                      'SkinThickness', 'Insulin', 'BMI', 
                                      'DiabetesPedigreeFunction', 'Age'], 
                        hue='Outcome')

    # Histogram of Glucose levels by Outcome using Plotly
    fig3 = go.Figure()
    
    fig3.add_trace(go.Histogram(
        x=data[data['Outcome'] == 0]['Glucose'],
        opacity=0.5,
        name='Outcome 0',
        hoverinfo='x+y',
        marker=dict(color='blue'),
        xbins=dict(start=0, end=200, size=10)
    ))

    fig3.add_trace(go.Histogram(
        x=data[data['Outcome'] == 1]['Glucose'],
        opacity=0.5,
        name='Outcome 1',
        hoverinfo='x+y',
        marker=dict(color='green'),
        xbins=dict(start=0, end=200, size=10)
    ))

    # Update layout
    fig3.update_layout(
        title='Distribution of Glucose by Outcome',
        xaxis_title='Glucose',
        yaxis_title='Frequency',
        barmode='overlay'
    )

    # Return figures
    return fig1, fig2, fig3


# Save the plotting function as a .pkl file
joblib.dump(plot_graph, "plot_graph.pkl")