import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('G:\My Drive\python\Med_Data_Visualizer\medical_examination.csv')

# Add 'overweight' column
df['overweight'] = ((df['weight'] / ((df['height'] / 100) ** 2)) > 25).astype(int)

# Normalize data by making 0 always good and 1 always bad.
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

df_cleaned = df[
    (df['ap_lo'] <= df['ap_hi']) & 
    (df['height'] >= df['height'].quantile(0.025)) &
    (df['height'] <= df['height'].quantile(0.975)) &
    (df['weight'] >= df['weight'].quantile(0.025)) &
    (df['weight'] <= df['weight'].quantile(0.975))
].copy()

# Draw Categorical Plot
def draw_cat_plot():
    # Ensure the order of variables matches the expected test case order
    df_cat = pd.melt(frame=df, id_vars=['cardio'], value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])
    fig = sns.catplot(x="variable", hue="value", col="cardio", data=df_cat, kind="count", order=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])
    fig.set_axis_labels("variable", "total")
    fig.savefig('catplot.png')
    return fig.fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(11, 9))

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", cmap='coolwarm', cbar_kws={"shrink": .5})
    
    # Save the plot
    plt.savefig('heatmap.png')
    return fig


# Uncomment below lines to execute the functions and generate plots
draw_cat_plot()
draw_heat_map()
