import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set the style for visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
# Set the figure size
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

# Load the dataset
air_data = pd.read_csv('air_quality_india.csv')


# Basic EDA
def basic_eda(data):
    print("Dataset Info:")
    print(data.info())
    print("\nDescriptive Stats:")
    print(data.describe())
    
    print("\nNull Values:")
    print(data.isnull().sum())

    print("\nTop Polluted Cities (Mean):")
    print(data.groupby("city")["pollutant_avg"].mean().sort_values(ascending=False).head(10))

    print("\nLeast Polluted Cities (Mean):")
    print(data.groupby("city")["pollutant_avg"].mean().sort_values().head(10))




# Data cleaning function
def clean_air_data(data):
    clean_data = data.copy()
    
    # Handle missing values for numeric columns
    numeric_cols = clean_data.select_dtypes(include=['float64', 'int64']).columns
    clean_data[numeric_cols] = clean_data[numeric_cols].fillna(clean_data[numeric_cols].mean())
    
    # Create or identify pollutant_type column
    if 'pollutant_type' not in clean_data.columns:
        # Look for columns that might contain pollutant type information
        potential_cols = ['pollutant_', 'pollutant', 'type', 'parameter']
        for col in clean_data.columns:
            if any(pc in col.lower() for pc in potential_cols):
                clean_data['pollutant_type'] = clean_data[col]
                break
        else:
            clean_data['pollutant_type'] = 'Unknown'
    
    # Create pollutant_avg column if it doesn't exist
    if 'pollutant_avg' not in clean_data.columns:
        # Find columns with numeric pollution values
        value_columns = [col for col in clean_data.columns 
                         if clean_data[col].dtype in ['int64', 'float64'] and 
                         col not in ['id', 'latitude', 'longitude']]
        
        clean_data['pollutant_avg'] = clean_data[value_columns].mean(axis=1)
    
    return clean_data

# Clean the data
clean_air_data = clean_air_data(air_data)
# 1. State-wise Average Pollution Level
def plot_state_pollution(data):
    plt.figure(figsize=(14, 8))
    
    state_averages = data.groupby('state')['pollutant_avg'].mean().sort_values(ascending=False)
    colors = sns.color_palette("YlOrRd", len(state_averages))
    
    ax = sns.barplot(x=state_averages.index, y=state_averages.values, hue=state_averages.index, palette=colors, legend=False)
    
    plt.title('Average Pollution Levels by State', fontsize=16, fontweight='bold')
    plt.xlabel('State', fontsize=14)
    plt.ylabel('Average Pollution Level', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for i, value in enumerate(state_averages.values):
        ax.text(i, value + 1, f'{value:.1f}', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# 2. Pollutant Type Distribution
def plot_pollutant_distribution(data):
    plt.figure(figsize=(12, 8))
    
    pollutant_counts = data['pollutant_type'].value_counts()
    
    plt.pie(pollutant_counts, labels=pollutant_counts.index, autopct='%1.1f%%', 
            startangle=90, shadow=True, explode=[0.05] * len(pollutant_counts),
            colors=sns.color_palette("Set3", len(pollutant_counts)))
    
    plt.title('Distribution of Pollutant Types', fontsize=16, fontweight='bold')
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()
    

# 3. Maximum Pollution by Pollutant Type
def plot_max_pollution_by_type(data):
    # Distribution boxplot
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='pollutant_type', y='pollutant_avg', data=data, hue='pollutant_type', palette="plasma", legend=False)

    
    plt.title('Distribution of Pollution Levels by Pollutant Type', fontsize=16, fontweight='bold')
    plt.xlabel('Pollutant Type', fontsize=14)
    plt.ylabel('Pollution Level', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

# 4. City Pollution Analysis
def plot_city_pollution(data):
    # Get top 15 cities by average pollution
    top_cities = data.groupby('city')['pollutant_avg'].mean().sort_values(ascending=False).head(15).index

    # Filter data for those cities
    top_data = data[data['city'].isin(top_cities)]

    # Pivot table: city vs pollutant_id (instead of pollutant_type)
    pollution_matrix = top_data.pivot_table(
        index='city',
        columns='pollutant_id',
        values='pollutant_avg',
        aggfunc='mean'
    ).fillna(0)

    # Sort cities by total pollution across pollutants
    pollution_matrix = pollution_matrix.loc[pollution_matrix.sum(axis=1).sort_values(ascending=False).index]

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(pollution_matrix, annot=True, cmap="YlOrRd", fmt=".1f", linewidths=0.5)

    plt.title('Pollution Heatmap - Top 15 Cities by Pollutant Type')
    plt.xlabel('Pollutant')
    plt.ylabel('City')
    plt.tight_layout()
    plt.show()
    


# 5. Top Polluted Monitoring Stations
def plot_polluted_stations(data):
    # Get top 20 stations by pollution level
    if 'city' in data.columns and 'state' in data.columns:
        top_stations = data.groupby(['station', 'city', 'state'])['pollutant_avg'].mean().sort_values(ascending=False).head(20)
        top_stations = top_stations.reset_index()
        
        # Create station labels with city and state
        station_labels = [f"{station} ({city}, {state})" for station, city, state in 
                         zip(top_stations['station'], top_stations['city'], top_stations['state'])]
    else:
        top_stations = data.groupby('station')['pollutant_avg'].mean().sort_values(ascending=False).head(20)
        top_stations = top_stations.reset_index()
        station_labels = top_stations['station']
    
    # Create colormap
    colors = sns.color_palette("YlOrRd", len(top_stations))
    
    # Create horizontal bar chart
    plt.figure(figsize=(14, 12))
    ax = sns.barplot(y=station_labels, x=top_stations['pollutant_avg'], hue=station_labels, palette=colors, orient='h', legend=False)

    
    plt.title('Top 20 Most Polluted Monitoring Stations', fontsize=16, fontweight='bold')
    plt.xlabel('Average Pollution Level', fontsize=14)
    plt.ylabel('Monitoring Station', fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add values at the end of bars
    for i, value in enumerate(top_stations['pollutant_avg']):
        ax.text(value + 1, i, f'{value:.1f}', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Create a scatter plot of stations if latitude/longitude exist
    if 'latitude' in data.columns and 'longitude' in data.columns:
        plt.figure(figsize=(14, 10))
        
        # Create a dataframe with station details
        group_cols = ['station']
        if 'city' in data.columns:
            group_cols.append('city')
        if 'state' in data.columns:
            group_cols.append('state')
            
        group_cols.extend(['latitude', 'longitude'])
        
        station_data = data.groupby(group_cols)['pollutant_avg'].mean().reset_index()
        
        # Create scatter plot
        scatter = plt.scatter(station_data['longitude'], station_data['latitude'], 
                             c=station_data['pollutant_avg'], cmap='YlOrRd', 
                             s=station_data['pollutant_avg']*3, alpha=0.7, edgecolors='black')
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Average Pollution Level', rotation=270, labelpad=20)
        
        # Add labels for top 10 most polluted stations
        top_10 = station_data.sort_values('pollutant_avg', ascending=False).head(10)
        for _, row in top_10.iterrows():
            plt.annotate(row['station'], 
                        (row['longitude'], row['latitude']),
                        xytext=(5, 5), textcoords='offset points',
                        bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.8),
                        fontsize=8)
        
        plt.title('Monitoring Stations by Location and Pollution Level', fontsize=16, fontweight='bold')
        plt.xlabel('Longitude', fontsize=14)
        plt.ylabel('Latitude', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()

def run_statistical_tests(data):
    print("\n🔬 Running Statistical Tests:")

    # 1. T-Test: Compare pollution between two states
    state1 = 'Delhi'
    state2 = 'Bihar'
    
    if state1 in data['state'].unique() and state2 in data['state'].unique():
        pollution_state1 = data[data['state'] == state1]['pollutant_avg']
        pollution_state2 = data[data['state'] == state2]['pollutant_avg']
        
        t_stat, p_val = stats.ttest_ind(pollution_state1, pollution_state2, equal_var=False)
        
        print(f"\nT-Test between {state1} and {state2}:")
        print(f"   T-statistic = {t_stat:.4f}")
        print(f"   P-value = {p_val:.4f}")
        if p_val < 0.05:
            print("   → Statistically significant difference in pollution levels.")
        else:
            print("   → No significant difference found.")
    else:
        print(f"One or both states not found in the dataset: {state1}, {state2}")
    
    # 2. Z-Test: Compare city mean to overall mean
    city = 'Patna'
    city_data = data[data['city'] == city]['pollutant_avg']
    overall_mean = data['pollutant_avg'].mean()
    overall_std = data['pollutant_avg'].std()
    sample_size = len(city_data)

    if sample_size > 0:
        z_score = (city_data.mean() - overall_mean) / (overall_std / np.sqrt(sample_size))
        p_val = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed
        
        print(f"\n📌 Z-Test for {city} vs overall pollution mean:")
        print(f"   Z-score = {z_score:.4f}")
        print(f"   P-value = {p_val:.4f}")
        if p_val < 0.05:
            print("   → Significant difference from the overall average.")
        else:
            print("   → No significant difference from the overall average.")
    else:
        print(f"   No data found for city: {city}")

# Run all visualizations
def run_all_visualizations():
    print("Generating visualizations for India's air quality data...")
    plot_state_pollution(clean_air_data)
    plot_pollutant_distribution(clean_air_data)
    plot_max_pollution_by_type(clean_air_data)
    plot_city_pollution(clean_air_data)
    plot_polluted_stations(clean_air_data)
    basic_eda(clean_air_data)
    print("All visualizations completed!")

# Run everything
run_all_visualizations()
run_statistical_tests(clean_air_data)
