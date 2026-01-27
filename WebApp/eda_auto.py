import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(csv_path, save_path):
    df = pd.read_csv(csv_path)
    df = df.drop(columns=['Unnamed: 0.1','Unnamed: 0'], errors='ignore')

    # 1. Accidents by Year
    plt.figure(figsize=(10,5))
    ax = sns.countplot(x="Year", data=df)
    ax.set_title("No. of Accidents (2016–2018)")
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x()+p.get_width()/2, p.get_height()),
                    ha='center', va='bottom')
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.figure(figsize=(10,5))

    plt.savefig(f"{save_path}/year.png")
    plt.close()

    # 2. Top States
    df['State'].value_counts()[:10].plot(kind='barh')
    plt.title("Top 10 States")
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.figure(figsize=(10,5))

    plt.savefig(f"{save_path}/state.png")
    plt.close()

    # 3. Top Cities
    df['City'].value_counts()[:10].plot(kind='barh')
    plt.title("Top 10 Cities")
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.figure(figsize=(10,5))

    plt.style.use("seaborn-v0_8-darkgrid")
    plt.figure(figsize=(10,5))

    plt.savefig(f"{save_path}/city.png")
    plt.close()

    # 4. Weather
    df['Weather_Condition'].value_counts()[:10].plot(kind='barh')
    plt.title("Top 10 Weather Conditions")
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.figure(figsize=(10,5))

    plt.savefig(f"{save_path}/weather.png")
    plt.close()

    # 5. Streets
    df['Street'].value_counts()[:20].plot(kind='barh')
    plt.title("Top 20 Streets")
    plt.savefig(f"{save_path}/street.png")
    plt.close()

    # 6. Severity Pie
    plt.figure(figsize=(6,6))
    df['Severity'].value_counts().plot.pie(autopct='%1.1f%%', shadow=True)
    plt.title("Severity Distribution")
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.figure(figsize=(10,5))

    plt.ylabel("")
    plt.savefig(f"{save_path}/severity.png")
    plt.close()

    # 7. Correlation Heatmap
    num_df = df.select_dtypes(include=np.number)
    corr = num_df.corr()
    plt.figure(figsize=(12,8))
    sns.heatmap(corr, annot=True, cmap='tab20c')
    plt.title("Correlation Heatmap")
    plt.savefig(f"{save_path}/corr.png")
    plt.close()

    # 8. Boxplot Severity vs Temperature
    sns.boxplot(data=df, x="Severity", y="Temperature(F)")
    plt.title("Severity vs Temperature")
    plt.savefig(f"{save_path}/box_temp.png")
    plt.close()
