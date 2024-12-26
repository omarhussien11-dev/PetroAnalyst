# libraries
import streamlit as st
import pandas as pd
import numpy as np
import lasio
import matplotlib.pyplot as plt
import seaborn as sns
import welly
from scipy.optimize import curve_fit
import statsmodels.api as sm
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingRegressor
from welly import Well
from welly import read_las
import pkg_resources
# Sidebar
st.sidebar.image("logo SPE.jpg")
st.sidebar.title("**⚒️Main Menu⚒️**")
section = st.sidebar.selectbox("**Choose:**",
    ["Home", "Well Logging","Core Analysis","Welly Multi Well",
     "Survey Data", "Decline Curve Analysis","Predection"])

# Home Section
if section == "Home":
    st.image("pixelcut-export.jpg")
    st.write("""<h1 style='text-align: center; color:rgb(46, 168, 255);
    '>Welcome to PetroAnalyst App</h1>""", unsafe_allow_html=True)
    st.write("## Contents of PetroAnalyst App")
    st.write("""- **⛏️ Well Logging**
- **📊 Core Analysis**
- **🔎 Survey Data**
- **📉 Decline Curve Analysis**
- **⚙️ Welly Multi Well Project**""")
    st.write("## About Me")
    st.write("""#### *****Name:** Omar Hussien Mahmoud***\n
 ***🔗FACEBOOK:** www.facebook.com/1omar.Hussein*\n
 ***🔗LINKEDIN:** www.linkedin.com/in/omar-hussien70*""")
    
# Well Logging Section
elif section == "Well Logging":
    st.header("Well Logging")
    st.sidebar.write("**Choose:**")
    dataPreview = st.sidebar.checkbox("Data Preview")
    customScatterplot = st.sidebar.checkbox("Custom Scatterplot")
    scatterThirdVariable = st.sidebar.checkbox("Scatterplot With Third Variable")
    subPlots = st.sidebar.checkbox("Subplots")
    ######################################################
    uploadedFile = st.file_uploader("Upload a LAS file:", type=["las"])
    if uploadedFile:
        @st.cache_data
        def loadLasData(uploadedFile):
            bytesData = uploadedFile.read()
            strIo = StringIO(bytesData.decode("Windows-1252"))
            lasFile = lasio.read(strIo)
            lasDf = lasFile.df()
            lasDf = lasDf.reset_index()
            return lasDf
        ######################################################    
        df = loadLasData(uploadedFile)
        if df is not None:
            df = df.rename(columns={'DEPT': 'Depth', 'DEN': 'Density', 'GR': 'Gamma Ray', 'NEU': 'Neutron Porosity'})
            if dataPreview:
                st.write("### Data Preview:")
                st.dataframe(df.head())
                st.write("### Data Describe:")
                st.write(df.describe())
            ######################################################
            if customScatterplot:
                st.subheader("Custom Scatterplot")
                xaxis = st.selectbox("Select X axis:", df.columns, index=df.columns.get_loc("Neutron Porosity"))
                yaxis = st.selectbox("Select Y axis:", df.columns, index=df.columns.get_loc("Density"))
                fig, ax = plt.subplots(figsize=(7, 4))
                sns.scatterplot(data=df, x=xaxis, y=yaxis, ax=ax, hue=yaxis, palette='Spectral', size=yaxis, sizes=(50, 150))
                ax.set_title(f"{xaxis} and {yaxis}")
                st.pyplot(fig)
            ######################################################
            if scatterThirdVariable:
                st.subheader("Scatterplot With Third Variable")
                fig, ax = plt.subplots(figsize=(7, 4))
                xaxis = st.selectbox("Select X axis:", df.columns, index=df.columns.get_loc('Neutron Porosity'))
                yaxis = st.selectbox("Select Y axis:", df.columns, index=df.columns.get_loc('Density'))
                zaxis = st.selectbox("Select Z axis:", df.columns, index=df.columns.get_loc('Gamma Ray'))
                scatter = ax.scatter(df[xaxis], df[yaxis], c=df[zaxis], cmap='viridis', edgecolor='k')
                ax.set_xlabel(f'{xaxis}')
                ax.set_ylabel(f'{yaxis}')
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(f'{zaxis}')
                st.pyplot(fig)
            ######################################################
            if subPlots:
                st.subheader("Subplots")
                plotlist = ['Gamma Ray', 'RDEP', ['Density', 'Neutron Porosity']]
                titles = ['Gamma Ray and Depth', 'Deep Resistivity and Depth']
                fig, axes = plt.subplots(1, 3, figsize=(15, 15))
                for i in range(len(plotlist)):
                    if plotlist[i] == plotlist[2]:
                        plt.subplot(1, 3, i + 1)
                        plt.plot(df[plotlist[i][0]], df['Depth'], label='Density')
                        plt.plot(df[plotlist[i][1]], df['Depth'], label='Neutron Porosity')
                        plt.title(f'{plotlist[i][0]} and {plotlist[i][1]} VS Depth')
                        plt.xlabel(f'{plotlist[i][0]} - {plotlist[i][1]}')
                    else:
                        plt.subplot(1, 3, i + 1)
                        plt.plot(df[plotlist[i]], df['Depth'])
                        plt.title(f"{titles[i]}")
                        plt.xlabel(f'{plotlist[i]}')
                        if plotlist[i] == 'Gamma Ray':
                            plt.ylabel('Depth', fontsize=12)
                st.pyplot(fig)

# Core Analysis Section
elif section == "Core Analysis":
    st.header("Core Analysis")
    st.sidebar.write("Choose:")
    dataPreview = st.sidebar.checkbox("Data Preview")
    subPlots = st.sidebar.checkbox("Subplots")
    scatterPlot = st.sidebar.checkbox("Scatterplot")
    interpretedPorosity = st.sidebar.checkbox("Interpreted Porosity")
    ######################################################
    uploadedFile = st.file_uploader("Upload a Core CSV file:", type=[".csv"])
    if uploadedFile:
        df = pd.read_csv(uploadedFile)
        if dataPreview:
            st.write("### Data Preview:")
            st.dataframe(df.head())
        ######################################################
        if subPlots:
            st.write('### Subplots')
            st.write('''* ##### Subplot 1: CPOR (Core Porosity) vs DEPTH
* ##### Subplot 2: CKHG (Core Permeability) vs DEPTH
* ##### Subplot 3: CPOR (Core Porosity) vs CKHG (Core Permeability)
* ##### Subplot 4: histogram of CPOR - Core Porosity
* ##### Subplot 5: histogram of CGD - Core Grain Density''')
            plotlistX = ['CPOR', 'CKHG', 'CPOR']
            plotlistY = ['DEPTH', 'DEPTH', 'CKHG']
            titles = ['CPOR vs DEPTH', 'CKHG vs DEPTH', 'CPOR vs CKHG']
            fig = plt.figure(figsize=(30, 17))
            for i in range(len(plotlistX)):
                plt.subplot(1, 5, i + 1)
                plt.plot(df[plotlistX[i]], df[plotlistY[i]])
                plt.title(f"{titles[i]}")
                plt.xlabel(f'{plotlistX[i]}')
                plt.ylabel(f'{plotlistY[i]}')
            plt.subplot(1, 5, 4)  
            plt.hist(df['CPOR'])
            plt.title("Core Porosity")
            plt.xlabel('CPOR')
            plt.subplot(1, 5, 5)  
            plt.hist(df['CGD'])
            plt.title("Core Grain Density")
            plt.xlabel('CGD')
            plt.show()
            st.pyplot(fig)
        ######################################################
        if scatterPlot:
            st.write("### Scatter Plot: Raw Core Porosity vs Depth")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=df['CPOR'], y=df['DEPTH'], hue=df['CPOR'], palette='Spectral', size=df['CPOR'], sizes=(50, 200), ax=ax)
            ax.set_title("Core Porosity vs Depth")
            ax.set_xlabel("Core Porosity (CPOR)")
            ax.set_ylabel("Depth")
            st.pyplot(fig)
        ######################################################
        if interpretedPorosity:
            st.write("### Interpreted Porosity")
            uploadedFile = st.file_uploader("Upload a LAS file for Interpreted Porosity:", type=["las"])
            ######################################################
            if uploadedFile:
                @st.cache_data
                def loadLasData(uploadedFile):
                    bytesData = uploadedFile.read()
                    strIo = StringIO(bytesData.decode("Windows-1252"))
                    lasFile = lasio.read(strIo)
                    lasDf = lasFile.df()
                    lasDf = lasDf.reset_index()
                    return lasDf
                lasDf = loadLasData(uploadedFile)
                fig, ax = plt.subplots(figsize=(20, 4))
                ax.plot(lasDf['DEPTH'], lasDf['PHIF'], label='Interpreted Porosity (PHIF)', color='blue', linewidth=2)
                ax.set_title("Interpreted Porosity (PHIF)", fontsize=16)
                ax.set_xlabel("Depth", fontsize=14)
                ax.set_ylabel("PHIF", fontsize=14)
                ax.grid(True)
                ax.legend()
                st.pyplot(fig)
                
# Welly Multi Well Section
elif section == "Welly Multi Well":
    st.header("Welly Multi Well")
    dataPreview = st.sidebar.checkbox("Preview Data")
    plotGR = st.sidebar.checkbox("Plot GR and Depth")
    plotRHOB = st.sidebar.checkbox("Plot RHOB and Depth")
    uploadedFiles = st.file_uploader("Upload 4 LAS files:", type=["las"], accept_multiple_files=True)
    if uploadedFiles:
        dataFrames = []
        for uploadedFile in uploadedFiles:
            bytesData = uploadedFile.read()
            strIo = StringIO(bytesData.decode("Windows-1252"))
            lasFile = lasio.read(strIo)
            lasDf = lasFile.df()
            lasDf = lasDf.reset_index()
            dataFrames.append(lasDf)
        ######################################################
        if dataPreview:
            st.write("## Data Preview:")
            for i, df in enumerate(dataFrames):
                st.write(f"### Well {i+1}:")
                st.dataframe(df.dropna().head())
                st.dataframe(df.dropna().describe().T)
        ######################################################
        titlesX= ['L05-B-01','L06-06','L06-07','L07-01']
        if plotGR:
            st.write("## Plot GR and Depth:")
            fig, axes = plt.subplots(1, len(dataFrames), figsize=(15, 15), sharey=True)
            for i, df in enumerate(dataFrames):
                ax = axes[i] if len(dataFrames) > 1 else axes
                ax.plot(df["GR"], df["DEPT"], label="GR")
                ax.set_title(f"{titlesX[i]}")
                ax.set_xlabel("GR")
                if i<1:   
                  ax.set_ylabel("Depth")
                ax.grid()
            st.pyplot(fig)
        ######################################################
        if plotRHOB:
            st.write("## Plot RHOB and Depth:")
            fig, axes = plt.subplots(1, len(dataFrames), figsize=(15, 15), sharey=True)
            for i, df in enumerate(dataFrames):
                if len(dataFrames) > 1:
                    ax = axes[i]
                df= df.fillna(0)    
                ax.plot(df["RHOB"], df["DEPT"], label="RHOB")
                ax.set_title(f"{titlesX[i]}")
                ax.set_xlabel("RHOB")
                if i<1:   
                  ax.set_ylabel("Depth")
                ax.grid()
            st.pyplot(fig)
 
# Survey Data Section
elif section == "Survey Data":
    st.header("Survey Data")
    st.sidebar.write("Choose:")
    dataPreview = st.sidebar.checkbox("Data Preview")
    locationPlots = st.sidebar.checkbox("Location Plots")
    wellPath3D = st.sidebar.checkbox("3D Well Path")
    curvePlots = st.sidebar.checkbox("Curve Plots")
    ######################################################
    uploadedFile = st.file_uploader("Upload a CSV File:", type=[".csv"])
    uploadedFilelas = st.file_uploader("Upload a LAS File:", type=["las"])
    if uploadedFilelas:
        @st.cache_data
        def loadLasData(uploadedFilelas):
            bytesData = uploadedFilelas.read()
            strIo = StringIO(bytesData.decode("Windows-1252"))
            return lasio.read(strIo)
        lasData = loadLasData(uploadedFilelas)
    ######################################################
    if uploadedFile:
        df = pd.read_csv(uploadedFile)
        if dataPreview:
            st.write("### Data Preview")
            st.dataframe(df.head())
            st.dataframe(df.describe().T)
            svdata = df[["MD", "INC", "AZI"]]
        ######################################################
        if locationPlots:
                svdata = df[["MD", "INC", "AZI"]]
                well = welly.Well.from_lasio(lasData)
                well.location.add_deviation(svdata.values)
                position = pd.DataFrame(well.location.position, columns=["X", "Y", "Z"])
                st.write("### Location Plots")
                plt.figure(figsize=(15, 4))
                plt.subplot(1, 3, 1)
                plt.title('X and Y')
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.plot(position['X'], position['Y'])
                plt.subplot(1, 3, 2)
                plt.title('X and TVD')
                plt.xlabel('X')
                plt.ylabel('TVD')
                plt.plot(position['X'], position['Z'])
                plt.subplot(1, 3, 3)
                plt.title('Y and TVD')
                plt.xlabel('Y')
                plt.ylabel('TVD')
                plt.plot(position['Y'], position['Z'])
                st.pyplot(plt)
                ######################################################
                def marker(x, y):
                    plt.scatter([position[x].iloc[0]], [position[y].iloc[0]], color='goldenrod', linewidth=6)
                    plt.scatter([position[x].iloc[-1]], [position[y].iloc[-1]], color="indigo", linewidth=6)
                fig = plt.figure(figsize=(15, 4))
                plt.subplot(1, 3, 1)
                plt.title('X and Y')
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.plot(position['X'], position['Y'])
                marker('X', 'Y')
                plt.subplot(1, 3, 2)
                plt.title('X and TVD')
                plt.xlabel('X')
                plt.ylabel('TVD')
                plt.plot(position['X'], position['Z'])
                plt.gca().invert_yaxis()
                marker('X', 'Z')
                plt.subplot(1, 3, 3)
                plt.title('Y and TVD')
                plt.xlabel('Y')
                plt.ylabel('TVD')
                plt.plot(position['Y'], position['Z'])
                plt.gca().invert_yaxis()
                marker('Y', 'Z')
                st.pyplot(fig)
        ######################################################
        if wellPath3D:
            svdata = df[["MD", "INC", "AZI"]]
            well = welly.Well.from_lasio(lasData)
            well.location.add_deviation(svdata.values)
            trajectory = well.location.trajectory(datum=[589075.56, 5963534.91, 0], elev=False)
            st.write("### 3D Well Path Plot")
            fig = plt.figure(figsize=(7, 7))
            ax = plt.axes(projection='3d')
            ax.plot3D(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color='red', linewidth=2)
            ax.set_title("3D Well Path")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("TVD")
            st.pyplot(fig)
    ######################################################
        if curvePlots:
            well = welly.Well.from_lasio(lasData)
            fig = well.plot(extents='curves')
            st.pyplot(fig)
# Decline Curve Analysis Section
elif section == "Decline Curve Analysis":
    st.title("Decline Curve Analysis")
    st.sidebar.write("Choose:")
    preview_data = st.sidebar.checkbox("Preview Data")
    exponential_model = st.sidebar.checkbox("Exponential Model")
    harmonic_model = st.sidebar.checkbox("Harmonic Model")
    hyperbolic_model = st.sidebar.checkbox("Hyperbolic Model")
    ultimate_recovery = st.sidebar.checkbox("Ultimate Recovery")
    uploaded_file = st.file_uploader("Upload a XLSX File:", type=[".xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        df = df[df['WELL_BORE_CODE'] == 'NO 15/9-F-14 H']
        df = df.drop('BORE_WI_VOL', axis=1)
        df = df.dropna(subset=['AVG_ANNULUS_PRESS'])
        df = df.fillna(df['AVG_CHOKE_SIZE_P'].median())
        ######################################################
        lowess = sm.nonparametric.lowess(df['BORE_OIL_VOL'], df['DATEPRD'].astype('int64'), frac=0.05)
        lowsX, lowsY = lowess.T
        lowsX = pd.to_datetime(lowsX)
        df['BOV Smoothed'] = lowess[:, 1]
        df['day'] = range(0, len(df))
        if preview_data:
            st.subheader("Data Info")
            st.write(df)
            st.write(df.describe().T)
            fig, ax = plt.subplots()
            ax.plot(df['DATEPRD'], df['BORE_OIL_VOL'], label="Raw Data")
            ax.set_xlabel("Date")
            ax.set_ylabel("BORE OIL VOL")
            ax.legend()
            st.pyplot(fig)
            fig, ax = plt.subplots()
            ax.plot(lowsX, lowsY, label="Lowess Smoothed", color="orange")
            ax.set_xlabel("Date")
            ax.set_ylabel("BORE OIL VOL")
            ax.legend()
            st.pyplot(fig)
        ######################################################
        if exponential_model:
            x_data = df['day'].to_numpy()
            y_data = df['BOV Smoothed'].to_numpy()
            def model_f(x, a, b, c):
                return a * (x - b)**2 + c
            popt, pcov = curve_fit(model_f, x_data, y_data, p0=[3, 2, -16], maxfev=2000)
            a_opt, b_opt, c_opt = popt
            x_model = np.linspace(min(x_data), max(x_data), 500)
            y_model = model_f(x_model, a_opt, b_opt, c_opt)
            fig, ax = plt.subplots()
            ax.scatter(x_data, y_data, label="Data")
            ax.plot(x_model, y_model, label="Exponential Fit", color="red")
            ax.set_xlabel("Time (days)")
            ax.set_ylabel("BORE OIL VOL")
            ax.set_title("Exponential Model")
            ax.legend()
            st.pyplot(fig)
            st.write("### Exponential Model Parameters")
            st.write(f"a: {a_opt}")
            st.write(f"b: {b_opt}")
            st.write(f"c: {c_opt}")
        ######################################################
        if harmonic_model:
            x_data = df['day'].to_numpy()
            y_data = df['BOV Smoothed'].to_numpy()
            def harmonic_decline(x, qi, Di):
                return qi / (1 + Di * x)
            popt_harmonic, pcov_harmonic = curve_fit(harmonic_decline, x_data, y_data, p0=[3000, 0.001])
            qi_harm, Di_harm = popt_harmonic
            x_model = np.linspace(min(x_data), max(x_data), 500)
            y_harmonic = harmonic_decline(x_model, qi_harm, Di_harm)
            fig, ax = plt.subplots()
            ax.scatter(x_data, y_data, label="Data")
            ax.plot(x_model, y_harmonic, label="Harmonic Fit", color="red")
            ax.set_xlabel("Time (days)")
            ax.set_ylabel("BORE OIL VOL")
            ax.set_title("Harmonic Model")
            ax.legend()
            st.pyplot(fig)
            st.write("### Harmonic Model Parameters")
            st.write(f"qi: {qi_harm}")
            st.write(f"Di: {Di_harm}")
        ######################################################
        if hyperbolic_model:
            x_data = df['day'].to_numpy()
            y_data = df['BOV Smoothed'].to_numpy()
            def hyperbolic_decline(x, qi, Di, b):
                return qi / (1 + b * Di * x)**(1/b)
            popt_hyperbolic, pcov_hyperbolic = curve_fit(hyperbolic_decline, x_data, y_data, p0=[3000, 0.001, 0.5])
            qi_opt, Di_opt, b_opt = popt_hyperbolic
            x_model = np.linspace(min(x_data), max(x_data), 500)
            y_hyperbolic = hyperbolic_decline(x_model, qi_opt, Di_opt, b_opt)
            fig, ax = plt.subplots()
            ax.scatter(x_data, y_data, label="Data")
            ax.plot(x_model, y_hyperbolic, label="Hyperbolic Fit", color="red")
            ax.set_xlabel("Time (days)")
            ax.set_ylabel("BORE OIL VOL")
            ax.set_title("Hyperbolic Model")
            ax.legend()
            st.pyplot(fig)
            st.write("### Hyperbolic Model Parameters")
            st.write(f"qi: {qi_opt}")
            st.write(f"Di: {Di_opt}")
        ######################################################
        if ultimate_recovery:
            st.subheader("Ultimate Recovery")
            df = df[df['BORE_GAS_VOL'] > 0]
            df = df[df['DATEPRD'] <= '2010-12-31']
            lowess = sm.nonparametric.lowess(df['BORE_GAS_VOL'], df['DATEPRD'].astype('int64'), frac=0.5)
            lowsX, lowsY = lowess.T
            lowsX = pd.to_datetime(lowsX)
            df['BGV Smoothed'] = lowess[:, 1]
            x_data = df['day'].to_numpy()
            y_data = df['BGV Smoothed'].to_numpy()
            def hyperbolic_decline(x, qi, Di, b):
                return qi / (1 + b * Di * x)**(1/b)
            popt, pcov = curve_fit(hyperbolic_decline, x_data, y_data, p0=[3000, 0.001, 0.5], maxfev=3000)
            qi_opt, Di_opt, b_opt = popt
            x_model = np.linspace(min(x_data), max(x_data), 500)
            y_model = hyperbolic_decline(x_model, qi_opt, Di_opt, b_opt)
            fig, ax = plt.subplots()
            ax.scatter(x_data, y_data, label="Data")
            ax.plot(x_model, y_model, color="red", label="Hyperbolic Fit")
            ax.set_xlabel("Time (days)")
            ax.set_ylabel("BORE GAS VOL")
            ax.set_title("Hyperbolic Model for Gas Production")
            ax.legend()
            st.pyplot(fig)
            st.write("### Hyperbolic Model Parameters for Gas Production")
            st.write(f"qi: {qi_opt}")
            st.write(f"Di: {Di_opt}")
            
# Predect Section
elif section == "Predection":
    st.header("Prediction Section")
    st.sidebar.write("Options:")
    dataPreview = st.sidebar.checkbox("Show Data Preview")
    trainModel = st.sidebar.checkbox("Train Model and Predict")
    uploadedFile = st.file_uploader("Upload the Excel file:", type=["xlsx"])
    #########################################################
    if uploadedFile is not None:
        df = pd.read_excel(uploadedFile)
        dfg = df
        df = df[df['WELL_BORE_CODE'] == 'NO 15/9-F-14 H']
        df = df.drop('BORE_WI_VOL', axis=1)
        df = df.dropna(subset=['AVG_ANNULUS_PRESS'])
        df = df.fillna(df['AVG_CHOKE_SIZE_P'].median())
        df['day'] = range(0, len(df))
        df = df[df['BORE_GAS_VOL'] > 0]
        df = df[['BORE_GAS_VOL', 'day']]
        df = df.iloc[1:].reset_index(drop=True)
        if dataPreview:
            st.write('### Original Data')
            st.dataframe(dfg.head())
            st.dataframe(dfg.describe())
            st.write("### Data For Predection")
            st.dataframe(df.head())
            st.dataframe(df.describe().T)
        #########################################################
        if trainModel:
            X = df[['BORE_GAS_VOL']]
            y = df['day']
            X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = HistGradientBoostingRegressor(random_state=42)
            model.fit(X_Train, y_Train)
            y_Pred = model.predict(X_Test)
            st.write("### Curve Plot")
            plt.figure(figsize=(10, 6))
            plt.scatter(y_Test, X_Test['BORE_GAS_VOL'], label='Actual', color='blue')
            plt.scatter(y_Pred, X_Test['BORE_GAS_VOL'], label='Predicted', color='red')
            plt.title('Production vs Time', fontsize=14)
            plt.xlabel('Time (Days)', fontsize=12)
            plt.ylabel('Production (BORE_GAS_VOL)', fontsize=12)
            plt.legend()
            st.pyplot(plt)
            #########################################################
            GASVOLTarget = st.number_input("**Insert the value of Gas Production rate:**", value=16500, step=100)
            GASVOLTargetDf = pd.DataFrame({'BORE_GAS_VOL': [GASVOLTarget]})
            predictedTime = model.predict(GASVOLTargetDf)[0]
            cumulativeProduction = df.loc[df['day'] <= predictedTime, 'BORE_GAS_VOL'].sum()
            st.write(f"### Predicted Time to reach {GASVOLTarget} rate: {predictedTime} days")
            st.write(f"### Cumulative Production at that time: {cumulativeProduction}")

                    