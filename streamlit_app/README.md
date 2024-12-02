# AP Outcomes vs. University: Streamlit App

We summarize the repository of code that generates the Streamlit app.

View the app here: [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ap-outcomes.streamlit.app/)

## How to run it on your own machine

1. Install the requirements

   ```
   $ conda install -c conda-forge pandas streamlit geopandas joblib scikit-learn streamlit-folium
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app/app.py
   ```

3. Editing with Live Preview

   In VSCode, type `Cmd + Shift + P`, then type in and select `Simple Browser: Show`, followed by entering the network URL. That URL will be something like `Network URL: http://10.184.235.232:8501`, which you can confirm in the terminal once you've run the `streamlit` command.

4. Stopping the server

   Simply press `Ctrl + C` in the terminal to stop locally serving the Streamlit app. 


## Directory Structure

```
streamlit_app/
├── ...    
├── data/               Dataframes used in the app
├── notebooks/          Notebooks that generate plots
├── app.py              Main Streamlit application
├── README.md        
└── requirements.txt    Python dependencies
```