AQI Index Predictor
Welcome to the AQI Index Predictor, a Streamlit-based web application that uses Linear Regression to predict Air Quality Index (AQI) based on pollutant data. This tool is designed to provide insights into air quality trends and enable interactive predictions, visualizations, and detailed reports. 🌱
Overview
The AQI Index Predictor leverages machine learning to analyze air quality data, including pollutants like PM2.5, PM10, NO2, and O3, along with city-specific factors. It offers:

Data Upload: Load your own CSV dataset (e.g., city_day.csv).
Feature Analysis: Identify key pollutant correlations with AQI.
Model Training: Train a Linear Regression model with cross-validation.
Visualizations: Explore actual vs. predicted AQI, residuals, and feature importance.
Interactive Prediction: Input pollutant values to predict AQI in real-time.
Reports: Generate comprehensive performance summaries.

Installation
To run the AQI Index Predictor locally, follow these steps:

Clone the Repository
git clone https://github.com/your-username/aqi-predictor.git
cd aqi-predictor


Install DependenciesEnsure you have Python 3.7+ installed. Then, install the required packages:
pip install -r requirements.txt

Note: Create a requirements.txt file with the following content and include it in your repository:
pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit


Download Sample Data

Use a dataset like city_day.csv (available from public sources like Kaggle) and place it in the project directory.
Ensure the dataset includes columns like AQI, City, Date, and pollutant features (e.g., PM2.5, NO2).


Run the ApplicationLaunch the app using Streamlit:
streamlit run app.py

Open your browser at http://localhost:8501 to access the app.


Usage

Upload Data

On the Home page, upload a CSV file containing air quality data.


Navigate

Use the Navigation Menu sidebar to switch between:
Home: View dataset info and credits.
Graphs: Explore visualizations of model performance.
Make a Prediction: Input pollutant values for AQI prediction.
View Model Report: See a detailed model analysis.




Interact

On the Make a Prediction page, enter pollutant levels and a city name to get an AQI forecast with category and color code.


Explore

Check the Credits & Help sidebar for team info, tool credits, and usage tips.



Credits & Help
Credits
🎉 Developed with ❤️ by Team GreenAir:

About Us: We are a passionate team dedicated to improving air quality awareness using data science. Our mission is to provide actionable insights through innovative tools like the AQI Predictor. 🌱
Tools Used:
Streamlit for the interactive UI 🌐
Pandas and NumPy for data processing 📊
Scikit-learn for machine learning 🤖
Matplotlib and Seaborn for visualizations 🎨


Special thanks to xAI for Grok assistance! 🚀

Help
📚 How to Use:

Upload a CSV file (e.g., city_day.csv) on the Home page 📤
Navigate using the left menu 🌍
Use 'Make a Prediction' to input pollutant values and predict AQI 🌡️
View graphs or reports for detailed analysis 📊
For issues, contact support at greenair.support@example.com ✉️

Contributing
We welcome contributions! To contribute:

Fork the repository.
Create a new branch (git checkout -b feature-branch).
Make your changes and commit (git commit -m "Add new feature").
Push to the branch (git push origin feature-branch).
Open a Pull Request.

License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

Inspired by open-source air quality datasets and machine learning communities.
Gratitude to the developers of the libraries used in this project.


Last updated: July 03, 2025
