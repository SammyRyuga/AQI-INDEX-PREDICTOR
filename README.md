# 🌍 AQI Index Predictor using Linear Regression

An interactive, feature-rich AQI (Air Quality Index) predictor powered by Linear Regression and built with **Streamlit**. This tool helps analyze air quality using pollutants data and provides real-time predictions with visual insights.

> 📌 Developed with ❤️ by [Akshat Goel](http://linkedin.com/in/akshat-goel-b13054323), [Anwita Padhi](https://www.linkedin.com/in/anwita-padhi-187576321/), [Samanyu Pattanayak](https://www.linkedin.com/in/samanyu-pattanayak-8757551a9/), [Samruddh Om Bahanwal](https://www.linkedin.com/in/samruddh-om-bahanwal-48a93a228?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app), and [Shruti Deepak](https://www.linkedin.com/in/shruti-deepak-956820362/)

---

## 🚀 Features

- 📈 Linear Regression model for AQI prediction
- 🌡️ Handles key pollutants like `PM2.5`, `PM10`, `NO2`, `SO2`, `CO`, `NH3`, etc.
- 📊 Visualizations:
  - Actual vs Predicted AQI
  - Residual plots
  - Q-Q plot for normality check
  - Feature importance (coefficients)
- 🏙️ City-wise one-hot encoding
- ✍️ Interactive AQI prediction form
- 📋 Comprehensive model performance report
- 📤 CSV upload and auto-cleaning
- 🔁 5-Fold Cross-Validation for robustness

---

## 📁 Project Structure

- `app.py`: Main Python script containing the Streamlit app and AQI Predictor class.
- `requirements.txt`: List of dependencies required to run the project.
- `README.md`: This file, providing project overview and instructions.

---

## 🛠️ Installation

To run the AQI Index Predictor locally, follow these steps:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/SammyRyuga/AQI-INDEX-PREDICTOR
   cd aqi-predictor
   ```

2. **Install Dependencies**
   Ensure you have Python 3.7+ installed. Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

   *Note*: The `requirements.txt` file should include:
   ```
   pandas
   numpy
   matplotlib
   seaborn
   scikit-learn
   streamlit
   ```

3. **Download Sample Data**
   - Use a dataset like `city_day.csv` (available from public sources like Kaggle) and place it in the project directory.
   - Ensure the dataset includes columns like `AQI`, `City`, `Date`, and pollutant features (e.g., `PM2.5`, `NO2`).

4. **Run the Application**
   Launch the app using Streamlit:
   ```bash
   streamlit run app.py
   ```
   Open your browser at `http://localhost:8501` to access the app.

---

## 🎮 Usage

1. **Upload Data**
   - On the **Home** page, upload a CSV file containing air quality data.

2. **Navigate**
   - Use the sidebar to switch between:
     - **Home**: View dataset info.
     - **Graphs**: Explore visualizations.
     - **Make a Prediction**: Input pollutant values for AQI prediction.
     - **View Model Report**: See detailed analysis.

3. **Interact**
   - On the **Make a Prediction** page, enter pollutant levels and a city name to get an AQI forecast.

4. **Explore**
   - Check the sidebar for credits and help information.

---

## 🤝 Contributing

We welcome contributions! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m "Add new feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Inspired by open-source air quality datasets from Kaggle community.
- Special thanks to ISTE for support and guidance.

---
