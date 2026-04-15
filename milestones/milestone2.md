## Milestone 2: Data Exploration

This project examines whether NYC properties can be classified as **undervalued, fairly valued, or overvalued** using structural, geographic, and tax-related features in order to identify systematic assessment inequities across neighborhoods and property types.

We use two datasets downloaded from the **NYC Department of Finance website**: the **Annualized Sales Data (2022–2024)** and the **2024 Property Assessment Roll**. After merging them using the **BBL identifier**, the final dataset contains **161,045 observations**, which is more than sufficient for model training.

The target variable is based on each property’s **assessment ratio relative to the tax-class benchmark**, producing three labels: undervalued, fairly valued, and overvalued.

Planned features include borough, neighborhood, ZIP code, square footage, land area, number of units, zoning, and prior-year assessment values. We also extract new features such as **building age**, **log-transformed area**, and **coverage ratio** to improve prediction quality.

We plan to test **logistic regression, decision trees, random forest, gradient boosting, and KNN**. 
