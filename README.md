# DataEngineering-Visualization
🚀 Project Overview
The goal of this project is to execute a complete data engineering pipeline using a real-world dataset on motor vehicle collisions in New York City.

The process includes:


Exploring the raw data to understand its structure and issues.




Cleaning the data by handling missing values, outliers, and inconsistencies .




Integrating the primary crashes dataset with a related dataset (e.g., 'Persons' involved).






Building a fully interactive website to visualize insights and generate dynamic reports.


📊 Datasets
This project uses the following datasets from NYC Open Data:


NYC Motor Vehicle Collisions Crashes: The primary dataset containing over 2 million crash records from 2012 to 2025.


NYC Motor Vehicle Collisions Person: A related dataset used for integration, joined via the COLLISION_ID column.


📁 Repository Contents
This repository contains all the required deliverables for Milestone 1:


Jupyter Notebook (.ipynb): A notebook detailing the full Exploratory Data Analysis (EDA), pre-integration cleaning, data integration, and post-integration cleaning steps.



Website Source Code: The complete source code for the interactive data visualization website.



README.md: This file, outlining the project, setup instructions, and team contributions.


✨ Website Features
The deployed website provides a dynamic way for users to explore and generate insights from the integrated dataset. Key features include:


Dynamic Filtering: Multiple dropdown filters (e.g., Borough, Year, Vehicle Type, Contributing Factor) to dynamically filter the data.


Search Mode: A search bar that allows users to type queries (e.g., "Brooklyn 2022 pedestrian crashes") to apply filters.


Generate Report Button: A central button that, when clicked, dynamically updates all visualizations based on the selected filters or search terms.



Interactive Visualizations: A variety of charts (bar charts, maps, heatmaps, etc.) that respond to user interaction with hover-tooltips, zoom, and panning.

💻 Technology Stack

Data Analysis & Cleaning: Python, Pandas 


Web Framework & Visualization: [e.g., Dash, Plotly, React, Flask] 


Deployment: [e.g., Vercel, Render, Heroku]
