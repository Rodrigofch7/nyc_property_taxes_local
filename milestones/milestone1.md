# <NYC Property Taxes>

## Members
- Ahmed Lodhi <ahmedlodhi@uchicago.edu>
- Faizan Imran <imranfaiz@uchicago.edu>
- Rodrigo Chaves <rchaves@uchicago.edu>

## Proposal

Property tax assessment accuracy is a critical issue of fiscal equity in New York City. When properties are systematically undervalued or overvalued relative to their actual market prices, the tax burden is distributed unfairly across property owners and neighborhoods. This project aims to classify NYC tax lots as undervalued, fairly valued, or overvalued by comparing Department of Finance assessed values against actual recorded sale prices from 2018 to 2022.
Using NYC's Property Valuation and Assessment Data (NYC Open Data) and Annualized Sales Data (NYC Department of Finance), we will engineer features such as price-per-square-foot, assessment ratios, building characteristics, and neighborhood-level variables to train classification models. Machine learning is the right approach here because the relationship between assessed and market value is highly nonlinear and varies across borough, building class, tax class, and neighborhood — patterns that are difficult to capture with simple rules or linear models. Our results could inform policy recommendations around assessment reform and help identify neighborhoods where systematic mis-assessment disproportionately affects residents.
