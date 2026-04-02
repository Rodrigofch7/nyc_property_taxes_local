# <NYC Property Taxes>

## Members
- Ahmed Raza Khan Lodhi <ahmedlodhi@uchicago.edu>
- Muhammad Faizan Imran <imranfaiz@uchicago.edu>
- Rodrigo Chaves <rchaves@uchicago.edu>

## Proposal

Property taxes have long been a potent instrument for local and state level government to generate viable revenue stream. Property taxes are so prevalent because they are hard to evade and relatively easy to administor. A policy backlash that property taxes often face is about their equity - due to fluctuations in property prices, the property tax department needs to keep updating value assessments. There is also some debate around how different property types should be assessed differently based on their use (commercial vs residential) that also complicates what the tax schedule looks like. When properties are systematically undervalued or overvalued relative to their actual market prices, the tax burden is distributed unfairly across property owners and neighborhoods.

For this project we will be focusing on New York City's property tax structure and attempt to use different models to get to a model that assess property values that is proportionate to their acutal values. New York's new Mayor Zohran Mamdani proposed a property tax increase of 9.5% if not allowed to raise wealth taxes to cover up city spending deficit. This highlights property taxes as a very policy relevant lever in hands of policymaker and places our research well in the current political milieu. We will aim to classify NYC property units as either undervalued, fairly valued, or overvalued by comparing Department of Finance assessed values against actual recorded sale prices from.

Using NYC's Property Valuation and Assessment Data (source: from NYC Open Data) and Annualized Sales Data (source: NYC Department of Finance), we will engineer features such as price-per-square-foot, assessment ratios, building characteristics, and neighborhood-level variables to train classification models. Machine learning is the right approach here because the relationship between assessed and market value is highly nonlinear and varies across borough, building class, tax class, and neighborhood — patterns that are difficult to capture with simple rules or linear models. Our results could inform policy recommendations around assessment reform and help identify neighborhoods where systematic mis-assessment disproportionately affects residents.
