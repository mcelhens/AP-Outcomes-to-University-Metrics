# Metrics to consider from reported datasets

## Carnegie University Categorization Dataset

1. Basic categorizations: exploratory
     * Look for trends to consider from the 2021 basic categories ignoring any potential time relationships
2. Basic categorization: time relationship
     1. Predictive within a category time frame
         * Categories last ~5 years can they predict the 5th year outcomes if trained on the first 4 years of AP data
         * With data limitations this would apply to most recent dataset only.
     2. Change of category distribution over time visualization
3. Minority categorizations
     1. General (HS, HBCU, tribal), i.e. any school with a distinction. Women's colleges are not included.
     2. Hispanic serving institutions (HS)
     3. Historically Black Colleges and Universities (HBCUs)
     4. Tribal colleges: probably too few
          * Tribal colleges are native/indigenous owned/operated not just native-serving to my understanding
     5. Women's: too small and few in number
4. STEM vs non-STEM category
5. Land-grant vs non-land grant correction
     * "Land-grant" university (wikipedia) "A land-grant university (also called land-grant college or land-grant institution) is an institution of higher education in the United States designated by a state to receive the benefits of the Morrill Acts of 1862 and 1890, or a beneficiarsy under the Equity in Educational Land-Grant Status Act of 1994. There are 57 institutions which fall under the 1862 Act, 19 under the 1890 Act, and 35 under the 1994 Act."
         * Colloquially: land-grant universities are a sub-category of public university.
6. Selectivity index categorization

### Correction for size disparities in US options

1. Correct for population of state
2. Correct for geographic area of state
3. Aggregate states into geographic regions
     * This might consider northeast regions where cross state accessibility is high.
     * Cross state accessibility is much lower in other parts of the US
4. Limiting to land grant universities
