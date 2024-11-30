# Metrics to consider from reported datasets

This tracked our initial list of potential metrics to consider. With early visualization and state-level data information we reduced this set to our final 5 types of universities and 17 total features.

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

## Final 17 features by district/county in each state

1. Population
2. Per capita income
3. Nearest 5 R1/R2 universities by average distance.
4. Nearest 5 R1/R2 universities by average enrollment.
5. Nearest 5 R1/R2 universities by average number of dormbeds.
6. Nearest 5 public universities by average distance.
7. Nearest 5 public universities by average enrollment.
8. Nearest 5 public universities by average number of dormbeds.
9. Nearest 5 private not for profit universities by average distance.
10. Nearest 5 private not for profit universities by average enrollment.
11. Nearest 5 private not for profit universities by average number of dormbeds.
12. Nearest 5 land-grant universities by average distance.
13. Nearest 5 land-grant universities by average enrollment.
14. Nearest 5 land-grant universities by average number of dormbeds.
15. Nearest 5 STEM-specialized universities by average distance.
16. Nearest 5 STEM-specialized universities by average enrollment.
17. Nearest 5 STEM-specialized universities by average number of dormbeds.
