# CCIHE2021-PublicData

"""
UNITID : Unique IPEDS identification number for an institution
"""
unit_features = ['unitid']

"""
MINORITIZED

MSI : Minority Serving Institution (HSI, HBCU, or Tribal College or University only)
HBCU : Historically Black College or University
HSI : Hispanic Serving Institution (Using composite lists)
TRIBAL : Tribal College flag
"""
minority_features = ['msi', 'hbcu', 'hsi', 'tribal']
minority_values = {
    'msi': {'yes': 1, 'no': 0},
    'hbcu': {'yes': 1, 'no': 2},
    'hsi': {'yes': 1, 'no': 0},
    'tribal': {'yes': 1, 'no': 2}
}

"""
WOMENS : Women's College flag
"""
womens_features = ['womens']

"""
BASIC CATEGORIZATIONS

BASIC2005 : 2005 Basic Classification (historical - not updated)
BASIC2010 : 2010 Basic Classification (historical - not updated)
BASIC2015 : 2015 Basic Classification (historical - not updated)
BASIC2018 : 2018 Basic Classification (historical - not updated)
BASIC2021 : 2021 Basic Classification 
"""
basic_features = ['basic2005', 'basic2010', 'basic2015', 'basic2018', 'basic2021']

"""
STEM_RSD : STEM research/scholarship doctoral degrees
"""
stem_features = ['stem_rsd']

"""
LANDGRNT : Land-grant institution
"""
land_grant_features = ['landgrnt']

"""
ACT

ACTCMP25 : ACT Composite Score, 25 percentile
ACTFINAL : Derived 25th percentile ACT score, weighting both ACT and equated SAT scores by number submitted
NACT : Number of first-time entering students who submitted ACT score
SATACTEQ25 : The ACT equivalent score for the combined 25th percentile SAT score
SATCMB25 : Combined SAT-Math and SAT-Verbal 25th percentiles scores
SATM25 : SAT-Math 25th percentile score
SATV25 : SAT-Verbal 25th percentile score
"""
act_features = ['actcmp25', 'actfinal', 'nact', 'satacteq25', 'satcmb25', 'satm25', 'satv25']

"""
SIZING

BACCDEG : Bachelor's degree total
SIZESET2021	: 2021 Size and Setting Classification
UGDSFT20 : Undergraduate degree-seeking full-time enrollment
UGDSPT20 : Undergraduate degree-seeking part-time enrollment
UGN1STTMFT20 : Undergraduate new first-time full-time students
UGN1STTMPT20 : Undergraduate new first-time part-time students
UGNDFT20 : Undergraduate non-degree full-time students
UGNDPT20 : Undergraduate non-degree part-time students
UGNTRFT20 : Undergraduate new transfer-in full-time students
UGNTRPT20 : Undergraduate new transfer-in part-time students
UGPROFILE2021 : 2021 Undergraduate Profile Classification
UGTENR20 : Undergraduate total enrollment, fall 2020
""" 
size_features = ['baccdeg', 'sizeset2021', 'ugdsft20', 'ugdspt20', 'ugn1sttmft20', 'ugn1sttmpt20', 'ugndft20', 'ugndpt20', 'ugntrft20', 'ugntrpt20', 'ugprofile2021', 'ugtenr20']

"""

INTER-STATE REGIONS

regions_UCSB: made using the US Census Beureau's latest metrpolitan and micropolitan data

"""
regions_USCB = {
    'Allentown-Bethlehem-East Stroudsburg, PA-NJ': ['NJ', 'PA'],
    'Atlanta--Athens-Clarke County--Sandy Springs, GA-AL': ['AL', 'GA'],
    'Boise City-Mountain Home-Ontario, ID-OR': ['ID', 'OR'],
    'Boston-Worcester-Providence, MA-RI-NH': ['MA', 'NH', 'RI'],
    'Brookings-Crescent City, OR-CA': ['CA', 'OR'],
    'Burlington-Fort Madison, IA-IL': ['IA', 'IL'],
    'Cape Girardeau-Sikeston, MO-IL': ['IL', 'MO'],
    'Charleston-Huntington-Ashland, WV-OH-KY': ['KY', 'OH', 'WV'],
    'Charlotte-Concord, NC-SC': ['NC', 'SC'],
    'Chattanooga-Cleveland-Dalton, TN-GA-AL': ['AL', 'GA', 'TN'],
    'Chicago-Naperville, IL-IN-WI': ['IL', 'IN', 'WI'],
    'Cincinnati-Wilmington, OH-KY-IN': ['IN', 'KY', 'OH'],
    'Columbus-Auburn-Opelika, GA-AL': ['AL', 'GA'],
    'Dallas-Fort Worth, TX-OK': ['OK', 'TX'],
    'Davenport-Moline, IA-IL': ['IA', 'IL'],
    'Duluth-Grand Rapids, MN-WI': ['MN', 'WI'],
    'El Paso-Las Cruces, TX-NM': ['NM', 'TX'],
    'Evansville-Henderson, IN-KY': ['IN', 'KY'],
    'Fargo-Wahpeton, ND-MN': ['MN', 'ND'],
    'Huntsville-Decatur-Albertville, AL-TN': ['AL', 'TN'],
    'Jacksonville-Kingsland-Palatka, FL-GA': ['FL', 'GA'],
    'Johnson City-Kingsport-Bristol, TN-VA': ['TN', 'VA'],
    'Joplin-Miami, MO-OK-KS': ['KS', 'MO', 'OK'],
    'Kansas City-Overland Park-Kansas City, MO-KS': ['KS', 'MO'],
    'Keene-Brattleboro, NH-VT': ['NH', 'VT'],
    'La Crosse-Onalaska-Sparta, WI-MN': ['MN', 'WI'],
    'Louisville/Jefferson County--Elizabethtown, KY-IN': ['IN', 'KY'],
    'Marinette-Iron Mountain, WI-MI': ['MI', 'WI'],
    'Memphis-Clarksdale-Forrest City, TN-MS-AR': ['AR', 'MS', 'TN'],
    'Minneapolis-St. Paul, MN-WI': ['MN', 'WI'],
    'New Orleans-Metairie-Slidell, LA-MS': ['LA', 'MS'],
    'New York-Newark, NY-NJ-CT-PA': ['CT', 'NJ', 'NY', 'PA'],
    'Omaha-Fremont, NE-IA': ['IA', 'NE'],
    'Paducah-Mayfield, KY-IL': ['IL', 'KY'],
    'Parkersburg-Marietta-Vienna, WV-OH': ['OH', 'WV'],
    'Philadelphia-Reading-Camden, PA-NJ-DE-MD': ['DE', 'MD', 'NJ', 'PA'],
    'Pittsburgh-Weirton-Steubenville, PA-OH-WV': ['OH', 'PA', 'WV'],
    'Portland-Vancouver-Salem, OR-WA': ['OR', 'WA'],
    'Pullman-Moscow, WA-ID': ['ID', 'WA'],
    'Quincy-Hannibal, IL-MO': ['IL', 'MO'],
    'Reno-Carson City-Gardnerville Ranchos, NV-CA': ['CA', 'NV'],
    'Salt Lake City-Provo-Orem, UT-ID': ['ID', 'UT'],
    'Sioux City-Le Mars, IA-NE-SD': ['IA', 'NE', 'SD'],
    'South Bend-Elkhart-Mishawaka, IN-MI': ['IN', 'MI'],
    "Spokane-Spokane Valley-Coeur d'Alene, WA-ID": ['ID', 'WA'],
    'St. Louis-St. Charles-Farmington, MO-IL': ['IL', 'MO'],
    'Tallahassee-Bainbridge, FL-GA': ['FL', 'GA'],
    'Virginia Beach-Chesapeake, VA-NC': ['NC', 'VA'],
    'Washington-Baltimore-Arlington, DC-MD-VA-WV-PA': ['DC', 'MD', 'PA', 'VA', 'WV']
 }

# Carnegie Classifications

## 2021
broad_carnegie_classification_2021 = {
    -2: 'Not Classified',
    1: "Associate's Colleges",
    2: "Associate's Colleges",
    3: "Associate's Colleges",
    4: "Associate's Colleges",
    5: "Associate's Colleges",
    6: "Associate's Colleges",
    7: "Associate's Colleges",
    8: "Associate's Colleges",
    9: "Associate's Colleges",
    10: 'Special Focus Institutions',
    11: 'Special Focus Institutions',
    12: 'Special Focus Institutions',
    13: 'Special Focus Institutions',
    14: "Associate's Colleges",
    15: 'R1 Universities',                    # Doctoral Universities: Very High Research Activity
    16: 'R2 Universities',                    # Doctoral Universities: High Research Activity
    17: 'Other Doctoral Universities',        # Doctoral/Professional Universities
    18: "Master's Colleges and Universities",
    19: "Master's Colleges and Universities",
    20: "Master's Colleges and Universities",
    21: 'Baccalaureate Colleges',
    22: 'Baccalaureate Colleges',
    23: 'Baccalaureate Colleges',
    24: 'Special Focus Institutions',
    25: 'Special Focus Institutions',
    26: 'Special Focus Institutions',
    27: 'Special Focus Institutions',
    28: 'Special Focus Institutions',
    29: 'Special Focus Institutions',
    30: 'Special Focus Institutions',
    31: 'Special Focus Institutions',
    32: 'Special Focus Institutions',
    33: 'Tribal Colleges and Universities'
}

broad_carnegie_classification_2018 = {
    -2: 'Not Classified',
    1: "Associate's Colleges",
    2: "Associate's Colleges",
    3: "Associate's Colleges",
    4: "Associate's Colleges",
    5: "Associate's Colleges",
    6: "Associate's Colleges",
    7: "Associate's Colleges",
    8: "Associate's Colleges",
    9: "Associate's Colleges",
    10: 'Special Focus Institutions',
    11: 'Special Focus Institutions',
    12: 'Special Focus Institutions',
    13: 'Special Focus Institutions',
    14: "Associate's Colleges",
    15: 'R1 Universities',                    # Doctoral Universities: Very High Research Activity
    16: 'R2 Universities',                    # Doctoral Universities: High Research Activity
    17: 'Other Doctoral Universities',        # Doctoral/Professional Universities
    18: "Master's Colleges and Universities",
    19: "Master's Colleges and Universities",
    20: "Master's Colleges and Universities",
    21: 'Baccalaureate Colleges',
    22: 'Baccalaureate Colleges',
    23: 'Baccalaureate Colleges',
    24: 'Special Focus Institutions',
    25: 'Special Focus Institutions',
    26: 'Special Focus Institutions',
    27: 'Special Focus Institutions',
    28: 'Special Focus Institutions',
    29: 'Special Focus Institutions',
    30: 'Special Focus Institutions',
    31: 'Special Focus Institutions',
    32: 'Special Focus Institutions',
    33: 'Tribal Colleges and Universities'
}

broad_carnegie_classification_2015 = {
    -2: 'Not Classified',
    1: "Associate's Colleges",
    2: "Associate's Colleges",
    3: "Associate's Colleges",
    4: "Associate's Colleges",
    5: "Associate's Colleges",
    6: "Associate's Colleges",
    7: "Associate's Colleges",
    8: "Associate's Colleges",
    9: "Associate's Colleges",
    10: 'Special Focus Institutions',
    11: 'Special Focus Institutions',
    12: 'Special Focus Institutions',
    13: 'Special Focus Institutions',
    14: "Associate's Colleges",
    15: 'R1 Universities',                    # Doctoral Universities: Highest Research Activity
    16: 'R2 Universities',                    # Doctoral Universities: Higher Research Activity
    17: 'Other Doctoral Universities',        # Doctoral Universities: Moderate Research Activity
    18: "Master's Colleges and Universities",
    19: "Master's Colleges and Universities",
    20: "Master's Colleges and Universities",
    21: 'Baccalaureate Colleges',
    22: 'Baccalaureate Colleges',
    23: 'Baccalaureate Colleges',
    24: 'Special Focus Institutions',
    25: 'Special Focus Institutions',
    26: 'Special Focus Institutions',
    27: 'Special Focus Institutions',
    28: 'Special Focus Institutions',
    29: 'Special Focus Institutions',
    30: 'Special Focus Institutions',
    31: 'Special Focus Institutions',
    32: 'Special Focus Institutions',
    33: 'Tribal Colleges and Universities'
}

broad_carnegie_classification_2005_and_2010 = {
    -2: 'Not Classified',
    1: "Associate's Colleges",
    2: "Associate's Colleges",
    3: "Associate's Colleges",
    4: "Associate's Colleges",
    5: "Associate's Colleges",
    6: "Associate's Colleges",
    7: "Associate's Colleges",
    8: "Associate's Colleges",
    9: "Associate's Colleges",
    10: "Associate's Colleges",
    11: "Associate's Colleges",
    12: "Associate's Colleges",
    13: "Associate's Colleges",
    14: "Associate's Colleges",
    15: 'R1 Universities',                    # RU/VH: Research Universities (very high research activity)
    16: 'R2 Universities',                    # RU/H: Research Universities (high research activity)
    17: 'Other Doctoral Universities',        # DRU: Doctoral/Research Universities
    18: "Master's Colleges and Universities",
    19: "Master's Colleges and Universities",
    20: "Master's Colleges and Universities",
    21: 'Baccalaureate Colleges',
    22: 'Baccalaureate Colleges',
    23: 'Baccalaureate Colleges',
    24: 'Special Focus Institutions',
    25: 'Special Focus Institutions',
    26: 'Special Focus Institutions',
    27: 'Special Focus Institutions',
    28: 'Special Focus Institutions',
    29: 'Special Focus Institutions',
    30: 'Special Focus Institutions',
    31: 'Special Focus Institutions',
    32: 'Special Focus Institutions',
    33: 'Tribal Colleges and Universities'
}

broad_carnegie_classification_2000 = {
    -2: 'Not Classified',
    15: 'R1 Universities',                    # Doctoral/Research Universities—Extensive
    16: 'R2 Universities',                    # Doctoral/Research Universities—Intensive
    21: "Master's Colleges and Universities",
    22: "Master's Colleges and Universities",
    31: 'Baccalaureate Colleges',
    32: 'Baccalaureate Colleges',
    33: 'Baccalaureate Colleges',
    40: "Associate's Colleges",
    51: 'Special Focus Institutions',
    52: 'Special Focus Institutions',
    53: 'Special Focus Institutions',
    54: 'Special Focus Institutions',
    55: 'Special Focus Institutions',
    56: 'Special Focus Institutions',
    57: 'Special Focus Institutions',
    58: 'Special Focus Institutions',
    59: 'Special Focus Institutions',
    60: 'Tribal Colleges and Universities'
}

broad_carnegie_classification = {
    2021: broad_carnegie_classification_2021,
    2018: broad_carnegie_classification_2018,
    2015: broad_carnegie_classification_2015,
    2010: broad_carnegie_classification_2005_and_2010,
    2005: broad_carnegie_classification_2005_and_2010, 
    2000: broad_carnegie_classification_2000
}