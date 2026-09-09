# Dataset

## Data Availability

The original dataset in **not included in this repo** due to data privacy and sharing restrictions.

The dataset was provided for academic research purposes and has been anonymised prior to use. It contained student-level educational data from 9 UK schools, including 7 schools part of a multi-academy trust. The original data set contained data from 2023-2026, although only the 2023 and 2023 cohorts were used as they had confirmed GCSE results included.

## Dataset Structure

The original data was provided as an Excel (.xlsx) file and was converted to CSV for use in Python analysis. Each row represented an individual student and there was 23 colums in each cohort.

| Category | Example Variables |
|---|---|
| Target outcomes | `GCSEEnglishResult`, `GCSEMathsResult` |
| Demographics | `Age`, `Gender`, `Ethnicity` |
| Student characteristics | `FSM`, `PP`, `EAL`, `SEND` |
| Attendance & behaviour | Attendance measures, `Suspensions` |
| Prior attainment | `KS2 Reading`, `KS2 Maths` |
| School assessments | GL English/Maths assessments |
| GCSE preparation | Year 10 and Year 11 mock results |