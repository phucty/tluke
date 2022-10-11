### Dataset:
T2D: 779 web tables, 237 tables are annotated with the 2014 version of DBpedia, the remaining 542 tables are negative samples. 

#### Semantic Table Interpretation Tasks:
- Row-to-instance (CEA): Annotate table rows (cells in the subject columns) with DBpedia instances
- Attribute-to-property (CPA): Annotate the relationship between the subject column and other table columns to DBpedia property
- Table-to-class (CPA): Annotate table subject columns with DBpedia instance type

#### Format:
The orginal data are splited, removed unavailbe instances and do mapping to Wikipedia

|Split | Ratio| Full | Pos | Neg | CEA | CTA | CPA | 
|---|---|---|---|---|---|---|---|
|Train | 70% | 545 | 165 | 380| - | - | - |
|Dev | 15% | 117 | 36 | 81| - | - | - |
|Test | 15% | 117 | 36 | 81| - | - | - |


