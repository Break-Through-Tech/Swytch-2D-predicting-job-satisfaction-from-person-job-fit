PR17 integration with 13 additional NSCG fields

74,016 respondents; 123 occupation categories; 63 columns.
The first 50 columns and row order are identical to the PR17 input.
The last 13 columns retain original NSCG values, joined by OBSNUM:
SALARY, HRSWK, EMSECSM, EMSIZE, SATADV, SATBEN, SATCHAL, SATIND, SATLOC, SATRESP, SATSAL, SATSEC, SATSOC

SAT fields: 1 = very satisfied, 2 = somewhat satisfied,
3 = somewhat dissatisfied, 4 = very dissatisfied.
EMSECSM and EMSIZE are category codes, not measured numeric amounts.
HRSWK value 96 represents 96 hours or more.
No cleaning or model training was performed. Validation checks do not alter data.
Variable definitions and codes: data/nscg/pcg23Public/Dpcg23.xlsx and Ppcg23.sas.
Reproduce: python build_integration63.py
Verification: integration63_verification.json. Original inputs remain unchanged.
