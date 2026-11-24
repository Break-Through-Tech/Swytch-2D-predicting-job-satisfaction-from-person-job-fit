import joblib, pandas as pd
from swytch.features import FAC, JOB, make_product_pipe

df = pd.read_csv("data/nscg2023_onet30_2_subset_1.csv", low_memory=False)
for c in FAC:
    df["IMP_" + c[3:]] = 5 - df[c]
y = df["JOBSATIS"].isin([1, 2]).astype(int)

model = make_product_pipe().fit(df, y)
joblib.dump(model, "artifacts/model.joblib")

# one row per occupation (O*NET values are constant within a profile)
occ = (df.groupby("ONET_PROFILE_ID")
         .agg(title=("NSCG_OCC_TITLE", "first"), n=("OBSNUM", "size"), **{c: (c, "first") for c in JOB}))
occ.to_csv("artifacts/occupations.csv")
print(occ.shape, "model saved")