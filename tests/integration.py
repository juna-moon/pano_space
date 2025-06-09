import glob, json, os, pandas as pd, humanize
rows=[]
for js in glob.glob("data/**/qc_stats.json", recursive=True):
    s=json.load(open(js))
    d=os.path.dirname(js).split('/')[-2]
    rows.append({"dataset":d,"imgs":s["n_total"],"mean":s["mean"],"std":s["std"]})
df=pd.DataFrame(rows)
print(df.to_markdown(index=False))