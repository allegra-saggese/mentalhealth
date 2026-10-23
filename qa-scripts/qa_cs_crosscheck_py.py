"""qa_cs_crosscheck_py.py -- csdid and moderndid on the balanced extract written by qa_cs_crosscheck.py.

Run with the isolated cross-check environment, not the pipeline environment (see requirements-crosscheck.txt):
    ./.venv/bin/python qa-scripts/qa_cs_crosscheck_py.py <extract.csv> <out_cells.csv> <out_overall.csv>
Settings mirror did_cs.py: outcome regression, not-yet-treated comparison, base year g-1 (universal), no bootstrap.
"""
import sys

import pandas as pd

import moderndid
from csdid.att_gt import ATTgt

extract, out_cells, out_overall = sys.argv[1:4]
d = pd.read_csv(extract)
xformla = "~ " + " + ".join(c for c in d.columns if c.startswith("x") and c[1:].isdigit())

cs = ATTgt(yname="y", tname="year", idname="fips", gname="g", data=d, control_group=["notyettreated"],
           xformla=xformla, panel=True, biters=99).fit(est_method="reg", base_period="universal", bstrap=False)
res = cs.results
cells = [pd.DataFrame({"package": "csdid", "cohort": res["group"], "year": res["year"], "att": res["att"]})]
overall = [{"package": "csdid", "overall_att": float(cs.aggte(typec="simple").atte["overall_att"])}]

md = moderndid.att_gt(data=d, yname="y", tname="year", idname="fips", gname="g", xformla=xformla, est_method="reg",
                      control_group="notyettreated", base_period="universal", boot=False)
cells.append(pd.DataFrame({"package": "moderndid", "cohort": md.groups, "year": md.times, "att": md.att_gt}))
overall.append({"package": "moderndid", "overall_att": float(moderndid.aggte(md, type="simple").overall_att)})

pd.concat(cells).to_csv(out_cells, index=False)
pd.DataFrame(overall).to_csv(out_overall, index=False)
