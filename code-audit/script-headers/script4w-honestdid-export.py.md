# Original header of `script4w-honestdid-export.py`

Verbatim module docstring before the 2026-10-22 header trim (see METHODS.md and the decision register for the current versions of this material).

````text

script4w-honestdid-export.py  --  inputs for the Rambachan-Roth sensitivity analysis (added 2026-10-17)

Refits the BASELINE TWFE event study of script4u (county + year FE, 8 controls, state-clustered covariance)
for each treatment definition and outcome, and writes what R's HonestDiD package needs:
    <design>__<outcome>__beta.csv     event_time, beta               (pre-periods then post-periods, reference e=-1 omitted)
    <design>__<outcome>__sigma.csv    full state-clustered covariance matrix of those coefficients
    manifest.csv                      one row per file pair with the number of pre and post periods

Only a CONTIGUOUS run of event times is exported (HonestDiD indexes periods by position): leads run back
from e=-2 as far as every lead is present, lags run forward from e=0 as far as every lag is present.

NOTE: this repeats the sample construction of script4u (copied, not imported, because script4u runs on
import). Merging the two is on the clean-up list.

Then run script4w-honestdid.R.

````
