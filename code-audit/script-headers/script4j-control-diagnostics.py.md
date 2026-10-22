# Original header of `script4j-control-diagnostics.py`

Verbatim module docstring before the 2026-10-22 header trim (see METHODS.md and the decision register for the current versions of this material).

````text

script4j-control-diagnostics.py  --  the 19 controls: coverage, causal role, pre-trends.

Three outputs.

1. COVERAGE + ROLE TABLE. For each control: share of the panel and of the
   estimating sample it covers, first/last year, a one-line description, which
   specifications it enters, and a causal classification.

   CLASSIFICATION -- what the three labels mean, since they are not interchangeable:
     confounder  Plausibly causes BOTH CAFO siting and the outcome, and is not
                 itself caused by CAFO presence. Conditioning on it REMOVES bias.
     mediator    Plausibly caused BY CAFO presence and in turn affects the
                 outcome. Conditioning on it BLOCKS part of the causal path and
                 biases the estimate toward zero.
     collider    Plausibly caused by BOTH the treatment and the outcome.
                 Conditioning on it INDUCES spurious association.
   Where a variable could be more than one, the label records the most likely
   role and the note says why. These are judgements about the causal graph, not
   statistical findings -- they are stated so a reviewer can disagree explicitly.

2. COVARIATE PRE-TREND TEST. Each control is used as the DEPENDENT variable in
   the event-study specification, with county and year fixed effects. If a
   control trends differentially in the pre-period, then treated and control
   counties were already diverging on it before treatment -- which undermines
   parallel trends and suggests the variable is not pre-determined with respect
   to treatment timing. Reported as the number of significant pre-period
   coefficients and the joint F-test on all of them.

3. VIF on the headline specification (TWFE, county + year FE, 19 controls).

````
