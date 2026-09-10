# V++_b3 public leaderboard result

Retrieved 2026-09-09T22:27:26.448237+00:00 from the [challenge leaderboard](https://rexrank.ai/ReXGroundingCT/challenge.html)
and its [public result document](https://firestore.googleapis.com/v1/projects/rexgrounding-challenge/databases/(default)/documents/leaderboard/81U1FmE4VxlJKVVjWOXa).

- Displayed rank at capture: **15** (page's best-per-submitter ranking).
- Submission: **V++_b3**; team: **INAIA**; submitter: **Chushu**; institution: **SHS**.
- Public cohort: **150 CTs / 302 findings**, `public-50pct`.
- Published result update: `2026-09-06T23:52:32.352Z`.
- External document ID: `81U1FmE4VxlJKVVjWOXa`.
- Public submission number: 1.

| Overall metric | Public source value |
| --- | ---: |
| Dice | 0.332147596250632 |
| Hit rate | 0.7350993377483444 |
| Instance precision | 0.13665338645418326 |
| Instance recall | 0.49 |
| Instance F1 | 0.21370716510903429 |

## Per-category results

| Category | n | Dice | Hit rate |
| --- | ---: | ---: | ---: |
| 1a — Bronchial wall thickening | 5 | 0.2638863206474939 | 0.8 |
| 1b — Bronchiectasis | 5 | 0.12983820979244445 | 0.2 |
| 1c — Emphysema (including Centrilobular, Paraseptal, Bullous) | 16 | 0.1763588010718259 | 0.4375 |
| 1d — Septal thickening (including Interlobular, Reticulation) | 6 | 0.16934231974487624 | 0.5 |
| 1e — Micronodules (including Centrilobular, Tree-in-bud, Perilymphatic) | 10 | 0.19509624200377867 | 0.5 |
| 1f — Other | 2 | 0.012016683338722927 | 0.0 |
| 2a — Linear (including subsegmental atelectasis, scarring, fibrosis) | 61 | 0.2699466288523313 | 0.7868852459016393 |
| 2b — Atelectasis, consolidation | 46 | 0.39158672437302594 | 0.782608695652174 |
| 2c — Groundglass opacity | 39 | 0.36336522594492915 | 0.7435897435897436 |
| 2d — Pulmonary nodules/masses | 97 | 0.3355990267285007 | 0.7938144329896907 |
| 2e — Pleural effusion or thickening | 13 | 0.3837239608949172 | 0.7692307692307693 |
| 2g — Pneumothorax | 1 | 0.5546214148650764 | 1.0 |
| 2h — Other | 1 | 0.4863036861857862 | 1.0 |

All 13 category rows and the three displayed overall scores match the user's
provided values when rounded to three decimal places. Category counts sum to
302. Category 2f is absent in this public record; no zero score is invented.

## Availability and interpretation

This is the complete public leaderboard document, including full-precision
category and aggregate metrics. It contains no per-case predictions, per-case
scores, distance metrics or private-half/full-300 results. Those values are
unavailable from this public entry.

The published overall Dice (0.332147596250632) differs from the finding-count-weighted
mean of the published category Dice values (0.31461865183074966). The source does
not explain that difference in this record. Preserve the published score;
do not replace it with a recomputed value. The category-weighted hit rate
matches the published aggregate.

The local b3 recipe is a candidate association based on the displayed name.
No uploaded ZIP hash or confirmed local submission event is available in this
record, so registry upload history is unchanged and the local link remains
unconfirmed. Public document creation time is not asserted to be upload time.

Files: `Vpp_b3.firestore.json` is the exact HTTP response for this public
result; `Vpp_b3.decoded.json` retains all decoded fields; the CSV holds the full
category values. `ranking_inputs.json` records extracted public ranking fields
from all pages; `challenge.html` preserves the page's display/ranking code.
