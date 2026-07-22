<!-- Archive metadata in this block is not source content. -->
> **Archive metadata:** Source: [https://rexrank.ai/explore/submission_guideline_ct.html](https://rexrank.ai/explore/submission_guideline_ct.html) | Retrieved: `2026-07-22T00:13:43Z` | Raw source: [`snapshots/2026-07-22T001343Z/raw/rexrank.ai/explore/submission_guideline_ct.html`](snapshots/2026-07-22T001343Z/raw/rexrank.ai/explore/submission_guideline_ct.html) | SHA-256: `771a84f15ea9696f59c653637541d1de96abde138992cbeaa48945090bbd1e7e`
> **Archive note:** Interactive controls do not function in Markdown; the raw HTML remains authoritative for page behavior.

---
- [ReXrankCT](https://rexrank.ai/explore/submission_guideline_ct.html#)
  - [ReXGroundingCT](https://rexrank.ai/ReXGroundingCT/index.html)
- [ReX-MLE](https://rexrank.ai/ReX-MLE/index.html)


 [ReXrank](https://rexrank.ai/)


 🏆 **ReXGroundingCT Challenge @ MICCAI 2026** — Registration is open! [Learn More →](https://rexrank.ai/ReXGroundingCT/challenge.html)


# ReXrankCT Submission Instructions


---


## How to Submit


To submit your model's predictions for evaluation on the ReXrankCT leaderboard, please follow these steps:


1. Run inference on the test set from the [ReXGroundingCT dataset](https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT). Your model's predictions should be saved as individual files with the same names as the original CT scan files.
2. Each prediction file should have the shape `(F, H, W, D)`, where `F` is the number of findings for that scan The shape should match the ground truth dimensions.
3. Compress all your prediction files into a single archive (e.g., .zip or .gz file).
4. Compose an email with the subject line: **"ReXrankCT Submission: [Your Model Name]"**.
5. Attach the compressed prediction file to the email.
6. In the body of the email, please include:
  - Your model's name.
  - A brief description of your model.
  - A link to the paper or code repository, if available.
  - The name of your institution.
7. Send the email to: **Mohammed Baharoon** [mohammed_baharoon@hms.harvard.edu](mailto:mohammed_baharoon@hms.harvard.edu)


We will evaluate your submission and add the results to the leaderboard. Thank you for your contribution to advancing medical imaging research!


---


## Removing Your Models from the Leaderboard


If you wish to have your model removed from the leaderboard, please send an email to Mohammed Baharoon with the subject line "ReXrankCT Leaderboard Removal Request" and include your model's name in the email body.
