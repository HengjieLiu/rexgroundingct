<!-- Archive metadata in this block is not source content. -->
> **Archive metadata:** Source: [https://arxiv.org/abs/2507.22030](https://arxiv.org/abs/2507.22030) | Retrieved: `2026-07-22T00:13:43Z` | Raw source: [`snapshots/2026-07-22T001343Z/raw/arxiv.org/abs/2507.22030.html`](snapshots/2026-07-22T001343Z/raw/arxiv.org/abs/2507.22030.html) | SHA-256: `2feb8dc0e8c0ddf40802ad86bc0c65c1cbfb9cb95aabb7b3f6c7ca7550b3823e`
> **Archive note:** Interactive controls do not function in Markdown; the raw HTML remains authoritative for page behavior.

---
# Electrical Engineering and Systems Science > Image and Video Processing


 **arXiv:2507.22030** (eess)


 [Submitted on 29 Jul 2025 ([v1](https://arxiv.org/abs/2507.22030v1)), last revised 27 Oct 2025 (this version, v2)]


# Title:ReXGroundingCT: A 3D Chest CT Dataset for Segmentation of Findings from Free-Text Reports


Authors:[Mohammed Baharoon](https://arxiv.org/search/eess?searchtype=author&query=Baharoon,+M), [Luyang Luo](https://arxiv.org/search/eess?searchtype=author&query=Luo,+L), [Michael Moritz](https://arxiv.org/search/eess?searchtype=author&query=Moritz,+M), [Abhinav Kumar](https://arxiv.org/search/eess?searchtype=author&query=Kumar,+A), [Sung Eun Kim](https://arxiv.org/search/eess?searchtype=author&query=Kim,+S+E), [Xiaoman Zhang](https://arxiv.org/search/eess?searchtype=author&query=Zhang,+X), [Miao Zhu](https://arxiv.org/search/eess?searchtype=author&query=Zhu,+M), [Mahmoud Hussain Alabbad](https://arxiv.org/search/eess?searchtype=author&query=Alabbad,+M+H), [Maha Sbayel Alhazmi](https://arxiv.org/search/eess?searchtype=author&query=Alhazmi,+M+S), [Neel P. Mistry](https://arxiv.org/search/eess?searchtype=author&query=Mistry,+N+P), [Lucas Bijnens](https://arxiv.org/search/eess?searchtype=author&query=Bijnens,+L), [Kent Ryan Kleinschmidt](https://arxiv.org/search/eess?searchtype=author&query=Kleinschmidt,+K+R), [Brady Chrisler](https://arxiv.org/search/eess?searchtype=author&query=Chrisler,+B), [Sathvik Suryadevara](https://arxiv.org/search/eess?searchtype=author&query=Suryadevara,+S), [Sri Sai Dinesh Jaliparthi](https://arxiv.org/search/eess?searchtype=author&query=Jaliparthi,+S+S+D), [Noah Michael Prudlo](https://arxiv.org/search/eess?searchtype=author&query=Prudlo,+N+M), [Mark David Marino](https://arxiv.org/search/eess?searchtype=author&query=Marino,+M+D), [Jeremy Palacio](https://arxiv.org/search/eess?searchtype=author&query=Palacio,+J), [Rithvik Akula](https://arxiv.org/search/eess?searchtype=author&query=Akula,+R), [Di Zhou](https://arxiv.org/search/eess?searchtype=author&query=Zhou,+D), [Hong-Yu Zhou](https://arxiv.org/search/eess?searchtype=author&query=Zhou,+H), [Ibrahim Ethem Hamamci](https://arxiv.org/search/eess?searchtype=author&query=Hamamci,+I+E), [Scott J. Adams](https://arxiv.org/search/eess?searchtype=author&query=Adams,+S+J), [Hassan Rayhan AlOmaish](https://arxiv.org/search/eess?searchtype=author&query=AlOmaish,+H+R), [Pranav Rajpurkar](https://arxiv.org/search/eess?searchtype=author&query=Rajpurkar,+P)

View a PDF of the paper titled ReXGroundingCT: A 3D Chest CT Dataset for Segmentation of Findings from Free-Text Reports, by Mohammed Baharoon and 24 other authors
 [View PDF](https://arxiv.org/pdf/2507.22030) [HTML (experimental)](https://arxiv.org/html/2507.22030v2)

> Abstract:We introduce ReXGroundingCT, the first publicly available dataset linking free-text findings to pixel-level 3D segmentations in chest CT scans. The dataset includes 3,142 non-contrast chest CT scans paired with standardized radiology reports from CT-RATE. Construction followed a structured three-stage pipeline. First, GPT-4 was used to extract and standardize findings, descriptors, and metadata from reports originally written in Turkish and machine-translated into English. Second, GPT-4o-mini categorized each finding into a hierarchical ontology of lung and pleural abnormalities. Third, 3D annotations were produced for all CT volumes: the training set was quality-assured by board-certified radiologists, and the validation and test sets were fully annotated by board-certified radiologists. Additionally, a complementary chain-of-thought dataset was created to provide step-by-step hierarchical anatomical reasoning for localizing findings within the CT volume, using GPT-4o and localization coordinates derived from organ segmentation models. ReXGroundingCT contains 16,301 annotated entities across 8,028 text-to-3D-segmentation pairs, covering diverse radiological patterns from 3,142 non-contrast CT scans. About 79% of findings are focal abnormalities and 21% are non-focal. The dataset includes a public validation set of 50 cases and a private test set of 100 cases, both annotated by board-certified radiologists. The dataset establishes a foundation for enabling free-text finding segmentation and grounded radiology report generation in CT imaging. Model performance on the private test set is hosted on a public leaderboard at [this https URL](https://rexrank.ai/ReXGroundingCT). The dataset is available at [this https URL](https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT).


| Subjects: | Image and Video Processing (eess.IV); Artificial Intelligence (cs.AI); Computer Vision and Pattern Recognition (cs.CV) |
| --- | --- |
| Cite as: | [arXiv:2507.22030](https://arxiv.org/abs/2507.22030) [eess.IV] |
| -- | (or [arXiv:2507.22030v2](https://arxiv.org/abs/2507.22030v2) [eess.IV] for this version) |
| -- | [https://doi.org/10.48550/arXiv.2507.22030](https://doi.org/10.48550/arXiv.2507.22030)<br>[Button: Focus to learn more]<br>arXiv-issued DOI via DataCite |


## Submission history

 From: Mohammed Baharoon [[view email](https://arxiv.org/show-email/9149b86e/2507.22030)]
 **[[v1]](https://arxiv.org/abs/2507.22030v1)** Tue, 29 Jul 2025 17:27:15 UTC (1,500 KB)
 **[v2]** Mon, 27 Oct 2025 17:51:47 UTC (1,704 KB)


 [](https://arxiv.org/abs/2507.22030) Full-text links:

## Access Paper:


- [View PDF](https://arxiv.org/pdf/2507.22030)
- [HTML (experimental)](https://arxiv.org/html/2507.22030v2)
- [TeX Source](https://arxiv.org/src/2507.22030)


[![license icon](https://arxiv.org/icons/licenses/by-nc-sa-4.0.png)](http://creativecommons.org/licenses/by-nc-sa/4.0/)


### Current browse context:


eess.IV

  [< prev](https://arxiv.org/prevnext?id=2507.22030&function=prev&context=eess.IV)   |   [next >](https://arxiv.org/prevnext?id=2507.22030&function=next&context=eess.IV)


 [new](https://arxiv.org/list/eess.IV/new)  |  [recent](https://arxiv.org/list/eess.IV/recent)  | [2025-07](https://arxiv.org/list/eess.IV/2025-07)

 Change to browse by:
 [cs](https://arxiv.org/abs/2507.22030?context=cs)
 [cs.AI](https://arxiv.org/abs/2507.22030?context=cs.AI)
 [cs.CV](https://arxiv.org/abs/2507.22030?context=cs.CV)
 [eess](https://arxiv.org/abs/2507.22030?context=eess)


### References & Citations


- [NASA ADS](https://ui.adsabs.harvard.edu/abs/arXiv:2507.22030)
- [Google Scholar](https://scholar.google.com/scholar_lookup?arxiv_id=2507.22030)
- [Semantic Scholar](https://api.semanticscholar.org/arXiv:2507.22030)


 [Button: export BibTeX citation] Loading...


## BibTeX formatted citation

 [Button: ×]

 [Textarea: loading the citation]

 Data provided by:  [](https://arxiv.org/abs/2507.22030)


### Bookmark


[![BibSonomy](https://arxiv.org/static/browse/0.3.4/images/icons/social/bibsonomy.png)](http://www.bibsonomy.org/BibtexHandler?requTask=upload&url=https://arxiv.org/abs/2507.22030&description=ReXGroundingCT: A 3D Chest CT Dataset for Segmentation of Findings from Free-Text Reports) [![Reddit](https://arxiv.org/static/browse/0.3.4/images/icons/social/reddit.png)](https://reddit.com/submit?url=https://arxiv.org/abs/2507.22030&title=ReXGroundingCT: A 3D Chest CT Dataset for Segmentation of Findings from Free-Text Reports)


[Input: tabs] Bibliographic Tools


# Bibliographic and Citation Tools


  [Input: checkbox]  Bibliographic Explorer Toggle

 Bibliographic Explorer *([What is the Explorer?](https://info.arxiv.org/labs/showcase.html#arxiv-bibliographic-explorer))*


  [Input: checkbox]  Connected Papers Toggle

 Connected Papers *([What is Connected Papers?](https://www.connectedpapers.com/about))*


  [Input: checkbox]  Litmaps Toggle

 Litmaps *([What is Litmaps?](https://www.litmaps.co/))*


  [Input: checkbox]  scite.ai Toggle

 scite Smart Citations *([What are Smart Citations?](https://www.scite.ai/))*


 [Input: tabs] Code, Data, Media


# Code, Data and Media Associated with this Article


  [Input: checkbox]  alphaXiv Toggle

 alphaXiv *([What is alphaXiv?](https://alphaxiv.org/))*


  [Input: checkbox]  Links to Code Toggle

 CatalyzeX Code Finder for Papers *([What is CatalyzeX?](https://www.catalyzex.com))*


  [Input: checkbox]  DagsHub Toggle

 DagsHub *([What is DagsHub?](https://dagshub.com/))*


  [Input: checkbox]  GotitPub Toggle

 Gotit.pub *([What is GotitPub?](http://gotit.pub/faq))*


  [Input: checkbox]  Huggingface Toggle

 Hugging Face *([What is Huggingface?](https://huggingface.co/huggingface))*


  [Input: checkbox]  ScienceCast Toggle

 ScienceCast *([What is ScienceCast?](https://sciencecast.org/welcome))*


 [Input: tabs] Demos


# Demos


  [Input: checkbox]  Replicate Toggle

 Replicate *([What is Replicate?](https://replicate.com/docs/arxiv/about))*


  [Input: checkbox]  Spaces Toggle

 Hugging Face Spaces *([What is Spaces?](https://huggingface.co/docs/hub/spaces))*


  [Input: checkbox]  Spaces Toggle

 TXYZ.AI *([What is TXYZ.AI?](https://txyz.ai))*


 [Input: tabs] Related Papers


# Recommenders and Search Tools


  [Input: checkbox]  Link to Influence Flower

 Influence Flower *([What are Influence Flowers?](https://influencemap.cmlab.dev/))*


  [Input: checkbox]  Core recommender toggle

 CORE Recommender *([What is CORE?](https://core.ac.uk/services/recommender))*


- [Author](https://arxiv.org/abs/2507.22030)
- [Venue](https://arxiv.org/abs/2507.22030)
- [Institution](https://arxiv.org/abs/2507.22030)
- [Topic](https://arxiv.org/abs/2507.22030)


 [Input: tabs]  About arXivLabs


# arXivLabs: experimental projects with community collaborators


arXivLabs is a framework that allows collaborators to develop and share new arXiv features directly on our website.


Both individuals and organizations that work with arXivLabs have embraced and accepted our values of openness, community, excellence, and user data privacy. arXiv is committed to these values and only works with partners that adhere to them.


Have an idea for a project that will add value for arXiv's community? [**Learn more about arXivLabs**](https://info.arxiv.org/labs/index.html).


 [Which authors of this paper are endorsers?](https://arxiv.org/auth/show-endorsers/2507.22030) | Disable MathJax ([What is MathJax?](https://info.arxiv.org/help/mathjax.html))
