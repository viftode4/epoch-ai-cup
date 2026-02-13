Overview
Welcome! This Kaggle competition hosts the AI Cup 2026 – Performance Track, hosted my Team Epoch. The performance track is used to evaluate the performance of the AI models developed as part of the entire AI cup. Thus, the leaderboard is not indicative for the entire AI Cup. More information can be found on the website.

The goal of this competition is to develop a model that classifies bird species based on radar track data. This data was gathered at windfarm Eemshaven in Groningen, originally for a different purpose.

Wind turbines are an important source of clean energy, but they can also pose a risk to birds. One promising mitigation strategy is to temporarily shut down turbines during periods of bird migration or when high-impact species are detected nearby. Radar offers a cost-effective, long-range solution for detecting, studying, and monitoring birds in flight. By accurately classifying the bird groups that pass through windfarms, monitoring systems can make better-informed decisions. This improves operational efficiency of windfarms, whilst reducing bird strikes.

Start

28 minutes ago
Close
a month to go
Timeline
February 13th at 10:00 CET: Start of the Kick-off day
February 13th at 14:00 CET: Start of the challenge
March 14th: Validation Deadline. Validate eligibility through the validation form
March 19th: Deadline for the performance track submissions
March 21st: Nomination day and private leaderboard reveal
April 14th: AIC4NL Congress
All deadlines are at 11:59 PM Central European Time (CET), unless stated otherwise.

Evaluation
Evaluation Metric
Submissions are evaluated using Mean (macro-averaged) Average Precision (mAP) over all nine classes. The final score is the arithmetic mean of the Average Precision (AP) calculated for each of the 9 columns (Clutter + 8 Bird Species).

The final score is calculated as:


Where 
 is the Average Precision for class 
. The nine classes are:

Clutter
Cormorants
Pigeons
Ducks
Geese
Gulls
Birds of Prey
Waders
Songbirds
Note: For Average Precision we use the SK-learn implementation.

A notebook with the implementation of the metric can be found in the following link.

Submission Format
Submissions in this competition must be done by uploading a single CSV or Parquet file. Each row corresponds to a unique radar track_id. The required columns are:

track_id: The unique identifier for the track.
9 Class Columns: A predicted probability (float between 0.0 and 1.0) for each of the 9 classes listed above.
For an exact template, refer to sample_submission.csv in the Data section.

Credits and Acknowledgments
Creators and Organiser: Team Epoch


Head Sponsor/Enabler: The AI Coalition for the Netherlands (AIC4NL)


Dataset and Challenge Partner: The Netherlands Organisation for Applied Scientific Research (TNO)


Venue, Networking and Logistic Partners: Mondai


Dream Team Sponsorship: TU Delft


About Team Epoch
Team Epoch is one of the seven Dreamteams of the Technical University of Delft. Team Epoch is a competitive machine learning team, who mainly concerns itself with the exploration of AI and ML on humanitarian, societal and environmental issues. Furthermore, the team actively strives to stimulate interest in AI within the wider student community. More information can be found here.

Citation
Team Epoch. AI Cup 2026 | Performance Track. https://kaggle.com/competitions/ai-cup-2026-performance, 2026. Kaggle.