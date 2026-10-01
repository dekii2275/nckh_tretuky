To: mohammad.kassir@utt.fr
Subject: Reproduction details for your ICCA 2025 BiLSTM attention paper

Dear Mohammad Kassir and co-authors,

We are attempting to reproduce “A BiLSTM with Attention Approach Using 2D
Skeletons for Autism Activity Recognition” (DOI:
10.1109/ICCA66035.2025.11431081). We have implemented the architecture in
Table II and the published training hyperparameters. Could you share the
original preprocessing/training code, or clarify the following details?

1. How were the 60-frame sequences constructed: uniform resampling of each
   clip, consecutive windows, or overlapping sliding windows? What were the
   stride, frame rate, short-sequence and trailing-frame policies? Were
   windows restricted to individual clips/person tracks?
2. What exactly defines a session ID? Could you provide the train/validation
   clip or session lists and random seed? Was splitting performed before
   constructing sequences?
3. Which clips were included in the final ten-class experiment? The paper
   describes 1,315 clips and eleven activities, while the model has ten
   outputs. Our local OpenPose release has 1,374 non-empty clip directories,
   including 343 Drumming clips, and no non-empty Frog Pose clips.
4. How did you select the person in multi-person OpenPose frames? Did you use
   one tracked child, the first detected person, or multiple people as samples?
5. How were low-confidence/missing joints and zero or very small shoulder
   distances handled? Was normalization performed independently per frame?
6. Were the reported validation accuracy and confusion matrix calculated per
   clip, per sequence/window, per frame, or aggregated over multiple runs?
   Could you provide the number of validation samples per class and the raw
   confusion-matrix counts?
7. Which metric controlled early stopping and checkpoint selection? What
   were the LSTM/attention dropout settings, optimizer weight decay, software
   versions, and any augmentation or additional preprocessing?

With one uniformly resampled sequence per clip and subject-session grouping,
our current validation accuracy is 67.84%. We want to resolve protocol
differences before making any performance comparison. A minimal notebook or
the data-loading and split-generation functions would be very helpful.

Thank you for your time and for your work.

---

Local draft only; no message has been sent. The recipient is listed on the
paper and on the author's official UTT page:
https://recherche.utt.fr/research-directory/mohammad-kassir
