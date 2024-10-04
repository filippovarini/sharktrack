# SharkTrack
A software that detects sharks and rays in underwater videos and computes MaxN 21 times faster, with machine learning.

## Guide Structure

Here we provide a guide to learn about SharkTrack. The guide is divided in the following 3 chapters. We suggest to follow the order. It should take less than 15m.

| Chapter | Description
|--|--|
| [Introduction to SharkTrack](./readme.md#introduction-to-sharktrack) | This page is for users who are just considering the use of AI in their workflow, and aren't even sure yet whether SharkTrack would be useful. It summarizes what we do with the model, how it can help you and how other people use it | 
| [SharkTrack User Guide](./sharktrack-user-guide.md) | How to download and run SharkTrack on your data. Jump to it if you're already familiar with SharkTrack and you have data it can analyse.
| [Annotation Pipeline](./annotation-pipelines.md) | Once you have run the model, this page illustrates how to convert the output into species-specific MaxN metrics.

[![watrch video](static/video_screenshot.png)](https://drive.google.com/file/d/1b_74wdPXyJPe2P-m1c45jjsV2C5Itr-R/view?usp=sharing)
*Click on the image above to watch a demo*


# Introduction to SharkTrack

## Overview
Shark and Ray (Elasmobranch) researchers monitor their populations using Baited Remote Underwater Video Systems (BRUVS). This is a time-consuming process, as each video needs to be manually annotated. 

SharkTrack is a Machine Learning model that uses computer vision to detect and track Elasmobranchii in BRUVS videos and compute the MaxN metrics, used by ecologists.

> Therefore SharkTrack is an AI-enhanced workflow to convert raw BRUVS videos to MaxN, which has been tested 21x faster than traditional methods.

## How is it useful
- 🏃‍♀️ Computes MaxN semi-automatically, 21x faster
- 🏝 Can detect any elasmobranch species (as a single 'elasmobranch' class) in any location with 89% accuracy
- 👨‍💻 Can run on a standard laptop - no experience or advanced tech requirement


## How does SharkTrack work?
SharkTrack analyses BRUVS in two steps, as showed below:
![](static/figure1.png)
*The SharkTrack pipeline to compute MaxN, divided into Step 1 (top) and Step 2 (bottom)*

##### Step 1: Automatic Processing
- (a) Ingests all underwater videos in a hard drive or folder
- (b) Automatically detects elasmobranchs 
- (c) Save sightings in a CSV
- (d) Save a screenshot for each detected elasmobranch with video
![](static/example_detection.jpg)
*Each detection image shows the video and time it was captured* 

##### Step 2: Manual Review
- (e) Classify the species of detected elasmobranchs by renaming the relative screenshot filename.
- (f) SharkTrack updates all sightings of detected elasmobranchs with the new species classification
- (g) And outputs the species-specific MaxN

## How it can help you?
- 👀 **Peek Mode** After a day of sampling, run it on your laptop overnight and automatically detect where and when sharks and rays appeared in the videos.
- 🔎 **Analyst Mode** Accurately analyse the footage to derive relative abundance with the MaxN metrics.

Both modes can be run on a standard laptop and do not require WIFI.

Overall, SharkTrack is the most helpful to process videos where elasmobranch appears rarely, as it extracts them from the mainly empty footage, removeing the need to watch it manually.


## How can I run SharkTrack?
SharkTrack is a publicly-available model and you can install and run it by following [this guide](./sharktrack-user-guide.md).

If you don't have experience with Python, we know it can be daunting but don't fear! By following the guide step-by-step you will have SharkTrack up and running on your videos in less than 10 minutes.

Reach out if have any quesions [here]((mailto:fppvrn@gmail.com?subject=SharkTrackHelp))


## Where has SharkTrack being used?
- [Anguilla by University of Exeter and Anguilla National Trust](https://www.linkedin.com/posts/filippo-varini_we-are-back-from-university-of-exeter-activity-7167899292593065985-dZLo?utm_source=share&utm_medium=member_desktop)
- [Revillagigedo Archipelago, Mexico by Pelagios Kakunja](https://youtu.be/NeBcpscTc3M?si=BfyYM4jQ0-NDCKZZ)
- La Paz, Mexico by Pelagios Kakunja
- Northern Gulf of California, Mexico by Pelagios Kakunja
- Maldives, by the University of Leeds
- Cabo Verde, by Dr. Francesco Garzon and Adam Whiting of the University of Exeter
- Red Sea, by Dr. Francesco Garzon, University of Exeter
- Cape Cod, by the Virginia Tech University
- Hawaii, by the Virginia Tech University
- Azores, by the Okeanos Institute of Marine Sciences

## Contributors
This software and related work was supported by the efforts of Filippo Varini, Joel H. Gayford, Jeremy Jenrette, Matthew J. Witt, Francesco Garzon, Francesco Ferretti, Sophie Wilday, Mark E. Bond, Michael R. Heithaus, Danielle Robinson, Devon Carter, Najee Gumbs, Vincent Webster, Ben Glocker, Fabio De Sousa Ribeiro, Rajat Rasal, Orlando Timmerman, Natalie Ng, Rui Wen Lim, Michael Sellgren, Lara Tse, Steven Chen, Maria Pia Donrelas, Manfredi Minervini, Xuen Bei (Bay) Chin, Adam Whiting, Aurora Crocini, Gabriele Bai, Stephanie Guerinfor.

## Collaborations
This repository is licensed with the [MIT license](https://opensource.org/license/mit). If you use SharkTrack, please cite:

Varini, F. et al (2024). SharkTrack. GitHub. Available at
https://github.com/filippovarini/sharktrack


The same citation, in BibTex format:

```BibTex
@article{varini2024sharktrack,
  title={SharkTrack},
  author={Filippo Varini et al},
  year={2024}
}
```

> We update the citation to reference the paper once it has been pubblished.
### Issues
Please submit any issue on [GitHub](https://github.com/filippovarini/sharktrack/issues). We aim to respond to it within a week.
### Contribution
This project welcomes contributions, as pull requests, issues, or suggestions by [email](mailto:fppvrn@gmail.com?subject=SharkTrackContribution).

This is the first step of a broader effort to develop generalisable marine species classifiers. We are looking for contributors for this project. If you want to get involved in AI-driven Ocean Conservation please email us.


## Next Stpes
Run the model on your BRUVS by following [this guide](./sharktrack-user-guide.md)
