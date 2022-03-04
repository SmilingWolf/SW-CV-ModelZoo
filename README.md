# SW-CV-ModelZoo
Repo for my Tensorflow/Keras CV experiments. Mostly revolving around the Danbooru20xx dataset

---

Framework: TF/Keras 2.7

Training SQLite DB built using fire-egg's tools: https://github.com/fire-eggs/Danbooru2019

Currently training on Danbooru2021, 512px SFW subset (sans the rating:q images that had been included in the 2022-01-21 release of the dataset)

## Reference:
Anonymous, The Danbooru Community, & Gwern Branwen; “Danbooru2021: A Large-Scale Crowdsourced and Tagged Anime Illustration Dataset”, 2022-01-21. Web. Accessed 2022-01-28 https://www.gwern.net/Danbooru2021

----

## Journal
**04/03/2022**:  
It is done! As the commit says, the first round of experiments on NFNets is finally over.

W&B reports this whole thing took about 112 days of compute time for the main runs, plus 11 days for experiments on higher image resolutions, and likely some more days on top accounting for failed experiments.  
All of the experiments were run on a fleet of TPUv2 and TPUv3 VMs offered for free by the TRC program. Thanks!

Few takeaways, going over the data:
- HSwish is a decent replacement for SiLU. In fact, based on the [b1092 datasheet](https://github.com/SmilingWolf/SW-CV-ModelZoo/blob/ffe897ae9946b474fb811c7378cce132d3a6833e/results/db2021_0000_0899_0950_0999_b0000.csv), which focuses on performance on minority classes, it can provide about 1% better metrics wrt. ReLU;
- contrary to [arXiv:2102.06171](https://arxiv.org/abs/2102.06171) I find SiLU to consistently and considerably boost network performance wrt. their ReLU counterparts. The most extreme example is how I haven't managed, despite a few different tries w/ smaller learning rates, to train a L0 ReLU network with MixUp = 0.3 without it becoming unstable and ultimately collapsing, while the HSwish and SiLU variants trained without an itch;
- increasing MixUp gives somewhat mixed results, I'm not too sure it is the path forward for better validation results on this dataset with those networks. I'd sooner explore different dropout/stochastic depth parameters.

Now a word about the validation dataset:  
- it has not been checked for duplicates against the training data
- images from danbooru2018 and back used black padding and alpha background, while images from danbooru2019+ use white. I think, with no proof outside of a few small tests, this may be skewing some classes to look for a black background or borders instead of paying attention to the actual details.

This means the validation metrics are likely overoptimistic.  
The first, obvious step would be to take the original images modulo 0900-0949, which I have reserved for this exact scenario, remove duplicates and "plausibly similar" images (say, maybe using embeddings or a very low threshold with a perceptual hash), preprocess them in a more uniform way, and use the output as a final test dataset.

**18/02/2022**:  
So far I'm incredibly pleased with the results of adding ECA to Lx+SiLU networks.  
At the meager cost of ~120 more parameters (give or take, depending on network depth) it is extremely effective at increasing network capacity.

On the other hand, I seem to have hit a wall with {L2,L1}+SiLU+ECA.  
They overfit by epoch ~70 and ~85 out of 100, respectively.  
I tried increasing MixUp to 0.3. This slowed down overfitting, but even then, by the end of training, the checkpoints with the best validation loss didn't display improved metrics over their MixUp 0.2 counterparts.  

Something that DID work was finetuning while increasing the image size from 320 to 384.  
I also tried a handful of learning rate schedules. In the end the one that worked best was starting off with max_learning_rate (0.1), no warmup, and letting cosine annealing do its thing over the course of 10 epochs.

**06/02/2022**:  
Great news crew! TRC allowed me to use a bunch of TPUs!

To make better use of this amount of compute I had to overhaul a number of components, so a bunch of things are likely to have fallen to bitrot in the process.  
I can only guarantee NFNet can work pretty much as before with the right arguments.  
NFResNet changes *should* have left it retrocompatible with the previous version.  
ResNet has been streamlined to be mostly in line with the Bag-of-Tricks paper ([arXiv:1812.01187](https://arxiv.org/abs/1812.01187)) with the exception of the stem. It is not compatible with the previous version of the code.

The training labels have been included in the 2021_0000_0899 folder for convenience.  
The list of files used for training is going to be uploaded as a GitHub Release.

Now for some numbers:  
compared to my previous best run, the one that resulted in [NFNetL1V1-100-0.57141](https://github.com/SmilingWolf/SW-CV-ModelZoo/releases/tag/NFNetL1V1-100-0.57141):
- I'm using 1.86x the amount of images: 2.8M vs 1.5M
- I'm training bigger models: 61M vs 45M params
- ... in less time: 232 vs 700 hours of processor time
- don't get me started on actual wall clock time
- with a few amenities thrown in: ECA for channel attention, SiLU activation

And it's all thanks to the folks at TRC, so shout out to them!

I currently have a few runs in progress across a couple of dimensions:
- effect of model size with NFNet L0/L1/L2, with SiLU and ECA for all three of them
- effect of activation function with NFNet L0, with SiLU/HSwish/ReLU, no ECA

Once the experiments are over, the plan is to select the network definitions that lay on the Pareto curve between throughput and F1 score and release the trained weights.

One last thing.  
I'd like to call your attention to the tools/cleanlab_stuff.py script.  
It reads two files: one with the binarized labels from the database, the other with the predicted probabilities.  
It then uses the [cleanlab](https://github.com/cleanlab/cleanlab) package to estimate whether if an image in a set could be missing a given label. At the end it stores its conclusions in a json file.  
This file could, potentially, be used in some tool to assist human intervention to add the missing tags.
