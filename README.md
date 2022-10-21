# SW-CV-ModelZoo
Repo for my Tensorflow/Keras CV experiments. Mostly revolving around the Danbooru20xx dataset

---

Framework: TF/Keras 2.10

Training SQLite DB built using fire-egg's tools: https://github.com/fire-eggs/Danbooru2019

Currently training on Danbooru2021, 512px SFW subset (sans the rating:q images that had been included in the 2022-01-21 release of the dataset)

## Reference:
Anonymous, The Danbooru Community, & Gwern Branwen; “Danbooru2021: A Large-Scale Crowdsourced and Tagged Anime Illustration Dataset”, 2022-01-21. Web. Accessed 2022-01-28 https://www.gwern.net/Danbooru2021

----

## Journal

**21/10/2022**:  
New release today, after a lot of time, many failed experiments, some successes and a whopping final 200 epochs and 24 days of training.  
TPU time courtesy of [TRC](https://sites.research.google/trc/about/). Thanks!  
ViT in particular was a pain, turns out it is quite sensitive to learning rate. I'm still not 100% sure I nailed it, but right now it works(TM).  
While it might look like ConvNext does better, the whole story is a bit more interesting.  

Full results below:

All 5500 tags:
| run_name                         | definition_name   | params_human   |   image_size |   thres |     F1 |     F2 |
|:---------------------------------|:------------------|:---------------|-------------:|--------:|-------:|-------:|
| ensenble_october                 | -                 | -              |          448 |  0.3611 | 0.7044 | 0.7044 |
| ConvNextBV1_09_25_2022_05h13m55s | B                 | 93.2M          |          448 |  0.3673 | 0.6941 | 0.6941 |
| ViTB16_09_25_2022_04h53m38s      | B16               | 90.5M          |          448 |  0.3663 | 0.6918 | 0.6918 |

All tags, starting from #2380 and below (sorted by most to least popular):
| run_name                         | definition_name   | params_human   |   image_size |   thres |     F1 |     F2 |
|:---------------------------------|:------------------|:---------------|-------------:|--------:|-------:|-------:|
| ensenble_october                 | -                 | -              |          448 |  0.3611 | 0.6107 | 0.5588 |
| ConvNextBV1_09_25_2022_05h13m55s | B                 | 93.2M          |          448 |  0.3673 | 0.5932 | 0.5425 |
| ViTB16_09_25_2022_04h53m38s      | B16               | 90.5M          |          448 |  0.3663 | 0.5993 | 0.5529 |

General tags (category 0), no character or series tags:
| run_name                         | definition_name   | params_human   |   image_size |   thres |     F1 |     F2 |
|:---------------------------------|:------------------|:---------------|-------------:|--------:|-------:|-------:|
| ensenble_october                 | -                 | -              |          448 |  0.3618 | 0.6878 | 0.6878 |
| ConvNextBV1_09_25_2022_05h13m55s | B                 | 93.2M          |          448 |  0.3682 | 0.6774 | 0.6774 |
| ViTB16_09_25_2022_04h53m38s      | B16               | 90.5M          |          448 |  0.3672 | 0.6748 | 0.6748 |

General tags (category 0), no character or series tags, starting from #2000 and below (sorted by most to least popular):
| run_name                         | definition_name   | params_human   |   image_size |   thres |     F1 |     F2 |
|:---------------------------------|:------------------|:---------------|-------------:|--------:|-------:|-------:|
| ensenble_october                 | -                 | -              |          448 |  0.3618 | 0.4515 | 0.3976 |
| ConvNextBV1_09_25_2022_05h13m55s | B                 | 93.2M          |          448 |  0.3682 | 0.4320 | 0.3804 |
| ViTB16_09_25_2022_04h53m38s      | B16               | 90.5M          |          448 |  0.3672 | 0.4416 | 0.3936 |

The numbers are obtained using [tools/analyze_metrics.py](https://github.com/SmilingWolf/SW-CV-ModelZoo/blob/main/tools/analyze_metrics.py) to first find the point where P ≈ R, then using that threshold to check what scores I get on the less popular tags.  
ViT blazes past ConvNext when it comes to rarer tags, so you might want to consider that when choosing what model to use.  
Personally, I ensemble them if I don't have time constraints. That would be `ensenble_october` in the tables above. Quite some gains.

Next, I'll be finetuning at least one of these models on the latest tags, and adding NSFW images and tags to the training set, so that it can be used in tandem with Waifu Diffusion.

**21/04/2022**:  
Checkpointing sweeps and conclusions so far:
- NFNets: ECA and SiLU are well worth the extra computational cost. ECA is incredibly cheap on parameters side, too. Using both ECA and SiLU.
- NFNets: tested MixUp with alpha = 0.2 and alpha = 0.3, found no particular reason to use alpha 0.3. Using alpha = 0.2.
- ConvNext: focal loss: tested a few parameter combinations from the original paper, defaults are good. Using alpha = 0.25 gamma = 2.
- ConvNext: losses: tested a few parameter combinations. Best results achieved with ASL with gamma_neg = gamma_pos = clip = 0, which boils down to BCE with the sum of the per-class losses instead of the average. Using ASL with gamma_neg = gamma_pos = clip = 0.
- ConvNext: tested cutout_rate = 0.0, cutout_rate = 0.25, cutout_rate = 0.5. Even training for 300 epochs, neither cutout_rate > 0 run ever displayed any advantage against the cutout_rate = 0 run, both overall and at the single class level. Using cutout_rate = 0.0.

**05/04/2022**:  
So, I trained a bunch of ConvNexts in the past month.  
Learned a few things too:
- on the importance of the learning rate: even simply halving it might lower a model's performance to the point of making it worthless.  
While running the ASL sweep I had a few stability issues. Among other things, I tried lowering the learning rate to half the usual one. Turned out the issue was elsewhere. Fixed that, I forgot to turn the learning rate back up.  
Lo and behold, the Focal Loss run I was using as "control group" came out a whopping 4% lower in F1 w.r.t. the old one (ConvNextTV1_03_05_2022_15h56m42s). Never making that mistake again.
- the ASL sweep results speak for themselves: neither Focal Loss nor ASL are a good fit for either the current dataset, or the current training settings. I suspect the culprit might be the very aggressive MixUp. I'll spin up a few more test runs and report back.

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
