# from sacrerouge.metrics import Rouge

# summary = "Dan walked to the bakery this morning."
# reference = "Dan went to buy scones earlier this morning."

# rouge = Rouge(max_ngram=2)
# rouge.score(summary, [reference])


# from sacrebleu.metrics import BLEU, CHRF

# refs = [  # First set of references
#     ["The dog bit the man.", "It was not unexpected.", "The man bit him first."],
#     # Second set of references
#     [
#         "The dog had bit the man.",
#         "No one was surprised.",
#         "The man had bitten the dog.",
#     ],
# ]
# sys = ["The dog bit the man.", "It wasn't surprising.", "The man had just bitten him."]
# bleu = BLEU()
# score = bleu.corpus_score(sys, refs)
# print(score)
# # BLEU = 48.53 82.4/50.0/45.5/37.5 (BP = 0.943 ratio = 0.944 hyp_len = 17 ref_len = 18)
# print(type(score))

# print(bleu.get_signature())
# # nrefs:2|case:mixed|eff:no|tok:13a|smooth:exp|version:2.0.0
# chrf = CHRF()
# print(chrf.corpus_score(sys, refs))
# # chrF2 = 59.73

import evaluate

predictions = [
    """
                Progress:
                Evaluation of QA models on datasets are done, and the online learning pipeline (gym environment) is also finished. 

                Findings:
                QA language models can perform well on out-of-distribution datasets, meaning that they can answer questions from other QA databases than their training sets. 
                This is very different from computer vision models, as these models cannot classify items outside of the pre-defined classes. 
                Therefore in the previous experiment, it is very straight forward which data point (e.g. cifar100 image) should be offloaded to which set of models (e.g. cifar100 classifiers). For the QA task, this is not the case, which leads to the following problem.

                Problems:
                I used a model trained on the datasets I am testing on, so it is outperforming other models a lot. Consequently, bandit algorithms tend to learn to only pull that arm, despite the context information. 
                Therefore, I have adjusted the settings by removing this model and injecting a new dataset on which no models are trained.
                I expect to see that contextual bandit algorithms outperform non-contextual ones and lose to the optimal benchmark. 
                """
]
references = [
    [
        """
Progress:
The evaluation of Question Answering (QA) models on various datasets has been completed, alongside the development of an online learning pipeline within a gym environment.

Findings:
QA language models exhibit strong performance even when faced with out-of-distribution datasets, demonstrating their ability to answer questions from QA databases not included in their training. In contrast, computer vision models struggle with items outside their predefined classes. This disparity was evident in an earlier experiment, where clear associations existed between data points (e.g., cifar100 images) and designated model sets (e.g., cifar100 classifiers). However, the QA task lacks such straightforward associations, leading to a distinct challenge.

Problems:
The use of a model trained on the testing datasets has resulted in its significant outperformance compared to other models. This phenomenon has biased bandit algorithms to consistently select this high-performing option, regardless of contextual information. To address this, the approach has been adjusted by removing the overly trained model and introducing a new dataset devoid of any model training. This alteration aims to showcase the superior performance of contextual bandit algorithms over non-contextual ones, while still falling short of the optimal benchmark. Anticipated results include improved contextual bandit algorithm performance and a concession to the optimal benchmark's superiority.
"""
    ],
]
sacrebleu = evaluate.load("sacrebleu")
results = sacrebleu.compute(predictions=predictions, references=references)
print(results)
print(list(results.keys()))
print(round(results["score"], 1))
