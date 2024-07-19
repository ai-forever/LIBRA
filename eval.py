import os
import argparse
import json
import numpy as np

from evaluation import dataset_loader, model_loader, answer_generator, metrics


def name_to_metric(name):
    dct = {
        "em": metrics.exact_match_score,
        "count_score": metrics.count_score,
        "f1": metrics.qa_f1_score,
    }
    return dct[name]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="Enter predictions path", required=True)
    args = parser.parse_args()
    
    predictions = json.load(open(args.path, "r"))
    datasets_params = json.load(open("configs/datasets_config.json", "r"))
    results = {}
    for dataset in predictions.keys():
        results[dataset] = {}
        metric_calculation = name_to_metric(datasets_params[dataset]["metric"])
        for prediction in predictions[dataset]:
            length = prediction["length"]
            if length not in results[dataset]:
                results[dataset][length] = []
            
            positive_scores = []
            for positive_output in prediction["positive_outputs"]:
                score = metric_calculation(str(prediction["model_answer"]), str(positive_output))
                positive_scores.append(score)
            score = max(positive_scores)
            if prediction["negative_outputs"]:
                negatives_count = 0
                for negative_output in prediction["negative_outputs"]:
                    if negative_output in prediction["model_answer"]:
                        negatives_count += 1
                
                if negatives_count == len(prediction["negative_outputs"]):
                    score = 0.0
                else:
                    score -= (1.0/len(prediction["negative_outputs"])) * negatives_count
                    if score < 0:
                        score = 0.0
            results[dataset][length].append(score)
            
    total_score = []
    for dataset in results.keys():
        dataset_score = []
        for length in datasets_params[dataset]["lengths"]:
            if length in results[dataset].keys():
                results[dataset][length] = np.mean(results[dataset][length])
            else:
                results[dataset][length] = 0
            dataset_score.append(results[dataset][length])
        results[dataset]["dataset_total_score"] = np.mean(dataset_score)
        total_score.append(results[dataset]["dataset_total_score"])
    results["total_score"] = np.mean(total_score)
        
    print(results)
    save_path = "results/" + args.path.split("/")[1]
    if not os.path.exists(save_path.split("/")[0]):
        os.makedirs(save_path.split("/")[0])
    with open(save_path, "w") as outfile:
        json.dump(results, outfile)
    print(f"evaluations were saved here: {save_path}.json")
