import os
import json
import glob
from tqdm import tqdm
import argparse

def run(args):
    retries = args.retries
    success_rate = args.success_rate

    tasks = args.tasks.split(",")  # Split tasks if passed as a comma-separated string
    results_folder = args.results_folder
    model_prefix = args.model_prefix

    all_avgs = {}

    for i in tqdm(range(retries + 1)):
        for task_name in tasks:
            print(f"Processing task {task_name}")
            files = sorted(glob.glob(f"{os.path.join(results_folder, task_name, model_prefix)}*"), reverse=True)
            for file_path in files: 
                display_name = file_path.split('/')[-1].split('.')[0]
                
                print(f'Processing model {display_name}')
                if display_name not in all_avgs:
                    all_avgs[display_name] = {t : [] for t in tasks}
                avgs = all_avgs[display_name]
                
                results = []
                with open(file_path, 'r') as f:
                    for line in f.readlines():
                        data = json.loads(line)
                        results.append(data)
                

                current_attempt_sum = 0
                counter = 0
                for r in results:
                    counter += 1
                    scores = r['success']
                    if success_rate:
                        scores = [1 if s == 1 else 0 for s in r['success']]
                    if i >= len(scores):
                        score = scores[-1]
                    else:
                        score = scores[i]
                        
                    current_attempt_sum += score
                
                current_attempt_avg = current_attempt_sum / counter
                avgs[task_name].append(current_attempt_avg)
            


    for display_name, avgs in all_avgs.items():     
        overall_avgs = []
        for i in range(retries + 1):
            attempt_avgs = []
            for task, avg in avgs.items():
                attempt_avgs.append(avg[i])
            overall_avgs.append(sum(attempt_avgs)/len(attempt_avgs))
            
        all_avgs[display_name]["overall_avg"] = overall_avgs
    

    all_avgs = sorted(all_avgs.items(), key=lambda x: x[1]["overall_avg"][-1], reverse=True)
    for key, value in all_avgs:
        print(f"{key}: {value}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retries", default=9, type=int, help="Number of retries")
    parser.add_argument("--success_rate", default=True, type=bool, help="Whether to calculate success rate as binary")
    parser.add_argument("--tasks", default="common_gen,style_transfer,num_planning,sentiment,story_generation,rationale_generation,paraphrase,clustering,panagram,program,mbpp,cmg_hard", 
                        type=str, help="Comma-separated list of tasks")
    parser.add_argument("--results_folder", default="./llama_3_results/test", type=str, help="Folder containing results")
    parser.add_argument("--model_prefix", default="Meta-Llama-3-8B-Instruct", type=str, help="Model prefix for results files")


    args = parser.parse_args()
    run(args)