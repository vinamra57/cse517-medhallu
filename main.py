import pandas as pd
from datasets import load_dataset
from llm import LLM
import os

#Hyperparams to tune
BATCH_SIZE = 9000
#Different model used for now
MODEL_NAME = "openai/gpt-oss-20b"

def generate_hallucinated_answer(question: str, knowledge: str, ground_truth: str, llm_model: LLM) -> str:
    prompt = f"#Question#: {question}\n\n#Knowledge#: {knowledge}\n\n#Ground truth answer#: {ground_truth}\n\n#Hallucinated Answer#:"
    
    #For now, we only get the response and print it. Code for quality control is required.
    response = llm_model.get_hallucinated_response(prompt)
    return response

def main():
    #Load the data locally, or if it isn't there, from hugging face
    if os.path.exists("medqa_data_artificial.csv") and os.path.exists("medqa_data_labeled.csv"):
        df_artificial = pd.read_csv("medqa_data_artificial.csv")
        df_labeled = pd.read_csv("medqa_data_labeled.csv")
        print("Loaded data from local machine")
    else:
        dataset_artificial = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split=f"train[:{BATCH_SIZE}]")
        df_artificial = dataset_artificial.to_pandas()

        dataset_labeled = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        df_labeled = dataset_labeled.to_pandas()

        print("Loaded data from web")

        df_artificial.to_csv("medqa_data_artificial.csv", index=False)
        df_labeled.to_csv("medqa_data_labeled.csv", index=False)
        print("Saved data locally")

    #Only one response for now.
    question = df_artificial['question'][0]
    knowledge = df_artificial['context'][0]
    ground_truth = df_artificial['long_answer'][0]
    model = LLM(MODEL_NAME)

    result = generate_hallucinated_answer(question, knowledge, ground_truth, model)
    print(f"Question: {question}")
    print(f"True Answer: {ground_truth}")
    print(f"Hallucinated Answer: {result}")

if __name__ == "__main__":
    main()
