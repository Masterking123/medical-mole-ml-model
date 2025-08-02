import numpy as np
import pandas as pd
import torch
from torch import nn
import argparse

class SymptomClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, input):
        return self.model(input)

def load_data_and_prepare():
    disease_data = pd.read_csv("data/DiseaseAndSymptoms.csv")

    diseases = disease_data.Disease.unique().tolist()
    symptoms_set = set()

    for col in disease_data.columns[1:]:
        col_values = disease_data[col].tolist()
        for val in col_values:
            if pd.notna(val):
                stripped_val = val.strip()
                symptoms_set.add(stripped_val)

    symptoms = list(symptoms_set)
    symptoms.sort()

    disease_indexes = {diseases[i]: i for i in range(len(diseases))}
    symptom_indexes = {symptoms[i]: i for i in range(len(symptoms))}
    
    return diseases, symptoms, disease_indexes, symptom_indexes

def predict_disease(symptom_list, model, symptom_index_map, disease_list):
    input_vector = np.zeros(len(symptom_index_map), dtype=np.float32)

    for symptom in symptom_list:
        if symptom in symptom_index_map:
            idx = symptom_index_map[symptom]
            input_vector[idx] = 1.0
        else:
            print(f"Warning: '{symptom}' not in symptom list")

    input_tensor = torch.tensor(input_vector).unsqueeze(0)

    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        predicted_index = torch.argmax(output, dim=1).item()

    return disease_list[predicted_index]

def main():
    parser = argparse.ArgumentParser(description="Predict disease based on symptoms")
    parser.add_argument('symptoms', nargs='+', help="List of symptoms to diagnose")

    args = parser.parse_args()
    symptom_input = args.symptoms

    diseases, symptoms, disease_indexes, symptom_indexes = load_data_and_prepare()

    model = SymptomClassifier(input_size=len(symptoms), num_classes=len(diseases))
    model.load_state_dict(torch.load("symptom_model_weights.pth", map_location=torch.device('cpu')))
    model.eval()

    predicted_disease = predict_disease(symptom_input, model, symptom_indexes, diseases)
    print(f"Predicted disease: {predicted_disease}")

if __name__ == "__main__":
    main()
