import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import your existing converter
from src.model_converter.converter import ModelConverter

# Import the new modules
from dataloader import create_dataloader
from inference_engines import ONNXInferenceEngine, TRTInferenceEngine, TFInferenceEngine

# --- Configuration ---
DATA_DIR = "data/test"
CALIBRATION_DIR = "data/calibration" # Use a subset of data for calibration
WEIGHTS_DIR = "weights"
MODELS_DIR = "models_generated"
RESULTS_FILE = "evaluation_results.csv"

# Batch size for evaluation (matches opt_batch for TRT)
BATCH_SIZE = 32
# Max batch size for TRT engine build
MAX_BATCH_SIZE = 32


def create_all_models(converter: ModelConverter):
    """
    Uses the ModelConverter to generate all required model variants.
    """
    print("="*50)
    print("Starting Model Generation Pipeline")
    print("="*50)
    
    base_models = ['DenseNet121', 'DenseNet201']
    
    for model_name in base_models:
        print(f"\n--- Processing {model_name} ---")
        model_out_dir = os.path.join(MODELS_DIR, model_name)
        weights_path = os.path.join(WEIGHTS_DIR, model_name, 'raw_weights.h5')
        
        if not os.path.exists(weights_path):
            print(f"WARNING: Weights file not found, skipping {model_name}: {weights_path}")
            continue
            
        os.makedirs(model_out_dir, exist_ok=True)
        
        # Define paths
        onnx_fp32_path = os.path.join(model_out_dir, f"{model_name}_fp32.onnx")
        onnx_fp16_path = os.path.join(model_out_dir, f"{model_name}_fp16.onnx")
        trt_fp32_path = os.path.join(model_out_dir, f"{model_name}_fp32.trt")
        trt_fp16_path = os.path.join(model_out_dir, f"{model_name}_fp16.trt")
        trt_int8_path = os.path.join(model_out_dir, f"{model_name}_int8.trt")
        int8_cache_path = os.path.join(model_out_dir, f"{model_name}_int8.cache")
        
        # --- 1. & 2. ONNX FP32 and FP16 ---
        try:
            if not os.path.exists(onnx_fp32_path):
                print(f"Creating ONNX FP32: {onnx_fp32_path}")
                converter.tf_to_onnx(
                    input_model=weights_path,
                    output_path=onnx_fp32_path,
                    precision='fp32',
                    only_weigths_of_model=model_name
                )
            else:
                print(f"Skipping, ONNX FP32 already exists: {onnx_fp32_path}")

            if not os.path.exists(onnx_fp16_path):
                print(f"Creating ONNX FP16: {onnx_fp16_path}")
                converter.tf_to_onnx(
                    input_model=weights_path,
                    output_path=onnx_fp16_path,
                    precision='fp16',
                    only_weigths_of_model=model_name
                )
            else:
                print(f"Skipping, ONNX FP16 already exists: {onnx_fp16_path}")

        except Exception as e:
            print(f"Error during ONNX conversion for {model_name}: {e}")
            continue # Skip to next model
            
        # --- 3. TRT FP32 ---
        try:
            if not os.path.exists(trt_fp32_path):
                print(f"Creating TRT FP32: {trt_fp32_path}")
                converter.onnx_to_trt(
                    input_model=onnx_fp32_path,
                    engine_file_path=trt_fp32_path,
                    precision='fp32',
                    opt_batch=BATCH_SIZE,
                    max_batch=MAX_BATCH_SIZE
                )
            else:
                print(f"Skipping, TRT FP32 already exists: {trt_fp32_path}")
        except Exception as e:
            print(f"Error building TRT FP32 for {model_name}: {e}")

        # --- 4. TRT FP16 ---
        try:
            if not os.path.exists(trt_fp16_path):
                print(f"Creating TRT FP16: {trt_fp16_path}")
                converter.onnx_to_trt(
                    input_model=onnx_fp32_path, # Use FP32 ONNX as base
                    engine_file_path=trt_fp16_path,
                    precision='fp16',
                    opt_batch=BATCH_SIZE,
                    max_batch=MAX_BATCH_SIZE
                )
            else:
                print(f"Skipping, TRT FP16 already exists: {trt_fp16_path}")
        except Exception as e:
            print(f"Error building TRT FP16 for {model_name}: {e}")

        # --- 5. TRT INT8 ---
        try:
            if not os.path.exists(trt_int8_path):
                print(f"Creating TRT INT8: {trt_int8_path}")
                if not os.path.exists(CALIBRATION_DIR):
                    print(f"WARNING: Calibration dir not found, skipping INT8: {CALIBRATION_DIR}")
                    continue
                converter.onnx_to_trt(
                    input_model=onnx_fp32_path, # Use FP32 ONNX as base
                    engine_file_path=trt_int8_path,
                    precision='int8',
                    calibration_images=CALIBRATION_DIR,
                    calibration_cache=int8_cache_path,
                    opt_batch=BATCH_SIZE,
                    max_batch=MAX_BATCH_SIZE
                )
            else:
                print(f"Skipping, TRT INT8 already exists: {trt_int8_path}")
        except Exception as e:
            print(f"Error building TRT INT8 for {model_name}: {e}")
            
    print("="*50)
    print("Model Generation Complete")
    print("="*50)


def run_evaluation():
    """
    Runs inference on all generated models and saves metrics to a CSV.
    """
    print("\n" + "="*50)
    print("Starting Model Evaluation Pipeline")
    print("="*50)

    # 1. Define all models to be evaluated
    models_to_evaluate = []
    for model_name in ['DenseNet121', 'DenseNet201']:
        model_dir = os.path.join(MODELS_DIR, model_name)
        weights_path = os.path.join(WEIGHTS_DIR, model_name, 'raw_weights.h5')
        
        if not os.path.exists(weights_path):
            print(f"Skipping {model_name}, weights not found at {weights_path}")
            continue

        # --- ADDED TF BASELINE MODEL ---
        models_to_evaluate.append(
            {'name': f'{model_name}_TF_FP32', 'type': 'tf', 'path': weights_path, 'base_model': model_name}
        )
        # ---------------------------------

        models_to_evaluate.extend([
            {'name': f'{model_name}_ONNX_FP32', 'type': 'onnx', 'path': os.path.join(model_dir, f"{model_name}_fp32.onnx"), 'base_model': model_name},
            {'name': f'{model_name}_ONNX_FP16', 'type': 'onnx', 'path': os.path.join(model_dir, f"{model_name}_fp16.onnx"), 'base_model': model_name},
            {'name': f'{model_name}_TRT_FP32', 'type': 'trt', 'path': os.path.join(model_dir, f"{model_name}_fp32.trt"), 'base_model': model_name},
            {'name': f'{model_name}_TRT_FP16', 'type': 'trt', 'path': os.path.join(model_dir, f"{model_name}_fp16.trt"), 'base_model': model_name},
            {'name': f'{model_name}_TRT_INT8', 'type': 'trt', 'path': os.path.join(model_dir, f"{model_name}_int8.trt"), 'base_model': model_name},
        ])

    # 2. Create the efficient data loader
    print(f"Loading test data from: {DATA_DIR}")
    try:
        # NOTE: Make sure to exclude the 'bothcells' folder
        test_loader = create_dataloader(DATA_DIR, BATCH_SIZE, exclude_folders=['bothcells'])
        total_images = len(test_loader.dataset)
        print(f"Successfully loaded {total_images} test images.")
    except Exception as e:
        print(f"Failed to create data loader: {e}")
        return

    all_results = []
    
    # --- LOAD EXISTING RESULTS TO APPEND ---
    if os.path.exists(RESULTS_FILE):
        try:
            all_results = pd.read_csv(RESULTS_FILE).to_dict('records')
            print(f"Loaded {len(all_results)} existing results from {RESULTS_FILE}")
        except Exception as e:
            print(f"Could not read existing results file, starting fresh: {e}")
            all_results = []
    
    # --- Track models already evaluated ---
    evaluated_models = {res['model_name'] for res in all_results}

    # 3. Loop through each model, run inference, and calculate metrics
    for model_info in models_to_evaluate:
        model_path = model_info['path']
        
        if model_info['name'] in evaluated_models:
            print(f"\nSkipping {model_info['name']} (already in results.csv)")
            continue

        if not os.path.exists(model_path):
            print(f"\nSkipping {model_info['name']} (File not found: {model_path})")
            continue
            
        print(f"\n--- Evaluating {model_info['name']} ---")
        
        try:
            # 3a. Load the inference engine
            
            # --- ADDED LOGIC FOR TF ENGINE ---
            if model_info['type'] == 'tf':
                engine = TFInferenceEngine(model_info['base_model'], model_path)
            # ---------------------------------
            elif model_info['type'] == 'onnx':
                engine = ONNXInferenceEngine(model_path)
            elif model_info['type'] == 'trt':
                engine = TRTInferenceEngine(model_path, max_batch_size=MAX_BATCH_SIZE)
            else:
                print(f"Unknown model type: {model_info['type']}")
                continue
                
        except Exception as e:
            print(f"Failed to load engine for {model_info['name']}: {e}")
            continue

        all_preds = []
        all_labels = []
        total_inference_time = 0.0

        # 3b. Run inference loop
        for images, labels in tqdm(test_loader, desc=f"Inferencing {model_info['name']}"):
            # Dataloader gives torch.Tensor, engine expects NumPy
            images_np = images.numpy()
            
            start_time = time.perf_counter()
            logits = engine(images_np)
            end_time = time.perf_counter()
            
            total_inference_time += (end_time - start_time)
            
            # Get predicted classes
            preds = np.argmax(logits, axis=1)
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

        del engine # Free up GPU memory
        
        # 3c. Calculate metrics
        try:
            accuracy = accuracy_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
            avg_inference_time_ms = (total_inference_time / total_images) * 1000

            print(f"Results for {model_info['name']}:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1-Score (Weighted): {f1:.4f}")
            print(f"  Precision (Weighted): {precision:.4f}")
            print(f"  Recall (Weighted): {recall:.4f}")
            print(f"  Total Inference Time: {total_inference_time:.2f} s")
            print(f"  Avg. Time per Image: {avg_inference_time_ms:.4f} ms")

            # 3d. Store results
            all_results.append({
                "model_name": model_info['name'],
                "base_model": model_info['base_model'],
                "quant_type": "_".join(model_info['name'].split('_')[1:]),
                "accuracy": accuracy,
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "total_time_s": total_inference_time,
                "avg_time_ms": avg_inference_time_ms
            })
            
        except Exception as e:
            print(f"Error calculating metrics for {model_info['name']}: {e}")

    # 4. Save all results to CSV
    if all_results:
        results_df = pd.DataFrame(all_results)
        # --- Remove duplicates ---
        results_df = results_df.drop_duplicates(subset=['model_name'], keep='last')
        # --- Sort by speed ---
        results_df = results_df.sort_values(by="avg_time_ms", ascending=True)
        
        results_df.to_csv(RESULTS_FILE, index=False)
        print("\n" + "="*50)
        print(f"Evaluation complete. Results saved to {RESULTS_FILE}")
        print("="*50)
        print(results_df.to_string())
    else:
        print("\nNo models were evaluated. Check configuration and file paths.")


if __name__ == "__main__":
    # Step 1: Create the converter instance
    converter = ModelConverter()
    
    # Step 2: Run the model generation pipeline
    # --- COMMENTED OUT - We already ran this ---
    create_all_models(converter)
    
    # Step 3: Run the evaluation pipeline on the generated models
    print("Skipping model creation, running evaluation only...")
    run_evaluation()