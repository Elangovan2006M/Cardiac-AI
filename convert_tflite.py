"""
TensorFlow Lite Micro Conversion & INT8 Quantization Script
Converts trained Keras model into INT8 Quantized .tflite format
and generates C Header array `main/model_data.h` for ESP32-S3 deployment.
"""

import numpy as np
import tensorflow as tf
import os


def representative_dataset_gen():
    """Generates representative Log-Mel features for INT8 quantization calibration."""
    from train_autoencoder import generate_synthetic_pcg_dataset
    X_normal, _ = generate_synthetic_pcg_dataset(num_normal=200)
    for sample in X_normal:
        yield [np.expand_dims(sample, axis=0).astype(np.float32)]


def convert_and_quantize(keras_model_path="models/conv_autoencoder.h5", output_tflite_path="models/model_int8.tflite"):
    print(f"Loading Keras model from {keras_model_path}...")
    model = tf.keras.models.load_model(keras_model_path)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    
    # Enforce INT8 Input and Output Tensors
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_quant_model = converter.convert()
    
    os.makedirs(os.path.dirname(output_tflite_path), exist_ok=True)
    with open(output_tflite_path, "wb") as f:
        f.write(tflite_quant_model)
        
    print(f"Successfully saved INT8 Quantized model to {output_tflite_path}")
    print(f"Model size: {len(tflite_quant_model) / 1024.0:.2f} KB")
    
    # Generate C Byte Array Header File
    generate_c_header(tflite_quant_model, header_path="main/model_data.h")


def generate_c_header(tflite_bytes, header_path="main/model_data.h"):
    """Converts binary TFLite array to C header array format."""
    os.makedirs(os.path.dirname(header_path), exist_ok=True)
    
    with open(header_path, "w") as f:
        f.write("// Auto-generated TensorFlow Lite Micro Model for ESP32-S3\n")
        f.write("// Quantization: Full INT8\n\n")
        f.write("#ifndef MODEL_DATA_H\n")
        f.write("#define MODEL_DATA_H\n\n")
        f.write("#include <cstdint>\n\n")
        f.write(f"constexpr unsigned int g_model_data_len = {len(tflite_bytes)};\n")
        f.write("alignas(16) const unsigned char g_model_data[] = {\n    ")
        
        hex_bytes = [f"0x{b:02x}" for b in tflite_bytes]
        for i in range(0, len(hex_bytes), 12):
            f.write(", ".join(hex_bytes[i:i+12]))
            if i + 12 < len(hex_bytes):
                f.write(",\n    ")
                
        f.write("\n};\n\n")
        f.write("#endif // MODEL_DATA_H\n")
        
    print(f"Successfully generated C header file at {header_path}")


if __name__ == "__main__":
    if not os.path.exists("models/conv_autoencoder.h5"):
        print("Model file not found. Running training pipeline first...")
        from train_autoencoder import train_and_evaluate
        train_and_evaluate()
        
    convert_and_quantize()
