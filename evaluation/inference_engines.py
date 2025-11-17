import onnxruntime as ort
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Manages context creation/destruction
import numpy as np
import time
import tensorflow as tf

# We need this to reconstruct the model
from src.model_converter.utils import make_model 

# ==================================================================
#                  NEW TENSORFLOW INFERENCE ENGINE
# ==================================================================

class TFInferenceEngine:
    """Inference engine for the original TensorFlow/Keras .h5 model."""
    
    def __init__(self, base_model_name: str, weights_path: str):
        print(f"Loading TensorFlow model: {base_model_name}")
        print(f"Loading weights from: {weights_path}")
        
        # Check for GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Restrict TensorFlow to only use the first GPU
                tf.config.set_visible_devices(gpus[0], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[0], True)
                print(f"TensorFlow will use GPU: {gpus[0]}")
            except RuntimeError as e:
                print(e)
        else:
            print("WARNING: No GPU found for TensorFlow. Running on CPU.")

        # Reconstruct the model structure
        self.model = make_model(num_classes=3, base_model=base_model_name)
        
        # Load the weights
        self.model.load_weights(weights_path)
        print("TensorFlow model loaded and weights set.")

    def __call__(self, batch_data: np.ndarray) -> np.ndarray:
        """
        Runs inference on a batch of data.
        
        Args:
            batch_data: NumPy array of shape (N, C, H, W) and dtype float32.
        
        Returns:
            NumPy array of model logits.
        """
        
        # --- CRITICAL STEP ---
        # The dataloader provides NCHW format [Batch, Channel, Height, Width]
        # Keras models expect NHWC format [Batch, Height, Width, Channel]
        # We must transpose the data before prediction.
        
        # Transpose from (N, C, H, W) to (N, H, W, C)
        batch_data_nhwc = np.transpose(batch_data, (0, 2, 3, 1))
        
        # Run prediction
        # verbose=0 stops it from printing progress bars
        return self.model.predict(batch_data_nhwc, verbose=0)


# ==================================================================
#                  UNCHANGED ONNX ENGINE
# ==================================================================

class ONNXInferenceEngine:
    """Inference engine for ONNX models using onnxruntime-gpu."""
    
    def __init__(self, model_path: str):
        print(f"Loading ONNX model from: {model_path}")
        try:
            # Use CUDAExecutionProvider
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kSameAsRequested',
                }),
                'CPUExecutionProvider'
            ]
            self.session = ort.InferenceSession(model_path, providers=providers)
            
            # Get input details
            input_details = self.session.get_inputs()[0]
            self.input_name = input_details.name
            
            # Check expected input type and store it
            if input_details.type == 'tensor(float16)':
                self.input_dtype = np.float16
                print("Model expects FP16 input.")
            else:
                self.input_dtype = np.float32
                print("Model expects FP32 input.")
                
            self.output_name = self.session.get_outputs()[0].name
            print(f"ONNX model loaded. Input: '{self.input_name}', Output: '{self.output_name}'")
            
        except Exception as e:
            print(f"Failed to load ONNX model: {e}")
            raise
            
    def __call__(self, batch_data: np.ndarray) -> np.ndarray:
        """
        Runs inference on a batch of data.
        """
        # Ensure batch_data type matches model's expected input type
        if batch_data.dtype != self.input_dtype:
            batch_data = batch_data.astype(self.input_dtype)
            
        return self.session.run([self.output_name], {self.input_name: batch_data})[0]


# ==================================================================
#                  UNCHANGED TRT ENGINE
# ==================================================================

class TRTInferenceEngine:
    """
    Inference engine for TensorRT (.trt) engines using the modern I/O API.
    """
    
    def __init__(self, engine_path: str, max_batch_size: int = 32):
        print(f"Loading TensorRT engine from: {engine_path}")
        self.max_batch_size = max_batch_size
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # Load and deserialize the engine
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine.")
            
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context.")

        self.stream = cuda.Stream()
        
        # --- Modern API for I/O and Buffer Allocation ---
        self.input_names = []
        self.output_names = []
        self.host_mems = {}      # Stores pagelocked host buffers (NumPy)
        self.device_mems = {}    # Stores device buffers (pycuda.DeviceAllocation)
        self.binding_addrs = {}  # Stores device buffer pointers (int)

        print("--- Allocating TRT Buffers (Modern API) ---")
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)
            
            # Handle dynamic batch size (-1)
            if shape[0] == -1:
                shape = (self.max_batch_size,) + shape[1:]
                print(f"  Binding '{name}' (dynamic): {shape} | {dtype}")
            else:
                print(f"  Binding '{name}' (static): {shape} | {dtype}")

            # Allocate page-locked host memory
            host_mem = cuda.pagelocked_empty(trt.volume(shape), dtype)
            
            # Allocate device memory
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Store buffers and pointers
            self.binding_addrs[name] = int(device_mem)
            self.host_mems[name] = host_mem
            self.device_mems[name] = device_mem
            
            # Classify as input or output
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        if not self.input_names or not self.output_names:
            raise ValueError("Could not find I/O tensors in the TRT engine.")
            
        self.input_name = self.input_names[0]
        self.output_name = self.output_names[0]
        
        print(f"TRT engine loaded. Input: '{self.input_name}', Output: '{self.output_name}'")
        print("---------------------------------------------")


    def __call__(self, batch_data: np.ndarray) -> np.ndarray:
        """
        Runs inference on a batch of data.
        """
        
        if not batch_data.flags['C_CONTIGUOUS'] or batch_data.dtype != np.float32:
            batch_data = np.ascontiguousarray(batch_data, dtype=np.float32)
            
        # 1. Set input shape
        self.context.set_input_shape(self.input_name, batch_data.shape)
        
        # 2. Get a view of the host buffer and copy data into it
        destination_view = self.host_mems[self.input_name][:batch_data.size]
        destination_view.shape = batch_data.shape 
        np.copyto(destination_view, batch_data)
        
        # 3. Copy data from page-locked (Host) to Device (GPU) asynchronously
        cuda.memcpy_htod_async(
            self.device_mems[self.input_name],  # dest: DeviceAllocation
            destination_view,                    # src: numpy.ndarray
            self.stream                          # stream: Stream
        )
        
        # 4. Set tensor addresses for execution
        for name in self.input_names + self.output_names:
            self.context.set_tensor_address(name, self.binding_addrs[name])
        
        # 5. Run inference asynchronously
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # 6. Copy data from Device (GPU) to page-locked (Host) asynchronously
        output_shape = self.context.get_tensor_shape(self.output_name)
        output_size = trt.volume(output_shape)
        
        output_view = self.host_mems[self.output_name][:output_size]
        output_view.shape = output_shape
        
        cuda.memcpy_dtoh_async(
            output_view,                         # dest: numpy.ndarray
            self.device_mems[self.output_name],  # src: DeviceAllocation
            self.stream                          # stream: Stream
        )
        
        # 7. Synchronize the stream to wait for all async ops to complete
        self.stream.synchronize()
        
        # 8. Return a copy of the output view
        return output_view.copy()


    def __del__(self):
        """Free CUDA memory."""
        if hasattr(self, 'device_mems'):
            for mem in self.device_mems.values():
                mem.free()
        if hasattr(self, 'TFInferenceEngine'):
            print("TF resources will be freed by TF/Keras.")
        else:
            print("TRTInferenceEngine resources freed.")