from inference_sdk import InferenceHTTPClient
import base64
import cv2
import numpy as np
import time
import random

# Roboflow inference setup
localInference = "http://localhost:9001"
roboflowInference = "https://detect.roboflow.com"
personalGPUInference = "http://192.168.0.55:9001"

# Roboflow inference setup
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="WgLYEMfa0WjwZWHJhwlO"
)

# Load and encode image
image_file = "botopia.jpg"
image_bgr = cv2.imread(image_file)

with open(image_file, "rb") as f:
    image_bytes = f.read()
image_base64 = base64.b64encode(image_bytes).decode("utf-8")

# Run the workflow and measure time
start_time = time.time()
result = client.run_workflow(
    workspace_name="visual-impaired-technology-ww0tt",
    workflow_id="custom-workflow-4",
    images={"image": image_base64},
    use_cache=True
)
inference_duration_ms = (time.time() - start_time) * 1000

# Access predictions
predictions = result[0]['model_predictions']['predictions']

# Draw polygons on image
overlay = image_bgr.copy()
for prediction in predictions:
    if "points" in prediction:
        # Convert list of {'x':..., 'y':...} to np.array
        points_np = np.array([[int(p["x"]), int(p["y"])] for p in prediction["points"]], dtype=np.int32)

        # Random color per polygon
        color = tuple(random.randint(100, 255) for _ in range(3))

        # Draw filled polygon and outline
        cv2.fillPoly(overlay, [points_np], color=color)
        cv2.polylines(image_bgr, [points_np], isClosed=True, color=color, thickness=2)

        # Optional: display class and confidence
        label = prediction.get("class", "object")
        confidence = prediction.get("confidence", 0)
        centroid = np.mean(points_np, axis=0).astype(int)
        cv2.putText(
            image_bgr, f"{label} ({confidence:.2f})", tuple(centroid),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, lineType=cv2.LINE_AA
        )

# Blend with transparency
alpha = 0.3
image_bgr = cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0)

# Overlay inference time
cv2.putText(
    image_bgr,
    f"Inference time: {inference_duration_ms:.1f} ms",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 0, 255),
    2,
    lineType=cv2.LINE_AA
)

# Display image
cv2.imshow("Roboflow Segmentation Overlay", image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
