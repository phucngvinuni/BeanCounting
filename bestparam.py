import cv2
import numpy as np
import random
import time
from tqdm import tqdm # Thư viện hiển thị thanh tiến trình (cài đặt: pip install tqdm)

# --- Configuration ---
IMAGE_PATH = 'Pic_3.jpg'        # Đường dẫn đến ảnh
TARGET_COUNT = 830              # Số lượng hạt đậu mục tiêu (ground truth)
NUM_ITERATIONS = 5000            # << SỐ LẦN THỬ NGẪU NHIÊN (Tăng để tìm kỹ hơn, giảm để chạy nhanh hơn)

# --- Parameter Ranges for Random Search ---
# Định nghĩa phạm vi hợp lý cho từng tham số bạn muốn tinh chỉnh.
# Dựa trên các giá trị tốt nhất bạn đã tìm thấy thủ công.
PARAM_RANGES = {
    # HSV Thresholds (Mở rộng nhẹ quanh giá trị tốt nhất [0, 58, 160] - [21, 191, 255])
    'h_low': (0, 5),           # Giữ gần 0
    'h_high': (18, 25),         # Quanh 21
    's_low': (50, 70),          # Quanh 58
    's_high': (180, 200),       # Quanh 191
    'v_low': (150, 175),        # Quanh 160
    'v_high': (245, 255),       # Giữ gần 255
    # Morphology (Kernel phải là số lẻ)
    'kernel_size': [3], # Chỉ thử kernel 3x3 (1x1 không hiệu quả, 5x5 có thể quá mạnh)
    'closing_iter': [0, 1],     # Thử không closing hoặc 1 lần
    'opening_iter': [1, 2],     # Thử 1 hoặc 2 lần opening
    # Watershed
    'dist_thresh_ratio': (0.3, 0.6) # Phạm vi quan trọng cần khám phá
}

# --- Evaluation Function ---
# Hàm này nhận một bộ tham số, chạy pipeline và trả về số lượng hạt đếm được.
def evaluate_parameters(params, img_bgr):
    """Runs the image processing pipeline with given parameters and returns the count."""
    try:
        # 1. Convert to HSV
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # 2. Get HSV bounds
        lower_bound = np.array([params['h_low'], params['s_low'], params['v_low']])
        upper_bound = np.array([params['h_high'], params['s_high'], params['v_high']])

        # 3. Create Mask
        mask_hsv = cv2.inRange(hsv, lower_bound, upper_bound)

        # 4. Morphology
        k_size = params['kernel_size']
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

        mask_processed = mask_hsv # Start with the original mask

        # Closing
        if params['closing_iter'] > 0:
            mask_processed = cv2.morphologyEx(mask_processed, cv2.MORPH_CLOSE, kernel, iterations=params['closing_iter'])

        # Opening
        if params['opening_iter'] > 0:
            mask_processed = cv2.morphologyEx(mask_processed, cv2.MORPH_OPEN, kernel, iterations=params['opening_iter'])

        # Check if mask is all black after morphology (avoid errors later)
        if cv2.countNonZero(mask_processed) == 0:
            return 0 # No beans found if mask is empty

        # 4.5 Watershed Prep
        sure_bg = cv2.dilate(mask_processed, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(mask_processed, cv2.DIST_L2, 5)

        # Check if dist_transform max is zero (can happen with empty/tiny mask)
        max_dist = dist_transform.max()
        if max_dist == 0:
            return 0 # Cannot threshold if max distance is 0

        ret, sure_fg = cv2.threshold(dist_transform, params['dist_thresh_ratio'] * max_dist, 255, 0)
        sure_fg = np.uint8(sure_fg)

        # Check if sure_fg is all black
        if cv2.countNonZero(sure_fg) == 0:
             # If no sure_fg, maybe count contours on mask_processed as fallback? Or return 0.
             # Let's return 0 for simplicity, watershed needs sure_fg.
             return 0

        unknown = cv2.subtract(sure_bg, sure_fg)
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        # 5. Watershed
        markers = cv2.watershed(img_bgr, markers)

        # 6. Count
        if markers.max() > 1:
            bean_count = markers.max() - 1
        else:
            bean_count = 0

        return bean_count

    except Exception as e:
        # print(f"Error during evaluation: {e}") # Optional: for debugging
        return -1 # Return an invalid count on error

# --- Main Tuning Loop ---
print(f"Bắt đầu Random Search với {NUM_ITERATIONS} lần thử...")
print(f"Mục tiêu số lượng: {TARGET_COUNT}")

# Load image once
img_original = cv2.imread(IMAGE_PATH)
if img_original is None:
    print(f"Lỗi: Không thể tải ảnh '{IMAGE_PATH}'")
    exit()

best_params = None
best_count = -1
# Lower score is better (closer to target)
# Initialize with infinity so the first valid result becomes the best
min_score = float('inf')

start_time = time.time()

# Use tqdm for progress bar
for i in tqdm(range(NUM_ITERATIONS), desc="Searching Parameters"):
    # --- Sample random parameters ---
    params = {}
    params['h_low'] = random.randint(PARAM_RANGES['h_low'][0], PARAM_RANGES['h_low'][1])
    params['h_high'] = random.randint(PARAM_RANGES['h_high'][0], PARAM_RANGES['h_high'][1])
    params['s_low'] = random.randint(PARAM_RANGES['s_low'][0], PARAM_RANGES['s_low'][1])
    params['s_high'] = random.randint(PARAM_RANGES['s_high'][0], PARAM_RANGES['s_high'][1])
    params['v_low'] = random.randint(PARAM_RANGES['v_low'][0], PARAM_RANGES['v_low'][1])
    params['v_high'] = random.randint(PARAM_RANGES['v_high'][0], PARAM_RANGES['v_high'][1])
    params['kernel_size'] = random.choice(PARAM_RANGES['kernel_size'])
    params['closing_iter'] = random.choice(PARAM_RANGES['closing_iter'])
    params['opening_iter'] = random.choice(PARAM_RANGES['opening_iter'])
    params['dist_thresh_ratio'] = random.uniform(PARAM_RANGES['dist_thresh_ratio'][0], PARAM_RANGES['dist_thresh_ratio'][1])

    # --- Evaluate the sampled parameters ---
    current_count = evaluate_parameters(params, img_original)

    # --- Calculate score and update best if necessary ---
    if current_count >= 0: # Check for valid count (not -1 from error)
        # Score: absolute difference from the target count
        score = abs(current_count - TARGET_COUNT)

        if score < min_score:
            min_score = score
            best_count = current_count
            best_params = params
            # Optional: Print improvement immediately
            # print(f"\nNew best found! Count: {best_count} (Score: {min_score}), Params: {best_params}")
        # Optional: Keep track of params that achieve exactly the target
        # if current_count == TARGET_COUNT:
        #    print(f"\nTarget count {TARGET_COUNT} achieved! Params: {params}")
        #    # You might want to store these params and continue searching or stop early

end_time = time.time()
print(f"\nHoàn thành tìm kiếm sau {end_time - start_time:.2f} giây.")

# --- Print Results ---
if best_params:
    print("\n=====================================")
    print(" Bộ tham số tốt nhất tìm thấy:")
    print(f"  - Số lượng đếm được: {best_count}")
    print(f"  - Độ lệch so với mục tiêu ({TARGET_COUNT}): {abs(best_count - TARGET_COUNT)} (Score: {min_score})")
    print("  - Chi tiết tham số:")
    for key, value in best_params.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.4f}") # Format float
        else:
            print(f"    {key}: {value}")
    print("=====================================")
    print("Lưu ý: Chạy lại có thể cho kết quả hơi khác do tính ngẫu nhiên.")
    print("Bạn có thể cần chạy với NUM_ITERATIONS lớn hơn hoặc điều chỉnh PARAM_RANGES để cải thiện.")
else:
    print("\nKhông tìm thấy bộ tham số hợp lệ nào.")
