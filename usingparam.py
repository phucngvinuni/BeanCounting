import cv2
import numpy as np
from matplotlib import pyplot as plt

# --- Tải ảnh ---
img_path = 'Pic_2.jpg' # Đảm bảo đúng tên file ảnh gốc
img_original = cv2.imread(img_path)

if img_original is None:
    print(f"Lỗi: Không thể tải ảnh từ đường dẫn '{img_path}'")
    exit()

img_display_watershed = img_original.copy() # Ảnh để vẽ kết quả watershed
print("Đã tải ảnh thành công.")

# --- Bước 1: Chuyển sang không gian màu HSV ---
print("Bước 1: Chuyển sang HSV...")
hsv = cv2.cvtColor(img_original, cv2.COLOR_BGR2HSV)

# --- Bước 2: Xác định dải màu HSV cho hạt đậu (SIÊU TỐI ƯU TỪ RANDOM SEARCH 5k) ---
print("Bước 2: Sử dụng ngưỡng HSV siêu tối ưu từ Random Search...")
lower_bound = np.array([5, 61, 173])   # Giá trị tối ưu tìm được (829 count)
upper_bound = np.array([18, 200, 251]) # Giá trị tối ưu tìm được (829 count)

# --- Bước 3: Tạo mặt nạ ---
print(f"Bước 3: Tạo mặt nạ với ngưỡng HSV: Low={lower_bound}, High={upper_bound}")
mask_hsv = cv2.inRange(hsv, lower_bound, upper_bound)

# --- Bước 4: Xử lý hậu kỳ mặt nạ (SIÊU TỐI ƯU TỪ RANDOM SEARCH 5k) ---
print("Bước 4: Xử lý mặt nạ (Kernel=3, Closing=0, Opening=1)...")
kernel_size = 3 # Giá trị tối ưu tìm được
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

# Closing (iterations = 0 nên bỏ qua)
closing_iterations = 0 # Giá trị tối ưu tìm được
mask_processed = mask_hsv # Bắt đầu với mask gốc vì không có closing

# Opening
opening_iterations = 1 # Giá trị tối ưu tìm được
if opening_iterations > 0:
    mask_processed = cv2.morphologyEx(mask_processed, cv2.MORPH_OPEN, kernel, iterations=opening_iterations)

# --- Bước 4.5: Chuẩn bị cho Watershed ---
print("Bước 4.5: Chuẩn bị markers cho Watershed...")

# a) Sure Background
sure_bg = cv2.dilate(mask_processed, kernel, iterations=3)

# b) Sure Foreground (Distance Transform)
dist_transform = cv2.distanceTransform(mask_processed, cv2.DIST_L2, 5)

# Phân ngưỡng distance transform (SIÊU TỐI ƯU TỪ RANDOM SEARCH 5k)
dist_threshold_ratio = 0.3769 # Giá trị tối ưu tìm được (829 count)
print(f"   Sử dụng dist_thresh_ratio = {dist_threshold_ratio:.4f}")
ret, sure_fg = cv2.threshold(dist_transform, dist_threshold_ratio * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)

# c) Unknown Region
unknown = cv2.subtract(sure_bg, sure_fg)

# d) Markers
# Kiểm tra xem sure_fg có trống không trước khi tạo marker
if cv2.countNonZero(sure_fg) == 0:
    print("CẢNH BÁO: Không tìm thấy vùng Sure Foreground nào! Kết quả sẽ là 0.")
    markers = np.zeros(mask_processed.shape, dtype=np.int32) # Tạo markers rỗng
    watershed_bean_count = 0
else:
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # --- Bước 5: Áp dụng thuật toán Watershed ---
    print("Bước 5: Áp dụng Watershed...")
    markers = cv2.watershed(img_original, markers)

    # --- Bước 6: Đếm kết quả từ Watershed ---
    if markers.max() > 1:
        watershed_bean_count = markers.max() - 1
    else:
        watershed_bean_count = 0

print(f"\n>>> Số lượng hạt đậu ước tính từ Watershed (Siêu Tối ưu): {watershed_bean_count}")

# --- Bước 7: Vẽ contours và hiển thị ---
print("Bước 7: Vẽ đường biên Watershed và hiển thị...")
# Chỉ vẽ nếu có kết quả watershed hợp lệ
if 'markers' in locals() and np.any(markers == -1): # Kiểm tra markers tồn tại và có đường biên
    img_display_watershed[markers == -1] = [0, 255, 0] # Màu xanh lá cây

plt.figure(figsize=(15, 10)) # Layout 2x3

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
plt.title('Ảnh Gốc')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(mask_processed, cmap='gray')
plt.title(f'Mặt nạ HSV siêu tối ưu\nLow={lower_bound}\nHigh={upper_bound}')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(dist_transform, cmap='gray')
plt.title(f'Distance Transform\nThresh Ratio={dist_threshold_ratio:.4f}')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(sure_fg, cmap='gray')
plt.title('Sure Foreground (Siêu tối ưu)')
plt.axis('off')

plt.subplot(2, 3, 5)
if 'markers' in locals(): # Chỉ hiển thị markers nếu chúng được tạo
    markers_display = markers.copy()
    markers_display[markers == -1] = 0
    plt.imshow(markers_display, cmap='tab20') # Thử cmap 'tab20' cho nhiều màu hơn
    plt.title('Markers (Sau Watershed)')
else:
    plt.title('Markers (Không được tạo)')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(cv2.cvtColor(img_display_watershed, cv2.COLOR_BGR2RGB))
plt.title(f'Kết quả Siêu Tối ưu ({watershed_bean_count} hạt)')
plt.axis('off')

plt.tight_layout()
plt.show()

# Lưu ảnh kết quả cuối cùng
output_path_final = f'dau_nanh_ket_qua_SUPER_OPTIMIZED2_{watershed_bean_count}.jpg'
cv2.imwrite(output_path_final, img_display_watershed)
print(f"Đã lưu ảnh kết quả siêu tối ưu vào '{output_path_final}'")
