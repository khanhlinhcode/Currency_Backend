import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from .config import IMG_SIZE

CLASS_NAMES = [
    "INR_10","INR_100","INR_20","INR_200","INR_2000","INR_50","INR_500",
    "INR_Background",
    "USA_1 Dollar","USA_10 Dollar","USA_100 Dollar","USA_2 Doolar",
    "USA_5 Dollar","USA_50 Dollar",
    "VN_000000","VN_000200","VN_000500","VN_001000","VN_002000",
    "VN_005000","VN_010000","VN_020000","VN_050000",
    "VN_100000","VN_200000","VN_500000"
]

CURRENCY_VALUE = {
    # ================= INR =================
    "INR_10": 10,
    "INR_20": 20,
    "INR_50": 50,
    "INR_100": 100,
    "INR_200": 200,
    "INR_500": 500,
    "INR_2000": 2000,
    "INR_Background": 0,

    # ================= USA =================
    "USA_1 Dollar": 1,
    "USA_2 Doolar": 2,   # giữ nguyên theo dataset
    "USA_5 Dollar": 5,
    "USA_10 Dollar": 10,
    "USA_50 Dollar": 50,
    "USA_100 Dollar": 100,

    # ================= VN =================
    "VN_000000": 0,
    "VN_000200": 200,
    "VN_000500": 500,
    "VN_001000": 1000,
    "VN_002000": 2000,
    "VN_005000": 5000,
    "VN_010000": 10000,
    "VN_020000": 20000,
    "VN_050000": 50000,
    "VN_100000": 100000,
    "VN_200000": 200000,
    "VN_500000": 500000,
}
def preprocess_image(image: Image.Image):
    image = image.resize(IMG_SIZE)
    arr = np.array(image)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr
