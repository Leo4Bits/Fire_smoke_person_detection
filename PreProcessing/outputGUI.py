def output_text_yolo(label_name="",conf_img=0) -> str:
    return f"{label_name} (YOLO:{conf_img:.2f})"

def output_text_rf(label_name="",rf_conf=0) -> str:
    return f"{label_name} (RF:{rf_conf:.2f})"