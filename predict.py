from PIL import Image
from yolo import YOLO


if __name__ == "__main__":
    
    mode = "dir_predict"
    crop            = False
    count           = False    
    dir_origin_path = "sample_RAFDB/"
    dir_save_path   = "img_out/"
    onnx_path = 'onnx_models/originalV2_320_finetune_woVSS_Unfreeze100.onnx'
    input_shape = [320,320]
    class_path = 'model_data/rafdb_classes.txt'
    

    yolo = YOLO(class_path=class_path,onnx_path=onnx_path,input_shape=input_shape)

    if mode == "img_predict":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                if img=='quit': break
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image, crop = crop, count=count)
                r_image.show()

    elif mode == "dir_predict":
        import os

        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = yolo.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name), quality=95, subsampling=0)
        
    else:
        raise AssertionError("Please specify the correct mode: 'img_predict', 'dir_predict'.")
