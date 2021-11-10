import argparse
import json
import os

labels = ["D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "B0", "B1", "B2", "B3", "B4", "H0", "H1", "H2", "H3", "H4", "Decimal", "KWH", "KW", "KVAH", "KVA", "PF", "CUM", "MD", "L1", "L2", "L3", "L4", "T1", "T2", "T3", "T4", "junk"]

def convert_coco_to_yolo_format(filename):
    f = open(filename)
    training_data = json.load(f)
    for image in training_data["images"]:
        img_id = image["id"]
        file_name = image["file_name"].split(".")[0]
        img_width = image["width"]
        image_height = image["height"]
        contents = []
        for annotation in training_data["annotations"]:
            if annotation["image_id"] == img_id:
                x = annotation['bbox'][0]
                y = annotation['bbox'][1]

                height = annotation['bbox'][3]
                width = annotation['bbox'][2]
                x_center = x + int(width / 2)
                y_center = y + int(height / 2)
                norm_x = x_center / img_width
                norm_y = y_center / image_height
                norm_width = width / img_width
                norm_height = height / image_height
                label = [category["name"] for category in training_data["categories"] if category["id"] == annotation["category_id"]]
                if label[0] not in labels:
                    print("Found new label! Kindly add it labels sequence and run")
                    return
                contents.append(str(labels.index(label[0])) + " " + "%0.6f"%norm_x + " " + "%0.6f"%norm_y + " " + "%0.6f"%norm_width + " " + "%0.6f"%norm_height)
        # write to file file
        file = open(os.path.join('coco_labels', "{}.txt".format(file_name)), "w")
        c = 0
        for content in contents:
            if c > 0:
                file.write("\n")
            file.write(content)
            c+=1
        file.close()




if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--infile', type=str, help='file name',
                      required=True)
  args = parser.parse_args()
  filename = args.infile
  print(labels)
  convert_coco_to_yolo_format(filename)





# import json
#
# f = open("/Users/rajeevreddypolepalli/Downloads/ahs63tuplij_1620880763285.json")
# training_data = json.load(f)
#
# import cv2
# check_set = set()
# def get_img_shape(path):
#     img = cv2.imread(path)
#     try:
#         return img.shape
#     except AttributeError:
#         print("error! ", path)
#         return (None, None, None)
# def convert_labels(path, x1, y1, x2, y2):
#
#     def sorting(l1, l2):
#         if l1 > l2:
#             lmax, lmin = l1, l2
#             return lmax, lmin
#         else:
#             lmax, lmin = l2, l1
#             return lmax, lmin
#     size = get_img_shape(path)
#     xmax, xmin = sorting(x1, x2)
#     ymax, ymin = sorting(y1, y2)
#     print(size)
#     dw = 720
#     dh = 1280
#     x = (xmin + xmax)/2.0
#     y = (ymin + ymax)/2.0
#     w = xmax - xmin
#     h = ymax - ymin
#     x = x*dw
#     w = w*dw
#     y = y*dh
#     h = h*dh
#     return (x,y,w,h)
#
#
# for i in range(len(training_data['annotations'])):
#     image_id = str(training_data['annotations'][i]['image_id'])
#     category_id = str(training_data['annotations'][i]['category_id'])
#     bbox = training_data['annotations'][i]['bbox']
#     image_path = "/Users/rajeevreddypolepalli/Downloads/ahs63tuplij_1620880763285.jpg"
#     kitti_bbox = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]
#     yolo_bbox = convert_labels(image_path, kitti_bbox[0], kitti_bbox[1], kitti_bbox[2], kitti_bbox[3])
#     filename = "abc" + ".txt"
#     content = category_id + " " + str(yolo_bbox[0]) + " " + str(yolo_bbox[1]) + " " + str(yolo_bbox[2]) + " " + str(yolo_bbox[3])
#     if image_id in check_set:
#         # Append to file files
#         file = open(filename, "a")
#         file.write("\n")
#         file.write(content)
#         file.close()
#     elif image_id not in check_set:
#         check_set.add(image_id)
#         # Write files
#         file = open(filename, "w")
#         file.write(content)
#         file.close()
# import numpy as np
# x={'bbox':[272,223,30,66]}
# box = np.array(x['bbox'], dtype=np.float64)
# box[:2] += box[2:] / 2  # xy top-left corner to center
# box[[0, 2]] /= 720  # normalize x
# box[[1, 3]] /= 1280  # normalize y
# print(box)
