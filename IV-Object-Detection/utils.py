def get_image_and_annotations(imgs, anns, img_id,):
    
    image_path = [img for img in imgs if img['id'] == img_id]
    assert len(image_path) == 1
    image_path = image_path[0]['file_name']
    ann = [{'id':a['id'],
            'image_id':a['image_id'],
            'category_id': a['category_id'],
            # change w, h to x1, y1 
            'bbox':[a['bbox'][0],
                    a['bbox'][1],
                    a['bbox'][0] + a['bbox'][2],
                    a['bbox'][1]+ a['bbox'][3]],
            'iscrowd':a['iscrowd']
           } for a in anns if a['image_id'] == img_id]
    
    return image_path, ann 