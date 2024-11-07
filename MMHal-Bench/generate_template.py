import json 
import re

def web_address_to_local(web_template_file, local_template_file):
    json_data = json.load(open(web_template_file, 'r'))

    for idx, line in enumerate(json_data):
        image_src_web = line['image_src']
        question = line['question']
        
        # Changes doing
        # https://c4.staticflickr.com/8/7211/7206072054_c53d53b97d_o.jpg -> 7206072054_c53d53b97d_o.jpg
        # https://farm2.staticflickr.com/5443/7413067568_06a8d1eb64_o.jpg -> 7413067568_06a8d1eb64_o.jpg
        image_src = re.sub(r'^https?://[^/]+/[^/]+/[^/]+/([^/]+)$', r'\1', image_src_web)
        image_src = re.sub(r'^https?://[^/]+/[^/]+/([^/]+)$', r'\1', image_src)
        
        print(idx, image_src_web, image_src)
        line['image_src'] = 'images/' + image_src
        
    with open(local_template_file, 'w') as f:
        json.dump(json_data, f, indent=2)

if __name__ == "__main__":
    web_template_file = 'response_template.json'
    local_template_file = 'response_template_local.json'
    web_address_to_local(web_template_file, local_template_file)