import json
from pprint import pprint
import os


def create_palette():

    dir_output = './color_palette.json'

    if os.path.isfile(dir_output):
        palette = json.load(open(dir_output))
        return palette

    palette = []

    color = {}
    color['id'] = 1
    color['color'] = 'blue'
    color['rgb'] = [255, 0, 0]
    palette.append(color)

    color = {}
    color['id'] = 2
    color['color'] = 'red'
    color['rgb'] = [0, 0, 255]
    palette.append(color)

    color = {}
    color['id'] = 3
    color['color'] = 'yellow'
    color['rgb'] = [0, 255, 25]
    palette.append(color)

    color = {}
    color['id'] = 4
    color['color'] = 'dark green'
    color['rgb'] = [0, 100, 0]
    palette.append(color)

    color = {}
    color['id'] = 5
    color['color'] = 'navy'
    color['rgb'] = [128, 0, 0]
    palette.append(color)

    color = {}
    color['id'] = 6
    color['color'] = 'deep pink'
    color['rgb'] = [147, 20, 255]
    palette.append(color)

    color = {}
    color['id'] = 7
    color['color'] = 'orange'
    color['rgb'] = [0, 165, 255]
    palette.append(color)

    color = {}
    color['id'] = 8
    color['color'] = 'brown'
    color['rgb'] = [19, 69, 139]
    palette.append(color)

    color = {}
    color['id'] = 9
    color['color'] = 'purple'
    color['rgb'] = [240, 32, 160]
    palette.append(color)

    color = {}
    color['id'] = 10
    color['color'] = 'gray'
    color['rgb'] = [128, 128, 128]
    palette.append(color)

    outfile = open(dir_output, 'w')
    x = json.dumps(palette)
    pprint(x)
    outfile.write(x)
    outfile.close()

    return palette
