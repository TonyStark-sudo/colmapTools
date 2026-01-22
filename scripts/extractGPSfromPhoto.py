import os
import re
import argparse
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def get_exif(filename):
    image = Image.open(filename)
    image.verify()
    return image._getexif()

def get_geotagging(exif):
    if not exif:
        raise ValueError("No EXIF metadata found")

    geotagging = {}
    for (idx, tag) in TAGS.items():
        if tag == 'GPSInfo':
            if idx not in exif:
                raise ValueError("No EXIF geotagging found")

            for (key, val) in GPSTAGS.items():
                if key in exif[idx]:
                    geotagging[val] = exif[idx][key]

    return geotagging

def get_decimal_from_dms(dms, ref):
    degrees = dms[0]
    minutes = dms[1] / 60.0
    seconds = dms[2] / 3600.0

    if ref in ['S', 'W']:
        degrees = -degrees
        minutes = -minutes
        seconds = -seconds

    return degrees + minutes + seconds

def get_image_gps_info(image_path):
    try:
        exif = get_exif(image_path)
        geotags = get_geotagging(exif)
        
        lat = get_decimal_from_dms(geotags['GPSLatitude'], geotags['GPSLatitudeRef'])
        lon = get_decimal_from_dms(geotags['GPSLongitude'], geotags['GPSLongitudeRef'])
        alt = geotags.get('GPSAltitude', 0) # Default to 0 if not present

        return {'lat': str(lat), 'lon': str(lon), 'alt': str(alt)}
    except (ValueError, KeyError, IndexError) as e:
        print(f"Could not get GPS info for {image_path}: {e}")
        return None

def extract_gps_from_images(image_dir):
    # Create the output directory if it doesn't exist
    output_dir = '../output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the output file
    with open(os.path.join(output_dir, 'gps.txt'), 'w') as f:
        # Custom sort key function to extract number from filename
        def sort_key(filename):
            match = re.search(r'_(\d+)\.jpg$', filename, re.IGNORECASE)
            if match:
                return int(match.group(1))
            return -1 # Or some other default for files that don't match

        # Get list of files and sort them using the custom key
        files = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]
        files.sort(key=sort_key)

        # Iterate over all files in the image directory in sorted order
        for filename in files:
            image_path = os.path.join(image_dir, filename)
            gps_info = get_image_gps_info(image_path)
            if gps_info:
                f.write(f"{filename} {gps_info['lat']} {gps_info['lon']} {gps_info['alt']}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract GPS data from images.')
    parser.add_argument('image_dir', type=str, help='The directory containing the images.')
    args = parser.parse_args()
    
    extract_gps_from_images(args.image_dir)