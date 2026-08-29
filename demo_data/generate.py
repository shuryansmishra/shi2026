import os
import numpy as np
import rasterio
from rasterio.transform import from_origin

def create_geotiff(filename, bands, width, height, dtype=np.uint8, is_sar=False):
    # Example coordinates for somewhere in India (e.g. Bangalore area)
    transform = from_origin(77.5, 12.9, 0.0001, 0.0001)
    
    # Generate some random spatial data that resembles satellite features
    # If SAR, typically 1 or 2 bands, high noise (speckle). If optical, 3 or 4 bands.
    
    data = []
    for _ in range(bands):
        # Base noise
        band_data = np.random.randint(0, 255, (height, width), dtype=dtype)
        # Add some structure (e.g. horizontal/vertical lines for roads, blocks for buildings)
        for _ in range(10):
            x = np.random.randint(0, width - 20)
            y = np.random.randint(0, height - 20)
            w = np.random.randint(10, 50)
            h = np.random.randint(10, 50)
            val = np.random.randint(0, 255)
            band_data[y:y+h, x:x+w] = val
            
        # If SAR, add speckle
        if is_sar:
            speckle = np.random.normal(1.0, 0.2, (height, width))
            band_data = np.clip(band_data * speckle, 0, 255).astype(dtype)
            
        data.append(band_data)

    data = np.array(data)

    with rasterio.open(
        filename,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=bands,
        dtype=dtype,
        crs='+proj=latlong',
        transform=transform,
    ) as dst:
        for i in range(bands):
            dst.write(data[i], i + 1)

    print(f"Generated {filename}")

if __name__ == "__main__":
    os.makedirs("/Users/shuryansmishra/Downloads/satquery-ai/demo_data", exist_ok=True)
    
    # 1. Single Image - Optical
    create_geotiff("/Users/shuryansmishra/Downloads/satquery-ai/demo_data/optical_single.tif", 3, 512, 512)
    
    # 2. Bi-Temporal Pair - Optical
    create_geotiff("/Users/shuryansmishra/Downloads/satquery-ai/demo_data/optical_t1.tif", 3, 512, 512)
    create_geotiff("/Users/shuryansmishra/Downloads/satquery-ai/demo_data/optical_t2.tif", 3, 512, 512)
    
    # 3. Fusion Pair - Optical and SAR
    create_geotiff("/Users/shuryansmishra/Downloads/satquery-ai/demo_data/fusion_optical.tif", 3, 512, 512)
    create_geotiff("/Users/shuryansmishra/Downloads/satquery-ai/demo_data/fusion_sar.tif", 1, 512, 512, is_sar=True)
    
    print("Demo data generation complete.")
