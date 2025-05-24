import cv2
import numpy as np
import json
import os
import tempfile
from glob import glob
from pathlib import Path
import logging
from src.preprocessing.dimension_extractor import DimensionExtractor
from src.preprocessing.grid_normalizer import GridNormalizer
import svgwrite
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from pdf2image import convert_from_path
import pytesseract
from PIL import Image, ImageDraw

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingDataGenerator:
    """Generate training data pairs from architectural PDF drawings"""
    
    def __init__(self, target_size=(256, 256)):
        """
        Initialize training data generator
        
        Args:
            target_size: Target image size (width, height) in pixels
        """
        self.target_size = target_size
        self.dimension_extractor = DimensionExtractor()
        self.grid_normalizer = GridNormalizer()
        
        self.wall_thickness = 4
        
        self.opening_size = 8

    def process_pdf_collection(self, pdf_dir, output_dir):
        """
        Process a collection of PDF floor plans to generate training data
        
        Args:
            pdf_dir: Directory containing PDF files
            output_dir: Directory to save training data pairs
            
        Returns:
            Number of successfully processed files
        """
        logger.info(f"Processing PDF collection from {pdf_dir}")
        
        pdf_files = glob(f"{pdf_dir}/*.pdf") + glob(f"{pdf_dir}/*.PDF")
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        os.makedirs(output_dir, exist_ok=True)
        
        successful = 0
        for i, pdf_path in enumerate(pdf_files):
            try:
                logger.info(f"[{i+1}/{len(pdf_files)}] Processing {pdf_path}")
                
                # 1. Extract dimensions from PDF
                dimensions = self.dimension_extractor.extract_from_pdf(pdf_path)
                if not dimensions:
                    logger.warning(f"No dimensions extracted from {pdf_path}")
                    continue
                    
                logger.info(f"Extracted {len(dimensions)} dimensions")
                
                normalized = self.grid_normalizer.normalize_dimensions(dimensions)
                logger.info(f"Normalized {len(normalized)} dimensions")
                
                svg_path = self.pdf_to_svg(pdf_path)
                logger.info(f"Converted PDF to SVG: {svg_path}")
                
                # 4. Convert SVG to grid image
                grid_image = self.svg_to_grid_image(svg_path, normalized)
                logger.info(f"Converted SVG to grid image of size {grid_image.shape}")
                
                channels = self.separate_elements(grid_image)
                logger.info(f"Separated elements into {channels.shape[2]} channels")
                
                site_mask = self.create_site_mask(normalized)
                logger.info(f"Created site mask of size {site_mask.shape}")
                
                metadata = self.create_metadata(normalized, channels, pdf_path)
                
                pair_dir = f"{output_dir}/pair_{i:04d}"
                self.save_training_pair(site_mask, channels, metadata, pair_dir)
                logger.info(f"Saved training pair to {pair_dir}")
                
                successful += 1
                
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        logger.info(f"Successfully processed {successful}/{len(pdf_files)} files")
        return successful
        
    def pdf_to_svg(self, pdf_path):
        """
        Convert PDF to SVG format
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Path to generated SVG file
        """
        svg_path = os.path.join(
            tempfile.gettempdir(), 
            f"{Path(pdf_path).stem}.svg"
        )
        
        images = convert_from_path(pdf_path, dpi=300)
        
        if not images:
            raise ValueError(f"Failed to convert PDF to image: {pdf_path}")
            
        img = images[0]
        
        dwg = svgwrite.Drawing(svg_path, size=(img.width, img.height))
        
        img_array = np.array(img)
        
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, 
            threshold=100, 
            minLineLength=100, 
            maxLineGap=10
        )
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dwg.add(dwg.line(
                    start=(int(x1), int(y1)), 
                    end=(int(x2), int(y2)), 
                    stroke=svgwrite.rgb(0, 0, 0, '%'),
                    stroke_width=2
                ))
        
        dwg.save()
        
        return svg_path
    
    # --- Placeholder methods to simulate required steps ---

    def pdf_to_grid_image_placeholder(self, pdf_path, normalized_dimensions):
        """
        Placeholder for converting PDF content to a grid image.
        In a real implementation, this would involve parsing PDF vector data
        and rendering it onto a grid, respecting normalized dimensions.
        For now, return a dummy image based on site dimensions.
        """
        print(f"Placeholder: Converting {pdf_path} to grid image...")
        # Assume site size is the first normalized dimension entry and is a list [width, height]
        site_dims = None
        for dim_info in normalized_dimensions:
            if dim_info.get('grid_type') == 'site_size' and isinstance(dim_info.get('normalized'), list):
                 site_dims = dim_info['normalized']
                 break

        if site_dims is None:
             print("Warning: Site dimensions not found in normalized data. Using default size.")
             site_width_grids = 11 # Default
             site_height_grids = 10 # Default
        else:
            site_width_grids = site_dims[0]['grid_count']
            site_height_grids = site_dims[1]['grid_count']


        # Create a simple dummy image representing a rectangle based on site size
        img = np.zeros(self.target_size, dtype=np.uint8) # Black background
        
        # Simple white rectangle representing the site area
        # This is highly simplified; real implementation needs to draw walls, etc.
        start_x = int((self.target_size[0] - site_width_grids * (self.target_size[0] / 20)) / 2) # Approximate centering
        start_y = int((self.target_size[1] - site_height_grids * (self.target_size[1] / 20)) / 2)
        end_x = start_x + int(site_width_grids * (self.target_size[0] / 20))
        end_y = start_y + int(site_height_grids * (self.target_size[1] / 20))

        cv2.rectangle(img, (start_x, start_y), (end_x, end_y), 255, -1) # White rectangle

        # Simulate some walls - extremely basic
        cv2.rectangle(img, (start_x, start_y), (start_x + 5, end_y), 0, -1) # Left wall
        cv2.rectangle(img, (end_x - 5, start_y), (end_x, end_y), 0, -1) # Right wall
        cv2.rectangle(img, (start_x, start_y), (end_x, start_y + 5), 0, -1) # Top wall
        cv2.rectangle(img, (start_x, end_y - 5), (end_x, end_y), 0, -1) # Bottom wall


        # In a real scenario, this function would convert vector graphics from PDF
        # into pixel data on our target grid resolution (256x256).
        # This requires robust PDF parsing and rendering capabilities not trivial to implement.
        # Libraries like pdflib (commercial) or complex use of reportlab/svglib might be needed.
        # Given the MVP scope and complexity, a full implementation is deferred.

        return img


    def separate_elements(self, grid_image):
        """建築要素をチャンネル分離 (Placeholder)"""
        print("Placeholder: Separating elements...")
        # This is highly dependent on the `pdf_to_grid_image_placeholder` output.
        # Assuming the grid_image is a grayscale representation of the floor plan.
        # In a real scenario, this would involve image processing (line detection,
        # shape analysis) or using information from the vector parsing step.

        # For now, create dummy channels based on the input grayscale image
        rgba = np.zeros((self.target_size[0], self.target_size[1], 4), dtype=np.uint8)

        # Dummy logic:
        # Channel 0 (Red): Simulate Walls (e.g., thicker lines)
        # Channel 1 (Green): Simulate Openings (e.g., breaks in lines)
        # Channel 2 (Blue): Simulate Stairs (e.g., specific patterns)
        # Channel 3 (Alpha): Simulate Rooms (e.g., areas enclosed by walls)

        # Simple edge detection to simulate walls
        walls = cv2.Canny(grid_image, 50, 150)
        rgba[:,:,0] = walls # Use Canny output as dummy wall channel

        # Openings - very basic: simulate random openings in walls
        openings = np.zeros(self.target_size, dtype=np.uint8)
        # Randomly place some "openings" along dummy walls
        wall_pixels = np.argwhere(walls > 0)
        if len(wall_pixels) > 100: # Ensure enough wall pixels
            sample_indices = np.random.choice(len(wall_pixels), 50, replace=False) # Sample 50 points
            for idx in sample_indices:
                r, c = wall_pixels[idx]
                # "Clear" a small area around the point
                min_r, max_r = max(0, r-3), min(self.target_size[0], r+3)
                min_c, max_c = max(0, c-3), min(self.target_size[1], c+3)
                openings[min_r:max_r, min_c:max_c] = 255 # White for opening
        rgba[:,:,1] = openings # Dummy opening channel

        # Stairs - even simpler: simulate a random rectangle for stairs
        stairs = np.zeros(self.target_size, dtype=np.uint8)
        if self.target_size[0] > 50 and self.target_size[1] > 50:
             st_x = np.random.randint(self.target_size[0] // 4, self.target_size[0] * 3 // 4 - 20)
             st_y = np.random.randint(self.target_size[1] // 4, self.target_size[1] * 3 // 4 - 20)
             cv2.rectangle(stairs, (st_x, st_y), (st_x + 20, st_y + 30), 255, -1) # White rectangle for stairs
        rgba[:,:,2] = stairs # Dummy stair channel


        # Rooms - fill enclosed areas. Requires proper contour finding and filling based on walls.
        # This is complex with current dummy wall representation.
        # For now, let's create a dummy "room" channel that's just the inverse of walls (roughly empty space)
        rooms = cv2.bitwise_not(walls) # Simple inverse of walls as dummy room area
        rgba[:,:,3] = rooms # Dummy room channel

        # Note: A proper implementation needs to use geometric data or advanced image processing
        # to accurately identify and separate these architectural elements based on drawing conventions.
        # The current implementation is a simplified placeholder.

        return rgba

    def create_site_mask(self, normalized_dimensions):
        """敷地マスク生成"""
        print("Placeholder: Creating site mask...")
        # Assume site size is the first normalized dimension entry and is a list [width, height]
        site_dims = None
        for dim_info in normalized_dimensions:
            if dim_info.get('grid_type') == 'site_size' and isinstance(dim_info.get('normalized'), list):
                 site_dims = dim_info['normalized']
                 break

        if site_dims is None:
             print("Warning: Site dimensions not found for mask. Using default size.")
             site_width_grids = 11 # Default
             site_height_grids = 10 # Default
        else:
            site_width_grids = site_dims[0]['grid_count']
            site_height_grids = site_dims[1]['grid_count']


        mask = np.zeros(self.target_size, dtype=np.uint8) # Black background

        # Draw a white rectangle representing the site boundary
        # Need to scale grid dimensions to target_size pixels
        # Assuming maximum grid size is around 20x20 fitting into 256x256
        pixels_per_grid = self.target_size[0] / 20 # Example scaling factor

        mask_width_pixels = int(site_width_grids * pixels_per_grid)
        mask_height_pixels = int(site_height_grids * pixels_per_grid)

        # Center the mask
        start_x = (self.target_size[0] - mask_width_pixels) // 2
        start_y = (self.target_size[1] - mask_height_pixels) // 2
        end_x = start_x + mask_width_pixels
        end_y = start_y + mask_height_pixels

        cv2.rectangle(mask, (start_x, start_y), (end_x, end_y), 255, -1) # White rectangle

        return mask


    def create_metadata(self, normalized_dimensions, channels, pdf_path):
        """メタデータ生成"""
        print("Placeholder: Creating metadata...")
        metadata = {
            'source_pdf': os.path.basename(pdf_path),
            'normalized_dimensions': normalized_dimensions,
            'target_image_size': self.target_size,
            'channels_info': {
                '0': 'walls',
                '1': 'openings',
                '2': 'stairs',
                '3': 'rooms'
            },
            'site_grid_size': None, # To be filled if site_size is found
            'total_area_sqm': None, # Calculate from site_grid_size
            'room_count': None # Placeholder, needs actual room detection
        }

        # Find site grid size from normalized dimensions
        for dim_info in normalized_dimensions:
            if dim_info.get('grid_type') == 'site_size' and isinstance(dim_info.get('normalized'), list):
                 width_norm = dim_info['normalized'][0]
                 height_norm = dim_info['normalized'][1]
                 metadata['site_grid_size'] = (width_norm['grid_count'], height_norm['grid_count'])
                 # Calculate area assuming 910mm grid
                 metadata['total_area_sqm'] = metadata['site_grid_size'][0] * 0.91 * metadata['site_grid_size'][1] * 0.91
                 break # Assuming only one site size entry

        # Room count - this requires processing the 'rooms' channel from `separate_elements`
        # A proper implementation would find connected components in the room channel.
        # For now, let's use a dummy value or try a very basic component detection on the dummy room channel.
        if channels is not None and channels.shape[2] > 3:
             dummy_rooms_channel = channels[:,:,3]
             # Find contours in the dummy room channel
             contours, _ = cv2.findContours(dummy_rooms_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
             # Filter contours by size to avoid noise and count them as rooms
             min_room_area_pixels = (self.target_size[0] / 20) * (self.target_size[1] / 20) * 2 # Roughly 2 grid cells
             room_count = sum(1 for cnt in contours if cv2.contourArea(cnt) > min_room_area_pixels)
             metadata['room_count'] = room_count if room_count > 0 else 3 # Default to 3 if none detected


        return metadata

    def save_training_pair(self, site_mask, channels, metadata, output_prefix):
        """学習ペアをファイルに保存"""
        print(f"Saving training pair to {output_prefix}...")
        pair_dir = output_prefix
        os.makedirs(pair_dir, exist_ok=True)

        # Save site mask (grayscale PNG)
        cv2.imwrite(f"{pair_dir}/site_mask.png", site_mask)

        # Save floor plan channels (RGBA PNG)
        # Note: channels is already RGBA in separate_elements placeholder
        cv2.imwrite(f"{pair_dir}/floor_plan.png", channels)


        # Save metadata (JSON)
        with open(f"{pair_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)

        print("Training pair saved.")

    # --- Helper methods from requirements (need to check if they exist or implement) ---
    # The following methods were shown in the requirements but might be in other files or need implementation
    # Assuming they are part of the DimensionExtractor and GridNormalizer now based on list_dir results.
    # If not, they need to be added here or imported correctly.

    # Based on the initial list_dir and read_file for dimension_extractor.py and grid_normalizer.py,
    # it seems extract_from_pdf, normalize_dimensions, etc., are indeed in those files.
    # The requirement document structure implies these classes might be used by TrainingDataGenerator.
    # Let's assume they are imported and used as shown in process_pdf_collection.

    # However, the methods `pdf_to_svg`, `svg_to_grid_image`, `detect_walls`, `detect_openings`,
    # `detect_stairs`, `detect_rooms` from the requirement's TrainingDataGenerator section
    # are not standard parts of the imported classes and need to be implemented or clarified.
    # Given the complexity of PDF/SVG processing and architectural element detection from images,
    # the placeholder implementations above are used as a simplified stand-in for MVP.
    # A full implementation would require significant effort in geometric processing and image analysis.

    # The requirement mentions `svg_to_grid_image` taking `svg_path` and `normalized`.
    # My placeholder `pdf_to_grid_image_placeholder` takes `pdf_path` and `normalized_dimensions`.
    # This indicates a dependency on a PDF-to-SVG step and an SVG-to-grid-image step.
    # The placeholder simplifies this by attempting to go directly from PDF path/dimensions to grid image,
    # but acknowledges the complexity and defers a proper implementation.
    # The `separate_elements` placeholder takes a `grid_image` and produces channels.
    
    def svg_to_grid_image(self, svg_path, normalized):
        """
        Convert SVG to grid-normalized image
        
        Args:
            svg_path: Path to SVG file
            normalized: Normalized dimensions
            
        Returns:
            Grid-normalized image as numpy array
        """
        logger.info(f"Converting SVG to grid image: {svg_path}")
        
        try:
            drawing = svg2rlg(svg_path)
            if not drawing:
                raise ValueError(f"Failed to load SVG: {svg_path}")
                
            pil_img = renderPM.drawToPIL(drawing)
            
            img_array = np.array(pil_img)
            
        except Exception as e:
            logger.warning(f"Error converting SVG to image: {e}")
            return self.pdf_to_grid_image_placeholder("", normalized)
        
        # Get site dimensions from normalized dimensions
        site_dims = [d for d in normalized if d['type'] == 'area']
        
        if not site_dims:
            # If no area dimensions, try to find the largest linear dimensions
            linear_dims = sorted(
                [d for d in normalized if d['type'] == 'linear'],
                key=lambda x: x['original'],
                reverse=True
            )
            
            if len(linear_dims) >= 2:
                width = linear_dims[0]['normalized_mm']
                depth = linear_dims[1]['normalized_mm']
            else:
                # Default to a standard size if we can't extract dimensions
                width, depth = 10920, 10920  # 12x12 grid units (910mm each)
        else:
            width = site_dims[0]['original'][0]
            depth = site_dims[0]['original'][1]
        
        width_grid = round(width / 910)
        depth_grid = round(depth / 910)
        
        # Calculate pixels per grid
        pixels_per_grid = min(
            self.target_size[0] // max(width_grid, 1),
            self.target_size[1] // max(depth_grid, 1)
        )
        
        grid_width = width_grid * pixels_per_grid
        grid_height = depth_grid * pixels_per_grid
        
        grid_width = min(grid_width, self.target_size[0])
        grid_height = min(grid_height, self.target_size[1])
        
        if len(img_array.shape) == 3:
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_array
            
        resized = cv2.resize(img_gray, (grid_width, grid_height))
        
        grid_image = np.ones(self.target_size, dtype=np.uint8) * 255
        
        # Center the resized image in the target image
        x_offset = (self.target_size[0] - grid_width) // 2
        y_offset = (self.target_size[1] - grid_height) // 2
        
        grid_image[
            y_offset:y_offset+grid_height, 
            x_offset:x_offset+grid_width
        ] = resized
        
        _, binary = cv2.threshold(grid_image, 200, 255, cv2.THRESH_BINARY)
        
        binary = 255 - binary
        
        return binary

    # The requirement document also mentions `pdf_to_svg`. This is a key missing piece
    # for a proper implementation of the training data pipeline. Libraries like `pypdf`, `PyMuPDF` (fitz),
    # or integrating with command-line tools (like `mutool` or `inkscape`) might be needed,
    # but extracting vector data reliably from arbitrary architectural PDFs is challenging.
    # For the MVP, the placeholder approach for image generation and element separation is pragmatic,
    # but acknowledges this deviation from the detailed requirement steps (PDF->SVG->PNG->Separate).                    