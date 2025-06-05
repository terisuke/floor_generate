import cv2
import numpy as np
import json
import os
import tempfile
from glob import glob
from pathlib import Path
import logging
from preprocessing.dimension_extractor import DimensionExtractor
from preprocessing.grid_normalizer import GridNormalizer
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
        
        if not os.path.exists(pdf_dir):
            logger.error(f"PDF directory does not exist: {pdf_dir}")
            raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
            
        pdf_files = glob(f"{pdf_dir}/*.pdf") + glob(f"{pdf_dir}/*.PDF")
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_dir}")
            return 0
            
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        try:
            os.makedirs(output_dir, exist_ok=True)
        except PermissionError:
            logger.error(f"Permission denied when creating output directory: {output_dir}")
            raise
        except Exception as e:
            logger.error(f"Failed to create output directory {output_dir}: {e}")
            raise
        
        successful = 0
        for i, pdf_path in enumerate(pdf_files):
            logger.info(f"[{i+1}/{len(pdf_files)}] Processing {pdf_path}")
            
            if not os.path.isfile(pdf_path):
                logger.error(f"PDF file does not exist or is not a file: {pdf_path}")
                continue
                
            if not os.access(pdf_path, os.R_OK):
                logger.error(f"PDF file is not readable: {pdf_path}")
                continue
            
            try:
                # 1. Extract dimensions from PDF
                dimensions = self.dimension_extractor.extract_from_pdf(pdf_path)
                
                if not dimensions:
                    logger.error(f"No dimensions extracted from {pdf_path}")
                    continue
                
                if len(dimensions) < 2:
                    logger.warning(f"Only {len(dimensions)} dimensions extracted from {pdf_path}. At least 2 recommended.")
                    
                logger.info(f"Extracted {len(dimensions)} dimensions")
                
                # 2. Normalize dimensions
                normalized = self.grid_normalizer.normalize_dimensions(dimensions)
                
                if not normalized:
                    logger.error(f"Failed to normalize dimensions from {pdf_path}")
                    continue
                    
                logger.info(f"Normalized {len(normalized)} dimensions")
                
                svg_path = self.pdf_to_svg(pdf_path)
                
                # Validate SVG conversion
                if not svg_path or not os.path.exists(svg_path):
                    logger.error(f"Failed to convert PDF to SVG: {pdf_path}")
                    continue
                    
                logger.info(f"Converted PDF to SVG: {svg_path}")
                
                # 4. Convert SVG to grid image
                grid_image = self.svg_to_grid_image(svg_path, normalized)
                
                # Validate grid image
                if grid_image is None or grid_image.size == 0:
                    logger.error(f"Failed to generate grid image from SVG: {svg_path}")
                    continue
                    
                logger.info(f"Converted SVG to grid image of size {grid_image.shape}")
                
                channels = self.separate_elements(grid_image)
                
                if channels is None or channels.size == 0 or channels.shape[2] != 4:
                    logger.error(f"Failed to separate elements from grid image for {pdf_path}")
                    continue
                    
                logger.info(f"Separated elements into {channels.shape[2]} channels")
                
                site_mask = self.create_site_mask(normalized)
                
                if site_mask is None or site_mask.size == 0:
                    logger.error(f"Failed to create site mask for {pdf_path}")
                    continue
                    
                logger.info(f"Created site mask of size {site_mask.shape}")
                
                metadata = self.create_metadata(normalized, channels, pdf_path)
                
                if not metadata:
                    logger.error(f"Failed to create metadata for {pdf_path}")
                    continue
                
                required_fields = ['source_pdf', 'normalized_dimensions', 'channels_info']
                missing_fields = [field for field in required_fields if field not in metadata]
                
                if missing_fields:
                    logger.error(f"Metadata missing required fields: {missing_fields}")
                    continue
                
                pair_dir = f"{output_dir}/pair_{i:04d}"
                
                try:
                    self.save_training_pair(site_mask, channels, metadata, pair_dir)
                except Exception as e:
                    logger.error(f"Failed to save training pair to {pair_dir}: {e}")
                    continue
                
                expected_files = [
                    f"{pair_dir}/site_mask.png",
                    f"{pair_dir}/floor_plan.png",
                    f"{pair_dir}/metadata.json"
                ]
                
                missing_files = [f for f in expected_files if not os.path.exists(f)]
                
                if missing_files:
                    logger.error(f"Failed to save all required files: {missing_files}")
                    continue
                
                logger.info(f"Saved training pair to {pair_dir}")
                successful += 1
                
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
                import traceback
                traceback.print_exc()
                break
        
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
        """建築要素をチャンネル分離 (Enhanced implementation)"""
        logger.info("Separating architectural elements...")
        
        rgba = np.zeros((self.target_size[0], self.target_size[1], 4), dtype=np.uint8)
        
        # Make a copy of the input image for processing
        img = grid_image.copy()
        
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img
            
        img_filtered = cv2.bilateralFilter(img_gray, 9, 75, 75)
        
        thresh = cv2.adaptiveThreshold(
            img_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        kernel_line = np.ones((3, 3), np.uint8)
        walls = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_line)
        
        edges = cv2.Canny(walls, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, 
            threshold=50, minLineLength=30, maxLineGap=10
        )
        
        walls_refined = np.zeros(self.target_size, dtype=np.uint8)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(walls_refined, (x1, y1), (x2, y2), 255, 2)
        
        if lines is None or len(lines) < 5:
            walls_refined = walls
            
        walls_refined = cv2.dilate(walls_refined, kernel_line, iterations=1)
        rgba[:,:,0] = walls_refined
        
        # Detect potential door/window openings by finding gaps in walls
        openings = np.zeros(self.target_size, dtype=np.uint8)
        
        wall_contours, _ = cv2.findContours(
            walls_refined, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in wall_contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) >= 4 and cv2.contourArea(approx) < 500:
                cv2.drawContours(openings, [approx], 0, 255, -1)
        
        if cv2.countNonZero(openings) < 50:
            # Find wall pixels
            wall_pixels = np.argwhere(walls_refined > 0)
            if len(wall_pixels) > 100:
                sample_count = min(30, len(wall_pixels) // 10)
                sample_indices = np.random.choice(len(wall_pixels), sample_count, replace=False)
                
                for idx in sample_indices:
                    r, c = wall_pixels[idx]
                    min_r, max_r = max(0, r-5), min(self.target_size[0], r+5)
                    min_c, max_c = max(0, c-5), min(self.target_size[1], c+5)
                    openings[min_r:max_r, min_c:max_c] = 255
        
        rgba[:,:,1] = openings
        
        stairs = np.zeros(self.target_size, dtype=np.uint8)
        
        if lines is not None and len(lines) > 10:
            angle_groups = {}
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 == 0:  # Vertical line
                    angle = 90
                else:
                    angle = np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi
                
                angle_key = round(angle / 5) * 5
                if angle_key not in angle_groups:
                    angle_groups[angle_key] = []
                angle_groups[angle_key].append(line[0])
            
            for angle, lines_group in angle_groups.items():
                if len(lines_group) >= 3:
                    if abs(angle) < 45:  # More horizontal
                        lines_group.sort(key=lambda l: l[1])  # Sort by y
                    else:  # More vertical
                        lines_group.sort(key=lambda l: l[0])  # Sort by x
                    
                    positions = [l[1] if abs(angle) < 45 else l[0] for l in lines_group]
                    diffs = np.diff(positions)
                    
                    if len(diffs) >= 2 and np.std(diffs) < np.mean(diffs) * 0.3:
                        for line in lines_group:
                            x1, y1, x2, y2 = line
                            cv2.line(stairs, (x1, y1), (x2, y2), 255, 3)
                        
                        min_x = min([l[0] for l in lines_group] + [l[2] for l in lines_group])
                        max_x = max([l[0] for l in lines_group] + [l[2] for l in lines_group])
                        min_y = min([l[1] for l in lines_group] + [l[3] for l in lines_group])
                        max_y = max([l[1] for l in lines_group] + [l[3] for l in lines_group])
                        
                        cv2.rectangle(stairs, (min_x, min_y), (max_x, max_y), 255, -1)
                        break  # Only detect one staircase for now
        
        if cv2.countNonZero(stairs) < 50:
            wall_distance = cv2.distanceTransform(cv2.bitwise_not(walls_refined), cv2.DIST_L2, 5)
            possible_locations = np.argwhere((wall_distance > 10) & (wall_distance < 30))
            
            if len(possible_locations) > 0:
                idx = np.random.randint(0, len(possible_locations))
                y, x = possible_locations[idx]
                
                stair_width = 20
                stair_height = 30
                stair_steps = 5
                step_height = stair_height // stair_steps
                
                for i in range(stair_steps):
                    y_pos = y + i * step_height
                    if 0 <= y_pos < self.target_size[0] and 0 <= x < self.target_size[1]:
                        cv2.line(stairs, (x, y_pos), (x + stair_width, y_pos), 255, 2)
                
                cv2.rectangle(stairs, (x, y), (x + stair_width, y + stair_height), 255, -1)
        
        rgba[:,:,2] = stairs
        
        # Create a mask excluding walls for flood filling
        non_wall_mask = cv2.bitwise_not(walls_refined)
        
        walls_dilated = cv2.dilate(walls_refined, kernel_line, iterations=2)
        
        # Create a copy for flood filling
        rooms = np.zeros(self.target_size, dtype=np.uint8)
        
        dist_transform = cv2.distanceTransform(non_wall_mask, cv2.DIST_L2, 5)
        _, dist_max_val, _, dist_max_loc = cv2.minMaxLoc(dist_transform)
        
        room_seeds = []
        dist_threshold = max(5, dist_max_val * 0.5)
        potential_seeds = np.argwhere(dist_transform > dist_threshold)
        
        if len(potential_seeds) > 0:
            grid_size = 4  # 4x4 grid
            cell_h = self.target_size[0] // grid_size
            cell_w = self.target_size[1] // grid_size
            
            for i in range(grid_size):
                for j in range(grid_size):
                    cell_seeds = [
                        (y, x) for y, x in potential_seeds 
                        if i*cell_h <= y < (i+1)*cell_h and j*cell_w <= x < (j+1)*cell_w
                    ]
                    
                    if cell_seeds:
                        room_seeds.append(cell_seeds[np.random.randint(0, len(cell_seeds))])
        
        if not room_seeds:
            for i in range(1, 4):
                for j in range(1, 4):
                    room_seeds.append((
                        i * self.target_size[0] // 4,
                        j * self.target_size[1] // 4
                    ))
        
        for seed_y, seed_x in room_seeds:
            if non_wall_mask[seed_y, seed_x] > 0:  # Only fill if not on a wall
                mask = np.zeros((self.target_size[0]+2, self.target_size[1]+2), np.uint8)
                
                cv2.floodFill(
                    rooms, mask, (seed_x, seed_y), 255, 
                    loDiff=0, upDiff=0, 
                    flags=8 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY
                )
        
        contours, _ = cv2.findContours(rooms, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 100:  # Minimum room size
                cv2.drawContours(rooms, [contour], 0, 0, -1)
        
        if cv2.countNonZero(rooms) < 100:
            rooms = cv2.bitwise_not(walls_dilated)
            
            contours, _ = cv2.findContours(rooms, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < 100:
                    cv2.drawContours(rooms, [contour], 0, 0, -1)
        
        rgba[:,:,3] = rooms
        
        return rgba

    def create_site_mask(self, normalized_dimensions):
        """
        敷地マスク生成 - Create site mask from normalized dimensions
        
        Args:
            normalized_dimensions: List of normalized dimension dictionaries
            
        Returns:
            Binary mask image representing buildable site area
        """
        logger.info("Creating site mask from normalized dimensions...")
        
        # Initialize grid normalizer for grid-to-pixel conversion
        grid_normalizer = GridNormalizer()
        
        # Extract site dimensions from normalized dimensions
        site_dims = [d for d in normalized_dimensions if d.get('type') == 'area']
        
        if not site_dims:
            # If no area dimensions found, try to find the two largest linear dimensions
            linear_dims = sorted(
                [d for d in normalized_dimensions if d.get('type') == 'linear'],
                key=lambda x: x.get('original', 0),
                reverse=True
            )
            
            if len(linear_dims) >= 2:
                logger.info("Using largest linear dimensions as site dimensions")
                width_mm = linear_dims[0].get('normalized_mm', 10920)  # Default to 12 grid units
                height_mm = linear_dims[1].get('normalized_mm', 10920)
                
                width_grid = round(width_mm / 910)
                height_grid = round(height_mm / 910)
            else:
                logger.warning("Insufficient dimensions found. Using default site size.")
                width_grid = 12  # Default to 12x12 grid units
                height_grid = 12
        else:
            logger.info("Using area dimensions for site mask")
            if isinstance(site_dims[0].get('original'), list) and len(site_dims[0]['original']) >= 2:
                width_mm = site_dims[0]['original'][0]
                height_mm = site_dims[0]['original'][1]
                
                width_grid = round(width_mm / 910)
                height_grid = round(height_mm / 910)
            else:
                logger.warning("Invalid area dimension format. Using default site size.")
                width_grid = 12
                height_grid = 12
        
        logger.info(f"Site dimensions: {width_grid}x{height_grid} grid units")
        
        # Calculate pixels per grid based on target size and grid dimensions
        max_grid_dimension = max(width_grid, height_grid)
        margin_factor = 0.9  # Leave 10% margin
        
        pixels_per_grid = min(
            int((self.target_size[0] * margin_factor) / max_grid_dimension),
            int((self.target_size[1] * margin_factor) / max_grid_dimension)
        )
        
        pixels_per_grid = max(pixels_per_grid, 8)
        
        logger.info(f"Using {pixels_per_grid} pixels per grid unit")
        
        # Calculate mask dimensions in pixels
        mask_width_pixels = int(width_grid * pixels_per_grid)
        mask_height_pixels = int(height_grid * pixels_per_grid)
        
        mask = np.zeros(self.target_size, dtype=np.uint8)
        
        # Center the mask in the target image
        start_x = (self.target_size[0] - mask_width_pixels) // 2
        start_y = (self.target_size[1] - mask_height_pixels) // 2
        end_x = start_x + mask_width_pixels
        end_y = start_y + mask_height_pixels
        
        # Draw the site boundary as a white rectangle
        cv2.rectangle(mask, (start_x, start_y), (end_x, end_y), 255, -1)
        
        # For now, we'll just use the rectangular boundary
        
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

    def generate_train_images(self, metadata:dict, png_path:str):
        """
        '*_integrated.json' metadataから、256x256pxの *_floor_plan.png, *_site_mask.png, *_conv.png を生成する

        Args:
            metadata: integrated metadata (json)
            png_path: base floor image(png) path 
            
        Returns:
            result_pair: success
            None: failure
        """
        try:
            grid_dimensions = metadata.get('grid_dimensions', None)
            if grid_dimensions is None or not isinstance(grid_dimensions, dict) or len(grid_dimensions) < 2:
                grid_dimensions = {'width_grids': 10, 'height_grids': 10}

            width_grids = grid_dimensions['width_grids']
            height_grids = grid_dimensions['height_grids']

            img_base = cv2.imread(png_path)
            img_conv = img_base.copy()
            img_conv = cv2.resize(img_conv, (width_grids*10, height_grids*10))

            structural_elements = metadata.get('structural_elements', None)
            if structural_elements is None or not isinstance(structural_elements, list):
                structural_elements = [
                    {"type": "stair", "grid_x": 1.0, "grid_y": 1.0, "grid_width": 2.0, "grid_height": 1.0, "name": "stair_1"},
                    {"type": "entrance", "grid_x": 8.0, "grid_y": 8.0, "grid_width": 2.0, "grid_height": 2.0, "name": "entrance_2"},
                    {"type": "balcony", "grid_x": 0.0, "grid_y": 7.0, "grid_width": 3.0, "grid_height": 3.0, "name": "entrance_2"}
                ]
        
            # 幅高さが小数点1位まであるため、グリッドの10倍で描画
            img_plan = np.zeros((height_grids*10, width_grids*10, 3), dtype=np.uint8)
            img_mask = np.ones((height_grids*10, width_grids*10, 3), dtype=np.uint8) * 255
            for item in structural_elements:
                element_type = item['type']
                grid_x1 = round(item['grid_x'] * 10)
                grid_y1 = round(item['grid_y'] * 10)
                grid_x2 = round((item['grid_x'] + item['grid_width']) * 10)
                grid_y2 = round((item['grid_y'] + item['grid_height']) * 10)
                # 階段は赤、玄関は緑、バルコニーは青、他は黒
                fill_color_dict = { "stair": (255, 0, 0), "entrance": (0, 255, 0), "balcony": (0, 0, 255) }
                fill_color = fill_color_dict.get(element_type, (0, 0, 0))
                cv2.rectangle(img_plan, (grid_x1, grid_y1), (grid_x2, grid_y2), fill_color, thickness=-1)
                cv2.rectangle(img_conv, (grid_x1, grid_y1), (grid_x2, grid_y2), fill_color, thickness=-1)

            img_plan = cv2.resize(img_plan, (256, 256))
            img_mask = cv2.resize(img_mask, (256, 256))
            img_conv = cv2.resize(img_conv, (256, 256))

            cv2.imwrite(f"{png_path.replace('.png', '_floor_plan.png')}", img_plan)
            cv2.imwrite(f"{png_path.replace('.png', '_site_mask.png')}", img_mask)
            cv2.imwrite(f"{png_path.replace('.png', '_conv.png')}", img_conv)

            result_pair = {
                "floor_plan": img_plan,
                "site_mask": img_mask,
                "conv": img_conv,
                "metadata": metadata
            }

            return result_pair

        except Exception as e:
            print(f"Error generate train images from metadata: {e}")

            return None