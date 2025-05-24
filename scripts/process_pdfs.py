"""
Process PDF files to extract dimensions
This is the first step in the data pipeline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import json
from src.preprocessing import DimensionExtractor, GridNormalizer
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_single_pdf(pdf_path: Path, output_dir: Path):
    """
    Process a single PDF file
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save results
    """
    logger.info(f"Processing: {pdf_path.name}")
    
    # Initialize processors
    extractor = DimensionExtractor()
    normalizer = GridNormalizer()
    
    # Extract dimensions
    dimensions = extractor.extract_from_pdf(str(pdf_path))
    
    if not dimensions:
        logger.warning(f"No dimensions found in {pdf_path.name}")
        return None
    
    # Normalize dimensions
    normalized = normalizer.normalize_dimensions(dimensions)
    
    # Prepare output
    result = {
        'pdf_file': pdf_path.name,
        'raw_dimensions_count': len(dimensions),
        'normalized_dimensions': normalized,
        'site_analysis': None
    }
    
    # Try to identify site dimensions (largest area dimensions)
    area_dims = [d for d in normalized if d['type'] == 'area']
    if area_dims:
        # Assume largest area is site dimension
        largest = max(area_dims, key=lambda x: x['original'][0] * x['original'][1])
        site_analysis = normalizer.analyze_site_dimensions(
            largest['original'][0],
            largest['original'][1]
        )
        result['site_analysis'] = site_analysis
    
    # Save results
    output_file = output_dir / f"{pdf_path.stem}_dimensions.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved results to: {output_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"PDF: {pdf_path.name}")
    print(f"Found {len(dimensions)} raw dimensions")
    print(f"Normalized to {len(normalized)} dimensions")
    
    if result['site_analysis']:
        site = result['site_analysis']
        print(f"\nSite Analysis:")
        print(f"  Width: {site['width']['original_mm']/1000:.1f}m "
              f"({site['width']['grid_count']} grids)")
        print(f"  Depth: {site['depth']['original_mm']/1000:.1f}m "
              f"({site['depth']['grid_count']} grids)")
        print(f"  Area: {site['area_sqm']:.1f}㎡")
        print(f"  Category: {site['size_category']}")
        print(f"  Suggested: {site['recommended_layout']['rooms']}")
    
    print(f"{'='*60}\n")
    
    return result


def main():
    """Process all PDFs in the raw_pdfs directory"""
    
    # Setup directories
    base_dir = Path(__file__).parent.parent
    pdf_dir = base_dir / "data" / "raw_pdfs"
    output_dir = base_dir / "data" / "extracted"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for PDFs
    if not pdf_dir.exists():
        print(f"Error: PDF directory not found: {pdf_dir}")
        print(f"Please place PDF files in: {pdf_dir}")
        return
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in: {pdf_dir}")
        print("Please add the 6 PDF floor plan files to this directory.")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF
    results = []
    for pdf_path in pdf_files:
        try:
            result = process_single_pdf(pdf_path, output_dir)
            if result:
                results.append(result)
        except Exception as e:
            logger.error(f"Error processing {pdf_path.name}: {e}")
            continue
    
    # Save summary
    summary = {
        'total_pdfs': len(pdf_files),
        'successful': len(results),
        'failed': len(pdf_files) - len(results),
        'results': results
    }
    
    summary_file = output_dir / "extraction_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Processing Complete!")
    print(f"Total PDFs: {summary['total_pdfs']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
