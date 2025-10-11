import os
import pandas as pd
import argparse
from tqdm import tqdm
import re
from vlmeval.config import supported_VLM
from vlmeval.smp import *

# Your AI detection prompt
# PROMPT_TEXT = (
#     "You are an image classification system. Analyze the image and give a probability score (0 to 100) for it being AI-generated. Always return ONLY a number without any explanation. "
#     "If you are uncertain, make your best estimate. Format: <number> Example: 72"
# )

PROMPT_TEXT = (
    "You are an deepfake detection system. Analyze the image and give a probability score (0.0000 to 1.0000) for it being fake. Always return ONLY a number without any explanation. There are real and fake images. lower score means more likely real, higher score means more likely fake. "
)

# PROMPT_TEXT = (
#     "Analyze the image and give a probability score (0.0000 to 1.0000) for it being fake. Always return ONLY a number. "
# )


def extract_score(response):
    """Extract numerical score from model response"""
    # Look for numbers in the response
    numbers = re.findall(r'\d+', str(response))
    if numbers:
        score = int(numbers[0])
        return min(max(score, 0), 100)  # Clamp between 0-100
    return None

def parse_args():
    parser = argparse.ArgumentParser(description='Run VLM inference on images from CSV')
    
    # CSV file arguments
    parser.add_argument('--csv', type=str, default='/mnt/ssd/project/lilin/politic_deepfakes/brightness_contrast_video.csv', 
                        help='Path to CSV file containing image paths')
    parser.add_argument('--image-column', type=str, default='Image Path',
                        help='Name of column containing image paths')
    parser.add_argument('--output', type=str, default='brightness_contrast_video_mPLUG-Owl2.csv',
                        help='Output CSV file path')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='mPLUG-Owl2',
                        help='Model name (e.g., llava_v1_5_7b, llava_v1_5_13b, etc.)')
    
    # Optional arguments
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for processing (currently only 1 supported)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed information')
    parser.add_argument('--work-dir', type=str, default='./csv_inference_outputs',
                        help='Working directory for outputs')
    
    return parser.parse_args()

def validate_csv(csv_path, image_column):
    """Validate CSV file and image column"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if image_column not in df.columns:
        raise ValueError(
            f"Column '{image_column}' not found in CSV. "
            f"Available columns: {df.columns.tolist()}"
        )
    
    return df

def validate_image_paths(df, image_column, verbose=False):
    """Check which image paths exist"""
    missing_images = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        image_path = row[image_column]
        if pd.isna(image_path):
            missing_images.append((idx, "NaN value"))
        elif not os.path.exists(image_path):
            missing_images.append((idx, image_path))
        else:
            valid_indices.append(idx)
    
    if verbose and missing_images:
        print(f"\nWarning: {len(missing_images)} images not found:")
        for idx, path in missing_images[:10]:  # Show first 10
            print(f"  Row {idx}: {path}")
        if len(missing_images) > 10:
            print(f"  ... and {len(missing_images) - 10} more")
    
    return valid_indices, missing_images

from PIL import Image

def preprocess_image_for_model(image_path, target_size=224):
    """
    Preprocess image to avoid dimension mismatches
    XComposer2 typically uses 448x448 images
    """
    try:
        img = Image.open(image_path).convert('RGB')
        
        # Resize image maintaining aspect ratio
        img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
        
        # Create square canvas with padding
        canvas = Image.new('RGB', (target_size, target_size), (255, 255, 255))
        offset = ((target_size - img.size[0]) // 2, (target_size - img.size[1]) // 2)
        canvas.paste(img, offset)
        
        # Save to temporary file
        temp_path = image_path.rsplit('.', 1)[0] + '_processed.jpg'
        canvas.save(temp_path, 'JPEG', quality=95)
        
        return temp_path
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {str(e)}")
        return image_path  # Return original if preprocessing fails

def load_model(model_name, verbose=False):
    """Load VLM model"""
    if model_name not in supported_VLM:
        raise ValueError(
            f"Model '{model_name}' not supported. "
            f"Available models: {list(supported_VLM.keys())}"
        )
    
    if verbose:
        print(f"\nLoading model: {model_name}")
        print("This may take a few minutes...")
    
    model = supported_VLM[model_name]()
    
    # model = supported_VLM[model_name](root='/mnt/ssd/project/lilin/politic_deepfakes/AI-Face-FairnessBench-main/Forensics-Bench/Yi')
    if verbose:
        print(f"✓ Model {model_name} loaded successfully")
    
    return model

def process_csv_images(args):
    """Main function to process images from CSV"""
    logger = get_logger('CSV_INFERENCE')
    
    # Step 1: Load and validate CSV
    logger.info(f"Loading CSV file: {args.csv}")
    df = validate_csv(args.csv, args.image_column)
    logger.info(f"✓ CSV loaded with {len(df)} rows")
    
    # Step 2: Validate image paths
    logger.info("Validating image paths...")
    valid_indices, missing_images = validate_image_paths(
        df, args.image_column, verbose=args.verbose
    )
    logger.info(f"✓ Found {len(valid_indices)} valid images, {len(missing_images)} missing")
    
    if len(valid_indices) == 0:
        logger.error("No valid images found. Exiting.")
        return
    
    # Step 3: Load model
    logger.info(f"Loading model: {args.model}")
    model = load_model(args.model, verbose=args.verbose)
    
    # Step 4: Create output directory
    os.makedirs(args.work_dir, exist_ok=True)
    
    # Step 5: Process images
    logger.info(f"Processing {len(valid_indices)} images...")
    
    # Initialize results columns
    df['ai_probability_score'] = None
    df['prob_fake'] = None
    df['processing_status'] = 'not_processed'
    
    # Process each valid image
    # for idx in tqdm(valid_indices, desc="Processing images"):
    #     try:
    #         image_path = df.loc[idx, args.image_column]
            
    #         if args.verbose:
    #             logger.info(f"\nProcessing row {idx}: {image_path}")

    #         # Preprocess image to handle dimension issues
    #         processed_path = preprocess_image_for_model(image_path)
            
    #         # Generate prediction using the model
    #         # response = model.generate([image_path, PROMPT_TEXT])
    #         response = model.generate([processed_path, PROMPT_TEXT])
    #         print(response)
            
    #         if args.verbose:
    #             logger.info(f"Raw response: {response}")
            
    #         # Extract score from response
    #         score = extract_score(response)
            
    #         # Store results
    #         df.loc[idx, 'prob_fake'] = str(response)
    #         df.loc[idx, 'ai_probability_score'] = score
    #         df.loc[idx, 'processing_status'] = 'success'
            
    #         if args.verbose:
    #             logger.info(f"Extracted score: {score}")
                
    #     except Exception as e:
    #         logger.error(f"Error processing row {idx}: {str(e)}")
    #         df.loc[idx, 'processing_status'] = f'error: {str(e)}'
    #         df.loc[idx, 'prob_fake'] = None
    #         df.loc[idx, 'ai_probability_score'] = None
    # Process each valid image
    for idx in tqdm(valid_indices, desc="Processing images"):
        image_path = None
        processed_path = None
        
        try:
            image_path = df.loc[idx, args.image_column]
            
            if args.verbose:
                logger.info(f"\nProcessing row {idx}: {image_path}")
            
            # Preprocess image to handle dimension issues (if you want to keep using it)
            # processed_path = preprocess_image_for_model(image_path)
            
            # Generate prediction using the model
            response = model.generate([image_path, PROMPT_TEXT])
            print(response)
            
            if args.verbose:
                logger.info(f"Raw response: {response}")
            
            # Extract score from response
            score = extract_score(response)
            
            # Store results
            df.loc[idx, 'prob_fake'] = str(response)
            df.loc[idx, 'ai_probability_score'] = score
            df.loc[idx, 'processing_status'] = 'success'
            
            if args.verbose:
                logger.info(f"Extracted score: {score}")
        
        except RuntimeError as e:
            error_msg = str(e)
            if "size of tensor" in error_msg:
                logger.warning(f"Row {idx}: Skipping due to tensor dimension mismatch - {error_msg}")
                df.loc[idx, 'processing_status'] = 'error: tensor_dimension_mismatch'
                df.loc[idx, 'prob_fake'] = None
                df.loc[idx, 'ai_probability_score'] = None
            else:
                logger.error(f"RuntimeError processing row {idx}: {error_msg}")
                df.loc[idx, 'processing_status'] = f'error: {error_msg}'
                df.loc[idx, 'prob_fake'] = None
                df.loc[idx, 'ai_probability_score'] = None
        
        except Exception as e:
            error_msg = str(e)
            # Double-check if it's a tensor size error that got wrapped
            if "size of tensor" in error_msg:
                logger.warning(f"Row {idx}: Skipping due to tensor dimension mismatch - {error_msg}")
                df.loc[idx, 'processing_status'] = 'error: tensor_dimension_mismatch'
            else:
                logger.error(f"Error processing row {idx}: {error_msg}")
                df.loc[idx, 'processing_status'] = f'error: {error_msg}'
            
            df.loc[idx, 'prob_fake'] = None
            df.loc[idx, 'ai_probability_score'] = None
        
    # Mark missing images
    for idx, _ in missing_images:
        df.loc[idx, 'processing_status'] = 'image_not_found'
    
    # Step 6: Save results
    logger.info(f"Saving results to: {args.output}")
    df.to_csv(args.output, index=False)
    logger.info("✓ Results saved successfully")
    
    # Step 7: Print summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total rows: {len(df)}")
    print(f"Successfully processed: {(df['processing_status'] == 'success').sum()}")
    print(f"Failed: {(df['processing_status'].str.startswith('error')).sum()}")
    print(f"Images not found: {(df['processing_status'] == 'image_not_found').sum()}")
    print(f"Not processed: {(df['processing_status'] == 'not_processed').sum()}")
    
    # Print score statistics
    valid_scores = df[df['ai_probability_score'].notna()]['ai_probability_score']
    if len(valid_scores) > 0:
        print("\nAI PROBABILITY SCORE STATISTICS")
        print(f"Average: {valid_scores.mean():.2f}")
        print(f"Median: {valid_scores.median():.2f}")
        print(f"Min: {valid_scores.min()}")
        print(f"Max: {valid_scores.max()}")
        print(f"Std Dev: {valid_scores.std():.2f}")
    
    print("="*60)
    print(f"\nResults saved to: {args.output}")

def main():
    """Entry point"""
    load_env()
    args = parse_args()
    
    try:
        process_csv_images(args)
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == '__main__':
    main()