import os
import pandas as pd
import shutil

# Path to the CSV file and spectrograms directory
CSV_PATH = 'spectrograms_balanced.csv'
SPECTROGRAMS_DIR = 'spectrograms'
OUTPUT_CSV_PATH = 'spectrograms_balanced_no_sirens.csv'

# Class to be removed (sirens)
CLASS_TO_REMOVE = 'siren'
CLASS_ID_TO_REMOVE = 8

def main():
    print("Reading the CSV file...")
    # Read the CSV file
    df = pd.read_csv(CSV_PATH)
    
    # Print some information about the dataset
    print(f"Original dataset shape: {df.shape}")
    print("\nClass distribution before removal:")
    print(df['class'].value_counts())
    
    # Identify files to delete (all images related to sirens)
    siren_records = df[df['class'] == CLASS_TO_REMOVE]
    print(f"\nFound {len(siren_records)} records of class '{CLASS_TO_REMOVE}' to remove")
    
    # Get unique image filenames to delete
    siren_images = siren_records['spec_file_name'].unique()
    print(f"Found {len(siren_images)} unique image files to delete")
    
    # Delete siren images from all folds
    deleted_count = 0
    for fold_num in range(1, 11):  # folds 1 through 10
        fold_dir = os.path.join(SPECTROGRAMS_DIR, f'fold{fold_num}')
        if os.path.exists(fold_dir):
            print(f"\nProcessing {fold_dir}...")
            fold_deleted = 0
            
            # Find all images in the fold directory
            for filename in os.listdir(fold_dir):
                # Images are named like {fsID}-{classID}-{augmentationType}.png
                # Check if the image name starts with an ID + "-8-" (classID 8 for sirens)
                if "-8-" in filename:
                    file_path = os.path.join(fold_dir, filename)
                    try:
                        os.remove(file_path)
                        fold_deleted += 1
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")
            
            deleted_count += fold_deleted
            print(f"Deleted {fold_deleted} images from fold{fold_num}")
    
    print(f"\nTotal images deleted: {deleted_count}")
    
    # Remove sirens from the DataFrame
    df_cleaned = df[df['class'] != CLASS_TO_REMOVE].copy()
    print(f"\nCleaned dataset shape: {df_cleaned.shape}")
    
    # Create a mapping for the new class IDs
    unique_classes = df_cleaned['class'].unique()
    class_mapping = {}
    old_to_new_class_id = {}
    
    for new_id, class_name in enumerate(sorted(unique_classes)):
        # Get the current class ID for this class
        old_id = df_cleaned[df_cleaned['class'] == class_name]['classID'].iloc[0]
        old_to_new_class_id[old_id] = new_id
        class_mapping[class_name] = new_id
    
    print("\nNew class mapping:")
    for class_name, new_id in class_mapping.items():
        # Find the old ID
        old_id = df_cleaned[df_cleaned['class'] == class_name]['classID'].iloc[0]
        print(f"{class_name}: {old_id} -> {new_id}")
    
    # Update the classID column with new IDs
    df_cleaned['classID'] = df_cleaned.apply(
        lambda row: old_to_new_class_id[row['classID']], 
        axis=1
    )
    
    # Save the cleaned DataFrame to a new CSV file
    df_cleaned.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nCleaned dataset saved to {OUTPUT_CSV_PATH}")
    
    print("\nClass distribution after removal and remapping:")
    print(df_cleaned['class'].value_counts())
    print("\nClass IDs after remapping:")
    print(df_cleaned.groupby(['class', 'classID']).size())

if __name__ == "__main__":
    main()
