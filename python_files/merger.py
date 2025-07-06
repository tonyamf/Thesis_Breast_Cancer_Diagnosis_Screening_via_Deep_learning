# combine_files.py
input_files = [
    "app.py",
    "utils/augmentation.py",
    "utils/metrics.py",
    "utils/preprocessing.py",
    "utils/visualization.py",
    "training/train_mpl.py",
    "training/train_segmentation.py",
    "training/train_classifier.py",
    "models/unet.py",
    "models/efficientnet.py",
    "models/meta_pseudo_labels.py",
]
output_file = "combined_output.py"

with open(output_file, 'w', encoding='utf-8') as outfile:
    for path in input_files:
        # Write header with file path
        header = f"""
# {'=' * 70}
# File: {path}
# {'=' * 70}\n\n
"""
        outfile.write(header)
        
        # Read with encoding handling
        try:
            # Try UTF-8 first, then fallback to latin-1 with error replacement
            try:
                with open(path, 'r', encoding='utf-8') as infile:
                    content = infile.read()
            except UnicodeDecodeError:
                with open(path, 'r', encoding='latin-1', errors='replace') as infile:
                    content = infile.read()
            
            outfile.write(content + '\n\n')
        except FileNotFoundError:
            outfile.write(f"# ERROR: File not found at {path}\n\n")
        except Exception as e:
            outfile.write(f"# ERROR: Could not read {path} - {str(e)}\n\n")

print("Combined file created:", output_file)