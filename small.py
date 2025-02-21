import argparse
import pandas as pd
import sys

def process_sku(sku):
    """Extract numeric part from SKU and validate range"""
    if pd.isna(sku):
        return False
    match = pd.Series(sku).str.extract(r'A\s*(\d+)$', expand=False)
    if match.empty or match[0] is None:
        return False
    return int(match[0]) <= 200

def process_excel(input_file, output_file):
    """Main processing function"""
    try:
        # Read all sheets from input Excel
        sheets = pd.read_excel(input_file, sheet_name=None)
        
        # Process each sheet
        processed_sheets = {}
        for sheet_name, df in sheets.items():
            # Filter rows where SKU is A1-A1000
            mask = df['SKU'].apply(process_sku)
            filtered_df = df[mask].reset_index(drop=True)
            processed_sheets[sheet_name] = filtered_df
        
        # Write to new Excel file
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            for sheet_name, df in processed_sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
        print(f"Successfully created filtered file: {output_file}")
        return 0
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter SKUs A1-A1000 from Excel sheets')
    parser.add_argument('input', help='Input Excel file path')
    parser.add_argument('output', help='Output Excel file path')
    args = parser.parse_args()
    
    sys.exit(process_excel(args.input, args.output))