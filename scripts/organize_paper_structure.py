"""
Organize paper folder structure.

This script moves statistics-related folders and publication figures
under the paper/ directory for better organization.

Usage:
    python scripts/organize_paper_structure.py --dry-run  # Preview changes
    python scripts/organize_paper_structure.py            # Execute moves
"""

import os
import shutil
from pathlib import Path
import argparse
import json
from datetime import datetime


class PaperOrganizer:
    """Reorganize project folders for paper preparation."""
    
    def __init__(self, project_root: str, dry_run: bool = True):
        self.project_root = Path(project_root)
        self.dry_run = dry_run
        self.moves = []
        self.created_dirs = []
        self.errors = []
        
    def log(self, message: str, level: str = "INFO"):
        """Print log message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = "🔍 DRY-RUN" if self.dry_run else "✅ EXECUTE"
        print(f"[{timestamp}] {prefix} [{level}] {message}")
    
    def create_directory(self, path: Path):
        """Create directory if it doesn't exist."""
        if not path.exists():
            self.log(f"Create directory: {path.relative_to(self.project_root)}")
            if not self.dry_run:
                path.mkdir(parents=True, exist_ok=True)
            self.created_dirs.append(str(path.relative_to(self.project_root)))
        return path
    
    def move_folder(self, source: Path, destination: Path, rename: str = None):
        """Move a folder from source to destination."""
        if not source.exists():
            self.log(f"SKIP: Source not found: {source.relative_to(self.project_root)}", "WARNING")
            return False
        
        # Determine final destination
        if rename:
            dest_path = destination / rename
        else:
            dest_path = destination / source.name
        
        # Check if destination already exists
        if dest_path.exists():
            self.log(f"SKIP: Destination exists: {dest_path.relative_to(self.project_root)}", "WARNING")
            self.errors.append(f"Destination exists: {dest_path}")
            return False
        
        self.log(f"Move: {source.relative_to(self.project_root)} → {dest_path.relative_to(self.project_root)}")
        
        self.moves.append({
            'source': str(source.relative_to(self.project_root)),
            'destination': str(dest_path.relative_to(self.project_root)),
            'type': 'folder'
        })
        
        if not self.dry_run:
            try:
                shutil.move(str(source), str(dest_path))
                self.log(f"✓ Moved successfully", "SUCCESS")
            except Exception as e:
                self.log(f"✗ Error: {e}", "ERROR")
                self.errors.append(f"Failed to move {source}: {e}")
                return False
        
        return True
    
    def move_file(self, source: Path, destination: Path, rename: str = None):
        """Move a file from source to destination."""
        if not source.exists():
            self.log(f"SKIP: File not found: {source.relative_to(self.project_root)}", "WARNING")
            return False
        
        # Determine final destination
        if rename:
            dest_path = destination / rename
        else:
            dest_path = destination / source.name
        
        # Check if destination already exists
        if dest_path.exists():
            self.log(f"SKIP: File exists: {dest_path.relative_to(self.project_root)}", "WARNING")
            return False
        
        self.log(f"Move file: {source.name} → {dest_path.relative_to(self.project_root)}")
        
        self.moves.append({
            'source': str(source.relative_to(self.project_root)),
            'destination': str(dest_path.relative_to(self.project_root)),
            'type': 'file'
        })
        
        if not self.dry_run:
            try:
                shutil.move(str(source), str(dest_path))
            except Exception as e:
                self.log(f"✗ Error: {e}", "ERROR")
                self.errors.append(f"Failed to move {source}: {e}")
                return False
        
        return True
    
    def organize_statistics(self):
        """Move statistics folders to paper/statistics/."""
        self.log("\n=== STEP 1: Organize Statistics Folders ===", "INFO")
        
        stats_source = self.project_root / "outputs" / "statistics"
        stats_dest = self.create_directory(self.project_root / "paper" / "statistics")
        
        # Define folder mappings (source name → new name)
        folder_mappings = {
            "Epochs_x_EBF": "epochs_x_ebf",
            "Epochs x EBF": "epochs_x_ebf_v2",
            "Epochs x Feeding": "epochs_x_feeding",
            "Epochs x Length of Stay": "epochs_x_los",
            "M.Age x EBF": "maternal_age_x_ebf",
            "M.Education x EBF": "maternal_education_x_ebf",
            "M.Occupation x EBF": "maternal_occupation_x_ebf",
            "Education x Epochs": "education_x_epochs",
            "ROC curves": "roc_curves"
        }
        
        moved_count = 0
        for old_name, new_name in folder_mappings.items():
            source = stats_source / old_name
            if self.move_folder(source, stats_dest, new_name):
                moved_count += 1
        
        self.log(f"Statistics folders: {moved_count}/{len(folder_mappings)} moved")
        return moved_count
    
    def organize_figures(self):
        """Move publication figures to paper/figures/."""
        self.log("\n=== STEP 2: Organize Publication Figures ===", "INFO")
        
        figures_source = self.project_root / "outputs" / "paper plots"
        figures_dest = self.create_directory(self.project_root / "paper" / "figures")
        figures_data_dest = self.create_directory(figures_dest / "data")
        
        # Define file mappings (source name → new name)
        file_mappings = {
            "bw-median-final.png": "birthweight_by_feeding.png",
            "mm-median-final.png": "breastmilk_by_feeding.png",
            "f-med_only-final.png": "formula_by_feeding.png",
        }
        
        # Data files (CSV)
        data_files = ["breastmilk_iqr.csv", "formula_iqr.csv", "weight_iqr.csv"]
        
        moved_count = 0
        
        # Move renamed publication figures
        for old_name, new_name in file_mappings.items():
            source = figures_source / old_name
            if self.move_file(source, figures_dest, new_name):
                moved_count += 1
        
        # Move data files
        for data_file in data_files:
            source = figures_source / data_file
            if self.move_file(source, figures_data_dest):
                moved_count += 1
        
        self.log(f"Figure files: {moved_count} moved")
        return moved_count
    
    def organize_analyses(self):
        """Move research analyses to paper/analyses/."""
        self.log("\n=== STEP 3: Organize Research Analyses ===", "INFO")
        
        analyses_source = self.project_root / "analyses" / "01_epochs_ebf"
        analyses_dest = self.create_directory(self.project_root / "paper" / "analyses")
        
        moved_count = 0
        if self.move_folder(analyses_source, analyses_dest, "epochs_ebf"):
            moved_count += 1
        
        self.log(f"Analysis folders: {moved_count} moved")
        return moved_count
    
    def create_readme_files(self):
        """Create README.md files in new directories."""
        self.log("\n=== STEP 4: Create README Files ===", "INFO")
        
        readmes = {
            "paper/statistics": """# Statistical Analysis Results

This directory contains all statistical analyses for the manuscript.

## Contents

- `epochs_x_ebf/`: Exclusive breastfeeding rates by study epochs
- `epochs_x_feeding/`: Feeding type distribution by epochs  
- `epochs_x_los/`: Length of stay analysis by epochs
- `maternal_*_x_ebf/`: Maternal factors associated with EBF
- `education_x_epochs/`: Education level across time periods
- `roc_curves/`: ROC curve analyses

## Usage

Each subfolder contains:
- CSV files with statistical test results
- Contingency tables and proportions
- Summary statistics
""",
            "paper/figures": """# Publication Figures

High-resolution figures for manuscript.

## Main Figures

- `birthweight_by_feeding.png`: Birth weight distribution by feeding outcome
- `breastmilk_by_feeding.png`: Breast milk intake by feeding outcome
- `formula_by_feeding.png`: Formula intake by feeding outcome

## Data Files

See `data/` subdirectory for source data (CSV format) used to generate figures.

## Requirements

All figures are 300+ DPI, publication-ready.
""",
            "paper/analyses": """# Research Analyses

Detailed research analyses conducted for the study.

## Contents

- `epochs_ebf/`: Comprehensive analysis of epochs and EBF outcomes

## Organization

Each analysis folder contains:
- Analysis scripts
- Results and outputs
- Documentation
"""
        }
        
        created_count = 0
        for dir_path, content in readmes.items():
            readme_path = self.project_root / dir_path / "README.md"
            
            if not readme_path.exists():
                self.log(f"Create README: {readme_path.relative_to(self.project_root)}")
                if not self.dry_run:
                    readme_path.write_text(content)
                created_count += 1
            else:
                self.log(f"SKIP: README exists: {readme_path.relative_to(self.project_root)}", "WARNING")
        
        self.log(f"README files: {created_count} created")
        return created_count
    
    def generate_summary(self):
        """Generate summary report."""
        self.log("\n" + "="*70)
        self.log("REORGANIZATION SUMMARY")
        self.log("="*70)
        
        print(f"\n📁 Directories created: {len(self.created_dirs)}")
        for d in self.created_dirs:
            print(f"   - {d}")
        
        print(f"\n📦 Items moved: {len(self.moves)}")
        for move in self.moves:
            print(f"   {move['source']} → {move['destination']}")
        
        if self.errors:
            print(f"\n❌ Errors: {len(self.errors)}")
            for error in self.errors:
                print(f"   - {error}")
        else:
            print(f"\n✅ No errors")
        
        # Save summary to file
        if not self.dry_run:
            summary_file = self.project_root / "paper" / "reorganization_summary.json"
            summary = {
                'timestamp': datetime.now().isoformat(),
                'created_directories': self.created_dirs,
                'moves': self.moves,
                'errors': self.errors
            }
            summary_file.write_text(json.dumps(summary, indent=2))
            print(f"\n💾 Summary saved to: {summary_file.relative_to(self.project_root)}")
    
    def run(self):
        """Execute the reorganization."""
        if self.dry_run:
            print("\n" + "="*70)
            print("🔍 DRY-RUN MODE - No files will be moved")
            print("="*70)
        else:
            print("\n" + "="*70)
            print("✅ EXECUTION MODE - Files will be moved")
            print("="*70)
        
        # Execute reorganization steps
        self.organize_statistics()
        self.organize_figures()
        self.organize_analyses()
        self.create_readme_files()
        
        # Generate summary
        self.generate_summary()
        
        if self.dry_run:
            print("\n" + "="*70)
            print("🔍 DRY-RUN COMPLETE")
            print("Run without --dry-run flag to execute moves")
            print("="*70)
        else:
            print("\n" + "="*70)
            print("✅ REORGANIZATION COMPLETE")
            print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Organize paper folder structure")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without executing"
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root directory (default: current directory)"
    )
    
    args = parser.parse_args()
    
    organizer = PaperOrganizer(
        project_root=args.project_root,
        dry_run=args.dry_run
    )
    organizer.run()


if __name__ == "__main__":
    main()
